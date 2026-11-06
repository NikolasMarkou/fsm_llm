"""Seeding an internal-prefix key does not warn (plan 0068b055, step 12.29).

Live evidence (findings/live-runs-3.md Defect 4): with library logging on, every
agent run printed ``Context update contains internal-prefix key:
'_max_iterations'``. The warning sat in ``FSMContext.update``. Every caller of
``update`` that can reach it writes a key BY NAME through a public write path:

* ``FSMManager.start_conversation`` seeding ``initial_context`` (agents seed
  their own ``_max_iterations`` etc.; ``push_fsm`` seeds inherited context);
* ``FSMManager.update_conversation_context`` (``API.update_context``: the HITL
  grant, the reasoning push flag, the meta-builder artifact).

The two ACCIDENTAL write paths never reach ``update``'s warning: an extraction
result goes through ``clean_context_keys`` (internal keys dropped) and a handler
delta is merged by ``merge_delta`` straight into ``context.data`` (reserved
keys are refused there, with their own WARNING). So the line was noise on every
documented path and a signal on none. These tests pin that, and pin the two
places where the signal really lives.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from fsm_llm import API
from fsm_llm.constants import RESERVED_CONTEXT_KEYS
from fsm_llm.definitions import FSMContext
from fsm_llm.handlers import HandlerTiming, create_handler
from tests.conftest import MockLLM2Interface


def _fsm() -> dict[str, Any]:
    return {
        "name": "seed",
        "description": "seeding probe",
        "initial_state": "a",
        "persona": "p",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "ask",
                "response_instructions": "Reply.",
                "field_extractions": [
                    {
                        "field_name": "mood",
                        "field_type": "str",
                        "extraction_instructions": "The user's mood.",
                    }
                ],
                "transitions": [{"target_state": "b", "description": "go"}],
            },
            "b": {
                "id": "b",
                "description": "b",
                "purpose": "end",
                "response_instructions": "Bye.",
            },
        },
    }


def _warnings(run: Any) -> list[str]:
    """Run ``run()`` with the library logger on; return the WARNING texts."""
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    logger.enable("fsm_llm")
    try:
        run()
    finally:
        logger.disable("fsm_llm")
        logger.remove(sink)
    return records


def _internal_warnings(records: list[str]) -> list[str]:
    return [r for r in records if "internal-prefix" in r]


def _api(**kw: Any) -> API:
    return API.from_definition(
        _fsm(), llm_interface=MockLLM2Interface(extraction_data={"mood": "ok"}), **kw
    )


class TestSeedingIsQuiet:
    def test_start_conversation_with_internal_initial_context(self):
        api = _api()
        holder: dict[str, str] = {}

        def run() -> None:
            holder["id"], _ = api.start_conversation(
                initial_context={"_max_iterations": 7, "system_note": "x", "ok": 1}
            )

        assert _internal_warnings(_warnings(run)) == []
        raw = api.fsm_manager.copy_raw_context_data(
            api.get_sub_conversation_id(holder["id"])
        )
        assert raw is not None
        assert raw["_max_iterations"] == 7  # seeded, not dropped
        assert raw["system_note"] == "x"

    def test_update_context_with_an_internal_key(self):
        api = _api()
        cid, _ = api.start_conversation()

        warnings = _warnings(lambda: api.update_context(cid, {"_flag": True}))

        assert _internal_warnings(warnings) == []
        raw = api.fsm_manager.copy_raw_context_data(api.get_sub_conversation_id(cid))
        assert raw is not None and raw["_flag"] is True

    def test_push_fsm_seeding_inherited_internal_keys(self):
        api = _api()
        cid, _ = api.start_conversation(initial_context={"_max_iterations": 3})

        def run() -> None:
            api.push_fsm(cid, _fsm(), inherit_context=True)

        assert _internal_warnings(_warnings(run)) == []

    def test_the_context_method_itself_is_quiet(self):
        ctx = FSMContext()
        assert _internal_warnings(_warnings(lambda: ctx.update({"_k": 1}))) == []
        assert ctx.data["_k"] == 1


class TestTheSignalStaysWhereItIs:
    """The accidental write paths: unchanged by the removal above."""

    def test_a_handler_overwriting_a_reserved_key_still_warns(self):
        api = _api()
        handler = (
            create_handler("clobber")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: {"_conversation_id": "other"})
        )
        api.register_handler(handler)
        cid, _ = api.start_conversation()
        assert "_conversation_id" in RESERVED_CONTEXT_KEYS

        records = _warnings(lambda: api.converse("hi", cid))

        assert any("reserved context key '_conversation_id'" in r for r in records)

    def test_a_handler_owned_internal_key_still_merges(self):
        api = _api()
        handler = (
            create_handler("owner")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: {"_replan_count": 2})
        )
        api.register_handler(handler)
        cid, _ = api.start_conversation()

        records = _warnings(lambda: api.converse("hi", cid))

        assert _internal_warnings(records) == []
        raw = api.fsm_manager.copy_raw_context_data(api.get_sub_conversation_id(cid))
        assert raw is not None and raw["_replan_count"] == 2

    def test_an_extracted_internal_key_is_dropped_not_stored(self):
        llm = MockLLM2Interface(extraction_data={"mood": "ok", "_planted": "x"})
        api = API.from_definition(_fsm(), llm_interface=llm)
        cid, _ = api.start_conversation()

        records = _warnings(lambda: api.converse("hi", cid))

        assert _internal_warnings(records) == []
        raw = api.fsm_manager.copy_raw_context_data(api.get_sub_conversation_id(cid))
        assert raw is not None and "_planted" not in raw
