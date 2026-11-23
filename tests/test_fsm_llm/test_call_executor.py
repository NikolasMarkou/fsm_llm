"""The call executor, Pass 1 of a ``call`` state (plan baa37f40, step 11, D-022, roadmap E3.2).

The executor makes no model call. It resolves the target through the pipeline's resolver and
either refuses (a FINAL ``failed`` record, so the same turn takes the failure edge) or writes the
``running`` request record as handler-only turn data, which rolls back with the turn like every
other Pass-1 write. It reads only the declared inputs. What happens to a ``running`` record
(the push, the pop, the merge) is settle's and is proven in ``test_call_settle.py``; every row
here reads the PARENT frame, which is the same before and after settle exists. Offline: a
scripted model and a one-class resolver.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, FSMManager
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMDefinition,
    FSMError,
    MissingExecutorError,
    handler_only_names,
    result_key_names,
)
from fsm_llm.core.state_kinds import CompletionStateKind, kind_of
from fsm_llm.engine.kinds import executor_for
from fsm_llm.engine.kinds.call import CallExecutor, refuse, write_call_request
from fsm_llm.engine.kinds.targets import TARGET_MODE_FRAME, ResolvedTarget
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_call_state import _call, _choose, _parent

_TARGET = "reasoning.analytical"


def _child() -> dict[str, Any]:
    """Two silent states (no greeting reaches the model): ``c1`` then the terminal ``c2``."""
    return kc._fsm(
        "Child",
        "c1",
        kc._state("c1", response_instructions="", transitions=[kc._go("c2")]),
        kc._state("c2", response_instructions=""),
    )


class _Resolver:
    """Answers one child definition for the ids it knows and records every question."""

    def __init__(self, *known: str) -> None:
        self.known = known or (_TARGET,)
        self.asked: list[tuple[str, str | None]] = []
        self.definition = FSMDefinition(**_child())

    def resolve_target(self, target_id: str, *, conversation_id: str | None = None):
        self.asked.append((target_id, conversation_id))
        if target_id not in self.known:
            return None
        return ResolvedTarget(
            self.definition, f"template:{target_id}", f"instance:{target_id}"
        )


async def _at_the_call(
    config: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
    resolver: _Resolver | None = None,
    llm: ScriptedLLM | None = None,
    **state_overrides: Any,
):
    """A parent API that has moved from ``start`` into the call state ``work`` (the next step
    runs the executor). Returns ``(api, cid, frame, resolver, llm)``; ``frame`` is the id of
    the PARENT frame, so a row reads the same conversation before and after settle."""
    resolver = _Resolver() if resolver is None else resolver
    llm = ScriptedLLM(default_response="r") if llm is None else llm
    state_overrides.setdefault(
        "response_instructions", ""
    )  # silent: `None` is not silent
    api = API.from_definition(
        _parent(config, **state_overrides), llm_interface=llm, targets=resolver
    )
    cid, _ = await api.astart_conversation({"topic": "why", **(context or {})})
    frame = api.conversation_stacks[cid][0].conversation_id
    await api.aadvance(cid)  # start -> work: the arrival runs nothing
    assert api.fsm_manager.get_conversation_state(frame) == "work"
    assert "call_result" not in api.fsm_manager.copy_raw_context_data(frame)
    return api, cid, frame, resolver, llm


def _record(api, frame) -> Any:
    return api.fsm_manager.copy_raw_context_data(frame).get("call_result")


class TestTheRequestRecord:
    async def test_a_request_is_a_running_record_with_both_identities(self):
        """Kills: a record with another shape, missing identities or a missing ``mode``."""
        api, cid, frame, resolver, _ = await _at_the_call()
        await api.aadvance(cid)
        record = _record(api, frame)
        assert record == {
            "status": "running",
            "target": _TARGET,
            "template_identity": f"template:{_TARGET}",
            "instance_identity": f"instance:{_TARGET}",
            "mode": "frame",
            "request": {
                "target": _TARGET,
                "inputs": {"topic": "why"},
                "max_steps": None,
                "max_seconds": None,
            },
        }
        # the executor asks first; settle (step 12) asks again when it pushes the child
        assert resolver.asked[0] == (_TARGET, frame)
        # the call stays in its state: no edge passes while the status is `running`
        assert api.fsm_manager.get_conversation_state(frame) == "work"

    async def test_the_budgets_of_the_call_are_in_the_request(self):
        api, cid, frame, *_ = await _at_the_call(_call(max_steps=3, max_seconds=2.5))
        await api.aadvance(cid)
        request = _record(api, frame)["request"]
        assert (request["max_steps"], request["max_seconds"]) == (3, 2.5)

    async def test_the_record_holds_only_the_declared_inputs(self):
        """The parent holds an internal grant, a secret-looking entry and plain extras; only
        the declared name reaches the record. Kills (D3): a ``request`` built from the whole
        context."""
        api, cid, frame, *_ = await _at_the_call(
            context={
                "_approval_granted": {"call": "x"},
                "api_key": "sk-secret-value",
                "unrelated": "do not copy me",
            }
        )
        await api.aadvance(cid)
        record = _record(api, frame)
        assert set(record["request"]["inputs"]) == {"topic"}
        text = repr(record)
        for leaked in ("_approval_granted", "sk-secret-value", "do not copy me"):
            assert leaked not in text

    async def test_a_large_input_is_cut_in_the_record_and_not_in_the_context(self):
        big = "x" * 50_000
        api, cid, frame, *_ = await _at_the_call()
        # set after the greeting: a prompt over 30000 characters is refused before any call
        api.fsm_manager.update_conversation_context(frame, {"topic": big})
        await api.aadvance(cid)
        raw = api.fsm_manager.copy_raw_context_data(frame)
        assert raw["topic"] == big
        shown = raw["call_result"]["request"]["inputs"]["topic"]
        assert isinstance(shown, str) and len(shown) < 200
        assert shown.startswith("<str, ") and "not recorded" in shown

    async def test_a_non_json_input_is_recorded_as_a_marker(self):
        api, cid, frame, *_ = await _at_the_call()
        api.fsm_manager.update_conversation_context(frame, {"topic": {1, 2}})
        await api.aadvance(cid)
        shown = _record(api, frame)["request"]["inputs"]["topic"]
        assert shown == "<set>"

    async def test_the_executor_makes_no_model_call(self):
        """The call turn sends nothing to the model (the call state is silent). Kills: an
        executor that completes, classifies or extracts."""
        api, cid, _frame, _, llm = await _at_the_call()
        before = list(llm.call_history)
        await api.aadvance(cid)
        assert list(llm.call_history) == before

    async def test_a_record_that_stands_is_never_rewritten_or_resolved_again(self):
        """Skip-if-set: a record of any status is final for the executor and the resolver is
        not asked. Kills: an executor that re-resolves or re-writes every turn."""
        api, cid, frame, resolver, _ = await _at_the_call()
        standing = {
            "status": "ok",
            "reason": None,
            "target": _TARGET,
            "template_identity": "marker",
            "instance_identity": "marker",
        }
        api.fsm_manager.update_conversation_context(frame, {"call_result": standing})
        step = await api.aadvance(cid)
        assert resolver.asked == [] and _record(api, frame) == standing
        assert (
            step.state_after == "done"
        )  # the success edge fired on the standing record


class TestChoose:
    @pytest.mark.parametrize(
        ("picked", "fallback", "want"),
        [
            ("reasoning.deductive", None, "reasoning.deductive"),
            ("reasoning.analytical", "reasoning.deductive", "reasoning.analytical"),
            ("not.in.the.list", "reasoning.deductive", "reasoning.deductive"),
            (None, "reasoning.deductive", "reasoning.deductive"),
            (["reasoning.analytical"], "reasoning.deductive", "reasoning.deductive"),
        ],
    )
    async def test_the_listed_value_or_the_fallback_is_called(
        self, picked, fallback, want
    ):
        config = {k: v for k, v in _call().items() if k != "target"}
        config["choose"] = _choose(**({"fallback": fallback} if fallback else {}))
        resolver = _Resolver("reasoning.analytical", "reasoning.deductive")
        context = {} if picked is None else {"picked": picked}
        api, cid, frame, *_ = await _at_the_call(
            config, context=context, resolver=resolver
        )
        await api.aadvance(cid)
        record = _record(api, frame)
        assert record["status"] == "running" and record["target"] == want
        assert resolver.asked[0] == (want, frame)
        assert {t for t, _ in resolver.asked} == {want}  # never another id

    async def test_a_value_outside_the_list_with_no_fallback_fails_the_call(self):
        """Kills: an executor that calls whatever the key holds."""
        config = {k: v for k, v in _call().items() if k != "target"}
        config["choose"] = _choose()
        resolver = _Resolver("reasoning.analytical", "reasoning.deductive", "evil.id")
        api, cid, frame, *_ = await _at_the_call(
            config, context={"picked": "evil.id"}, resolver=resolver
        )
        step = await api.aadvance(cid)
        assert resolver.asked == []  # an id outside the list is never even resolved
        assert _record(api, frame)["reason"] == "unknown_target"
        assert step.state_after == "apology"


class TestRefusals:
    @pytest.mark.parametrize(
        ("config", "context", "resolver", "reason", "target", "identity"),
        [
            # an id the resolver does not know
            (
                _call(target="no.such.id"),
                {},
                _Resolver(),
                "unknown_target",
                "no.such.id",
                None,
            ),
            # a declared input the context does not hold
            (
                _call(inputs=["absent"]),
                {},
                _Resolver(),
                "inputs_missing",
                _TARGET,
                "set",
            ),
            # a declared input that is None
            (_call(), {"topic": None}, _Resolver(), "inputs_missing", _TARGET, "set"),
        ],
        ids=["unknown-target", "input-absent", "input-none"],
    )
    async def test_each_refusal_lands_in_the_same_turn_and_takes_the_failure_edge(
        self, config, context, resolver, reason, target, identity
    ):
        """The final record is written in Pass 1, so the same turn's evaluation sees it.
        Kills: a refusal that waits for the next turn, one that stays, a wrong reason."""
        api, cid, frame, *_ = await _at_the_call(
            config, context=context, resolver=resolver
        )
        step = await api.aadvance(cid)
        record = _record(api, frame)
        assert record == {
            "status": "failed",
            "reason": reason,
            "target": target,
            "template_identity": None if identity is None else f"template:{target}",
            "instance_identity": None if identity is None else f"instance:{target}",
        }
        assert (step.state_before, step.state_after) == ("work", "apology")
        assert api.get_stack_depth(cid) == 1  # a refusal pushes nothing

    async def test_a_refusal_is_final_and_the_next_visit_skips_it(self):
        api, cid, frame, resolver, _ = await _at_the_call(_call(target="no.such.id"))
        await api.aadvance(cid)
        assert resolver.asked == [("no.such.id", frame)]


class TestRollback:
    async def test_a_pass_two_failure_after_the_record_rolls_it_back(self):
        """The call state speaks, its Pass 2 fails, the whole turn is restored: no record, no
        resolver trace in the context. Kills: a record committed outside the turn snapshot."""
        api, cid, frame, _resolver, llm = await _at_the_call(
            response_instructions="Working on it."
        )
        with patch.object(llm, "generate_response", side_effect=RuntimeError("down")):
            with pytest.raises(FSMError):
                await api.aadvance(cid)
        assert "call_result" not in api.fsm_manager.copy_raw_context_data(frame)
        assert api.fsm_manager.get_conversation_state(frame) == "work"
        # the retry writes it again
        await api.aadvance(cid)
        assert _record(api, frame)["status"] == "running"


class TestTheWiring:
    def test_the_call_kind_has_its_executor(self):
        state = FSMDefinition(**_parent()).states["work"]
        assert kind_of(state).name == CompletionStateKind.CALL.value == "call"
        executor = executor_for(state)
        assert isinstance(executor, CallExecutor)
        assert executor.kind == "call" and executor.cleans_extracted is False

    def test_a_bare_manager_with_no_resolver_refuses_at_the_first_pass_one(self):
        """The call executor refuses itself when the pipeline holds no resolver (a bare
        ``FSMManager`` has no construction site). Kills: a silent skip."""
        definition = _parent()
        manager = FSMManager(
            fsm_loader=lambda fsm_id: FSMDefinition(**definition),
            llm_interface=ScriptedLLM(default_response="r"),
        )
        cid, _ = manager.start_conversation("any", initial_context={"topic": "t"})
        manager.advance(cid)  # start -> work
        with pytest.raises(MissingExecutorError, match="work"):
            manager.advance(cid)

    def test_the_two_shared_bodies_return_one_key(self):
        """``write_call_request`` and ``refuse`` are what ``SynthesizeExecutor`` reuses (step
        20): each returns a response with exactly the result key."""
        resolved = ResolvedTarget(
            FSMDefinition(**_child()), "template", "instance", TARGET_MODE_FRAME
        )
        request = write_call_request(
            "out",
            target="t",
            resolved=resolved,
            inputs={"a": 1},
            max_steps=2,
            max_seconds=None,
        )
        assert isinstance(request, DataExtractionResponse)
        assert request.confidence == 1.0 and list(request.extracted_data) == ["out"]
        assert request.extracted_data["out"]["request"]["inputs"] == {"a": 1}
        refused = refuse("out", "inputs_missing", target="t", resolved=resolved)
        assert refused.extracted_data == {
            "out": {
                "status": "failed",
                "reason": "inputs_missing",
                "target": "t",
                "template_identity": "template",
                "instance_identity": "instance",
            }
        }
        bare = refuse("out", "unknown_target")
        assert bare.extracted_data["out"]["target"] is None

    async def test_a_call_result_is_closed_to_every_extraction_channel(self):
        """The framework result key is handler-only like ``synthesis_result``
        (``result_key_names``): a conversation state that declares a field of that name gets
        nothing written, though the model answers it. Kills: a call result an extraction can
        forge."""
        definition = _parent()
        assert result_key_names(FSMDefinition(**definition)) == {"call_result"}
        assert "call_result" in handler_only_names(FSMDefinition(**definition))
        definition["states"]["start"].update(
            required_context_keys=["call_result"],
            extraction_instructions="Extract the call result.",
        )
        llm = ScriptedLLM(
            default_response="r", extraction_data={"call_result": {"status": "ok"}}
        )
        api = API.from_definition(definition, llm_interface=llm, targets=_Resolver())
        cid, _ = await api.astart_conversation({"topic": "t"})
        await api.aadvance(cid)
        frame = api.conversation_stacks[cid][0].conversation_id
        assert "call_result" not in api.fsm_manager.copy_raw_context_data(frame)
