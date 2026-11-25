"""The purpose of a model call and usage by purpose (plan 3f4530d3 step 6, D-011).

``purpose`` is a ContextVar read where the events are written. The rows pin: every
``llm/request``, ``llm/response`` and ``provider/error`` of a scripted run carries one (the
tie-break is ``tie_break`` while the same ``classification`` call type through
``classification_extractions`` is ``classify``), a scope wins over the call type table and the
table never leaves a call unlabeled, the purpose reaches no provider request byte, a scope is
never held across a ``yield``, and ``usage_by_purpose`` is the same fold as ``usage_from_events``
keyed by purpose.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API
from fsm_llm.agents.composition import _default_complete
from fsm_llm.constants import USAGE_KIND_BY_CALL_TYPE
from fsm_llm.context.compactor import ContextCompactor
from fsm_llm.core.codes import Purpose
from fsm_llm.core.definitions import (
    CompletionRequest,
    Conversation,
    LLMResponseError,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.llm.purpose import (
    PURPOSE_BY_CALL_TYPE,
    current_purpose,
    purpose_of,
    purpose_scope,
    resolve_purpose,
)
from fsm_llm.runtime import EventLog, bind
from fsm_llm.runtime.usage import usage_by_purpose, usage_from_events
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm.event_log_support import greeting_request as _request
from tests.test_fsm_llm.event_log_support import usage_reply as _reply
from tests.test_fsm_llm.support import _aiter1_ambiguous_fsm
from tests.test_fsm_llm_regression.support import REPO

ROOT = "root-1"
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_VALUES = {p.value for p in Purpose}


def _events(log: EventLog, cid: str, *types: str) -> list[dict[str, Any]]:
    return list(log.query(cid, types=types or None))


def _classify(intent: str) -> Any:
    return final_reply(
        json.dumps({"reasoning": "r", "intent": intent, "confidence": 0.9})
    )


def _scripted_run(log: EventLog) -> str:
    """One greeting, one turn with bulk-free field extraction, a classification extraction
    and an AMBIGUOUS tie (the three ``classification`` call sites share one call type)."""
    fsm = _aiter1_ambiguous_fsm()
    hub = fsm["states"]["hub"]
    hub["required_context_keys"] = ["topic"]
    hub["classification_extractions"] = [
        {
            "field_name": "mood",
            "intents": [
                {"name": "calm", "description": "calm"},
                {"name": "upset", "description": "upset"},
            ],
            "fallback_intent": "calm",
            "confidence_threshold": 0.5,
        }
    ]
    llm = ScriptedLLM(
        [_classify("calm"), _classify("billing")],
        extraction_data={"topic": "x"},
        responses={"hub": "hi", "billing": "bill"},
    )
    api = API(fsm, llm_interface=llm, event_log=log)
    cid, _ = api.start_conversation()
    api.converse("hello", cid)
    return cid


class TestEveryCallSiteIsLabeled:
    def test_the_events_of_a_scripted_run_carry_the_expected_purposes(
        self, log: EventLog
    ) -> None:
        cid = _scripted_run(log)
        requests = _events(log, cid, "llm/request")
        assert [(e["call_type"], e["purpose"]) for e in requests] == [
            ("greeting", "reply"),
            ("field_extraction", "extract"),
            ("classification", "classify"),
            ("classification", "tie_break"),
            ("response_generation", "reply"),
        ]

    def test_a_response_and_an_error_carry_the_purpose_of_their_request(
        self, log: EventLog
    ) -> None:
        cid = _scripted_run(log)
        by_seq = {e["seq"]: e for e in _events(log, cid, "llm/request")}
        responses = _events(log, cid, "llm/response")
        assert len(responses) == len(by_seq)
        for response in responses:
            assert response["purpose"] == by_seq[response["request_seq"]]["purpose"]
        assert {e["purpose"] for e in _events(log, cid, "llm/response")} <= _VALUES

    def test_a_failing_engine_call_writes_the_purpose_on_the_error(
        self, log: EventLog
    ) -> None:
        class Down(ScriptedLLM):
            def generate_response(self, request: Any) -> Any:
                raise LLMResponseError("down")

        llm = Down([], responses={"hub": "hi"})
        api = API(_aiter1_ambiguous_fsm(), llm_interface=llm, event_log=log)
        with pytest.raises(Exception):
            api.start_conversation()
        errors = list(log.query(next(iter(_roots(log))), types=["provider/error"]))
        assert [(e["call_type"], e["purpose"]) for e in errors] == [
            ("greeting", "reply")
        ]

    def test_a_completion_call_is_complete_and_the_provider_path_is_labeled(
        self, log: EventLog
    ) -> None:
        llm = LiteLLMInterface(model="gpt-4o")
        with bind(log, ROOT), patch(COMPLETION, lambda **kw: _reply()):
            llm.complete(CompletionRequest(messages=[{"role": "user", "content": "x"}]))
            llm.generate_response(_request())
        events = _events(log, ROOT, "llm/request", "llm/response")
        assert [(e["call_type"], e["purpose"]) for e in events] == [
            ("completion", "complete"),
            ("completion", "complete"),
            ("response_generation", "reply"),
            ("response_generation", "reply"),
        ]

    def test_a_scope_wins_over_the_call_type_on_request_response_and_error(
        self, log: EventLog
    ) -> None:
        llm = LiteLLMInterface(model="gpt-4o")

        def down(**kwargs: Any) -> Any:
            raise RuntimeError("down")

        with bind(log, ROOT):
            with purpose_scope(Purpose.JUDGE), patch(COMPLETION, lambda **kw: _reply()):
                llm.generate_response(_request())
            with purpose_scope(Purpose.COMPILE), patch(COMPLETION, down):
                with pytest.raises(LLMResponseError):
                    llm.generate_response(_request())
        assert [e["purpose"] for e in _events(log, ROOT)] == [
            "judge",
            "judge",
            "compile",
            "compile",
        ]

    def test_a_scope_is_read_when_the_call_starts_not_when_the_reply_arrives(
        self, log: EventLog
    ) -> None:
        llm = LiteLLMInterface(model="gpt-4o")
        with bind(log, ROOT), patch(COMPLETION, lambda **kw: _reply()):
            with purpose_scope(Purpose.SIMULATE):
                call = llm.generate_response(_request())
        assert call is not None
        assert {e["purpose"] for e in _events(log, ROOT)} == {"simulate"}


def _roots(log: EventLog) -> list[str]:
    return sorted(p.stem for p in Path(log.root).glob("**/*.jsonl"))


class TestSettersAtTheRealCallSites:
    def test_the_judge_and_the_summary_are_labeled_through_their_real_code(
        self, log: EventLog
    ) -> None:
        with (
            bind(log, ROOT),
            patch(COMPLETION, lambda **kw: _reply('{"message": "ok"}')),
        ):
            _default_complete("gpt-4o", "grade this")
            conversation = Conversation(max_history_size=100)
            conversation.add_user_message("Hello, I am Alice")
            conversation.add_system_message("Nice to meet you")
            ContextCompactor().summarize(conversation, LiteLLMInterface(model="gpt-4o"))
        requests = _events(log, ROOT, "llm/request")
        assert [(e["call_type"], e["purpose"]) for e in requests] == [
            ("completion", "judge"),
            ("response_generation", "summarize"),
        ]

    @pytest.mark.parametrize(
        ("path", "purpose", "callee"),
        [
            ("synthesis/synthesizer.py", "COMPILE", "complete"),
            ("synthesis/repair.py", "COMPILE", "complete"),
            ("synthesis/simulation.py", "SIMULATE", "complete"),
            ("engine/turn/transitions.py", "TIE_BREAK", "classify"),
        ],
    )
    def test_the_scope_wraps_the_model_call_it_names(
        self, path: str, purpose: str, callee: str
    ) -> None:
        source = (REPO / "src" / "fsm_llm" / path).read_text(encoding="utf-8")
        scopes = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.With)
            and any(
                f"Purpose.{purpose}" in ast.unparse(i.context_expr) for i in node.items
            )
        ]
        assert scopes, f"{path}: no purpose_scope(Purpose.{purpose})"
        assert all(callee in ast.unparse(node) for node in scopes)

    def test_no_scope_is_held_across_a_yield_in_the_source(self) -> None:
        offenders: list[str] = []
        for path in (REPO / "src" / "fsm_llm").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.With, ast.AsyncWith)):
                    continue
                if not any(
                    "purpose_scope" in ast.unparse(i.context_expr) for i in node.items
                ):
                    continue
                if any(
                    isinstance(inner, (ast.Yield, ast.YieldFrom))
                    for stmt in node.body
                    for inner in ast.walk(stmt)
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == []


class TestTheScopeAndTheTable:
    def test_a_scope_resets_on_exit_and_on_error(self) -> None:
        assert current_purpose() is None
        with purpose_scope(Purpose.JUDGE):
            assert current_purpose() is Purpose.JUDGE
            with purpose_scope(Purpose.COMPILE):
                assert current_purpose() is Purpose.COMPILE
            assert current_purpose() is Purpose.JUDGE
        with pytest.raises(RuntimeError):
            with purpose_scope(Purpose.SIMULATE):
                raise RuntimeError("x")
        assert current_purpose() is None

    def test_resolve_prefers_the_scope_then_the_table_then_complete(self) -> None:
        assert resolve_purpose("classification") is Purpose.CLASSIFY
        with purpose_scope(Purpose.TIE_BREAK):
            assert resolve_purpose("classification") is Purpose.TIE_BREAK
        assert resolve_purpose("a call type nobody listed") is Purpose.COMPLETE
        assert purpose_of("") is Purpose.COMPLETE

    def test_the_table_covers_every_usage_call_type(self) -> None:
        assert set(USAGE_KIND_BY_CALL_TYPE) <= set(PURPOSE_BY_CALL_TYPE)
        assert "response_stream" in PURPOSE_BY_CALL_TYPE
        assert set(PURPOSE_BY_CALL_TYPE.values()) <= set(Purpose)

    def test_embedding_has_no_purpose(self) -> None:
        # the embedder has no observer: an embed call writes no event, so no member names it
        assert "embed" not in _VALUES


class TestNoRequestByteMoves:
    def test_the_provider_request_is_equal_with_and_without_a_scope(self) -> None:
        seen: list[dict[str, Any]] = []

        def completion(**kwargs: Any) -> Any:
            seen.append(json.loads(json.dumps(kwargs, default=str)))
            return _reply()

        llm = LiteLLMInterface(model="gpt-4o", temperature=0.2)
        with patch(COMPLETION, completion):
            llm.generate_response(_request())
            with purpose_scope(Purpose.JUDGE):
                llm.generate_response(_request())
            llm.complete(CompletionRequest(messages=[{"role": "user", "content": "x"}]))
            with purpose_scope(Purpose.COMPILE):
                llm.complete(
                    CompletionRequest(messages=[{"role": "user", "content": "x"}])
                )
        assert seen[0] == seen[1] and seen[2] == seen[3]
        assert not any("purpose" in call for call in seen)

    def test_purpose_is_a_field_of_no_request_model(self) -> None:
        from fsm_llm.core import definitions

        for name in dir(definitions):
            model = getattr(definitions, name)
            fields = getattr(model, "model_fields", None)
            if isinstance(fields, dict) and name.endswith("Request"):
                assert "purpose" not in fields, name


class TestUsageByPurpose:
    def _run(self, log: EventLog) -> LiteLLMInterface:
        llm = LiteLLMInterface(model="gpt-4o")
        with bind(log, ROOT), patch(COMPLETION, lambda **kw: _reply()):
            llm.generate_response(_request())
            llm.generate_response(_request())
            with purpose_scope(Purpose.JUDGE):
                llm.complete(
                    CompletionRequest(messages=[{"role": "user", "content": "x"}])
                )

            def down(**kwargs: Any) -> Any:
                raise RuntimeError("down")

            with patch(COMPLETION, down), purpose_scope(Purpose.COMPILE):
                with pytest.raises(LLMResponseError):
                    llm.complete(
                        CompletionRequest(messages=[{"role": "user", "content": "x"}])
                    )
        return llm

    def test_the_sums_equal_the_totals_of_the_same_log_and_of_the_meter(
        self, log: EventLog
    ) -> None:
        llm = self._run(log)
        events = _events(log, ROOT)
        by_purpose = usage_by_purpose(events)
        total = usage_from_events(events)
        for name in (
            "calls",
            "errors",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            assert sum(getattr(c, name) for c in by_purpose.values()) == getattr(
                total, name
            )
        meter = llm.usage()
        assert (total.calls, total.errors, total.total_tokens) == (
            meter.calls,
            meter.errors,
            meter.total_tokens,
        )
        assert total.calls == 4 and total.errors == 1

    def test_the_counts_per_purpose(self, log: EventLog) -> None:
        self._run(log)
        by_purpose = usage_by_purpose(_events(log, ROOT))
        assert sorted(by_purpose) == ["compile", "judge", "reply"]
        assert (by_purpose["reply"].calls, by_purpose["reply"].total_tokens) == (2, 30)
        assert (by_purpose["judge"].calls, by_purpose["judge"].prompt_tokens) == (1, 11)
        assert (by_purpose["compile"].calls, by_purpose["compile"].errors) == (1, 1)

    def test_an_old_event_without_a_purpose_folds_through_its_call_type(self) -> None:
        events = [
            {
                "type": "llm/response",
                "call_type": "classification",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            {"type": "llm/response", "call_type": "never heard of it"},
            {"type": "provider/error", "call_type": "data_extraction"},
            {"type": "llm/response", "call_type": "x", "purpose": "not a purpose"},
            {"type": "llm/request", "call_type": "classification", "purpose": "judge"},
        ]
        by_purpose = usage_by_purpose(events)
        assert sorted(by_purpose) == ["classify", "complete", "extract"]
        assert (
            by_purpose["complete"].calls == 2
        )  # the unknown type and the foreign value
        assert by_purpose["classify"].total_tokens == 3
        assert by_purpose["extract"].errors == 1

    def test_a_shape_log_folds_too(self, tmp_path: Path) -> None:
        with EventLog(tmp_path / "shape", privacy="shape") as shape:
            llm = LiteLLMInterface(model="gpt-4o")
            with bind(shape, ROOT), patch(COMPLETION, lambda **kw: _reply()):
                with purpose_scope(Purpose.SUMMARIZE):
                    llm.generate_response(_request())
            by_purpose = usage_by_purpose(shape.query(ROOT))
        assert {k: v.calls for k, v in by_purpose.items()} == {"summarize": 1}
