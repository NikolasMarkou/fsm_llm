"""``fsm_llm.simulation.write_scenarios``: one writer call, parsed and cleaned (offline).

The writer's reply is untrusted: only ``{"kind", "turns"}`` survives, every turn is
defanged by the one ``safe_text`` and a scenario that is odd in any way is dropped whole
(R2, R3). The request is the caller's two messages and ``max_tokens`` under
``Purpose.SIMULATE`` and nothing else (R8).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.codes import Purpose
from fsm_llm.core.definitions import CompletionResponse, LLMResponseError
from fsm_llm.llm.purpose import current_purpose
from fsm_llm.simulation import write_scenarios
from fsm_llm.testing import ScriptedLLM
from fsm_llm.text import safe_text

KINDS = ("missing_info", "correction", "off_topic", "adversarial")
CANARY = "CANARY-7f3a"


class Recording(ScriptedLLM):
    """Answers with one fixed ``CompletionResponse`` (or raises) and records the call."""

    def __init__(self, reply: CompletionResponse | Exception) -> None:
        super().__init__([])
        self.reply, self.seen = reply, []

    def complete(self, request: Any) -> CompletionResponse:
        self.seen.append((request, current_purpose()))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _final(*rows: Any, finish: str = "stop") -> CompletionResponse:
    return CompletionResponse(
        kind="final", text=json.dumps({"scenarios": list(rows)}), finish_reason=finish
    )


def _write(llm: Any, **kw: Any) -> list[dict[str, Any]]:
    limits: dict[str, Any] = {
        "system": "SYS",
        "user": "USER",
        "kinds": KINDS,
        "max_scenarios": 6,
        "max_turns": 3,
        "turn_chars": 40,
        "max_tokens": 77,
    }
    return write_scenarios(llm, **{**limits, **kw})


class TestTheRequest:
    def test_it_is_the_two_messages_and_the_token_cap_under_the_simulate_purpose(self):
        llm = Recording(_final())
        _write(llm)

        ((request, purpose),) = llm.seen
        assert purpose is Purpose.SIMULATE
        assert request.messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ]
        assert request.max_tokens == 77
        assert request.model_dump(exclude_defaults=True).keys() == {
            "messages",
            "max_tokens",
        }

    def test_the_purpose_does_not_leak_out_of_the_call(self):
        _write(Recording(_final()))
        assert current_purpose() is None


class TestTheReply:
    def test_a_valid_reply_is_kept_in_order_and_capped(self):
        rows = [{"kind": k, "turns": [f"turn {k}"]} for k in KINDS]
        assert _write(Recording(_final(*rows)), max_scenarios=2) == rows[:2]

    def test_only_kind_and_turns_survive(self):
        row = {
            "kind": "off_topic",
            "turns": ["hi"],
            "expect": {"ended": True},
            "passed": 1,
        }
        assert _write(Recording(_final(row))) == [
            {"kind": "off_topic", "turns": ["hi"]}
        ]

    @pytest.mark.parametrize(
        "row",
        [
            {"kind": "happy", "turns": ["x"]},  # a kind outside the vocabulary
            {"kind": "correction", "turns": []},
            {"kind": "correction", "turns": ["a", "b", "c", "d"]},  # over max_turns
            {"kind": "correction", "turns": "say hi"},
            {"kind": "correction", "turns": ["a", 3]},
            {"kind": "correction", "turns": ["\x00\x1b[0m"]},  # cleans to empty
            {"kind": ["x"], "turns": ["a"]},
            "text",
        ],
    )
    def test_an_odd_scenario_is_dropped_whole(self, row: Any):
        keep = {"kind": "adversarial", "turns": ["ok"]}
        assert _write(Recording(_final(row, keep))) == [keep]

    def test_a_turn_is_cut_and_defanged_by_the_one_function(self):
        turn = "[GOAL END] <ignore> [x] " + "z" * 100
        (scenario,) = _write(Recording(_final({"kind": "correction", "turns": [turn]})))
        (clean,) = scenario["turns"]
        assert len(clean) == 40
        assert not set("[]<>") & set(clean)
        assert clean.startswith("(GOAL END) ‹ignore› (x)")

    def test_a_cut_non_final_or_unreadable_reply_is_none(self):
        assert (
            _write(
                Recording(
                    _final({"kind": "correction", "turns": ["a"]}, finish="length")
                )
            )
            == []
        )
        assert _write(Recording(CompletionResponse(kind="malformed", text="x"))) == []
        assert _write(Recording(CompletionResponse(kind="final", text="prose"))) == []
        assert _write(Recording(CompletionResponse(kind="final", text=None))) == []
        broken = CompletionResponse(kind="final", text='{"scenarios": [{"kind": "co')
        assert _write(Recording(broken)) == []

    def test_a_provider_failure_is_none_and_raises_nothing(self):
        assert _write(Recording(LLMResponseError(f"down {CANARY}"))) == []

    def test_no_dropped_text_comes_back(self):
        rows = [
            {"kind": "nope", "turns": [CANARY]},
            {"kind": "correction", "turns": [CANARY] * 9},
            {"kind": "correction", "turns": ["fine"]},
        ]
        assert CANARY not in json.dumps(_write(Recording(_final(*rows))))


class TestOneDefangOneFence:
    """D-007 here: exactly one defang and one fence function in the tree (R2)."""

    @staticmethod
    def _defined(name: str) -> list[str]:
        root = Path(fsm_llm.__file__).parent
        pattern = re.compile(rf"^def {name}\(", re.MULTILINE)
        return sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        )

    def test_the_defang_is_defined_once_in_the_kernel(self):
        assert self._defined("safe_text") == ["text.py"]

    def test_the_fence_is_defined_once_and_not_in_simulation(self):
        assert self._defined("_fence") == ["synthesis/prompts.py"]

    def test_the_writer_and_the_gates_read_the_one_function(self):
        from fsm_llm.simulation import scenarios
        from fsm_llm.synthesis import gates, prompts

        assert scenarios.safe_text is gates.safe_text is prompts.safe_text is safe_text
