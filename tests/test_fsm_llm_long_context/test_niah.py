from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.niah — M5 slice 1 SC1+SC2."""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.lam import Executor, Oracle, PlanInputs, plan
from fsm_llm.lam.combinators import ReduceOp
from fsm_llm.stdlib.long_context import best_answer_op, make_size_bucket, niah


class _ScriptedOracle:
    """Minimal Oracle: pops scripted responses, records prompts."""

    def __init__(self, responses: list[Any], K: int = 10_000) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self._K = K

    def invoke(
        self,
        prompt: str,
        schema: Any = None,
        *,
        model_override: str | None = None,
    ) -> Any:
        self.calls.append(prompt)
        return self._responses.pop(0)

    def tokenize(self, text: str) -> int:
        return max(1, len(text))

    def context_window(self) -> int:
        return self._K


def test_niah_smoke() -> None:
    """T1 — synthetic 8-char doc with one needle char.

    Doc ``"AAAAXAAA"``, τ=1, k=2. Recursion produces 8 leaves, each a
    single character. Scripted oracle returns ``"X"`` for the needle char,
    ``"NOT_FOUND"`` for non-needles. ``best_answer_op`` picks the
    informative answer.
    """
    doc = "AAAAXAAA"
    # Order of leaf invocations follows split's left-to-right traversal,
    # which matches doc-position order for clean splits.
    responses = ["NOT_FOUND" if ch != "X" else "X" for ch in doc]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = niah("Where is the X?", tau=1, k=2)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    assert result == "X"
    assert ex.oracle_calls == 8
    assert len(oracle.calls) == 8
    # Every leaf prompt should be the templated NIAH prompt with the
    # question baked in and a single character substituted for {P}.
    for ch, prompt in zip(doc, oracle.calls, strict=True):
        assert "Where is the X?" in prompt
        assert f"\nText:\n{ch}\n" in prompt


def test_niah_cost_equality_sc2() -> None:
    """T2 — SC2 invariant: ``ex.oracle_calls == predicted_calls``.

    For ``n = τ · k^d`` (here 8 = 1 · 2^3), the planner predicts exactly
    ``k^d`` Leaf calls and the executor delivers exactly that.
    """
    doc = "AAAAXAAA"  # n=8, τ=1, k=2 → d=3, leaves=8
    responses = ["NOT_FOUND"] * 7 + ["X"]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = niah("question", tau=1, k=2)
    ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert predicted.k_star == 2
    assert predicted.d == 3
    assert predicted.predicted_calls == 8
    assert ex.oracle_calls == predicted.predicted_calls == 8


def test_niah_degenerate_small_input() -> None:
    """T3 — input length ≤ τ: single Leaf call, depth 0, no recursion."""
    oracle = _ScriptedOracle(responses=["the answer is 42"])
    ex = Executor(oracle=oracle)

    program = niah("what?", tau=100, k=2)
    result = ex.run(
        program,
        {
            "document": "tiny doc",
            "size_bucket": make_size_bucket(100),
            "best": best_answer_op(),
        },
    )
    assert result == "the answer is 42"
    assert ex.oracle_calls == 1


def test_niah_purity() -> None:
    """T4 — stdlib.long_context must not import fsm_llm.{llm,fsm,pipeline}."""
    pkg_dir = Path(__file__).resolve().parents[2] / "src" / "fsm_llm" / "stdlib" / "long_context"
    assert pkg_dir.is_dir(), f"long_context package not found at {pkg_dir}"

    forbidden = ("fsm_llm.llm", "fsm_llm.fsm", "fsm_llm.pipeline")
    offenders: list[str] = []
    for py in pkg_dir.glob("*.py"):
        text = py.read_text()
        for needle in forbidden:
            if f"from {needle}" in text or f"import {needle}" in text:
                offenders.append(f"{py.name}: {needle}")
    assert offenders == [], f"purity violation(s): {offenders}"


def test_niah_oracle_protocol_conformance() -> None:
    """Sanity: scripted oracle satisfies the runtime-checkable Oracle protocol."""
    assert isinstance(_ScriptedOracle(responses=["x"]), Oracle)


def test_niah_validates_args() -> None:
    """Constructor surfaces obviously-bad args as ValueError, not silent."""
    with pytest.raises(ValueError, match="tau"):
        niah("q", tau=0)
    with pytest.raises(ValueError, match="k"):
        niah("q", k=1)


def test_make_size_bucket() -> None:
    sb = make_size_bucket(5)
    assert sb("hi") == "small"
    assert sb("hello") == "small"  # boundary: len == tau
    assert sb("hellos") == "big"
    assert sb(42) == "small"  # non-sized → small


def test_best_answer_op() -> None:
    op = best_answer_op()
    # discard sentinel
    assert op.fn("NOT_FOUND", "real answer") == "real answer"
    assert op.fn("real answer", "NOT_FOUND") == "real answer"
    # both bad → sentinel
    assert op.fn("NOT_FOUND", "") == "NOT_FOUND"
    # both real → longer wins
    assert op.fn("short", "longer answer") == "longer answer"
    # unit
    assert op.unit == "NOT_FOUND"


def test_niah_custom_reduce_op_name() -> None:
    """Caller can supply a different reduce_op_name and bind it in env."""
    doc = "ab"
    oracle = _ScriptedOracle(responses=["A", "B"])
    ex = Executor(oracle=oracle)

    # Use a custom op name "my_op" that concatenates strings.
    custom_op = ReduceOp(
        name="my_op", fn=lambda a, b: str(a) + str(b), associative=True, unit=""
    )
    program = niah("q", tau=1, k=2, reduce_op_name="my_op")
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "my_op": custom_op,
        },
    )
    assert result == "AB"
    assert ex.oracle_calls == 2
