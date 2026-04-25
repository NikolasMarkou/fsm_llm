from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.pairwise — M5 slice 3."""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.lam import Executor, Oracle, PlanInputs, plan
from fsm_llm.lam.combinators import ReduceOp
from fsm_llm.stdlib.long_context import compare_op, make_size_bucket, pairwise


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


def test_pairwise_smoke() -> None:
    """T1 — synthetic 8-char doc; oracle picks the most relevant segment.

    Doc ``"AAAAXAAA"``, τ=1, k=2. Recursion produces 8 leaves, each a
    single character. Scripted oracle returns ``"X"`` (winner) for the
    needle char and ``"NOT_FOUND"`` for non-relevant chars. ``compare_op``
    propagates the winner to the root.
    """
    doc = "AAAAXAAA"
    responses = ["NOT_FOUND" if ch != "X" else "X" for ch in doc]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = pairwise("Which segment mentions X?", tau=1, k=2)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "compare": compare_op(),
        },
    )

    assert result == "X"
    assert ex.oracle_calls == 8
    assert len(oracle.calls) == 8
    # Every leaf prompt should embed the question and feed a single
    # character into {P}.
    for ch, prompt in zip(doc, oracle.calls, strict=True):
        assert "Which segment mentions X?" in prompt
        assert f"\nPassage:\n{ch}\n" in prompt


def test_pairwise_cost_equality_sc2() -> None:
    """T2 — SC2 invariant: ``ex.oracle_calls == predicted_calls``.

    For ``n = τ · k^d`` (8 = 1 · 2^3), planner predicts ``k^d = 8``
    leaf calls; executor delivers exactly that.
    """
    doc = "AAAAXAAA"
    responses = ["NOT_FOUND"] * 7 + ["X"]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = pairwise("question", tau=1, k=2)
    ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "compare": compare_op(),
        },
    )

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert predicted.k_star == 2
    assert predicted.d == 3
    assert predicted.predicted_calls == 8
    assert ex.oracle_calls == predicted.predicted_calls == 8


def test_pairwise_degenerate_small_input() -> None:
    """T3 — input length ≤ τ: single Leaf call, depth 0, no recursion."""
    oracle = _ScriptedOracle(responses=["winner segment"])
    ex = Executor(oracle=oracle)

    program = pairwise("which?", tau=100, k=2)
    result = ex.run(
        program,
        {
            "document": "tiny doc",
            "size_bucket": make_size_bucket(100),
            "compare": compare_op(),
        },
    )
    assert result == "winner segment"
    assert ex.oracle_calls == 1


def test_pairwise_purity() -> None:
    """T4 — pairwise.py must not import fsm_llm.{llm,fsm,pipeline}."""
    pkg_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fsm_llm"
        / "stdlib"
        / "long_context"
    )
    assert pkg_dir.is_dir(), f"long_context package not found at {pkg_dir}"

    forbidden = ("fsm_llm.llm", "fsm_llm.fsm", "fsm_llm.pipeline")
    pairwise_src = (pkg_dir / "pairwise.py").read_text()
    offenders: list[str] = []
    for needle in forbidden:
        if f"from {needle}" in pairwise_src or f"import {needle}" in pairwise_src:
            offenders.append(f"pairwise.py: {needle}")
    assert offenders == [], f"purity violation(s): {offenders}"


def test_pairwise_oracle_protocol_conformance() -> None:
    """T5 — scripted oracle satisfies the runtime-checkable Oracle protocol."""
    assert isinstance(_ScriptedOracle(responses=["x"]), Oracle)


def test_pairwise_validates_args() -> None:
    """T6 — bad args surface as ValueError, not silent."""
    with pytest.raises(ValueError, match="tau"):
        pairwise("q", tau=0)
    with pytest.raises(ValueError, match="k"):
        pairwise("q", k=1)


def test_pairwise_custom_reduce_op_name() -> None:
    """T7 — caller can supply a different reduce_op_name and bind it in env."""
    doc = "ab"
    oracle = _ScriptedOracle(responses=["A", "B"])
    ex = Executor(oracle=oracle)

    custom_op = ReduceOp(
        name="my_op", fn=lambda a, b: str(a) + str(b), associative=True, unit=""
    )
    program = pairwise("q", tau=1, k=2, reduce_op_name="my_op")
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


def test_compare_op() -> None:
    """T8 — compare_op specifics: longer-non-sentinel-wins, sentinel handling."""
    op = compare_op()
    # discard sentinel
    assert op.fn("NOT_FOUND", "real segment") == "real segment"
    assert op.fn("real segment", "NOT_FOUND") == "real segment"
    # both bad → sentinel
    assert op.fn("NOT_FOUND", "") == "NOT_FOUND"
    assert op.fn("", "NOT_FOUND") == "NOT_FOUND"
    assert op.fn(None, "NOT_FOUND") == "NOT_FOUND"
    # both real, different lengths → longer wins
    assert op.fn("short", "longer segment") == "longer segment"
    assert op.fn("longer segment", "short") == "longer segment"
    # equal-length tie → first arg wins (>= comparison)
    assert op.fn("aaa", "bbb") == "aaa"
    # unit + name + associative flag
    assert op.unit == "NOT_FOUND"
    assert op.name == "compare"
    assert op.associative is True
    # custom sentinel
    op2 = compare_op(sentinel="NONE")
    assert op2.unit == "NONE"
    assert op2.fn("NONE", "x") == "x"
