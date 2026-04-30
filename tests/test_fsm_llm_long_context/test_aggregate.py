from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.aggregate_term — M5 slice 2."""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.runtime import Executor, Oracle, PlanInputs, plan
from fsm_llm.runtime.combinators import ReduceOp
from fsm_llm.stdlib.long_context import (
    aggregate_op,
    aggregate_term,
    make_size_bucket,
)


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


def test_aggregate_smoke() -> None:
    """T1 — synthetic 8-char doc, 8 leaves, 4 informative + 4 sentinel.

    Verify the bullet-joined output contains every informative leaf
    response, separator-joined, and excludes sentinels.
    """
    doc = "ABCDEFGH"  # 8 chars, τ=1 → 8 leaves
    # Alternate informative chunks with NOT_FOUND sentinels.
    responses = [
        "fact A",
        "NOT_FOUND",
        "fact C",
        "NOT_FOUND",
        "fact E",
        "NOT_FOUND",
        "fact G",
        "NOT_FOUND",
    ]
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = aggregate_term("question?", tau=1, k=2)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "merge": aggregate_op(),
        },
    )

    assert isinstance(result, str)
    # All four real facts must appear; no sentinel leaks through.
    for fact in ("fact A", "fact C", "fact E", "fact G"):
        assert fact in result
    assert "NOT_FOUND" not in result
    assert ex.oracle_calls == 8


def test_aggregate_cost_equality_sc2() -> None:
    """T2 — Theorem-2 invariant: ex.oracle_calls == predicted_calls."""
    doc = "abcdefgh"
    responses = ["x"] * 8
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = aggregate_term("q", tau=1, k=2)
    ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "merge": aggregate_op(),
        },
    )

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert predicted.k_star == 2
    assert predicted.d == 3
    assert predicted.predicted_calls == 8
    assert ex.oracle_calls == predicted.predicted_calls == 8


def test_aggregate_degenerate_small_input() -> None:
    """T3 — input length ≤ τ: single Leaf call, depth 0."""
    oracle = _ScriptedOracle(responses=["a single chunk's summary"])
    ex = Executor(oracle=oracle)

    program = aggregate_term("q", tau=100, k=2)
    result = ex.run(
        program,
        {
            "document": "tiny doc",
            "size_bucket": make_size_bucket(100),
            "merge": aggregate_op(),
        },
    )
    assert result == "a single chunk's summary"
    assert ex.oracle_calls == 1


def test_aggregate_purity() -> None:
    """T4 — long_context package must not import fsm_llm.{llm,fsm,pipeline}."""
    pkg_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fsm_llm"
        / "stdlib"
        / "long_context"
    )
    assert pkg_dir.is_dir(), f"missing: {pkg_dir}"

    forbidden = ("fsm_llm.llm", "fsm_llm.fsm", "fsm_llm.pipeline")
    offenders: list[str] = []
    for py in pkg_dir.glob("*.py"):
        text = py.read_text()
        for needle in forbidden:
            if f"from {needle}" in text or f"import {needle}" in text:
                offenders.append(f"{py.name}: {needle}")
    assert offenders == [], f"purity violation(s): {offenders}"


def test_aggregate_oracle_protocol_conformance() -> None:
    assert isinstance(_ScriptedOracle(responses=["x"]), Oracle)


def test_aggregate_validates_args() -> None:
    with pytest.raises(ValueError, match="tau"):
        aggregate_term("q", tau=0)
    with pytest.raises(ValueError, match="k"):
        aggregate_term("q", k=1)


def test_aggregate_op_unit() -> None:
    """T7 — aggregate_op behaves correctly: filters, joins, units."""
    op = aggregate_op()

    # Sentinel filtering: bad on either side returns the other.
    assert op.fn("NOT_FOUND", "kept") == "kept"
    assert op.fn("kept", "NOT_FOUND") == "kept"
    assert op.fn("", "kept") == "kept"
    assert op.fn(None, "kept") == "kept"

    # Both bad → sentinel.
    assert op.fn("NOT_FOUND", "") == "NOT_FOUND"
    assert op.fn(None, "NOT_FOUND") == "NOT_FOUND"

    # Both real → separator-joined.
    assert op.fn("a", "b") == "a\n- b"

    # Unit element.
    assert op.unit == "NOT_FOUND"
    assert op.associative is True

    # Custom separator + sentinel.
    op2 = aggregate_op(separator=" | ", sentinel="MISS")
    assert op2.fn("a", "b") == "a | b"
    assert op2.fn("MISS", "x") == "x"
    assert op2.unit == "MISS"


def test_aggregate_custom_reduce_op_name() -> None:
    """Caller can supply a different reduce_op_name and bind it in env."""
    doc = "ab"
    oracle = _ScriptedOracle(responses=["alpha", "beta"])
    ex = Executor(oracle=oracle)

    custom = ReduceOp(
        name="my_join",
        fn=lambda a, b: f"{a};{b}",
        associative=True,
        unit="",
    )
    program = aggregate_term("q", tau=1, k=2, reduce_op_name="my_join")
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "my_join": custom,
        },
    )
    assert result == "alpha;beta"
    assert ex.oracle_calls == 2
