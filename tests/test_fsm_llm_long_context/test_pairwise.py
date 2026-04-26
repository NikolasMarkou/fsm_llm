from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.pairwise — M5 slice 3."""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.lam import Executor, Oracle, PlanInputs, plan
from fsm_llm.lam.combinators import ReduceOp
from fsm_llm.stdlib.long_context import (
    compare_op,
    make_size_bucket,
    oracle_compare_op,
    pairwise,
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


class _SmartOracle:
    """T9-T11 helper.

    - Leaf prompts: extract the ``{P}`` chunk and return it verbatim
      (echo). When ``sparse=True``, leaves whose chunk does not contain
      'X' return ``"NOT_FOUND"`` instead — used by T11 to exercise the
      sentinel short-circuit path.
    - Compare prompts (``oracle_compare_op``): pick whichever Segment
      contains 'X'; deterministic 'A'-tiebreak otherwise.

    Records every prompt for inspection.
    """

    def __init__(self, *, sparse: bool = False, K: int = 10_000) -> None:
        self.calls: list[str] = []
        self._K = K
        self._sparse = sparse

    def invoke(
        self,
        prompt: str,
        schema: Any = None,
        *,
        model_override: str | None = None,
    ) -> str:
        self.calls.append(prompt)
        # Compare-prompt path: emitted by oracle_compare_op.
        if "Reply with exactly one character: A or B." in prompt:
            # Extract Segment A and Segment B blocks.
            try:
                a_block = prompt.split("Segment A:\n", 1)[1].split(
                    "\n\nSegment B:\n", 1
                )[0]
                b_block = prompt.split("Segment B:\n", 1)[1].split(
                    "\n\nReply with", 1
                )[0]
            except IndexError:
                return "A"
            if "X" in a_block and "X" not in b_block:
                return "A"
            if "X" in b_block and "X" not in a_block:
                return "B"
            return "A"  # deterministic tiebreak
        # Leaf prompt path: return the {P} value verbatim.
        # The pairwise leaf template has "Passage:\n{P}\n\n".
        try:
            chunk = prompt.split("Passage:\n", 1)[1].split("\n\n", 1)[0]
        except IndexError:
            return "NOT_FOUND"
        # T11 (sparse): leaves with no 'X' return sentinel; only the
        # X-leaf returns its content. Drives reduce short-circuit.
        if self._sparse and "X" not in chunk:
            return "NOT_FOUND"
        # T9/T10 (dense): every leaf returns its non-sentinel chunk so
        # every reduce node has two real arms → strict T2 holds.
        return chunk

    def tokenize(self, text: str) -> int:
        return max(1, len(text))

    def context_window(self) -> int:
        return self._K


# --------------------------------------------------------------------------
# M5 slice 5 — oracle_compare_op contract: counter increments per call
# --------------------------------------------------------------------------


def test_oracle_compare_op_counter_contract() -> None:
    """Contract: oracle_compare_op increments executor._oracle_calls
    exactly once per non-sentinel pair invocation; sentinel short-circuit
    does not tick the counter (D-S5-001)."""
    oracle = _ScriptedOracle(responses=["A", "B", "A"])
    ex = Executor(oracle=oracle)
    op = oracle_compare_op("which segment is more relevant?", ex)

    assert isinstance(op, ReduceOp)
    assert op.name == "oracle_compare"
    assert op.associative is True
    assert op.unit == "NOT_FOUND"

    # 3 real-pair invocations → 3 oracle calls.
    assert ex.oracle_calls == 0
    op.fn("alpha segment", "beta segment")
    assert ex.oracle_calls == 1
    op.fn("gamma", "delta")
    assert ex.oracle_calls == 2
    op.fn("epsilon", "zeta")
    assert ex.oracle_calls == 3

    # Sentinel short-circuit: no oracle call.
    op.fn("NOT_FOUND", "real segment")
    op.fn("real segment", "NOT_FOUND")
    op.fn("NOT_FOUND", "NOT_FOUND")
    op.fn("", "real")
    op.fn(None, "real")
    assert ex.oracle_calls == 3


def test_oracle_compare_op_smoke() -> None:
    """T9 — full Executor.run + oracle_compare_op smoke.

    Doc "AAAAXAAA", τ=1, k=2 → 8 leaves, 7 reduce nodes (all non-sentinel
    arms thanks to leaves echoing chunk text). Smart oracle picks the
    arm containing 'X'. Result == "X". ex.oracle_calls == 8 + 7 = 15.
    """
    doc = "AAAAXAAA"
    oracle = _SmartOracle()
    ex = Executor(oracle=oracle)

    program = pairwise("Which segment mentions X?", tau=1, k=2)
    op = oracle_compare_op("Which segment mentions X?", ex)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "compare": op,
        },
    )

    assert result == "X"
    assert ex.oracle_calls == 15  # 8 leaf + 7 reduce
    # 8 leaf prompts + 7 compare prompts = 15 invokes.
    assert len(oracle.calls) == 15


def test_oracle_compare_op_t2_cost_equality() -> None:
    """T10 — Theorem-2 strict equality with reduce_calls_per_node=1.

    For aligned input (n = τ·k^d), planner predicts:
        leaf_calls   = k^d
        reduce_calls = (k^d - 1) · 1
        predicted    = 2·k^d - 1
    and ex.oracle_calls matches exactly.
    """
    doc = "AAAAXAAA"  # n=8, τ=1, k=2 → d=3 → leaf=8, reduce=7, total=15
    oracle = _SmartOracle()
    ex = Executor(oracle=oracle)
    program = pairwise("Which segment mentions X?", tau=1, k=2)
    op = oracle_compare_op("Which segment mentions X?", ex)
    ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "compare": op,
        },
    )

    predicted = plan(
        PlanInputs(
            n=len(doc),
            K=10_000,
            tau=1,
            alpha=1.0,
            max_k=2,
            reduce_calls_per_node=1,
        )
    )
    assert predicted.k_star == 2
    assert predicted.d == 3
    assert predicted.leaf_calls == 8
    assert predicted.reduce_calls == 7
    assert predicted.predicted_calls == 15
    # Theorem-2 hard gate.
    assert ex.oracle_calls == predicted.predicted_calls == 15


def test_oracle_compare_op_sentinel_short_circuit() -> None:
    """T11 — sparse-needle case: T2 strict equality relaxes to upper bound.

    Only 1 of 8 leaves has 'X'; the others return sentinel ("NOT_FOUND").
    The oracle_compare_op short-circuits whenever EITHER arm is sentinel
    (returns the other arm without invoking the oracle — D-004). Sparse
    inputs therefore never produce a two-real-arms reduce node:

        leaves [N,N,N,N,X,N,N,N]
        level 0 (4 nodes): N,N→N; N,N→N; X,N→X (short-circuit); N,N→N
        level 1 (2 nodes): N,N→N; X,N→X (short-circuit)
        level 2 (1 node) : N,X→X (short-circuit)
        total reduce oracle calls = 0 ; total = 8 leaves

    Documents that strict T2 (predicted = 15 with reduce_calls_per_node=1)
    becomes an UPPER BOUND on sparse-needle inputs: actual ≤ predicted.
    Strict equality requires every reduce input to have two non-sentinel
    arms at the leaf level (e.g. T9/T10's dense oracle).
    """
    doc = "AAAAXAAA"
    oracle = _SmartOracle(sparse=True)  # leaves without 'X' return NOT_FOUND
    ex = Executor(oracle=oracle)
    program = pairwise("Which segment mentions X?", tau=1, k=2)
    op = oracle_compare_op("Which segment mentions X?", ex)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "compare": op,
        },
    )
    assert result == "X"
    # 8 leaf calls + 0 reduce calls (every reduce had ≥1 sentinel arm).
    assert ex.oracle_calls == 8

    # Planner upper bound at reduce_calls_per_node=1 = 15. Strict equality
    # relaxes to ≤ here.
    predicted = plan(
        PlanInputs(
            n=len(doc),
            K=10_000,
            tau=1,
            alpha=1.0,
            max_k=2,
            reduce_calls_per_node=1,
        )
    )
    assert ex.oracle_calls < predicted.predicted_calls  # 8 < 15 — upper bound
    # And ≥ leaf_calls (the lower bound — leaves always fire).
    assert ex.oracle_calls == predicted.leaf_calls == 8


def test_oracle_compare_op_associativity_caveat() -> None:
    """T12 — declarative associativity: matches slice-3 compare_op.

    Oracle responses are not provably associative; we declare True to
    allow caller-side reorder optimisations (LESSONS — same caveat as
    compare_op). This test pins the contract so future refactors that
    flip the flag are caught.
    """
    ex = Executor(oracle=_ScriptedOracle(responses=[]))
    op = oracle_compare_op("q", ex)
    assert isinstance(op, ReduceOp)
    assert op.name == "oracle_compare"
    assert op.associative is True
    assert op.unit == "NOT_FOUND"
    # Custom sentinel honored.
    op2 = oracle_compare_op("q", ex, sentinel="NONE")
    assert op2.unit == "NONE"
