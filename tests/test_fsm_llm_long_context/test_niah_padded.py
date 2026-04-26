from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.niah_padded — M5 slice 4.

Mirrors the canonical ``test_niah.py`` template (smoke / cost-equality /
degenerate / purity / oracle-conformance / validates-args / custom-op)
and adds 4 padding-specific tests:

- T9  ``test_aligned_size_math`` — pure math contract.
- T10 ``test_pad_to_aligned_correctness`` — string-padding helper.
- T11 ``test_no_op_when_already_aligned`` — aligned ``n`` ≡ plain ``niah``.
- T12 ``test_worst_case_padding_factor`` — ``n = τ·k^d + 1`` worst case.
"""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.lam import Executor, Oracle, PlanInputs, plan
from fsm_llm.lam.combinators import ReduceOp
from fsm_llm.stdlib.long_context import (
    aligned_size,
    best_answer_op,
    make_pad_callable,
    make_size_bucket,
    niah,
    niah_padded,
    pad_to_aligned,
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


def _env(doc: str, tau: int, k: int, *, op_name: str = "best", op: Any = None) -> dict:
    """Build the standard env for niah_padded with raw_document binding."""
    return {
        "raw_document": doc,
        "pad_to_aligned": make_pad_callable(tau, k),
        "size_bucket": make_size_bucket(tau),
        op_name: op if op is not None else best_answer_op(),
    }


# ---------------------------------------------------------------------
# T1 — smoke
# ---------------------------------------------------------------------
def test_niah_padded_smoke() -> None:
    """T1 — unaligned 5-char doc with one needle char, τ=1, k=2.

    Raw doc ``"AAXAA"`` (n=5) pads to ``"AAXAA   "`` (N*=8) → 8 leaves.
    Scripted oracle returns ``"X"`` for the needle char, ``"NOT_FOUND"``
    elsewhere (including padding spaces).
    """
    doc = "AAXAA"
    padded = "AAXAA   "  # 3 spaces of padding
    assert aligned_size(len(doc), 1, 2) == len(padded) == 8

    responses = ["NOT_FOUND" if ch != "X" else "X" for ch in padded]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = niah_padded("Where is the X?", tau=1, k=2)
    result = ex.run(program, _env(doc, 1, 2))

    assert result == "X"
    assert ex.oracle_calls == 8
    assert len(oracle.calls) == 8
    # Every leaf prompt should carry the templated NIAH question.
    for prompt in oracle.calls:
        assert "Where is the X?" in prompt


# ---------------------------------------------------------------------
# T2 — cost equality on UNALIGNED n (the slice's primary novelty)
# ---------------------------------------------------------------------
def test_niah_padded_cost_equality_unaligned() -> None:
    """T2 — ``ex.oracle_calls == plan(N*).predicted_calls`` for unaligned n.

    Raw n=5 (not τ·k^d-aligned for τ=1, k=2). N*=8 = τ·k^3.
    Planner predicts 8 leaves on the padded size; executor delivers 8.
    """
    doc = "AAXAA"  # n=5
    n_star = aligned_size(len(doc), 1, 2)
    assert n_star == 8

    responses = ["NOT_FOUND"] * 7 + ["X"]
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = niah_padded("question", tau=1, k=2)
    ex.run(program, _env(doc, 1, 2))

    predicted = plan(
        PlanInputs(n=n_star, K=10_000, tau=1, alpha=1.0, max_k=2)
    )
    assert predicted.k_star == 2
    assert predicted.d == 3
    assert predicted.predicted_calls == 8
    assert ex.oracle_calls == predicted.predicted_calls == 8


# ---------------------------------------------------------------------
# T3 — degenerate small input
# ---------------------------------------------------------------------
def test_niah_padded_degenerate_small_input() -> None:
    """T3 — ``n ≤ τ``: single Leaf call, no padding (aligned_size returns n)."""
    oracle = _ScriptedOracle(responses=["the answer is 42"])
    ex = Executor(oracle=oracle)

    # n=8 ≤ τ=100 → no padding, single leaf.
    program = niah_padded("what?", tau=100, k=2)
    result = ex.run(program, _env("tiny doc", 100, 2))
    assert result == "the answer is 42"
    assert ex.oracle_calls == 1


# ---------------------------------------------------------------------
# T4 — purity (covers niah_padded.py too)
# ---------------------------------------------------------------------
def test_niah_padded_purity() -> None:
    """T4 — stdlib.long_context (incl. niah_padded.py) must not import
    fsm_llm.{llm,fsm,pipeline}."""
    pkg_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fsm_llm"
        / "stdlib"
        / "long_context"
    )
    assert pkg_dir.is_dir(), f"long_context package not found at {pkg_dir}"

    forbidden = ("fsm_llm.llm", "fsm_llm.fsm", "fsm_llm.pipeline")
    offenders: list[str] = []
    for py in pkg_dir.glob("*.py"):
        text = py.read_text()
        for needle in forbidden:
            if f"from {needle}" in text or f"import {needle}" in text:
                offenders.append(f"{py.name}: {needle}")
    assert offenders == [], f"purity violation(s): {offenders}"


# ---------------------------------------------------------------------
# T5 — oracle protocol conformance
# ---------------------------------------------------------------------
def test_niah_padded_oracle_protocol_conformance() -> None:
    """Sanity: scripted oracle satisfies the runtime-checkable Oracle protocol."""
    assert isinstance(_ScriptedOracle(responses=["x"]), Oracle)


# ---------------------------------------------------------------------
# T6 — validates_args
# ---------------------------------------------------------------------
def test_niah_padded_validates_args() -> None:
    """T6 — bad tau/k surface as ValueError; bad pad_char rejected on helpers."""
    # Factory-level guards delegated to _recursive_long_context.
    with pytest.raises(ValueError, match="tau"):
        niah_padded("q", tau=0)
    with pytest.raises(ValueError, match="k"):
        niah_padded("q", k=1)

    # Helper-level guards.
    with pytest.raises(ValueError, match="n must be >= 0"):
        aligned_size(-1, 256, 2)
    with pytest.raises(ValueError, match="tau must be >= 1"):
        aligned_size(100, 0, 2)
    with pytest.raises(ValueError, match="k must be >= 2"):
        aligned_size(100, 256, 1)

    with pytest.raises(ValueError, match="pad_char"):
        pad_to_aligned("hi", 4, 2, pad_char="")
    with pytest.raises(ValueError, match="pad_char"):
        pad_to_aligned("hi", 4, 2, pad_char="..")
    with pytest.raises(ValueError, match="pad_char"):
        make_pad_callable(4, 2, pad_char="ab")
    with pytest.raises(ValueError, match="tau must be >= 1"):
        make_pad_callable(0, 2)


# ---------------------------------------------------------------------
# T7 — aligned_size math
# ---------------------------------------------------------------------
def test_aligned_size_math() -> None:
    """T7 — pure math contract for ``aligned_size``."""
    # No-op when already aligned.
    assert aligned_size(2048, 256, 2) == 2048
    assert aligned_size(256, 256, 2) == 256
    # Round up to next τ·k^d boundary.
    assert aligned_size(2000, 256, 2) == 2048
    assert aligned_size(257, 256, 2) == 512
    # n ≤ τ branch — return n unchanged.
    assert aligned_size(0, 256, 2) == 0
    assert aligned_size(1, 100, 2) == 1
    assert aligned_size(100, 100, 2) == 100
    # Worst case: n = τ·k^d + 1 forces +1 depth.
    assert aligned_size(9, 1, 2) == 16  # = 1·2^4
    assert aligned_size(8, 1, 2) == 8   # = 1·2^3 (no pad)
    # Non-power-of-2 k.
    assert aligned_size(28, 1, 3) == 81  # 1·3^4 = 81 (since 27 < 28)
    assert aligned_size(27, 1, 3) == 27  # exact match


# ---------------------------------------------------------------------
# T8 — custom reduce_op_name
# ---------------------------------------------------------------------
def test_niah_padded_custom_reduce_op_name() -> None:
    """T8 — caller can supply a custom reduce_op_name and bind it in env."""
    doc = "ab"  # n=2 ≤ τ=2 → no padding, but exercising the env path.
    # Use τ=1, k=2 to force splitting; doc 'ab' → padded 'ab' (already 2=τ·k^1).
    custom_op = ReduceOp(
        name="my_op", fn=lambda a, b: str(a) + str(b), associative=True, unit=""
    )
    oracle = _ScriptedOracle(responses=["A", "B"])
    ex = Executor(oracle=oracle)
    program = niah_padded("q", tau=1, k=2, reduce_op_name="my_op")
    result = ex.run(
        program,
        {
            "raw_document": doc,
            "pad_to_aligned": make_pad_callable(1, 2),
            "size_bucket": make_size_bucket(1),
            "my_op": custom_op,
        },
    )
    assert result == "AB"
    assert ex.oracle_calls == 2


# ---------------------------------------------------------------------
# T9 — pad_to_aligned correctness
# ---------------------------------------------------------------------
def test_pad_to_aligned_correctness() -> None:
    """T9 — output length == N*, raw doc preserved, padding char is correct."""
    doc = "hello"  # n=5
    padded = pad_to_aligned(doc, 4, 2)  # N* = 8
    assert len(padded) == aligned_size(len(doc), 4, 2)
    assert padded.startswith(doc)
    assert padded[len(doc):] == " " * (8 - 5)

    # Default pad_char is a single space.
    assert pad_to_aligned("xy", 1, 2) == "xy"  # n=2 = τ·k^1, no pad
    assert pad_to_aligned("xyz", 1, 2) == "xyz" + " " * (4 - 3)

    # n ≤ τ branch — no padding.
    assert pad_to_aligned("ab", 100, 2) == "ab"


# ---------------------------------------------------------------------
# T10 — no-op preservation when already aligned
# ---------------------------------------------------------------------
def test_no_op_when_already_aligned() -> None:
    """T10 — for aligned ``n`` (e.g. 8 = τ·k^3 with τ=1,k=2) niah_padded
    produces the same scripted-oracle result + call count as plain ``niah``
    on the unpadded doc."""
    doc = "AAAAXAAA"  # n=8, τ=1, k=2 → already aligned, no padding.
    assert aligned_size(len(doc), 1, 2) == 8

    # Plain niah baseline.
    base_responses = [
        "NOT_FOUND" if ch != "X" else "X" for ch in doc
    ]
    oracle_base = _ScriptedOracle(responses=base_responses)
    ex_base = Executor(oracle=oracle_base)
    prog_base = niah("q", tau=1, k=2)
    res_base = ex_base.run(
        prog_base,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    # Padded variant on the same (already-aligned) doc.
    padded_responses = [
        "NOT_FOUND" if ch != "X" else "X" for ch in doc
    ]
    oracle_pad = _ScriptedOracle(responses=padded_responses)
    ex_pad = Executor(oracle=oracle_pad)
    prog_pad = niah_padded("q", tau=1, k=2)
    res_pad = ex_pad.run(prog_pad, _env(doc, 1, 2))

    assert res_base == res_pad == "X"
    assert ex_base.oracle_calls == ex_pad.oracle_calls == 8


# ---------------------------------------------------------------------
# T11 — worst-case padding factor
# ---------------------------------------------------------------------
def test_worst_case_padding_factor() -> None:
    """T11 — ``n = τ·k^d + 1`` forces N* = τ·k^(d+1) (factor ~k leaves)."""
    # τ=1, k=2, d=3 → τ·k^d = 8; n=9 forces N*=16.
    n = 9
    n_star = aligned_size(n, 1, 2)
    assert n_star == 16

    doc = "A" * 8 + "X"  # n=9
    # Padded to 16: 'AAAAAAAAX' + 7 spaces → leaves: 8 A's, 1 X, 7 spaces.
    padded = pad_to_aligned(doc, 1, 2)
    assert padded == "AAAAAAAAX" + " " * 7
    assert len(padded) == 16

    responses = ["NOT_FOUND" if ch != "X" else "X" for ch in padded]
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = niah_padded("q", tau=1, k=2)
    result = ex.run(program, _env(doc, 1, 2))

    assert result == "X"
    assert ex.oracle_calls == 16
    predicted = plan(
        PlanInputs(n=n_star, K=10_000, tau=1, alpha=1.0, max_k=2)
    )
    assert ex.oracle_calls == predicted.predicted_calls == 16


# ---------------------------------------------------------------------
# T12 — custom pad_char
# ---------------------------------------------------------------------
def test_custom_pad_char() -> None:
    """T12 — ``pad_char='.'`` produces dots in padded slots."""
    doc = "AAXAA"  # n=5 → N*=8
    pad_fn = make_pad_callable(1, 2, pad_char=".")
    padded = pad_fn(doc)
    assert padded == "AAXAA..."
    assert len(padded) == 8

    # End-to-end: factory still produces correct call count when the env's
    # pad callable is a custom-char one (the factory itself is char-agnostic).
    responses = ["NOT_FOUND" if ch not in ("X",) else "X" for ch in padded]
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = niah_padded("q", tau=1, k=2)
    result = ex.run(
        program,
        {
            "raw_document": doc,
            "pad_to_aligned": pad_fn,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )
    assert result == "X"
    assert ex.oracle_calls == 8
