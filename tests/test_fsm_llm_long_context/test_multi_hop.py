from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.multi_hop — M5 slice 3."""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.lam import Executor, Oracle, PlanInputs, plan
from fsm_llm.stdlib.long_context import (
    best_answer_op,
    make_size_bucket,
    multi_hop,
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


def test_multi_hop_smoke_two_hops() -> None:
    """T1 — 2-hop scripted-oracle smoke.

    Doc length τ·k^d = 8 (τ=1, k=2, d=3). Each hop sweeps 8 leaves.
    Hop 0: 7 NOT_FOUND + 1 "entity_X" (the winner).
    Hop 1: 7 NOT_FOUND + 1 "fact_about_entity_X".
    Total oracle calls = 2 * 8 = 16. Final result = hop-1 winner.
    """
    doc = "aabbccdd"
    hop0_responses = ["NOT_FOUND"] * 7 + ["entity_X"]
    hop1_responses = ["NOT_FOUND"] * 7 + ["fact_about_entity_X"]
    responses = hop0_responses + hop1_responses

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    program = multi_hop("What about X?", hops=2, tau=1, k=2)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    assert result == "fact_about_entity_X"
    assert ex.oracle_calls == 16
    assert len(oracle.calls) == 16


def test_multi_hop_cost_equality_sc2() -> None:
    """T2 — SC2 invariant: ex.oracle_calls == hops * predicted_calls."""
    doc = "aabbccdd"
    responses = (["NOT_FOUND"] * 7 + ["winner_h0"]) + (
        ["NOT_FOUND"] * 7 + ["winner_h1"]
    )

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = multi_hop("q", hops=2, tau=1, k=2)
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
    assert ex.oracle_calls == 2 * predicted.predicted_calls == 16


def test_multi_hop_degenerate_hops_one() -> None:
    """T3 — hops=1 reduces to a single niah-shaped sweep.

    Same doc shape; same call count as a niah sweep over the same doc.
    """
    doc = "aabbccdd"
    responses = ["NOT_FOUND"] * 7 + ["the_answer"]

    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)
    program = multi_hop("q", hops=1, tau=1, k=2)
    result = ex.run(
        program,
        {
            "document": doc,
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert result == "the_answer"
    assert ex.oracle_calls == predicted.predicted_calls == 8


def test_multi_hop_degenerate_tau() -> None:
    """T4 — len(doc) <= τ and hops=2: 1 leaf per hop, 2 calls total."""
    oracle = _ScriptedOracle(responses=["entity_A", "fact_about_A"])
    ex = Executor(oracle=oracle)

    program = multi_hop("q", hops=2, tau=100, k=2)
    result = ex.run(
        program,
        {
            "document": "tiny doc",
            "size_bucket": make_size_bucket(100),
            "best": best_answer_op(),
        },
    )

    assert result == "fact_about_A"
    assert ex.oracle_calls == 2


def test_multi_hop_purity() -> None:
    """T5 — multi_hop.py must not import fsm_llm.{llm,fsm,pipeline}."""
    pkg_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fsm_llm"
        / "stdlib"
        / "long_context"
    )
    assert pkg_dir.is_dir(), f"long_context package not found at {pkg_dir}"

    forbidden = ("fsm_llm.llm", "fsm_llm.fsm", "fsm_llm.pipeline")
    src = (pkg_dir / "multi_hop.py").read_text()
    offenders: list[str] = []
    for needle in forbidden:
        if f"from {needle}" in src or f"import {needle}" in src:
            offenders.append(f"multi_hop.py: {needle}")
    assert offenders == [], f"purity violation(s): {offenders}"


def test_multi_hop_oracle_protocol_conformance() -> None:
    """T6 — scripted oracle satisfies the runtime-checkable Oracle protocol."""
    assert isinstance(_ScriptedOracle(responses=["x"]), Oracle)


def test_multi_hop_validates_args() -> None:
    """T7 — bad args surface as ValueError, not silent."""
    with pytest.raises(ValueError, match="hops"):
        multi_hop("q", hops=0)
    with pytest.raises(ValueError, match="tau"):
        multi_hop("q", hops=2, tau=0)
    with pytest.raises(ValueError, match="k"):
        multi_hop("q", hops=2, k=1)


def test_multi_hop_hop1_prompt_threads_prior_result() -> None:
    """T8 — hop-1 leaf prompts must contain hop-0's result (env threading)."""
    # Use degenerate-tau so each hop fires exactly one leaf — easier to
    # inspect prompts.
    oracle = _ScriptedOracle(responses=["entity_ZZZ", "fact_about_ZZZ"])
    ex = Executor(oracle=oracle)

    program = multi_hop("What is ZZZ?", hops=2, tau=100, k=2)
    ex.run(
        program,
        {
            "document": "tiny doc",
            "size_bucket": make_size_bucket(100),
            "best": best_answer_op(),
        },
    )

    assert len(oracle.calls) == 2
    hop0_prompt, hop1_prompt = oracle.calls
    # Hop 0 prompt: contains the question, does NOT mention prior finding.
    assert "What is ZZZ?" in hop0_prompt
    assert "Prior finding:" not in hop0_prompt
    # Hop 1 prompt: contains the question AND the prior hop's result.
    assert "What is ZZZ?" in hop1_prompt
    assert "Prior finding:" in hop1_prompt
    assert "entity_ZZZ" in hop1_prompt
