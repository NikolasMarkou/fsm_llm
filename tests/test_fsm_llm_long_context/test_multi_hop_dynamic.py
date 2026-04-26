from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context.multi_hop_dynamic — M5 slice 6."""

from typing import Any

import pytest

from fsm_llm.lam import Executor, PlanInputs, plan
from fsm_llm.stdlib.long_context import (
    best_answer_op,
    make_dynamic_hop_runner,
    make_size_bucket,
    multi_hop,
    multi_hop_dynamic,
    not_found_gate,
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


def _peer_env(tau: int) -> dict[str, Any]:
    return {"size_bucket": make_size_bucket(tau), "best": best_answer_op()}


def test_term_closed() -> None:
    """SC2: multi_hop_dynamic builds a closed Term whose shape is App(Var, Var)."""
    t = multi_hop_dynamic("question?", max_hops=4)
    assert t.kind == "App"
    assert t.fn.kind == "Var"
    assert t.fn.name == "dynamic_hop_runner"
    assert t.arg.kind == "Var"
    assert t.arg.name == "document"


def test_validation() -> None:
    with pytest.raises(ValueError, match="max_hops must be >= 1"):
        multi_hop_dynamic("q", max_hops=0)


def test_early_stop() -> None:
    """SC3: gate fires hop-0 (concrete answer) → actual_hops=1, oracle_calls=k^d.

    Doc length τ·k^d = 8 (τ=1, k=2, d=3). Hop 0 returns concrete answer
    on the 8th leaf; gate fires after hop 0 completes; no hop 1 runs.
    """
    doc = "aabbccdd"
    hop0_responses = ["NOT_FOUND"] * 7 + ["entity_X"]
    oracle = _ScriptedOracle(responses=hop0_responses)
    ex = Executor(oracle=oracle)

    actual_hops_cell = [0]
    runner = make_dynamic_hop_runner(
        ex,
        "What about X?",
        max_hops=4,
        peer_env=_peer_env(1),
        tau=1,
        k=2,
        actual_hops_cell=actual_hops_cell,
    )
    program = multi_hop_dynamic("What about X?", max_hops=4)
    result = ex.run(program, {"document": doc, "dynamic_hop_runner": runner})

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert result == "entity_X"
    assert actual_hops_cell[0] == 1
    assert ex.oracle_calls == 1 * predicted.predicted_calls
    assert len(oracle.calls) == 8


def test_full_run_no_gate_fire() -> None:
    """SC4: gate never fires (all NOT_FOUND) → actual_hops=max_hops, oracle_calls=max_hops*k^d."""
    doc = "aabbccdd"
    max_hops = 3
    # 8 leaves per hop * max_hops hops = 24 NOT_FOUND
    responses = ["NOT_FOUND"] * (8 * max_hops)
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    actual_hops_cell = [0]
    runner = make_dynamic_hop_runner(
        ex,
        "What about X?",
        max_hops=max_hops,
        peer_env=_peer_env(1),
        tau=1,
        k=2,
        actual_hops_cell=actual_hops_cell,
    )
    program = multi_hop_dynamic("What about X?", max_hops=max_hops)
    result = ex.run(program, {"document": doc, "dynamic_hop_runner": runner})

    predicted = plan(PlanInputs(n=len(doc), K=10_000, tau=1, alpha=1.0, max_k=2))
    assert actual_hops_cell[0] == max_hops
    assert ex.oracle_calls == max_hops * predicted.predicted_calls
    # No concrete result was found; runner returns last hop's result.
    assert result == "NOT_FOUND"


def test_max_hops_1_degenerate() -> None:
    """SC5: max_hops=1 reduces to a single niah-shape sweep.

    Cost equals one niah pass (multi_hop(hops=1) per slice 3).
    """
    doc = "aabbccdd"
    hop0_responses = ["NOT_FOUND"] * 7 + ["found_it"]

    # multi_hop_dynamic path
    oracle_dyn = _ScriptedOracle(responses=list(hop0_responses))
    ex_dyn = Executor(oracle=oracle_dyn)
    runner = make_dynamic_hop_runner(
        ex_dyn, "Q?", max_hops=1, peer_env=_peer_env(1), tau=1, k=2
    )
    program_dyn = multi_hop_dynamic("Q?", max_hops=1)
    result_dyn = ex_dyn.run(
        program_dyn, {"document": doc, "dynamic_hop_runner": runner}
    )

    # multi_hop(hops=1) reference path
    oracle_ref = _ScriptedOracle(responses=list(hop0_responses))
    ex_ref = Executor(oracle=oracle_ref)
    program_ref = multi_hop("Q?", hops=1, tau=1, k=2)
    result_ref = ex_ref.run(
        program_ref,
        {"document": doc, "size_bucket": make_size_bucket(1), "best": best_answer_op()},
    )

    assert result_dyn == result_ref == "found_it"
    assert ex_dyn.oracle_calls == ex_ref.oracle_calls == 8


def test_gate_exception_continues() -> None:
    """E4: gate exception → continue to next hop (don't kill the run)."""
    doc = "aabbccdd"
    # Hop 0 returns concrete answer; gate raises; hop 1 returns NOT_FOUND;
    # iteration completes max_hops=2 and returns last_concrete from hop 0.
    responses = (["NOT_FOUND"] * 7 + ["concrete_h0"]) + (["NOT_FOUND"] * 8)
    oracle = _ScriptedOracle(responses=responses)
    ex = Executor(oracle=oracle)

    def _bad_gate(result: Any, hop_index: int) -> bool:
        raise RuntimeError("gate boom")

    actual_hops_cell = [0]
    runner = make_dynamic_hop_runner(
        ex,
        "Q?",
        max_hops=2,
        peer_env=_peer_env(1),
        confidence_gate=_bad_gate,
        tau=1,
        k=2,
        actual_hops_cell=actual_hops_cell,
    )
    program = multi_hop_dynamic("Q?", max_hops=2)
    result = ex.run(program, {"document": doc, "dynamic_hop_runner": runner})

    assert actual_hops_cell[0] == 2
    assert ex.oracle_calls == 16
    # last_concrete from hop 0 preserved despite hop-1 sentinel
    assert result == "concrete_h0"


def test_not_found_gate_helper() -> None:
    """not_found_gate matches the documented contract."""
    g = not_found_gate()
    # Sentinel-form (any case + trailing punct/whitespace): gate does NOT fire.
    assert g("NOT_FOUND", 0) is False
    assert g("NOT_FOUND.", 0) is False
    assert g("not_found", 0) is False
    assert g("  NOT_FOUND  ", 0) is False
    # Anything that doesn't START WITH the sentinel literal is concrete.
    assert g("not found", 0) is True  # space, not underscore — concrete
    assert g("ACCESS_CODE: SECRET-7421", 0) is True
    assert g("real answer", 5) is True

    g_custom = not_found_gate(sentinel="UNKNOWN")
    assert g_custom("UNKNOWN", 0) is False
    assert g_custom("NOT_FOUND", 0) is True
