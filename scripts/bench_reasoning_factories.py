#!/usr/bin/env python
"""Bench script — M3 slice 2 reasoning λ-factories.

Runs all 9 strategy factories + ``classifier_term`` sequentially against a
real LLM (default: ``ollama_chat/qwen3.5:4b``), measuring per-cell
``oracle_calls``, ``predicted_calls`` (from ``Plan``), wall time, and
emits ``evaluation/m3_slice2_reasoning_scorecard.json``.

Theorem-2 (strict): ``oracle_calls == predicted_calls == leaf_count``
for every non-Fix factory in this slice.

Usage::

    .venv/bin/python scripts/bench_reasoning_factories.py
    .venv/bin/python scripts/bench_reasoning_factories.py --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fsm_llm.lam import App, Executor, Leaf, Let, LiteLLMOracle, Term, Var
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.reasoning.lam_factories import (
    abductive_term,
    analogical_term,
    analytical_term,
    calculator_term,
    classifier_term,
    creative_term,
    critical_term,
    deductive_term,
    hybrid_term,
    inductive_term,
)

PROBLEM = (
    "How does compound interest amplify long-term wealth growth, and "
    "what assumptions can break that effect?"
)

CALC_PROBLEM = "Compute: 17 * 23 + 14"


def _3leaf_kwargs(b1: str, b2: str) -> dict[str, str]:
    return {
        "prompt_a": "Stage 1.\nProblem: {problem}",
        "prompt_b": f"Stage 2.\nProblem: {{problem}}\n{b1.title()}: {{{b1}}}",
        "prompt_c": f"Stage 3.\nProblem: {{problem}}\n{b2.title()}: {{{b2}}}",
    }


def _static_leaf_count(term: Term) -> int:
    """Recursively count Leaf nodes — skips Var/App-fn/App-arg sub-trees
    that are not Leaves themselves. For non-Fix factories this is the
    Theorem-2 ``predicted_calls``."""
    if isinstance(term, Leaf):
        return 1
    if isinstance(term, Let):
        return _static_leaf_count(term.value) + _static_leaf_count(term.body)
    if isinstance(term, App):
        return _static_leaf_count(term.fn) + _static_leaf_count(term.arg)
    if isinstance(term, Var):
        return 0
    return 0


def build_cells() -> list[dict[str, Any]]:
    """Return list of {factory, leaves, term, env} entries."""
    return [
        {
            "factory": "analytical_term",
            "leaves": 3,
            "term": analytical_term(**_3leaf_kwargs("decomposition", "analysis")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "deductive_term",
            "leaves": 3,
            "term": deductive_term(**_3leaf_kwargs("premises", "inference")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "inductive_term",
            "leaves": 3,
            "term": inductive_term(**_3leaf_kwargs("examples", "pattern")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "abductive_term",
            "leaves": 3,
            "term": abductive_term(**_3leaf_kwargs("observation", "hypothesis")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "analogical_term",
            "leaves": 3,
            "term": analogical_term(**_3leaf_kwargs("source_domain", "mapping")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "creative_term",
            "leaves": 3,
            "term": creative_term(**_3leaf_kwargs("divergence", "combination")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "critical_term",
            "leaves": 3,
            "term": critical_term(**_3leaf_kwargs("examination", "evaluation")),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "hybrid_term",
            "leaves": 4,
            "term": hybrid_term(
                facets_prompt="List 2-3 facets.\nProblem: {problem}",
                strategies_prompt=(
                    "Pick a strategy per facet.\nProblem: {problem}\nFacets: {facets}"
                ),
                execute_prompt=(
                    "Execute briefly.\nProblem: {problem}\nStrategies: {strategies}"
                ),
                integrate_prompt=(
                    "Integrate.\nProblem: {problem}\nExecution: {execution}"
                ),
            ),
            "env": {"problem": PROBLEM},
        },
        {
            "factory": "calculator_term",
            "leaves": 2,
            "term": calculator_term(
                parse_prompt="Parse the expression.\nProblem: {problem}",
                compute_prompt=(
                    "Compute the result.\nProblem: {problem}\nParsed: {parsed}"
                ),
            ),
            "env": {"problem": CALC_PROBLEM},
        },
        {
            "factory": "classifier_term",
            "leaves": 4,
            "term": classifier_term(
                domain_prompt="Domain in one phrase.\nProblem: {problem}",
                structure_prompt=(
                    "Structure briefly.\nProblem: {problem}\nDomain: {domain}"
                ),
                needs_prompt=(
                    "Reasoning needs in 1-2 lines.\n"
                    "Problem: {problem}\nStructure: {structure}"
                ),
                recommend_prompt=(
                    "Recommend ONE strategy.\nProblem: {problem}\nNeeds: {needs}"
                ),
            ),
            "env": {"problem": PROBLEM},
        },
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ollama_chat/qwen3.5:4b")
    p.add_argument(
        "--out",
        default="evaluation/m3_slice2_reasoning_scorecard.json",
    )
    args = p.parse_args()

    llm = LiteLLMInterface(model=args.model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)

    rows: list[dict[str, Any]] = []
    cells = build_cells()
    for cell in cells:
        ex = Executor(oracle=oracle)
        # Predicted calls = static leaf count of the term (Theorem-2 LHS).
        predicted = _static_leaf_count(cell["term"])

        t0 = time.perf_counter()
        try:
            _ = ex.run(cell["term"], cell["env"])
            wall = time.perf_counter() - t0
            actual = ex.oracle_calls
            ok = (actual == cell["leaves"]) and (
                predicted is None or predicted == cell["leaves"]
            )
            err = None
        except Exception as e:  # pragma: no cover - bench resilience
            wall = time.perf_counter() - t0
            actual = ex.oracle_calls
            ok = False
            err = repr(e)
        row = {
            "factory": cell["factory"],
            "leaves": cell["leaves"],
            "oracle_calls": actual,
            "predicted_calls": predicted,
            "theorem2_holds": ok,
            "wall_seconds": round(wall, 2),
            "error": err,
        }
        rows.append(row)
        print(
            f"  [{cell['factory']:<18}] leaves={cell['leaves']} "
            f"oracle={actual} predicted={predicted} "
            f"theorem2={'yes' if ok else 'NO'} wall={wall:.1f}s"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "cells": rows,
        "all_theorem2_hold": all(r["theorem2_holds"] for r in rows),
        "total_wall_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
        "n_cells": len(rows),
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nWrote {out_path} (n_cells={len(rows)}, "
        f"theorem2_all={summary['all_theorem2_hold']}, "
        f"total_wall={summary['total_wall_seconds']}s)"
    )
    return 0 if summary["all_theorem2_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
