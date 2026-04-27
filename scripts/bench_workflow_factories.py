#!/usr/bin/env python
"""Bench script — M3 slice 3 workflow λ-factories.

Runs the 5 workflow factories sequentially against a real LLM
(default: ``ollama_chat/qwen3.5:4b``), measuring per-cell ``oracle_calls``,
runtime upper-bound (when applicable), and wall time. Emits
``evaluation/m3_slice3_workflow_scorecard.json``.

Theorem-2 forms:
- ``linear_term`` / ``parallel_term``: strict — ``oracle == sum(leaves)``
- ``branch_term`` / ``switch_term``: runtime — ``oracle == leaves(taken_arm)``
  (we record both ``oracle_calls`` and ``static_upper_bound``)
- ``retry_term``: ``oracle == 0`` (host-callable body)

Usage::

    .venv/bin/python scripts/bench_workflow_factories.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fsm_llm.lam import (
    App,
    Case,
    Combinator,
    Executor,
    Leaf,
    Let,
    LiteLLMOracle,
    ReduceOp,
    Term,
    Var,
    leaf,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.workflows.lam_factories import (
    branch_term,
    linear_term,
    parallel_term,
    retry_term,
    switch_term,
)


def _static_leaf_upper_bound(term: Term) -> int:
    if isinstance(term, Leaf):
        return 1
    if isinstance(term, Let):
        return _static_leaf_upper_bound(term.value) + _static_leaf_upper_bound(
            term.body
        )
    if isinstance(term, App):
        return _static_leaf_upper_bound(term.fn) + _static_leaf_upper_bound(term.arg)
    if isinstance(term, Case):
        n = _static_leaf_upper_bound(term.scrutinee)
        for v in term.branches.values():
            n += _static_leaf_upper_bound(v)
        if term.default is not None:
            n += _static_leaf_upper_bound(term.default)
        return n
    if isinstance(term, Combinator):
        return sum(_static_leaf_upper_bound(o) for o in (term.args or []))
    if isinstance(term, Var):
        return 0
    return 0


def build_cells() -> list[dict[str, Any]]:
    a = leaf(template="A: {input}", input_vars=("input",))
    b = leaf(template="B: {a}", input_vars=("a",))
    c = leaf(template="C: {b}", input_vars=("b",))
    linear_t = linear_term(("a", a), ("b", b), ("c", c))

    t_arm = leaf(template="THEN: {input}", input_vars=("input",))
    e_arm = leaf(template="ELSE: {input}", input_vars=("input",))
    branch_t = branch_term("cond", t_arm, e_arm)

    arms = {
        "a": leaf(template="A: {input}", input_vars=("input",)),
        "b": leaf(template="B: {input}", input_vars=("input",)),
        "c": leaf(template="C: {input}", input_vars=("input",)),
    }
    switch_def = leaf(template="DEF: {input}", input_vars=("input",))
    switch_t = switch_term("clf", arms, switch_def)

    p1 = leaf(template="P1: {input}", input_vars=("input",))
    p2 = leaf(template="P2: {input}", input_vars=("input",))
    p3 = leaf(template="P3: {input}", input_vars=("input",))
    parallel_t = parallel_term(
        [("p1", p1), ("p2", p2), ("p3", p3)],
        reduce_op_name="my_concat",
    )

    retry_t = retry_term("body", "success")

    my_concat = ReduceOp(
        name="my_concat",
        fn=lambda a, b: f"{a}|{b}",
        associative=True,
        unit="",
    )

    return [
        {
            "factory": "linear_term",
            "term": linear_t,
            "env": {"input": "ocean tides"},
            "expected_oracle_calls": 3,
            "theorem2_form": "strict",
        },
        {
            "factory": "branch_term",
            "term": branch_t,
            "env": {"input": "Is it raining?", "cond": lambda x: "true"},
            "expected_oracle_calls": 1,
            "theorem2_form": "runtime_arm_only",
        },
        {
            "factory": "switch_term",
            "term": switch_t,
            "env": {"input": "test", "clf": lambda x: "b"},
            "expected_oracle_calls": 1,
            "theorem2_form": "runtime_arm_only",
        },
        {
            "factory": "parallel_term",
            "term": parallel_t,
            "env": {
                "input": "rain",
                "build_branch_list": lambda last: [last],
                "identity": lambda x: x,
                "my_concat": my_concat,
            },
            "expected_oracle_calls": 3,
            "theorem2_form": "strict",
        },
        {
            "factory": "retry_term",
            "term": retry_t,
            "env": {
                "input": "task",
                "body": lambda x: f"done-{x}",
                "success": lambda _a: "true",
            },
            "expected_oracle_calls": 0,
            "theorem2_form": "no_leaf",
        },
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ollama_chat/qwen3.5:4b")
    p.add_argument(
        "--out",
        default="evaluation/m3_slice3_workflow_scorecard.json",
    )
    args = p.parse_args()

    llm = LiteLLMInterface(model=args.model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)

    rows: list[dict[str, Any]] = []
    for cell in build_cells():
        ex = Executor(oracle=oracle)
        upper = _static_leaf_upper_bound(cell["term"])
        t0 = time.perf_counter()
        try:
            _ = ex.run(cell["term"], cell["env"])
            wall = time.perf_counter() - t0
            actual = ex.oracle_calls
            ok = actual == cell["expected_oracle_calls"]
            err = None
        except Exception as e:  # pragma: no cover
            wall = time.perf_counter() - t0
            actual = ex.oracle_calls
            ok = False
            err = repr(e)

        row = {
            "factory": cell["factory"],
            "theorem2_form": cell["theorem2_form"],
            "expected_oracle_calls": cell["expected_oracle_calls"],
            "actual_oracle_calls": actual,
            "static_leaf_upper_bound": upper,
            "actual_le_upper_bound": actual <= upper,
            "theorem2_holds": ok,
            "wall_seconds": round(wall, 2),
            "error": err,
        }
        rows.append(row)
        print(
            f"  [{cell['factory']:<14}] expected={cell['expected_oracle_calls']} "
            f"actual={actual} upper={upper} ({cell['theorem2_form']}) "
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
