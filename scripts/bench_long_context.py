#!/usr/bin/env python3
"""
Multi-model benchmark for fsm_llm.stdlib.long_context factories.

Sequential in-process runner (D-S2-004): for each (model x factory) cell,
build a synthetic haystack, run the factory under a LiteLLMOracle for
that model, capture telemetry (oracle calls, tokens, wall time, Theorem-2
equality), and emit a JSON scorecard.

Usage::

    python scripts/bench_long_context.py \
        --models ollama_chat/qwen3.5:4b,ollama_chat/qwen3.5:9b \
        --factories niah,aggregate \
        --doc-size 2048 --tau 256

Output: ``evaluation/bench_long_context_<UTC-isoformat>.json`` with shape::

    {
        "date": "2026-04-25T...Z",
        "git_commit": "abc1234",
        "doc_size": 2048,
        "tau": 256,
        "k": 2,
        "records": [
            {
                "model": "...", "factory": "niah", "ok": true,
                "oracle_calls_actual": 8, "oracle_calls_predicted": 8,
                "theorem2_holds": true,
                "tokens_in": 2103, "tokens_out": 156, "total_cost": 0.0,
                "wall_time_s": 12.34,
                "needle_found": true, "heuristic_pass": null,
                "error": null
            },
            ...
        ]
    }

This is slice 2's headline deliverable: SC7 requires every record's
``theorem2_holds`` to be ``true``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    aggregate,
    aggregate_op,
    best_answer_op,
    make_size_bucket,
    niah,
)

NEEDLE = "ACCESS_CODE: SECRET-7421"
NEEDLE_OFFSET = 1024


def build_haystack(doc_size: int) -> str:
    """Reusable haystack with one needle at chunk-aligned offset 1024."""
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    )
    body = (filler * ((doc_size // len(filler)) + 2))[:doc_size]
    needle_padded = " " + NEEDLE + " "
    if NEEDLE_OFFSET + len(needle_padded) > doc_size:
        # Needle won't fit at the canonical offset; tuck at end instead.
        offset = max(0, doc_size - len(needle_padded))
    else:
        offset = NEEDLE_OFFSET
    doc = (body[:offset] + needle_padded + body[offset + len(needle_padded) :])[
        :doc_size
    ]
    assert len(doc) == doc_size
    return doc


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_cell(
    model: str,
    factory_name: str,
    doc_size: int,
    tau: int,
    k: int,
) -> dict[str, Any]:
    """Run one (model x factory) cell. Captures telemetry, never raises."""
    record: dict[str, Any] = {
        "model": model,
        "factory": factory_name,
        "ok": False,
        "oracle_calls_actual": None,
        "oracle_calls_predicted": None,
        "theorem2_holds": None,
        "tokens_in": None,
        "tokens_out": None,
        "total_cost": None,
        "wall_time_s": None,
        "needle_found": None,
        "heuristic_pass": None,
        "error": None,
    }

    haystack = build_haystack(doc_size)
    predicted = plan(PlanInputs(n=doc_size, K=10_000, tau=tau, alpha=1.0, max_k=k))
    record["oracle_calls_predicted"] = predicted.predicted_calls

    if factory_name == "niah":
        program = niah(
            "What is the access code? Reply with just the code value.",
            tau=tau,
            k=k,
        )
        env_op_name = "best"
        env_op = best_answer_op()
    elif factory_name == "aggregate":
        program = aggregate(
            "What is the most important fact in this document?",
            tau=tau,
            k=k,
        )
        env_op_name = "merge"
        env_op = aggregate_op()
    else:
        record["error"] = f"unknown factory: {factory_name}"
        return record

    try:
        llm = LiteLLMInterface(model=model)
        oracle = LiteLLMOracle(llm, context_window_tokens=8192)
        ex = Executor(oracle=oracle)

        env = {
            "document": haystack,
            "size_bucket": make_size_bucket(tau),
            env_op_name: env_op,
        }

        t0 = time.perf_counter()
        result = ex.run(program, env)
        wall = time.perf_counter() - t0

        record["ok"] = True
        record["wall_time_s"] = round(wall, 3)
        record["oracle_calls_actual"] = ex.oracle_calls
        record["theorem2_holds"] = ex.oracle_calls == predicted.predicted_calls
        record["tokens_in"] = ex.cost_accumulator.total_tokens_in
        record["tokens_out"] = ex.cost_accumulator.total_tokens_out
        record["total_cost"] = ex.cost_accumulator.total_cost

        if factory_name == "niah":
            record["needle_found"] = isinstance(result, str) and "SECRET-7421" in result
        elif factory_name == "aggregate":
            record["heuristic_pass"] = (
                isinstance(result, str) and len(result) > 50 and result != "NOT_FOUND"
            )
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument(
        "--models",
        default="ollama_chat/qwen3.5:4b",
        help="Comma-separated list of models. Default: ollama_chat/qwen3.5:4b",
    )
    p.add_argument(
        "--factories",
        default="niah,aggregate",
        help="Comma-separated factories from {niah,aggregate}. "
        "Default: niah,aggregate",
    )
    p.add_argument("--doc-size", type=int, default=2048)
    p.add_argument("--tau", type=int, default=256)
    p.add_argument("--k", type=int, default=2)
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: evaluation/bench_long_context_<ts>.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    factories = [f.strip() for f in args.factories.split(",") if f.strip()]
    valid = {"niah", "aggregate"}
    bad = [f for f in factories if f not in valid]
    if bad:
        print(f"Unknown factories: {bad}; valid={sorted(valid)}", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(
        args.out or f"evaluation/bench_long_context_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Bench: {len(models)} model(s) x {len(factories)} factor"
        f"{'ies' if len(factories) != 1 else 'y'} = {len(models) * len(factories)} cells"
    )
    print(f"  doc_size={args.doc_size} tau={args.tau} k={args.k}")
    print(f"  out={out_path}")
    print("-" * 60)

    records: list[dict[str, Any]] = []
    all_holds = True
    for model in models:
        for factory in factories:
            r = run_cell(model, factory, args.doc_size, args.tau, args.k)
            records.append(r)
            status = "OK" if r["ok"] else "FAIL"
            if r["ok"]:
                line = (
                    f"[{status}] model={model} factory={factory} "
                    f"calls={r['oracle_calls_actual']}/{r['oracle_calls_predicted']} "
                    f"theorem2={r['theorem2_holds']} "
                    f"tokens={r['tokens_in']}+{r['tokens_out']} "
                    f"t={r['wall_time_s']}s"
                )
            else:
                line = f"[{status}] model={model} factory={factory} error={r['error']}"
            print(line)
            if r["ok"] and not r["theorem2_holds"]:
                all_holds = False
            elif not r["ok"]:
                all_holds = False
            # Persist incrementally so a crash doesn't lose work.
            scorecard = {
                "date": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
                "doc_size": args.doc_size,
                "tau": args.tau,
                "k": args.k,
                "records": records,
            }
            out_path.write_text(json.dumps(scorecard, indent=2))

    print("-" * 60)
    print(f"Wrote scorecard: {out_path}")
    print(f"Theorem-2 holds across all cells: {all_holds}")
    return 0 if all_holds else 1


if __name__ == "__main__":
    sys.exit(main())
