#!/usr/bin/env python3
"""
Multi-model benchmark for fsm_llm.stdlib.long_context factories.

Per-cell runner: for each (model x factory) cell, build a synthetic haystack,
run the factory under a LiteLLMOracle for that model, capture telemetry
(oracle calls, tokens, wall time, Theorem-2 equality), and emit a JSON
scorecard. Records are written incrementally per cell so a crash never loses
prior work.

Usage::

    python scripts/bench_long_context.py \\
        --models ollama_chat/qwen3.5:4b,ollama_chat/qwen3.5:9b \\
        --factories niah,aggregate,pairwise,multi_hop \\
        --doc-size 2048 --tau 256 --workers 1

Worker semantics (D-S3-003)
---------------------------

``--workers N`` (default 1):

- ``N == 1`` (default): sequential nested loop, model-then-factory order. The
  legacy slice-2 behaviour, byte-identical record output.
- ``N > 1``: ``concurrent.futures.ProcessPoolExecutor(max_workers=N)`` with
  ``as_completed()``. Each cell is submitted as a primitive-arg future
  (``run_cell(model, factory_name, doc_size, tau, k, git_commit)``); the
  main process accumulates results and writes the scorecard incrementally
  on every completion. ``run_cell`` constructs LiteLLM/Executor objects
  inside the worker — no closure state crosses the pickle boundary.

Caveat: ``--workers > 1`` benefits cloud APIs (OpenAI / Anthropic / Gemini /
Bedrock / Azure) only. Ollama serializes requests server-side; parallel
workers there yield no speedup but also do not crash.

Cloud env-var preflight
-----------------------

For each non-``ollama_chat/*`` model in ``--models``, the harness emits a
warning to stderr if the expected env var is unset. The mapping is:

==============   =====================
Model prefix     Expected env var
==============   =====================
``gpt*``         ``OPENAI_API_KEY``
``anthropic/*``  ``ANTHROPIC_API_KEY``
``gemini/*``     ``GOOGLE_API_KEY``
``bedrock/*``    ``AWS_ACCESS_KEY_ID``
``azure/*``      ``AZURE_API_KEY``
==============   =====================

Preflight is advisory only — LiteLLM may fall back to other env vars or
config files. The harness does not abort.

Output
------

``evaluation/bench_long_context_<UTC-isoformat>.json`` (or ``--out``) with
shape::

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
                "hops": null, "pairwise_winner_len": null,
                "error": null
            },
            ...
        ]
    }

SC5 requires every record's ``theorem2_holds`` to be ``true``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    aggregate,
    aggregate_op,
    aligned_size,
    best_answer_op,
    compare_op,
    make_pad_callable,
    make_size_bucket,
    multi_hop,
    niah,
    niah_padded,
    pairwise,
)

NEEDLE = "ACCESS_CODE: SECRET-7421"
NEEDLE_OFFSET = 1024

# Slice-4 niah_padded cell uses a deliberately UN-aligned doc size:
# 1024 < 2000 < 2048 = τ·k^3 (with τ=256, k=2). The factory pads to 2048
# internally so the planner-executor cost-equality contract holds against
# N* = aligned_size(NIAH_PADDED_DOC_SIZE, tau, k).
NIAH_PADDED_DOC_SIZE = 2000

# Cloud-model env-var preflight map. Prefix → expected env var name.
# DECISION D-S3-003: warning-only; LiteLLM may fall back to other config.
CLOUD_ENV_MAP: list[tuple[str, str]] = [
    ("gpt", "OPENAI_API_KEY"),
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("gemini/", "GOOGLE_API_KEY"),
    ("bedrock/", "AWS_ACCESS_KEY_ID"),
    ("azure/", "AZURE_API_KEY"),
]

# Multi-hop slice-3 default (hardcoded per plan step 5 spec).
MULTI_HOP_HOPS = 2


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


def preflight_env(models: list[str]) -> None:
    """Warn (do not abort) when cloud models are missing expected env vars."""
    for model in models:
        if model.startswith("ollama_chat/"):
            continue
        for prefix, env_var in CLOUD_ENV_MAP:
            # Prefix match; "gpt" matches "gpt-4o" etc., "anthropic/" matches
            # "anthropic/claude-3-5-sonnet" etc.
            if model.startswith(prefix) and not os.environ.get(env_var):
                print(
                    f"WARN: model {model} expects env var {env_var}; "
                    "not set in environment",
                    file=sys.stderr,
                )


def run_cell(
    model: str,
    factory_name: str,
    doc_size: int,
    tau: int,
    k: int,
    git_commit_str: str,
) -> dict[str, Any]:
    """Run one (model x factory) cell. Captures telemetry, never raises.

    Picklable: all args are primitives. All LiteLLM/Executor objects are
    constructed inside this function so the worker process is self-contained.
    """
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
        "hops": None,
        "pairwise_winner_len": None,
        "padded_size": None,
        "raw_doc_size": None,
        "git_commit": git_commit_str,
        "error": None,
    }

    haystack = build_haystack(doc_size)
    predicted = plan(PlanInputs(n=doc_size, K=10_000, tau=tau, alpha=1.0, max_k=k))

    # Build program + env per factory.
    if factory_name == "niah":
        program = niah(
            "What is the access code? Reply with just the code value.",
            tau=tau,
            k=k,
        )
        env = {
            "document": haystack,
            "size_bucket": make_size_bucket(tau),
            "best": best_answer_op(),
        }
        predicted_calls = predicted.predicted_calls
    elif factory_name == "aggregate":
        program = aggregate(
            "What is the most important fact in this document?",
            tau=tau,
            k=k,
        )
        env = {
            "document": haystack,
            "size_bucket": make_size_bucket(tau),
            "merge": aggregate_op(),
        }
        predicted_calls = predicted.predicted_calls
    elif factory_name == "pairwise":
        program = pairwise(
            question="Which segment discusses topic X?",
            tau=tau,
            k=k,
        )
        env = {
            "document": haystack,
            "size_bucket": make_size_bucket(tau),
            "compare": compare_op(),
        }
        predicted_calls = predicted.predicted_calls
    elif factory_name == "multi_hop":
        program = multi_hop(
            question="Find the first entity, then a fact about it.",
            hops=MULTI_HOP_HOPS,
            tau=tau,
            k=k,
        )
        env = {
            "document": haystack,
            "size_bucket": make_size_bucket(tau),
            "best": best_answer_op(),
        }
        # DECISION D-S3-002: each hop is an independent Fix call; cost
        # equality is hops * predicted_calls.
        predicted_calls = MULTI_HOP_HOPS * predicted.predicted_calls
        record["hops"] = MULTI_HOP_HOPS
    elif factory_name == "niah_padded":
        # DECISION D-S4-001: niah_padded cell uses an UN-aligned raw size
        # (NIAH_PADDED_DOC_SIZE) and re-plans against N* (the padded
        # boundary). The factory binds raw_document and resolves the inner
        # document via Let(pad_to_aligned(raw_document)).
        n_star = aligned_size(NIAH_PADDED_DOC_SIZE, tau, k)
        predicted_padded = plan(
            PlanInputs(n=n_star, K=10_000, tau=tau, alpha=1.0, max_k=k)
        )
        program = niah_padded(
            "What is the access code? Reply with just the code value.",
            tau=tau,
            k=k,
        )
        env = {
            "raw_document": build_haystack(NIAH_PADDED_DOC_SIZE),
            "pad_to_aligned": make_pad_callable(tau, k),
            "size_bucket": make_size_bucket(tau),
            "best": best_answer_op(),
        }
        # Override the haystack: the cell measures the padded variant,
        # not the doc_size used by the other factories.
        haystack = env["raw_document"]
        predicted_calls = predicted_padded.predicted_calls
        record["padded_size"] = n_star
        record["raw_doc_size"] = NIAH_PADDED_DOC_SIZE
    else:
        record["error"] = f"unknown factory: {factory_name}"
        return record

    record["oracle_calls_predicted"] = predicted_calls

    try:
        llm = LiteLLMInterface(model=model)
        oracle = LiteLLMOracle(llm, context_window_tokens=8192)
        ex = Executor(oracle=oracle)

        t0 = time.perf_counter()
        result = ex.run(program, env)
        wall = time.perf_counter() - t0

        record["ok"] = True
        record["wall_time_s"] = round(wall, 3)
        record["oracle_calls_actual"] = ex.oracle_calls
        record["theorem2_holds"] = ex.oracle_calls == predicted_calls
        record["tokens_in"] = ex.cost_accumulator.total_tokens_in
        record["tokens_out"] = ex.cost_accumulator.total_tokens_out
        record["total_cost"] = ex.cost_accumulator.total_cost

        if factory_name == "niah":
            record["needle_found"] = (
                isinstance(result, str) and "SECRET-7421" in result
            )
        elif factory_name == "niah_padded":
            record["needle_found"] = (
                isinstance(result, str) and "SECRET-7421" in result
            )
        elif factory_name == "aggregate":
            record["heuristic_pass"] = (
                isinstance(result, str) and len(result) > 50 and result != "NOT_FOUND"
            )
        elif factory_name == "pairwise":
            record["pairwise_winner_len"] = (
                len(result) if isinstance(result, str) else None
            )
        elif factory_name == "multi_hop":
            record["heuristic_pass"] = (
                isinstance(result, str) and result != "NOT_FOUND" and len(result) > 0
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
        help="Comma-separated factories from {niah,aggregate,pairwise,"
        "multi_hop,niah_padded}. Default: niah,aggregate",
    )
    p.add_argument("--doc-size", type=int, default=2048)
    p.add_argument("--tau", type=int, default=256)
    p.add_argument("--k", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Default 1 (sequential). "
        ">1 uses ProcessPoolExecutor; benefits cloud APIs only — Ollama "
        "serializes server-side. (D-S3-003)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: evaluation/bench_long_context_<ts>.json",
    )
    return p.parse_args(argv)


def _format_status_line(model: str, factory: str, r: dict[str, Any]) -> str:
    status = "OK" if r["ok"] else "FAIL"
    if r["ok"]:
        return (
            f"[{status}] model={model} factory={factory} "
            f"calls={r['oracle_calls_actual']}/{r['oracle_calls_predicted']} "
            f"theorem2={r['theorem2_holds']} "
            f"tokens={r['tokens_in']}+{r['tokens_out']} "
            f"t={r['wall_time_s']}s"
        )
    return f"[{status}] model={model} factory={factory} error={r['error']}"


def _write_scorecard(
    out_path: Path,
    args: argparse.Namespace,
    git_commit_str: str,
    records: list[dict[str, Any]],
) -> None:
    scorecard = {
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_str,
        "doc_size": args.doc_size,
        "tau": args.tau,
        "k": args.k,
        "workers": args.workers,
        "records": records,
    }
    out_path.write_text(json.dumps(scorecard, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    factories = [f.strip() for f in args.factories.split(",") if f.strip()]
    valid = {"niah", "aggregate", "pairwise", "multi_hop", "niah_padded"}
    bad = [f for f in factories if f not in valid]
    if bad:
        print(f"Unknown factories: {bad}; valid={sorted(valid)}", file=sys.stderr)
        return 2

    if args.workers < 1:
        print(
            f"--workers must be >= 1, got {args.workers}",
            file=sys.stderr,
        )
        return 2

    # Cloud env-var preflight (warnings only).
    preflight_env(models)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(args.out or f"evaluation/bench_long_context_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = len(models) * len(factories)
    print(
        f"Bench: {len(models)} model(s) x {len(factories)} factor"
        f"{'ies' if len(factories) != 1 else 'y'} = {n_cells} cells "
        f"(workers={args.workers})"
    )
    print(f"  doc_size={args.doc_size} tau={args.tau} k={args.k}")
    print(f"  out={out_path}")
    print("-" * 60)

    git_commit_str = git_commit()
    records: list[dict[str, Any]] = []
    all_holds = True

    if args.workers == 1:
        # Sequential path — preserves slice-2 behaviour exactly.
        for model in models:
            for factory in factories:
                r = run_cell(
                    model, factory, args.doc_size, args.tau, args.k, git_commit_str
                )
                records.append(r)
                print(_format_status_line(model, factory, r))
                if not (r["ok"] and r["theorem2_holds"]):
                    all_holds = False
                _write_scorecard(out_path, args, git_commit_str, records)
    else:
        # Parallel path (D-S3-003) — submit all cells, drain via as_completed,
        # main-process-only writes preserve crash-safe protocol.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_cell = {}
            for model in models:
                for factory in factories:
                    fut = pool.submit(
                        run_cell,
                        model,
                        factory,
                        args.doc_size,
                        args.tau,
                        args.k,
                        git_commit_str,
                    )
                    future_to_cell[fut] = (model, factory)

            for fut in as_completed(future_to_cell):
                model, factory = future_to_cell[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "model": model,
                        "factory": factory,
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
                        "hops": None,
                        "pairwise_winner_len": None,
                        "padded_size": None,
                        "raw_doc_size": None,
                        "git_commit": git_commit_str,
                        "error": f"worker raised: {type(e).__name__}: {e}",
                    }
                records.append(r)
                print(_format_status_line(model, factory, r))
                if not (r["ok"] and r["theorem2_holds"]):
                    all_holds = False
                _write_scorecard(out_path, args, git_commit_str, records)

    print("-" * 60)
    print(f"Wrote scorecard: {out_path}")
    print(f"Theorem-2 holds across all cells: {all_holds}")
    return 0 if all_holds else 1


if __name__ == "__main__":
    sys.exit(main())
