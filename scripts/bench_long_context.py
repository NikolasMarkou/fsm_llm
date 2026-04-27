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
    make_dynamic_hop_runner,
    make_pad_callable,
    make_size_bucket,
    multi_hop,
    multi_hop_dynamic,
    niah,
    niah_padded,
    oracle_compare_op,
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


# M5 slice 5 — pairwise oracle-mode dense haystack. Each chunk holds a
# concrete, easily-recognised factual sentence so the leaf prompt yields
# non-sentinel content for every chunk (D-S5-001 sentinel short-circuit
# caveat). The compare question targets the marine-biology chunk; the
# tournament selects it over the other facts.
_PAIRWISE_DENSE_BANK: tuple[str, ...] = (
    " The Sun is a G-type main-sequence star at the centre of our solar system. ",
    " The English alphabet has 26 letters and is descended from Latin script. ",
    " Deep-sea hydrothermal vents host chemosynthetic tubeworms drawing "
    "energy from sulfide-rich fluids vented at oceanic spreading ridges. ",
    " The Eiffel Tower in Paris stands 330 metres tall and was completed in 1889. ",
    " The Great Wall of China was built across multiple dynasties over centuries. ",
    " Medieval guilds regulated apprenticeships and craft skills in walled cities. ",
    " Mount Everest reaches 8,849 metres above sea level on the Nepal-Tibet border. ",
    " Honeybees communicate flower locations to hivemates through a waggle dance. ",
)


def _build_pairwise_dense_haystack(doc_size: int, tau: int) -> str:
    """Synthesise a doc where every τ-sized chunk has its own factual
    statement (M5 slice 5 oracle-mode requirement). Falls back to the
    standard ``build_haystack`` if doc_size < tau (single-leaf case)."""
    if doc_size < tau:
        return build_haystack(doc_size)
    n_chunks = doc_size // tau
    chunks: list[str] = []
    for i in range(n_chunks):
        topic = _PAIRWISE_DENSE_BANK[i % len(_PAIRWISE_DENSE_BANK)]
        if len(topic) >= tau:
            chunk = topic[:tau]
        else:
            chunk = topic + " " * (tau - len(topic))
        chunks.append(chunk)
    doc = "".join(chunks)
    # Pad/truncate to exact doc_size.
    if len(doc) < doc_size:
        doc = doc + " " * (doc_size - len(doc))
    return doc[:doc_size]


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


# M5 slice 6 — labelled-dataset support.
#
# DECISION D-S6-003: scoring is plain string comparison (exact / substring /
# token-F1) on the model's free-form output. No semantic-similarity
# scorer; no dependency on a judge LLM. Real OOLONG ingestion is deferred
# (D-003); slice 6 ships infrastructure.

# Map factory name → task tag in the JSONL schema.
_FACTORY_TASK_MAP = {
    "niah": "niah",
    "niah_padded": "niah",
    "aggregate": "aggregate",
    "pairwise": "pairwise",
    "multi_hop": "multi_hop",
    "multi_hop_dynamic": "multi_hop",
}


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file. Each line must be a single JSON object with at
    least ``id`` and ``task`` keys; per-task fields validated downstream."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def score_answer(actual: str, expected: str, *, mode: str = "exact") -> bool:
    """Score one answer against the labelled ground truth.

    Modes:
    - ``exact``: case-insensitive equality after strip.
    - ``substring``: ``expected.lower()`` appears in ``actual.lower()``.
    - ``f1_token``: token-level F1 ≥ 0.5 (whitespace-tokenised, lowercase).
    """
    a = str(actual).strip().lower()
    e = str(expected).strip().lower()
    if mode == "exact":
        return a == e
    if mode == "substring":
        return e in a
    if mode == "f1_token":
        a_tok = set(a.split())
        e_tok = set(e.split())
        if not a_tok or not e_tok:
            return False
        common = a_tok & e_tok
        if not common:
            return False
        precision = len(common) / len(a_tok)
        recall = len(common) / len(e_tok)
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5
    raise ValueError(f"unknown score mode: {mode!r}")


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
    pairwise_mode: str = "length",
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
        "max_hops": None,  # M5 slice 6: filled for multi_hop_dynamic only
        "predicted_upper_bound": None,  # M5 slice 6: ditto
        "pairwise_winner_len": None,
        "padded_size": None,
        "raw_doc_size": None,
        "pairwise_mode": None,
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
        if pairwise_mode == "oracle":
            # M5 slice 5: build a dense haystack — every chunk is a
            # distinct factual statement so leaf prompts yield
            # non-sentinel content (D-S5-001 sentinel short-circuit
            # caveat — strict Theorem-2 equality requires this). The
            # leaf prompt asks the *broad* question (every chunk has
            # an answer); the oracle_compare_op uses the *specific*
            # question (tournament selects the chunk most relevant to
            # the target topic).
            haystack = _build_pairwise_dense_haystack(doc_size, tau)
            leaf_question = "What single factual statement is asserted in this passage?"
            compare_question = (
                "Which segment is more directly about marine biology "
                "(undersea organisms, hydrothermal vents, ocean ecology)?"
            )
            program = pairwise(question=leaf_question, tau=tau, k=k)
            # Re-plan with reduce_calls_per_node=1 so
            # predicted_calls = leaf + reduce = 2·k^d - 1.
            predicted = plan(
                PlanInputs(
                    n=doc_size,
                    K=10_000,
                    tau=tau,
                    alpha=1.0,
                    max_k=k,
                    reduce_calls_per_node=1,
                )
            )
            # Env binds a placeholder; the real op (closing over the
            # Executor) is bound below after the Executor is constructed.
            env = {
                "document": haystack,
                "size_bucket": make_size_bucket(tau),
                # "compare" injected post-Executor in the try-block below
            }
        else:
            question = "Which segment discusses topic X?"
            compare_question = question  # unused in length mode
            program = pairwise(question=question, tau=tau, k=k)
            env = {
                "document": haystack,
                "size_bucket": make_size_bucket(tau),
                "compare": compare_op(),
            }
        predicted_calls = predicted.predicted_calls
        record["pairwise_mode"] = pairwise_mode
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
    elif factory_name == "multi_hop_dynamic":
        # M5 slice 6 — confidence-gated dynamic hops. Theorem-2 reformulated
        # as upper bound per D-S6-002: predicted_upper = max_hops * k^d;
        # strict per actual_hops verified post-run. Late binding: the runner
        # closes over `ex`, built inside the try-block below.
        program = multi_hop_dynamic(
            "Find the first entity, then a fact about it.",
            max_hops=4,
        )
        env = {"document": haystack}
        predicted_calls = 4 * predicted.predicted_calls  # max_hops * k^d
        record["hops"] = None  # actual_hops filled post-run
        record["max_hops"] = 4
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

        # Late binding for pairwise oracle mode: the op closes over `ex`.
        if factory_name == "pairwise" and pairwise_mode == "oracle":
            env["compare"] = oracle_compare_op(compare_question, ex)

        # Late binding for multi_hop_dynamic — runner closes over ex._eval.
        actual_hops_cell = [0]
        if factory_name == "multi_hop_dynamic":
            runner = make_dynamic_hop_runner(
                ex,
                "Find the first entity, then a fact about it.",
                max_hops=4,
                peer_env={
                    "size_bucket": make_size_bucket(tau),
                    "best": best_answer_op(),
                },
                tau=tau,
                k=k,
                actual_hops_cell=actual_hops_cell,
            )
            env["dynamic_hop_runner"] = runner

        t0 = time.perf_counter()
        result = ex.run(program, env)
        wall = time.perf_counter() - t0

        record["ok"] = True
        record["wall_time_s"] = round(wall, 3)
        record["oracle_calls_actual"] = ex.oracle_calls
        # M5 slice 6: dynamic-hop T2 uses strict per-actual-hops equality.
        if factory_name == "multi_hop_dynamic":
            actual_hops = actual_hops_cell[0]
            record["hops"] = actual_hops
            strict_predicted = actual_hops * predicted.predicted_calls
            record["theorem2_holds"] = ex.oracle_calls == strict_predicted
            record["oracle_calls_predicted"] = strict_predicted
            record["predicted_upper_bound"] = predicted_calls
        else:
            record["theorem2_holds"] = ex.oracle_calls == predicted_calls
        record["tokens_in"] = ex.cost_accumulator.total_tokens_in
        record["tokens_out"] = ex.cost_accumulator.total_tokens_out
        record["total_cost"] = ex.cost_accumulator.total_cost

        if factory_name == "niah":
            record["needle_found"] = isinstance(result, str) and "SECRET-7421" in result
        elif factory_name == "niah_padded":
            record["needle_found"] = isinstance(result, str) and "SECRET-7421" in result
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
        elif factory_name == "multi_hop_dynamic":
            record["heuristic_pass"] = (
                isinstance(result, str) and result != "NOT_FOUND" and len(result) > 0
            )
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def run_dataset_cell(
    model: str,
    factory_name: str,
    dataset_path: str,
    tau: int,
    k: int,
    git_commit_str: str,
    score_mode: str,
    max_hops: int,
    pairwise_mode: str = "length",
) -> dict[str, Any]:
    """Run one (model x factory) cell over a labelled dataset (M5 slice 6).

    Iterates over JSONL records whose ``task`` matches the factory's task
    tag (per ``_FACTORY_TASK_MAP``). For each record: builds the factory
    program over the record's document + question, runs the executor,
    scores against ``record["answer"]``, accumulates per-record outcomes.

    Returns a cell record with ``record_count``, ``accuracy`` (= correct
    fraction over runnable records), ``pass_rate`` (= fraction with
    ``theorem2_holds``), and a truncated ``per_record`` list (first 20).

    Picklable: re-loads dataset inside the worker (no record dicts cross
    the pickle boundary).
    """
    record: dict[str, Any] = {
        "model": model,
        "factory": factory_name,
        "ok": False,
        "record_count": 0,
        "accuracy": None,
        "pass_rate": None,
        "per_record": [],
        "score_mode": score_mode,
        "git_commit": git_commit_str,
        "error": None,
    }
    task_tag = _FACTORY_TASK_MAP.get(factory_name)
    if task_tag is None:
        record["error"] = f"unknown factory: {factory_name}"
        return record

    try:
        dataset = load_dataset(dataset_path)
        eligible = [r for r in dataset if r.get("task") == task_tag]
        record["record_count"] = len(eligible)
        if not eligible:
            return record

        llm = LiteLLMInterface(model=model)
        oracle = LiteLLMOracle(llm, context_window_tokens=8192)
        ex = Executor(oracle=oracle)

        n_correct = 0
        n_holds = 0
        per_record: list[dict[str, Any]] = []
        for rec in eligible:
            outcome = _run_one_dataset_record(
                ex, factory_name, rec, tau, k, max_hops, pairwise_mode
            )
            score_pass = score_answer(outcome["answer"], rec["answer"], mode=score_mode)
            outcome["score_pass"] = score_pass
            outcome["expected"] = rec["answer"]
            if score_pass:
                n_correct += 1
            if outcome["theorem2_holds"]:
                n_holds += 1
            per_record.append(outcome)

        record["ok"] = True
        record["accuracy"] = n_correct / len(eligible)
        record["pass_rate"] = n_holds / len(eligible)
        record["per_record"] = per_record[:20]
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def _run_one_dataset_record(
    ex: Any,
    factory_name: str,
    rec: dict[str, Any],
    tau: int,
    k: int,
    max_hops: int,
    pairwise_mode: str,
) -> dict[str, Any]:
    """Build + run one factory invocation against one dataset record."""
    out: dict[str, Any] = {
        "record_id": rec.get("id"),
        "answer": None,
        "oracle_calls": None,
        "predicted": None,
        "theorem2_holds": False,
        "actual_hops": None,
        "error": None,
    }
    try:
        if factory_name == "pairwise":
            doc = rec["segment_a"] + " <SEP> " + rec["segment_b"]
        else:
            doc = rec["document"]
        question = rec["question"]
        n = len(doc)
        predicted = plan(PlanInputs(n=n, K=10_000, tau=tau, alpha=1.0, max_k=k))

        if factory_name in ("niah", "niah_padded"):
            program = niah(question, tau=tau, k=k)
            env = {
                "document": doc,
                "size_bucket": make_size_bucket(tau),
                "best": best_answer_op(),
            }
            predicted_calls = predicted.predicted_calls
        elif factory_name == "aggregate":
            program = aggregate(question, tau=tau, k=k)
            env = {
                "document": doc,
                "size_bucket": make_size_bucket(tau),
                "merge": aggregate_op(),
            }
            predicted_calls = predicted.predicted_calls
        elif factory_name == "pairwise":
            program = pairwise(question=question, tau=tau, k=k)
            env = {
                "document": doc,
                "size_bucket": make_size_bucket(tau),
                "compare": (
                    oracle_compare_op(question, ex)
                    if pairwise_mode == "oracle"
                    else compare_op()
                ),
            }
            predicted_calls = predicted.predicted_calls
            if pairwise_mode == "oracle":
                # Re-plan with reduce_calls_per_node=1.
                predicted_calls = plan(
                    PlanInputs(
                        n=n,
                        K=10_000,
                        tau=tau,
                        alpha=1.0,
                        max_k=k,
                        reduce_calls_per_node=1,
                    )
                ).predicted_calls
        elif factory_name == "multi_hop":
            program = multi_hop(question, hops=2, tau=tau, k=k)
            env = {
                "document": doc,
                "size_bucket": make_size_bucket(tau),
                "best": best_answer_op(),
            }
            predicted_calls = 2 * predicted.predicted_calls
        elif factory_name == "multi_hop_dynamic":
            actual_hops_cell = [0]
            runner = make_dynamic_hop_runner(
                ex,
                question,
                max_hops=max_hops,
                peer_env={
                    "size_bucket": make_size_bucket(tau),
                    "best": best_answer_op(),
                },
                tau=tau,
                k=k,
                actual_hops_cell=actual_hops_cell,
            )
            program = multi_hop_dynamic(question, max_hops=max_hops)
            env = {"document": doc, "dynamic_hop_runner": runner}
            predicted_calls = max_hops * predicted.predicted_calls  # upper bound
        else:
            out["error"] = f"unsupported factory: {factory_name}"
            return out

        result = ex.run(program, env)
        out["answer"] = result if isinstance(result, str) else str(result)
        out["oracle_calls"] = ex.oracle_calls

        if factory_name == "multi_hop_dynamic":
            actual_hops = actual_hops_cell[0]
            out["actual_hops"] = actual_hops
            strict_predicted = actual_hops * predicted.predicted_calls
            out["predicted"] = strict_predicted
            out["theorem2_holds"] = ex.oracle_calls == strict_predicted
        else:
            out["predicted"] = predicted_calls
            out["theorem2_holds"] = ex.oracle_calls == predicted_calls
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


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
        "--pairwise-mode",
        choices=("length", "oracle"),
        default="length",
        help="Pairwise compare op: 'length' (slice-3 best_answer-equivalent, "
        "default; preserves existing eval baselines) or 'oracle' "
        "(M5 slice 5 oracle-mediated tournament — re-plans with "
        "reduce_calls_per_node=1, predicted=2·k^d - 1). Affects "
        "pairwise factory only; ignored for other factories.",
    )
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
    # M5 slice 6 — labelled-dataset path.
    p.add_argument(
        "--dataset",
        default=None,
        help="Path to a labelled JSONL dataset (e.g. evaluation/datasets/"
        "oolong_synth.jsonl). When set, each cell iterates over records "
        "whose 'task' matches the factory; reports per-cell accuracy + "
        "pass_rate alongside Theorem-2 telemetry. When unset, uses "
        "synthetic single-haystack mode (slice 1-5 behaviour).",
    )
    p.add_argument(
        "--score-mode",
        choices=("exact", "substring", "f1_token"),
        default="substring",
        help="Scoring function for --dataset mode. 'exact': "
        "case-insensitive equality. 'substring' (default): "
        "expected.lower() in actual.lower(). 'f1_token': token-level "
        "F1 >= 0.5.",
    )
    p.add_argument(
        "--max-hops",
        type=int,
        default=4,
        help="Cap on hops for multi_hop_dynamic factory. Default 4.",
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


def _format_dataset_status_line(model: str, factory: str, r: dict[str, Any]) -> str:
    status = "OK" if r["ok"] else "FAIL"
    if not r["ok"]:
        return f"[{status}] model={model} factory={factory} error={r['error']}"
    acc = r.get("accuracy")
    pr = r.get("pass_rate")
    return (
        f"[{status}] model={model} factory={factory} "
        f"records={r['record_count']} "
        f"accuracy={acc:.2f} pass_rate={pr:.2f}"
        if (acc is not None and pr is not None)
        else f"[{status}] model={model} factory={factory} "
        f"records={r['record_count']} (none eligible)"
    )


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
    valid = {
        "niah",
        "aggregate",
        "pairwise",
        "multi_hop",
        "multi_hop_dynamic",
        "niah_padded",
    }
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
                if args.dataset is not None:
                    r = run_dataset_cell(
                        model,
                        factory,
                        args.dataset,
                        args.tau,
                        args.k,
                        git_commit_str,
                        args.score_mode,
                        args.max_hops,
                        args.pairwise_mode,
                    )
                    records.append(r)
                    line = _format_dataset_status_line(model, factory, r)
                    print(line)
                    # T2 holds iff every record in the cell holds.
                    if not (r["ok"] and (r["pass_rate"] or 0) == 1.0):
                        all_holds = False
                else:
                    r = run_cell(
                        model,
                        factory,
                        args.doc_size,
                        args.tau,
                        args.k,
                        git_commit_str,
                        args.pairwise_mode,
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
                    if args.dataset is not None:
                        fut = pool.submit(
                            run_dataset_cell,
                            model,
                            factory,
                            args.dataset,
                            args.tau,
                            args.k,
                            git_commit_str,
                            args.score_mode,
                            args.max_hops,
                            args.pairwise_mode,
                        )
                    else:
                        fut = pool.submit(
                            run_cell,
                            model,
                            factory,
                            args.doc_size,
                            args.tau,
                            args.k,
                            git_commit_str,
                            args.pairwise_mode,
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
                        "pairwise_mode": None,
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
