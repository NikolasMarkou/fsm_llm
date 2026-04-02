#!/usr/bin/env python3
"""
Parallel evaluation runner for FSM-LLM examples.

Discovers all examples under examples/, runs them in parallel using a
process pool, captures stdout+stderr per example, and writes individual
log files plus a summary scorecard.

Usage:
    # Run with defaults (auto-detect examples, 4 workers)
    .venv/bin/python scripts/eval.py

    # Custom model and parallelism
    .venv/bin/python scripts/eval.py --model ollama_chat/qwen3.5:4b --workers 6

    # Filter by category
    .venv/bin/python scripts/eval.py --category agents

    # Custom timeout and output directory
    .venv/bin/python scripts/eval.py --timeout 180 --output-dir evaluation/run_012

    # List discovered examples without running
    .venv/bin/python scripts/eval.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "ollama_chat/qwen3.5:4b")
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 120  # seconds

# Timeout overrides per category (seconds)
CATEGORY_TIMEOUTS: dict[str, int] = {
    "agents": 180,
    "reasoning": 300,
    "workflows": 180,
    "advanced": 180,
    "meta": 120,
}

# Per-example timeout overrides (for known slow examples)
EXAMPLE_TIMEOUTS: dict[str, int] = {
    "advanced/e_commerce": 300,
    "reasoning/math_tutor": 300,
    "agents/reflexion": 240,
    "agents/debate": 300,
    "agents/evaluator_optimizer": 300,
    "agents/orchestrator": 300,
    "agents/adapt": 240,
    "agents/memory_agent": 300,
    "agents/concurrent_react": 240,
    "agents/agent_as_tool": 300,
    "agents/structured_output": 240,
    "agents/react_hitl_combined": 240,
    "agents/skill_loader": 240,
    "agents/orchestrator_specialist": 300,
    "agents/hierarchical_orchestrator": 300,
    "agents/pipeline_review": 300,
    "agents/adapt_with_memory": 300,
    "agents/react_structured_pipeline": 300,
    "agents/multi_debate_panel": 300,
    "agents/reflexion_code_gen": 240,
    "agents/agent_memory_chain": 300,
    "agents/debate_with_tools": 240,
    "agents/eval_opt_structured": 300,
    "agents/legal_document_review": 300,
    "agents/investment_portfolio": 300,
    "agents/security_audit": 300,
    "agents/medical_literature": 300,
    "agents/architecture_review": 300,
    "agents/supply_chain_optimizer": 300,
    "agents/regulatory_compliance": 300,
    "intermediate/adaptive_quiz": 300,
    "meta/meta_review_loop": 240,
    "meta/meta_from_spec": 240,
    "meta/build_fsm": 240,
    "meta/build_agent": 240,
    "meta/build_workflow": 240,
    "workflows/workflow_agent_loop": 300,
    "workflows/loan_processing": 300,
    "workflows/release_management": 300,
    "workflows/customer_onboarding": 300,
}

# Stdin inputs for interactive examples (those with input() calls).
# Each value is a newline-separated string piped to stdin.
EXAMPLE_INPUTS: dict[str, str] = {
    # basic
    "basic/simple_greeting": "Hello there!\nMy name is Alex\nquit\n",
    "basic/form_filling": "My name is John Smith\njohn@example.com\n30\nSoftware engineer\nyes\nquit\n",
    "basic/story_time": "That's a silly choice, straw is way too weak!\nSticks are barely any better, the wolf will blow them down\nNo way, bricks are too strong for the wolf\nThey should put a pot of boiling water in the fireplace\nGreat story, I loved it!\nquit\n",
    # intermediate
    "intermediate/book_recommendation": "I like science fiction\nSomething like Dune\nquit\n",
    "intermediate/product_recommendation": "I need a laptop for programming\nAround 1500 dollars\nquit\n",
    "intermediate/adaptive_quiz": "My name is Sam\nParis\n42\nH2O\nThat was fun!\nquit\n",
    # advanced
    "advanced/yoga_instructions": "I want to do some yoga\nI'm a beginner\nquit\n",
    "advanced/support_pipeline": "Hi, I'm Sarah and my order hasn't arrived\nOrder number 12345\nyes that fixed it\nno thanks\nquit\nquit\nquit\n",
    # classification
    "classification/intent_routing": "I want to cancel my subscription\nI need help with billing\nquit\n",
    "classification/smart_helpdesk": "My internet is down\nI've already tried restarting the router\nquit\n",
    "classification/classified_transitions": "I want to buy something\nA new laptop\nquit\n",
    # agents (interactive)
    "agents/react_search": "What is the population of Tokyo?\nquit\n",
    "agents/hitl_approval": "Send an email to bob@example.com saying hello\ny\nquit\n",
    "agents/react_hitl_combined": "Search for weather in Paris\ny\nquit\nquit\n",
    "agents/classified_dispatch": "Search for information about climate change\nquit\n",
    "agents/classified_tools": "Calculate the sum of 15, 23, and 100\nquit\n",
    "agents/full_pipeline": "Look up the latest news about AI\nquit\n",
    "agents/hierarchical_tools": "What is 15 times 23?\nquit\n",
    "agents/reasoning_stacking": "What is the square root of 144?\nquit\n",
    "agents/reasoning_tool": "Solve: if x + 5 = 12, what is x?\nquit\n",
    "agents/tool_decorator": "What is 5 plus 3?\nquit\n",
    "agents/skill_loader": "What time is it?\nquit\n",
    # reasoning
    "reasoning/math_tutor": "What is 15 + 27?\nquit\n",
    # meta
    "meta/build_fsm": "Build a simple greeting bot\nyes\nquit\n",
    "meta/build_workflow": "Build a workflow for order processing with validation payment and fulfillment\nyes\nquit\n",
    "meta/build_agent": "Build a research agent that can search the web and summarize using ReAct pattern\nyes\nquit\n",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExampleInfo:
    """Metadata about a single example."""

    name: str  # e.g. "agents/debate"
    category: str  # e.g. "agents"
    path: Path  # absolute path to run.py
    interactive: bool  # has input() calls
    stdin_data: str | None  # data to pipe, or None


@dataclass
class ExampleResult:
    """Result of running a single example."""

    name: str
    category: str
    exit_code: int | None  # None = timeout
    duration: float  # seconds
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_examples(
    category_filter: str | None = None,
    name_filter: str | None = None,
) -> list[ExampleInfo]:
    """Find all run.py files under examples/ and classify them."""
    examples = []

    for run_py in sorted(EXAMPLES_DIR.rglob("run.py")):
        rel = run_py.relative_to(EXAMPLES_DIR)
        parts = rel.parts  # e.g. ("agents", "debate", "run.py")
        if len(parts) < 3:
            continue

        category = parts[0]
        name = f"{parts[0]}/{parts[1]}"

        if category_filter and category != category_filter:
            continue
        if name_filter and name_filter not in name:
            continue

        # Check for input() calls
        try:
            source = run_py.read_text()
            interactive = "input(" in source
        except OSError:
            interactive = False

        stdin_data = EXAMPLE_INPUTS.get(name)

        examples.append(
            ExampleInfo(
                name=name,
                category=category,
                path=run_py,
                interactive=interactive,
                stdin_data=stdin_data,
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def get_timeout(example: ExampleInfo, default: int) -> int:
    """Get the timeout for an example, checking per-example then per-category."""
    if example.name in EXAMPLE_TIMEOUTS:
        return EXAMPLE_TIMEOUTS[example.name]
    if example.category in CATEGORY_TIMEOUTS:
        return CATEGORY_TIMEOUTS[example.category]
    return default


def run_example(
    example: ExampleInfo,
    model: str,
    timeout: int,
) -> ExampleResult:
    """Run a single example in a subprocess. Safe to call from a worker process."""
    effective_timeout = get_timeout(example, timeout)

    env = os.environ.copy()
    env["LLM_MODEL"] = model
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Always pipe stdin when configured — input() may live in imported
    # modules (e.g. meta_builder.run_interactive), not in run.py itself.
    stdin_data = example.stdin_data

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [VENV_PYTHON, str(example.path)],
            cwd=str(ROOT),
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env=env,
        )
        duration = time.monotonic() - t0
        return ExampleResult(
            name=example.name,
            category=example.category,
            exit_code=result.returncode,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - t0
        return ExampleResult(
            name=example.name,
            category=example.category,
            exit_code=None,
            duration=duration,
            stdout=e.stdout or ""
            if isinstance(e.stdout, str)
            else (e.stdout or b"").decode("utf-8", errors="replace"),
            stderr=e.stderr or ""
            if isinstance(e.stderr, str)
            else (e.stderr or b"").decode("utf-8", errors="replace"),
            timed_out=True,
            error=f"Timeout after {effective_timeout}s",
        )
    except Exception as e:
        duration = time.monotonic() - t0
        return ExampleResult(
            name=example.name,
            category=example.category,
            exit_code=-1,
            duration=duration,
            stdout="",
            stderr=str(e),
            timed_out=False,
            error=str(e),
        )


def _run_wrapper(args: tuple) -> ExampleResult:
    """Wrapper for ProcessPoolExecutor (must be top-level picklable)."""
    example_dict, model, timeout = args
    example = ExampleInfo(**example_dict)
    return run_example(example, model, timeout)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_example_log(result: ExampleResult, output_dir: Path) -> Path:
    """Write individual log file for an example. Returns path."""
    cat_dir = output_dir / "logs" / result.category
    cat_dir.mkdir(parents=True, exist_ok=True)

    example_slug = result.name.replace("/", "_")
    log_path = cat_dir / f"{example_slug}.log"

    with open(log_path, "w") as f:
        f.write(f"# Example: {result.name}\n")
        f.write(f"# Exit code: {result.exit_code}\n")
        f.write(f"# Duration: {result.duration:.1f}s\n")
        f.write(f"# Timed out: {result.timed_out}\n")
        if result.error:
            f.write(f"# Error: {result.error}\n")
        f.write(f"# {'=' * 60}\n\n")

        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)

    return log_path


def _parse_extraction_rate(stdout: str) -> tuple[int, int] | None:
    """Parse 'Extraction rate: X/Y (Z%)' from output. Returns (extracted, total) or None."""
    import re

    m = re.search(r"Extraction rate:\s*(\d+)/(\d+)", stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _check_state_stuck(stdout: str) -> bool:
    """Check if FSM is stuck in initial state by comparing first and last state mentions."""
    import re

    # Look for "State: <name>" lines (printed during conversation)
    states = re.findall(r"State:\s+(\S+)", stdout)
    if not states:
        return False
    # If all state mentions are the same, the FSM never transitioned
    return len(set(states)) == 1


def _count_missing_fields(stdout: str) -> tuple[int, int]:
    """Count [EXTRACTED] vs [MISSING] fields in output. Returns (missing, total)."""
    import re

    extracted = len(re.findall(r"\[EXTRACTED\]", stdout))
    missing = len(re.findall(r"\[MISSING\]", stdout))
    total = extracted + missing
    return missing, total


def _check_empty_handler_metrics(stdout: str) -> bool:
    """Check if handler analytics show zero activity (empty arrays, 0 counts)."""
    import re

    # Look for handler sections with empty arrays like "phases: []" or "stages: []"
    if "HANDLER" not in stdout and "HOOK" not in stdout:
        return False  # No handler section — not applicable

    # Count empty list indicators in handler/analytics sections
    empty_lists = len(re.findall(r":\s*\[\]", stdout))
    # Count zero-value metrics like "0 updates", "0 entries"
    zero_metrics = len(re.findall(r":\s*0\b", stdout))

    return empty_lists >= 2 or zero_metrics >= 2


def classify_result(result: ExampleResult) -> tuple[int, list[str]]:
    """
    Heuristic scoring and failure classification based on output analysis.

    Returns (score 0-4, list of failure codes).

    Checks (in order):
    1. Crash detection (import errors, validation errors, instant exit)
    2. Timeout detection
    3. Error signals in output (tracebacks, tool failures, parse errors)
    4. Extraction quality (extraction rate, field completion, state movement)
    5. Handler activity (for examples with handler analytics)
    """
    failures: list[str] = []
    combined = result.stdout + result.stderr

    # ------------------------------------------------------------------
    # Score 0: crash / can't start
    # ------------------------------------------------------------------
    if result.exit_code == -1 or (
        result.exit_code is not None
        and result.exit_code != 0
        and result.duration < 3.0
        and not result.stdout.strip()
    ):
        if "ImportError" in combined or "ModuleNotFoundError" in combined:
            failures.append("F-CODE")
        elif "ValidationError" in combined or "pydantic" in combined.lower():
            failures.append("F-SCHEMA")
        else:
            failures.append("F-CODE")
        return 0, failures

    # ------------------------------------------------------------------
    # Score 1: timeout
    # ------------------------------------------------------------------
    if result.timed_out:
        failures.append("F-LOOP")
        if len(result.stdout.strip()) > 200:
            return 1, failures
        return 1, failures

    # ------------------------------------------------------------------
    # Check for error signals in output
    # ------------------------------------------------------------------
    lower = combined.lower()

    if "traceback" in lower and "error" in lower:
        failures.append("F-CODE")

    if any(
        kw in lower
        for kw in ["no tool was selected", "0 tools called", "tool_status: skipped"]
    ):
        failures.append("F-TOOL")

    if any(
        kw in lower
        for kw in ["0 keys extracted", "extraction produced no", "none extracted"]
    ):
        failures.append("F-EXTRACT")

    if "budget" in lower and "exhaust" in lower:
        failures.append("F-LOOP")

    if any(kw in lower for kw in ["parse error", "jsondecodeerror", "json.decoder"]):
        if "F-CODE" not in failures:
            failures.append("F-PARSE")

    # ------------------------------------------------------------------
    # Check extraction quality (FSM examples print extraction summaries)
    # ------------------------------------------------------------------
    extraction = _parse_extraction_rate(result.stdout)
    if extraction is not None:
        extracted, total = extraction
        if total > 0 and extracted == 0:
            # Zero extraction — nothing was captured despite running
            if "F-EXTRACT" not in failures:
                failures.append("F-EXTRACT")
        elif total > 0 and extracted < total * 0.5:
            # Less than 50% extraction — partial failure
            if "F-EXTRACT" not in failures:
                failures.append("F-EXTRACT")

    # Check if all printed fields are MISSING
    missing, field_total = _count_missing_fields(result.stdout)
    if field_total > 0 and missing == field_total:
        if "F-EXTRACT" not in failures:
            failures.append("F-EXTRACT")

    # Check if FSM never transitioned (stuck in initial state)
    # Only flag as F-TRANS if there are also extraction failures — on its own,
    # some examples legitimately stay in one state (stacking, classification, agents)
    state_stuck = _check_state_stuck(result.stdout)
    if state_stuck and "F-EXTRACT" in failures:
        if "F-TRANS" not in failures:
            failures.append("F-TRANS")

    # Check for empty handler metrics (handlers registered but never fired)
    # Same rule: only flag if extraction also failed
    if _check_empty_handler_metrics(result.stdout) and "F-EXTRACT" in failures:
        if "F-TRANS" not in failures:
            failures.append("F-TRANS")

    # ------------------------------------------------------------------
    # Non-zero exit without crash (ran but failed)
    # ------------------------------------------------------------------
    if result.exit_code is not None and result.exit_code != 0:
        if not failures:
            failures.append("F-CODE")
        if len(result.stdout.strip()) > 300:
            return 2, failures
        return 1, failures

    # ------------------------------------------------------------------
    # Exit code 0 — ran to completion, score based on failure signals
    # ------------------------------------------------------------------
    if not failures:
        return 4, failures

    # Determine severity based on failure types and counts
    # Zero extraction + stuck state = fundamentally broken despite clean exit
    has_extract_fail = "F-EXTRACT" in failures
    has_trans_fail = "F-TRANS" in failures

    if has_extract_fail and has_trans_fail:
        # Both extraction and transitions failed — broken
        return 1, failures

    if has_extract_fail:
        # Extraction failed but may have transitioned (or no transitions expected)
        if extraction and extraction[0] == 0:
            return 1, failures  # Zero extraction = broken
        return 2, failures  # Partial extraction = partial

    if len(failures) == 1:
        return 3, failures
    return 2, failures


def write_scorecard(
    results: list[ExampleResult],
    model: str,
    output_dir: Path,
) -> Path:
    """Write a markdown scorecard summarizing all results."""
    now = datetime.now()
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        pass

    scores: list[tuple[ExampleResult, int, list[str]]] = []
    for r in sorted(results, key=lambda r: r.name):
        score, failures = classify_result(r)
        scores.append((r, score, failures))

    total = len(scores)
    score_sum = sum(s for _, s, _ in scores)
    max_possible = total * 4
    health = (score_sum / max_possible * 100) if max_possible > 0 else 0

    # Distribution
    dist = {i: 0 for i in range(5)}
    for _, s, _ in scores:
        dist[s] += 1

    # Category breakdown
    cat_scores: dict[str, list[int]] = {}
    for r, s, _ in scores:
        cat_scores.setdefault(r.category, []).append(s)

    # Failure code counts
    failure_counts: dict[str, int] = {}
    for _, _, fails in scores:
        for f in fails:
            failure_counts[f] = failure_counts.get(f, 0) + 1

    # Build scorecard
    lines: list[str] = []
    lines.append(f"# Evaluation: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"- **Date**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Git commit**: {git_hash}")
    lines.append(f"- **Model**: {model}")
    lines.append(f"- **Example count**: {total}")
    lines.append("- **Workers**: parallel execution")
    lines.append("- **Evaluator**: scripts/eval.py (automated)")
    lines.append("")

    # Scores table
    lines.append("## Scores")
    lines.append("")
    lines.append("| # | Example | Score | Duration | Failures | Notes |")
    lines.append("|---|---------|-------|----------|----------|-------|")
    for i, (r, score, fails) in enumerate(scores, 1):
        label = ["CRASH", "BROKEN", "PARTIAL", "MOSTLY", "PASS"][score]
        fail_str = ", ".join(fails) if fails else ""
        note = ""
        if r.timed_out:
            note = f"Timeout ({r.duration:.0f}s)"
        elif r.exit_code != 0 and r.exit_code is not None:
            note = f"Exit {r.exit_code}"
        elif score == 4:
            note = f"{r.duration:.1f}s"
        else:
            note = f"{r.duration:.1f}s"
        lines.append(
            f"| {i} | {r.name} | {score} ({label}) | {r.duration:.1f}s | {fail_str} | {note} |"
        )

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total examples**: {total}")
    lines.append(
        f"- **Score distribution**: {dist[4]}x4, {dist[3]}x3, {dist[2]}x2, {dist[1]}x1, {dist[0]}x0"
    )
    lines.append(f"- **Health Score**: {score_sum}/{max_possible} = **{health:.1f}%**")
    lines.append("- **Category breakdown**:")
    for cat in sorted(cat_scores):
        cat_list = cat_scores[cat]
        cat_sum = sum(cat_list)
        cat_max = len(cat_list) * 4
        cat_pct = cat_sum / cat_max * 100 if cat_max > 0 else 0
        lines.append(f"  - {cat}: {cat_sum}/{cat_max} ({cat_pct:.0f}%)")
    if failure_counts:
        top_failures = sorted(failure_counts.items(), key=lambda x: -x[1])
        lines.append(
            "- **Top failure codes**: "
            + ", ".join(f"{code} ({cnt})" for code, cnt in top_failures)
        )

    # Timing
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    durations = [r.duration for r in results]
    lines.append(
        f"- **Total wall time**: {sum(durations):.1f}s (sequential equivalent)"
    )
    lines.append(f"- **Fastest**: {min(durations):.1f}s")
    lines.append(f"- **Slowest**: {max(durations):.1f}s")
    lines.append(f"- **Mean**: {sum(durations) / len(durations):.1f}s")

    lines.append("")

    scorecard_path = output_dir / "scorecard.md"
    scorecard_path.write_text("\n".join(lines))

    # Also write machine-readable JSON
    json_data = {
        "date": now.isoformat(),
        "git_commit": git_hash,
        "model": model,
        "health_score": round(health, 1),
        "total_examples": total,
        "distribution": dist,
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "score": s,
                "failures": f,
                "duration": round(r.duration, 1),
                "exit_code": r.exit_code,
                "timed_out": r.timed_out,
            }
            for r, s, f in scores
        ],
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(json_data, indent=2))

    return scorecard_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel evaluation runner for FSM-LLM examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all with defaults
  %(prog)s --model gpt-4o-mini --workers 8    # Cloud model, 8 workers
  %(prog)s --category agents --timeout 240    # Only agents, longer timeout
  %(prog)s --filter react                     # Only examples matching 'react'
  %(prog)s --list                             # Just list discovered examples
        """,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Default timeout per example in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by category (basic, intermediate, advanced, classification, agents, reasoning, workflows, meta)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        dest="name_filter",
        help="Filter examples by substring match on name",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: evaluation/<timestamp>_<hash>_<model>)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered examples and exit",
    )

    args = parser.parse_args()

    # Discover examples
    examples = discover_examples(
        category_filter=args.category,
        name_filter=args.name_filter,
    )

    if not examples:
        print("No examples found matching filters.", file=sys.stderr)
        return 1

    if args.list:
        print(f"Discovered {len(examples)} examples:\n")
        for ex in examples:
            mode = "interactive" if ex.interactive else "automated"
            has_input = "has stdin" if ex.stdin_data else "no stdin"
            print(f"  {ex.name:45s} [{mode}, {has_input}]")
        return 0

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        git_hash = "unknown"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(ROOT),
                text=True,
            ).strip()
        except Exception:
            pass
        model_slug = args.model.split("/")[-1].replace(":", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = ROOT / "evaluation" / f"{timestamp}_{git_hash}_{model_slug}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Print banner
    print("FSM-LLM Evaluation Runner")
    print(f"{'=' * 60}")
    print(f"  Model:    {args.model}")
    print(f"  Workers:  {args.workers}")
    print(f"  Timeout:  {args.timeout}s (default)")
    print(f"  Examples: {len(examples)}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 60}")
    print()

    # Prepare work items (dicts for pickling across processes)
    work_items = [
        (
            {
                "name": ex.name,
                "category": ex.category,
                "path": ex.path,
                "interactive": ex.interactive,
                "stdin_data": ex.stdin_data,
            },
            args.model,
            args.timeout,
        )
        for ex in examples
    ]

    # Run in parallel
    results: list[ExampleResult] = []
    completed = 0
    wall_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_name = {}
        for item in work_items:
            future = pool.submit(_run_wrapper, item)
            future_to_name[future] = item[0]["name"]

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)

                # Write individual log
                write_example_log(result, output_dir)

                # Progress indicator
                score, failures = classify_result(result)
                label = ["CRASH", "BROKEN", "PARTIAL", "MOSTLY", "PASS"][score]
                fail_str = f" [{', '.join(failures)}]" if failures else ""
                timeout_str = " TIMEOUT" if result.timed_out else ""
                print(
                    f"  [{completed:2d}/{len(examples)}] {result.name:45s} "
                    f"{score} ({label:7s}) {result.duration:6.1f}s{timeout_str}{fail_str}"
                )
            except Exception as e:
                print(
                    f"  [{completed:2d}/{len(examples)}] {name:45s} ERROR: {e}",
                    file=sys.stderr,
                )

    wall_time = time.monotonic() - wall_start

    # Write scorecard
    scorecard_path = write_scorecard(results, args.model, output_dir)

    # Print summary
    print()
    print(f"{'=' * 60}")
    score_sum = sum(classify_result(r)[0] for r in results)
    max_possible = len(results) * 4
    health = score_sum / max_possible * 100 if max_possible > 0 else 0
    print(f"  Health Score: {score_sum}/{max_possible} = {health:.1f}%")
    print(
        f"  Wall time:   {wall_time:.1f}s ({len(results)} examples, {args.workers} workers)"
    )
    print(f"  Scorecard:   {scorecard_path}")
    print(f"  Logs:        {output_dir / 'logs'}/")
    print(f"  JSON:        {output_dir / 'results.json'}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
