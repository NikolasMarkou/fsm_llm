"""
Heuristic 0-4 scoring of one example run, from its stdout, stderr and exit.

``classify_result`` and its helpers were moved verbatim from ``scripts/eval.py``
(commit 0facf56); the only edit is hoisting the ``import re`` each helper
repeated. The rubric keys on the repository examples' print conventions
(``Extraction rate: X/Y``, ``[EXTRACTED]``/``[MISSING]``, ``State: <name>``,
HANDLER/HOOK sections) and is pinned by a golden parity table in
``tests/test_fsm_llm_eval/test_scoring.py``: change it only for rubric
reasons, and never together with a structural change, or new scores stop
being comparable with the historical runs in ``EVALUATE.md``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; examples imports this module
    from .examples import ExampleResult


def _parse_extraction_rate(stdout: str) -> tuple[int, int] | None:
    """Parse 'Extraction rate: X/Y (Z%)' from output. Returns (extracted, total) or None."""
    m = re.search(r"Extraction rate:\s*(\d+)/(\d+)", stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _check_state_stuck(stdout: str) -> bool:
    """Check if FSM is stuck in initial state by comparing first and last state mentions."""
    # Look for "State: <name>" lines (printed during conversation)
    states = re.findall(r"State:\s+(\S+)", stdout)
    if not states:
        return False
    # If all state mentions are the same, the FSM never transitioned
    return len(set(states)) == 1


def _count_missing_fields(stdout: str) -> tuple[int, int]:
    """Count [EXTRACTED] vs [MISSING] fields in output. Returns (missing, total)."""
    extracted = len(re.findall(r"\[EXTRACTED\]", stdout))
    missing = len(re.findall(r"\[MISSING\]", stdout))
    total = extracted + missing
    return missing, total


def _check_empty_handler_metrics(stdout: str) -> bool:
    """Check if handler analytics show zero activity (empty arrays, 0 counts)."""
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
