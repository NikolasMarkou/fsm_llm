"""
Handler implementations for the reasoning engine.
Enhanced with retry logic, context pruning, and standardized handling.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from fsm_llm import API, clear_keys_on_entry
from fsm_llm.constants import MESSAGE_TRUNCATION_SUFFIX
from fsm_llm.logging import logger

from .calculator import calculate
from .constants import (
    CALCULATION_STATE,
    HYBRID_CLEARED_KEYS,
    HYBRID_EVALUATION_STATE,
    HYBRID_FIRST_STATE,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    ReasoningType,
)
from .strategy_files import solution_keys

# Stop-words filtered out during solution-relevance checking in validate_solution.
# Without this, common words (the, is, a) cause near-universal overlap, making the
# "addresses_problem" validation check meaningless.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "if",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
    }
)


def _is_solution(value: Any) -> bool:
    """True when ``value`` is a solution: anything but ``None`` or ``""``.

    Contract (callers: ``ReasoningHandlers.validate_solution`` and
    ``OutputFormatter.extract_final_solution``): a numeric ``0``, ``False``
    or an empty list IS a solution (an answer can be zero or false). Never
    raises for a JSON value.
    """
    return value is not None and value != ""


def retries_exhausted(retry_count: int) -> bool:
    """True when ``retry_count`` has reached ``Defaults.MAX_RETRIES``.

    Contract: the one retry-limit rule, shared by ``validate_solution`` (after
    its increment) and ``ReasoningEngine._check_retry_limit``.
    """
    return retry_count >= Defaults.MAX_RETRIES


@cache
def _solution_keys() -> tuple[str, ...]:
    """Orchestrator keys ``extract_final_solution`` reads, in order: the two the
    SYNTHESIZE/FINAL states set, then one per strategy in ``STRATEGY_ORDER`` (the keys the
    call state merges under their parent names, not the raw sub-FSM keys)."""
    return (
        ContextKeys.FINAL_SOLUTION,
        ContextKeys.PROPOSED_SOLUTION,
        *solution_keys(),
    )


def _head_and_tail(trace: list[Any], limit: int) -> list[Any]:
    """The oldest ``Defaults.TRACE_KEEP_START`` and the newest steps, at most
    ``limit`` in all: how the run started is what ``total_steps`` and the
    reasoning types read, so both prune rules of the trace key keep it."""
    if len(trace) <= limit:
        return trace
    return (
        trace[: Defaults.TRACE_KEEP_START]
        + trace[len(trace) - (limit - Defaults.TRACE_KEEP_START) :]
    )


class ReasoningHandlers:
    """Collection of handlers for the reasoning engine."""

    @staticmethod
    def validate_solution(context: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the proposed solution with proper handling for different problem types.

        :param context: Current conversation context
        :return: Validation results with retry handling
        """
        solution = context.get(ContextKeys.PROPOSED_SOLUTION, "")
        insights = context.get(ContextKeys.KEY_INSIGHTS, [])
        # A non-str problem_type (a caller's initial_context, a model value)
        # reads as "", so the validator never crashes on it (D-061).
        raw_problem_type = context.get(ContextKeys.PROBLEM_TYPE)
        problem_type = (
            raw_problem_type.lower() if isinstance(raw_problem_type, str) else ""
        )
        reasoning_strategy = context.get(ContextKeys.REASONING_STRATEGY, "")
        retry_count = context.get(ContextKeys.RETRY_COUNT, 0)

        # Determine if this is a simple problem
        is_simple_problem = (
            "arithmetic" in problem_type
            or "calculation" in problem_type
            or reasoning_strategy == ReasoningType.SIMPLE_CALCULATOR.value
            or context.get(ContextKeys.REASONING_TYPE_SELECTED)
            == ReasoningType.SIMPLE_CALCULATOR.value
        )

        # Validation checks. A refused calculation has no solution, whatever the
        # model wrote beside it (D-052): the bounded retry runs.
        has_solution = _is_solution(solution) and not context.get(
            ContextKeys.CALCULATION_ERROR_DETAILS
        )
        has_insights = (
            bool(insights) or is_simple_problem
        )  # Simple problems may not need insights

        # Check solution detail appropriately
        if is_simple_problem:
            # For simple problems, even a single number is sufficient
            sufficient_detail = has_solution
        else:
            # For complex problems, require a substantive solution (more than a trivial answer)
            solution_str = str(solution).strip()
            sufficient_detail = (
                has_solution and len(solution_str) > Defaults.MIN_SOLUTION_LENGTH
            )

        # Check if solution addresses the problem (keyword overlap check).
        # Filter common stop-words so the check is meaningful — without this,
        # words like "the", "is", "a" cause near-universal overlap.
        addresses_problem: bool
        problem_statement = context.get(ContextKeys.PROBLEM_STATEMENT, "")
        if not isinstance(problem_statement, str):  # reads as "", as D-061 above
            problem_statement = ""
        if has_solution and problem_statement and not is_simple_problem:

            def _content_words(text: str) -> set[str]:
                """Extract content words: lowercase, strip punctuation, remove stop words."""
                words = set()
                for w in text.lower().split():
                    cleaned = w.strip(".,;:!?\"'()[]{}/-")
                    if cleaned and cleaned not in _STOP_WORDS:
                        words.add(cleaned)
                return words

            problem_words = _content_words(problem_statement)
            solution_words = _content_words(str(solution))
            # At least some content-word overlap between problem and solution.
            # If the problem has no content words (all stop words, e.g. "What is this?"),
            # we can't assess relevance — give benefit of the doubt.
            if not problem_words:
                addresses_problem = has_solution
            else:
                addresses_problem = len(problem_words & solution_words) > 0
        else:
            # Simple problems (e.g. "What is 2+2?" → "4") won't have word overlap
            addresses_problem = has_solution

        validation_checks = {
            "has_solution": has_solution,
            "has_insights": has_insights,
            "sufficient_detail": sufficient_detail,
            "addresses_problem": addresses_problem,
        }

        # Calculate overall validity
        is_valid = all(validation_checks.values())

        # Handle retry logic. Increment first, then derive the flag from the
        # post-increment count so MAX_RETRIES_REACHED stays consistent with the
        # returned RETRY_COUNT and with engine._check_retry_limit (both use
        # retries_exhausted).
        if not is_valid and not retries_exhausted(retry_count):
            retry_count += 1
            logger.info(
                LogMessages.RETRY_ATTEMPT.format(
                    current=retry_count, max=Defaults.MAX_RETRIES
                )
            )

        # Confidence is the fraction of checks passed
        confidence = sum(validation_checks.values()) / len(validation_checks)

        logger.info(
            LogMessages.VALIDATION_RESULT.format(valid=is_valid, confidence=confidence)
        )

        return {
            ContextKeys.RETRY_COUNT: retry_count,
            ContextKeys.VALIDATION_CHECKS: validation_checks,
            ContextKeys.VALIDATION_RESULT: is_valid,
            ContextKeys.SOLUTION_CONFIDENCE: confidence,
            ContextKeys.MAX_RETRIES_REACHED: retries_exhausted(retry_count),
        }

    @staticmethod
    def execute_calculation(context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the ``calculation_expression`` (on entry to ``perform_calculation``).

        No LLM call, never raises (it is critical). Returns exactly
        ``{calculation_result: value, calculation_error: None}`` or
        ``{calculation_result: None, calculation_error: reason}``.
        """
        value, error = calculate(context.get(ContextKeys.CALCULATION_EXPRESSION))
        if error is not None:
            logger.warning(f"Calculator refused the expression: {error}")
        return {
            ContextKeys.CALCULATION_RESULT: value,
            ContextKeys.CALCULATION_ERROR: error,
        }

    # DECISION plan-2026-11-19T025425-51fde95b/D-052
    # A refused calculation (``calculation_error_details``) is NO solution and ends
    # with the fixed ``ErrorMessages.CALCULATION_REFUSED``, never the model's own
    # arithmetic; the result and error keys are orchestrator handler-only keys,
    # cleared on every attempt. Do NOT accept ``proposed_solution`` or
    # ``final_solution`` text beside an error, do NOT make the keys claimable again,
    # and do NOT grow ``engine.py`` for it (799 lines, cap 800).
    @staticmethod
    def finalize_calculation(context: dict[str, Any]) -> dict[str, Any]:
        """On entry to ``final_answer``: a calculator run answers with its result.

        ``final_solution`` is read first and is otherwise model-written, which cuts a
        long number. Returns ``{final_solution: str(result)}`` for a ``simple_calculator``
        solve with a numeric result (``0`` included, ``bool`` not); for one whose
        expression was refused (``calculation_error_details``) the fixed
        ``ErrorMessages.CALCULATION_REFUSED``, never model text; else ``{}``. Both keys
        are handler-only and cleared on every attempt (D-052).
        """
        if (
            context.get(ContextKeys.REASONING_TYPE_SELECTED)
            != ReasoningType.SIMPLE_CALCULATOR.value
        ):
            return {}
        reason = context.get(ContextKeys.CALCULATION_ERROR_DETAILS)
        if reason:
            message = ErrorMessages.CALCULATION_REFUSED.format(reason=reason)
            return {ContextKeys.FINAL_SOLUTION: message}
        result = context.get(ContextKeys.CALCULATION_RESULT)
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            return {}
        return {ContextKeys.FINAL_SOLUTION: str(result)}

    @staticmethod
    def validate_when_insights_ready(context: dict[str, Any]) -> dict[str, Any]:
        """The registered validator: ``validate_solution``, but only once the
        attempt has its insights.

        The validator fires when ``proposed_solution`` is committed; when
        ``key_insights`` arrive in a later commit that early verdict is a stale
        False. It returns ``{}`` then, so no retry is spent and the attempt is
        judged at ``validate_refine`` entry (``ReasoningEngine._check_retry_limit``).
        Every other verdict is ``validate_solution``'s, unchanged.
        """
        # The engine's fallback (D-054 of plan 944e2692) judges an attempt
        # with no verdict. Do NOT re-validate on a key_insights update
        # instead: an invalid attempt would count twice.
        verdict = ReasoningHandlers.validate_solution(context)
        if not verdict[ContextKeys.VALIDATION_CHECKS]["has_insights"]:
            return {}
        return verdict

    @staticmethod
    def count_hybrid_loop(context: dict[str, Any]) -> dict[str, Any]:
        """Count one exit of the hybrid strategy's critical_evaluation state.

        Registered on that state's exit (PRE_TRANSITION, after the transition
        was chosen), so the back edge reads the count of earlier exits: with
        the count starting at 0 it runs at most ``Defaults.MAX_HYBRID_LOOPS``
        times. Returns ``{hybrid_loop_count: previous + 1}`` (an unset count
        is 0). Never raises.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-054: the count is a
        # handler-only key. Do NOT ask the model to increment it again: a
        # message-free bulk fills unset keys only, so a model-written count
        # never moves and the back edge loops until the forced pop.
        previous = context.get(ContextKeys.HYBRID_LOOP_COUNT) or 0
        return {ContextKeys.HYBRID_LOOP_COUNT: int(previous) + 1}

    @staticmethod
    def update_reasoning_trace(context: dict[str, Any]) -> dict[str, Any]:
        """
        Update reasoning trace with size management.

        :param context: Current conversation context
        :return: Updated trace with pruning if needed
        """
        current_state = context.get("_current_state", "")
        previous_state = context.get("_previous_state", "")

        trace = context.get(ContextKeys.REASONING_TRACE, [])

        # Prune old traces if getting too long
        if len(trace) > Defaults.MAX_TRACE_STEPS:
            trace = _head_and_tail(trace, Defaults.MAX_TRACE_STEPS)
            logger.debug(f"Pruned reasoning trace to {len(trace)} steps")

        if previous_state and current_state:
            # Create minimal context snapshot
            snapshot_keys_to_include = [
                ContextKeys.PROBLEM_TYPE,
                ContextKeys.REASONING_STRATEGY,
                ContextKeys.REASONING_TYPE_SELECTED,
                ContextKeys.VALIDATION_RESULT,
                ContextKeys.RETRY_COUNT,
            ]

            context_snapshot = {
                k: v for k, v in context.items() if k in snapshot_keys_to_include
            }

            step = {
                "from": previous_state,
                "to": current_state,
                "context_snapshot": context_snapshot,
            }

            trace.append(step)
            logger.debug(f"Added reasoning step: {previous_state} -> {current_state}")

        return {ContextKeys.REASONING_TRACE: trace}

    @staticmethod
    def prune_context(context: dict[str, Any]) -> dict[str, Any]:
        """
        Prune context to prevent explosion.

        :param context: Current context
        :return: Pruning recommendations
        """
        # Estimate context size. `default=str` is safe HERE and in the two
        # size checks in `filter_context` below: the serialized text is only
        # measured with `len()` and never emitted, so no `__str__` reaches a
        # prompt, a file or a socket (the emitting writers use
        # `context.filtering.redacting_json_default`).
        context_str = json.dumps(context, default=str)
        context_size = len(context_str)

        if context_size <= Defaults.CONTEXT_PRUNE_THRESHOLD:
            return {}

        # Keys that can be pruned if large (none overlap with preserve_keys)
        prune_candidates = [
            ContextKeys.REASONING_TRACE,
            ContextKeys.LOGICAL_STEPS,
            ContextKeys.OBSERVATIONS,
            ContextKeys.CREATIVE_IDEAS,
        ]

        pruned_updates: dict[str, Any] = {}
        for key in prune_candidates:
            value = context.get(key)
            if value is None:
                continue
            if isinstance(value, list) and len(value) > Defaults.PRUNE_LIST_MAX_LENGTH:
                logger.warning(
                    f"Pruning context key '{key}': list truncated from "
                    f"{len(value)} to {Defaults.PRUNE_LIST_MAX_LENGTH} items"
                )
                pruned_updates[key] = (
                    _head_and_tail(value, Defaults.PRUNE_LIST_MAX_LENGTH)
                    if key == ContextKeys.REASONING_TRACE
                    else value[-Defaults.PRUNE_LIST_MAX_LENGTH :]
                )
            elif (
                isinstance(value, str) and len(value) > Defaults.PRUNE_STRING_MAX_LENGTH
            ):
                logger.warning(
                    f"Pruning context key '{key}': string truncated from "
                    f"{len(value)} to {Defaults.PRUNE_STRING_MAX_LENGTH} chars"
                )
                pruned_updates[key] = (
                    value[: Defaults.PRUNE_STRING_MAX_LENGTH]
                    + MESSAGE_TRUNCATION_SUFFIX
                )

        if pruned_updates:
            logger.info(LogMessages.CONTEXT_PRUNED.format(original=context_size))

        return pruned_updates


def register_strategy_handlers(api: API) -> None:
    """Register the handlers two strategy FSMs need on ``api``.

    The calculator strategy needs its executor (``perform_calculation``) and
    finalizer (``final_answer``); the hybrid strategy needs its key clearer and
    its back-edge counter. ``ReasoningEngine`` calls this on its orchestrator
    API; the handlers are keyed by state name, so an FSM without those states
    never fires them. Not idempotent: a second call on one ``api`` registers
    the four handlers again.
    """
    api.register_handler(
        api.create_handler(HandlerNames.CALCULATOR_EXECUTOR)
        .on_state_entry(CALCULATION_STATE)
        .critical()
        .do(ReasoningHandlers.execute_calculation)
    )

    api.register_handler(
        api.create_handler(HandlerNames.CALCULATOR_FINALIZER)
        .on_state_entry(OrchestratorStates.FINAL_ANSWER)
        .do(ReasoningHandlers.finalize_calculation)
    )

    # DECISION plan-2026-11-05T134105-5ed71cf2/D-016: clear the hybrid
    # body keys and needs_refinement on EVERY entry to the first hybrid
    # state so the critical_evaluation back edge re-runs the analysis
    # (core extracts only unset keys; without this the loop is a no-op
    # and needs_refinement stays True). Do NOT add hybrid_loop_count
    # here: it is handler-only and the counter bounds the loop (D-054).
    # Same shape as the retry clear of ``ReasoningEngine._register_handlers``
    # (D-053); the first entry finds nothing to clear.
    api.register_handler(
        clear_keys_on_entry(
            HYBRID_CLEARED_KEYS,
            state=HYBRID_FIRST_STATE,
            name=HandlerNames.HYBRID_KEY_CLEARER,
        )
    )

    # Hybrid back-edge counter: strategy FSMs run on the orchestrator API,
    # and only the hybrid FSM has this state.
    api.register_handler(
        api.create_handler(HandlerNames.HYBRID_LOOP_COUNTER)
        .on_state_exit(HYBRID_EVALUATION_STATE)
        .critical()
        .do(ReasoningHandlers.count_hybrid_loop)
    )


class ContextManager:
    """Manages context size and content."""

    @staticmethod
    def extract_relevant_context(
        source_context: dict[str, Any],
        target_keys: list[str],
        max_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Extract only relevant context keys.

        :param source_context: Source context
        :param target_keys: Keys to extract
        :param max_size: Maximum size limit
        :return: Filtered context
        """
        filtered = {
            k: v
            for k, v in source_context.items()
            if k in target_keys and v is not None
        }

        # Enforce size limit by removing lowest-priority keys until under budget
        if max_size:
            size = len(json.dumps(filtered, default=str))
            if size > max_size:
                logger.warning(
                    f"Context size {size} exceeds limit {max_size}, truncating"
                )
                # Remove keys in reverse order (last-added = lowest priority) until under limit
                keys_by_priority = list(filtered.keys())
                while size > max_size and keys_by_priority:
                    removed_key = keys_by_priority.pop()
                    # Estimate size reduction from removed key instead of re-serializing
                    removed_size = len(
                        json.dumps({removed_key: filtered[removed_key]}, default=str)
                    )
                    del filtered[removed_key]
                    size -= removed_size

        return filtered


class OutputFormatter:
    """Formats and extracts final outputs."""

    @staticmethod
    def extract_final_solution(context: dict[str, Any]) -> str:
        """
        Extract the final solution from context with proper fallbacks.

        :param context: Final context
        :return: Formatted solution string
        """
        for key in _solution_keys():
            solution = context.get(key)
            if _is_solution(solution):
                logger.debug(f"Extracted final solution from context key '{key}'")
                return str(solution)

        # Check if max retries were hit
        if context.get(ContextKeys.MAX_RETRIES_REACHED):
            return ErrorMessages.MAX_RETRIES_EXCEEDED

        return "Solution process completed, but no explicit solution found."

    @staticmethod
    def format_reasoning_summary(trace_info: dict[str, Any]) -> str:
        """
        Format a human-readable reasoning summary.

        :param trace_info: Reasoning trace information
        :return: Formatted summary
        """
        lines = [
            "Reasoning Summary:",
            f"- Total steps: {trace_info.get('total_steps', 0)}",
            f"- Reasoning types: {', '.join(trace_info.get('reasoning_types_used', ['unknown']))}",
            f"- Confidence: {trace_info.get('final_confidence', 0):.2%}",
        ]

        return "\n".join(lines)
