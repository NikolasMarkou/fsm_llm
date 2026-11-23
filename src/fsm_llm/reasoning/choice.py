"""
The strategy choice and the classifier's output of a reasoning solve
====================================================================

Three functions that read no engine state: which strategies of a registry a solve may
choose (``_eligible_choices``), which one a solve runs (``_strategy_for``) and the
normalised result of a finished classifier conversation (``_classification_from``).
They moved here, text for text, from ``engine.py`` (plan baa37f40 step 19.1, D-036) so
that module stays under its size cap; ``engine.py`` imports them back.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.logging import logger

from .constants import ContextKeys, ReasoningType
from .definitions import ReasoningClassificationResult
from .exceptions import ReasoningRegistryError
from .selection import ANALYTICAL, StrategyChoice, find_choice, strategy_choices


def _eligible_choices(
    registry: ExecutableRegistry,
) -> tuple[list[StrategyChoice], StrategyChoice]:
    """The strategies of *registry* a solve may choose and the one it falls back to.

    Contract: ``selection.strategy_choices`` decides (skipped ones are warned there); raises
    ``ReasoningRegistryError`` naming what is missing when no eligible strategy is named
    ``analytical`` (the fallback) or fewer than two are eligible (a closed classification needs
    two intents). Returns ``(choices, the analytical choice)``.
    """
    choices = strategy_choices(registry)
    analytical = find_choice(choices, ANALYTICAL)
    if analytical is None or analytical.name != ANALYTICAL:
        raise ReasoningRegistryError(
            f"the registry has no eligible strategy named {ANALYTICAL!r} (the fallback of "
            f"every solve); eligible: {[choice.name for choice in choices]}"
        )
    if len(choices) < 2:
        raise ReasoningRegistryError(
            "the registry has fewer than two eligible strategies: the choice is a closed "
            f"classification of at least two; eligible: {[c.name for c in choices]}"
        )
    return choices, analytical


def _strategy_for(
    recommended: str,
    preferred: object,
    choices: list[StrategyChoice],
    analytical: StrategyChoice,
) -> StrategyChoice:
    """The strategy a solve runs: the classified one, then the preferred one.

    Contract: ``recommended`` is the classifier's recommendation, found among ``choices`` by name
    or alias (an unknown spelling is ``analytical``, with a WARNING). ``preferred`` is
    ``initial_context["preferred_reasoning_type"]``: when it names a strategy of ``choices`` (a
    name or an alias, ANY eligible one, not only the intents the classifier saw) it WINS over the
    classified one; when it names none, ONE WARNING and it is ignored. Never raises for any value.
    """
    chosen = find_choice(choices, recommended)
    if chosen is None:
        logger.warning(
            f"Unknown reasoning type '{recommended}', defaulting to analytical"
        )
        chosen = analytical
    if preferred:
        named = find_choice(choices, str(preferred))
        if named is None:
            logger.warning(
                f"preferred_reasoning_type {preferred!r} names no known strategy; ignoring"
            )
        else:
            logger.debug(f"Using preferred_reasoning_type from context: {named.name!r}")
            chosen = named
    return chosen


def _classification_from(result: dict[str, Any]) -> ReasoningClassificationResult:
    """The classification of a finished classifier conversation's ``result``.

    Contract: never raises for any JSON value the model wrote. A
    ``recommended_reasoning_type`` that is missing (a classification below
    its confidence threshold, a cut reply or a provider outage writes
    nothing), not a ``str`` or blank becomes ``"analytical"`` (an unknown
    name is mapped to analytical later, by ``map_reasoning_type``); a
    non-``str`` justification or domain becomes ``""``;
    ``alternative_approaches`` keeps its non-blank ``str`` items (a bare
    non-blank ``str`` becomes a one-item list), anything else is dropped.
    Each replaced value, a missing one included, is logged as one WARNING;
    well-shaped values pass unchanged.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-060: the classifier handler
    # is critical (D-058), so the model's values are normalised here. Do NOT
    # build ReasoningClassificationResult from the raw values again: a badly
    # shaped auxiliary field (a str alternative_approaches, seen live) would
    # fail the whole solve. Do NOT drop `.critical()` to absorb it instead:
    # that also hides a spent classifier budget or a provider error.
    recommended = result.get(ContextKeys.RECOMMENDED_REASONING_TYPE)
    if not (isinstance(recommended, str) and recommended.strip()):
        if recommended is None:
            logger.warning(
                "No recommended reasoning type was produced (low confidence, a cut "
                "reply or a failed classification call): using analytical"
            )
        else:
            logger.warning(f"Malformed recommended reasoning type: {recommended!r}")
        recommended = ReasoningType.ANALYTICAL.value
    texts: dict[str, str] = {}
    for key in (ContextKeys.STRATEGY_JUSTIFICATION, ContextKeys.PROBLEM_DOMAIN):
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            logger.warning(f"Malformed classifier field {key} dropped: {value!r}")
        texts[key] = value if isinstance(value, str) else ""
    approaches = result.get(ContextKeys.ALTERNATIVE_APPROACHES)
    items = [approaches] if isinstance(approaches, str) else approaches
    alternatives = (
        [item for item in items if isinstance(item, str) and item.strip()]
        if isinstance(items, list)
        else []
    )
    if approaches is not None and alternatives != approaches:
        logger.warning(f"Malformed alternative approaches normalised: {approaches!r}")
    return ReasoningClassificationResult(
        recommended_type=recommended,
        justification=texts[ContextKeys.STRATEGY_JUSTIFICATION],
        domain=texts[ContextKeys.PROBLEM_DOMAIN],
        alternatives=alternatives,
    )
