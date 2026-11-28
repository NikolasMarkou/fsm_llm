"""Which strategies a reasoning solve may choose (plan baa37f40 step 17, D-033; roadmap E3.4).

One function decides it for every consumer: the closed ``choose.targets`` list of the orchestrator's
``execute_reasoning`` call, the classifier's closed intents (step 17b), the ``--type`` choices of
the CLI, the spelling table of ``map_reasoning_type`` and the solution keys of the final answer. A
second filter on any of those paths is a defect: the lists would drift apart and a model could be
shown a strategy the call refuses.

Interface contract:

- ``strategy_choices(registry)``: the eligible strategies of an ``ExecutableRegistry`` as a list of
  ``StrategyChoice`` (frozen: ``name``, ``target_id``, ``description``, ``aliases``,
  ``solution_key``), the nine built-ins first in ``STRATEGY_ORDER``, the rest by id. Eligible: kind
  ``strategy`` AND capability ``strategy``, the latest version of its id, a document strategy (an
  envelope with a strategy declaration), lifecycle not ``deprecated``, ``contract.inputs`` within
  ``PROBLEM_KEYS``, a ``name`` that is an ASCII identifier and unique (case-insensitively). What
  passed the kind and capability filter and fails a rule is skipped with ONE WARNING naming its id
  and why; a ``procedure`` (model-written) executable is never listed whatever it claims. An alias
  that equals a name or an earlier alias is dropped with a WARNING (names win). Never raises for a
  registry it can read; the id is only a registry key, no target is ever resolved by a string
  import.
- ``classification_intents(choices)``: the closed intents of the classifier as ``{name,
  description}`` dicts, derived from the SAME ``choices`` the call's ``choose.targets`` came from
  (plan baa37f40 step 17b, D-034): in list order, at most ``MAX_CLASSIFICATION_INTENTS`` (15);
  when more are eligible the first 15 are kept and, if ``analytical`` is not among them, the 15th
  is replaced by it; ``ValueError`` when no choice is named ``analytical``. The description of a
  choice is already one control-stripped line, so a description cannot forge another intent line.
- ``find_choice(choices, text)``: the choice a spelling names (the name or an alias, case and
  padding ignored; a name beats an alias), or ``None``.
- ``builtin_choices()``: the choices of the nine built-ins, from one private cached tuple (a fresh
  list each call), what ``registry=None`` means in ``utilities``.
- ``choices_for(registry)``: ``builtin_choices()`` for ``None``, else ``strategy_choices(registry)``.
- ``ANALYTICAL``: the name of the strategy every solve falls back to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    MAX_CLASSIFICATION_INTENTS,
)
from fsm_llm.logging import logger
from fsm_llm.text import capped_line

from .constants import ContextKeys, Defaults, ReasoningType
from .strategy_files import STRATEGY_ORDER, strategy_id

if TYPE_CHECKING:
    from fsm_llm.engine.executables import ExecutableRegistry

__all__ = [
    "ANALYTICAL",
    "PROBLEM_KEYS",
    "StrategyChoice",
    "builtin_choices",
    "choices_for",
    "classification_intents",
    "find_choice",
    "strategy_choices",
]

ANALYTICAL = ReasoningType.ANALYTICAL.value

#: The keys a strategy may declare as inputs: the four a solve has (the orchestrator hands a pushed
#: strategy exactly these, whichever it reads).
PROBLEM_KEYS = frozenset(
    {
        ContextKeys.PROBLEM_STATEMENT,
        ContextKeys.PROBLEM_COMPONENTS,
        ContextKeys.CONSTRAINTS,
        ContextKeys.PROBLEM_TYPE,
    }
)

_IDENTIFIER = re.compile(ASCII_IDENTIFIER_PATTERN)


@dataclass(frozen=True)
class StrategyChoice:
    """One strategy a solve may choose. ``aliases`` are lowercase and already deduplicated."""

    name: str
    target_id: str
    description: str
    aliases: tuple[str, ...]
    solution_key: str


# DECISION plan-2026-11-23T122249-baa37f40/D-033
# This is the ONE rule for which registered executables a solve may choose (the closed
# `choose.targets`, the classifier's intents, `--type`, the spelling table). Do NOT filter a second
# time on any of those paths (the model could be offered a strategy the call refuses), do NOT
# resolve a target through `importlib` or any string import (the id is a registry key only), do NOT
# make a `procedure` (model-written) executable eligible by widening the kind filter, and do NOT
# let a lifecycle claim decide eligibility beyond refusing `deprecated` (the envelope default
# `draft` must stay selectable). The registry's `get` serves the latest version, so older versions
# of an id are skipped silently.
def strategy_choices(registry: ExecutableRegistry) -> list[StrategyChoice]:
    """The eligible strategies of *registry*, built-ins first (see the module docstring)."""
    rank = {strategy_id(name): index for index, name in enumerate(STRATEGY_ORDER)}
    listed: list[Any] = [
        executable
        for executable in registry.list(kind="strategy", capability="strategy")
        if registry.get(executable.id) is executable
    ]
    listed.sort(
        key=lambda executable: (rank.get(executable.id, len(rank)), executable.id)
    )
    owners: dict[
        str, str
    ] = {}  # lowercase name or alias -> id of the strategy that owns it
    eligible: list[Any] = []
    seeded = False
    for executable in listed:
        if executable.id not in rank and not seeded:
            # the built-ins are judged (they sort first): their aliases are taken before any
            # other name is, so a user strategy named `math` cannot take the alias (D-044 (3))
            seeded = True
            for built_in in eligible:
                for alias in built_in.envelope.body.strategy.aliases:
                    owners.setdefault(alias.lower().strip(), built_in.id)
        reason = _skip_reason(executable, owners)
        if reason is None:
            eligible.append(executable)
            owners[executable.envelope.name.lower()] = executable.id
        else:
            logger.warning(f"strategy {executable.id!r} is not selectable: {reason}")
    choices: list[StrategyChoice] = []
    for executable in eligible:  # a second pass: a name always beats an alias
        envelope = executable.envelope
        aliases: list[str] = []
        for alias in envelope.body.strategy.aliases:
            key = alias.lower().strip()
            if not key or key == envelope.name.lower() or key in aliases:
                continue
            if owners.get(key, executable.id) != executable.id:
                logger.warning(
                    f"alias {alias!r} of strategy {executable.id!r} is dropped: "
                    f"it already names {owners[key]!r}"
                )
                continue
            owners[key] = executable.id
            aliases.append(key)
        choices.append(
            StrategyChoice(
                name=envelope.name,
                target_id=executable.id,
                description=capped_line(
                    envelope.description, Defaults.STRATEGY_DESCRIPTION_CHARS
                )
                or envelope.name,
                aliases=tuple(aliases),
                solution_key=envelope.body.strategy.solution_key,
            )
        )
    return choices


def _skip_reason(executable: Any, owners: dict[str, str]) -> str | None:
    """Why *executable* is not a selectable strategy, or ``None`` when it is."""
    envelope = getattr(executable, "envelope", None)
    if (
        envelope is None
        or getattr(getattr(envelope, "body", None), "strategy", None) is None
    ):
        return "it has no strategy declaration (only a document strategy can be chosen)"
    if executable.lifecycle is Lifecycle.DEPRECATED:
        return "it is deprecated"
    outside = sorted(set(executable.contract.inputs) - PROBLEM_KEYS)
    if outside:
        return f"its contract asks for inputs outside the problem keys: {outside}"
    name = envelope.name
    if not _IDENTIFIER.fullmatch(name):
        return f"its name {name!r} is not an ASCII identifier"
    if name.lower() in owners:
        return f"its name {name!r} is already taken by {owners[name.lower()]!r}"
    return None


def find_choice(choices: list[StrategyChoice], text: str) -> StrategyChoice | None:
    """The choice a spelling names: a name or an alias, case and padding ignored.

    A name beats an alias (the first pass of ``strategy_choices`` keeps them apart, so the two
    loops below can never disagree); ``None`` for an unknown spelling. Never raises for a ``str``.
    """
    key = text.lower().strip()
    for choice in choices:
        if choice.name.lower() == key:
            return choice
    for choice in choices:
        if key in choice.aliases:
            return choice
    return None


def classification_intents(choices: list[StrategyChoice]) -> list[dict[str, str]]:
    """The classifier's closed intents derived from *choices* (see the module docstring)."""
    kept = choices[:MAX_CLASSIFICATION_INTENTS]
    if not any(choice.name == ANALYTICAL for choice in kept):
        analytical = next((c for c in choices if c.name == ANALYTICAL), None)
        if analytical is None:
            raise ValueError(
                f"no strategy named {ANALYTICAL!r}: the fallback intent must be in the list"
            )
        kept = [*choices[: MAX_CLASSIFICATION_INTENTS - 1], analytical]
    return [{"name": c.name, "description": c.description} for c in kept]


@cache
def _builtin() -> tuple[StrategyChoice, ...]:
    # lazy: `executables` imports the engine, which imports this module
    from .executables import strategy_registry

    return tuple(strategy_choices(strategy_registry()))


def builtin_choices() -> list[StrategyChoice]:
    """The nine built-in strategies as choices, in ``STRATEGY_ORDER`` (a fresh list)."""
    return list(_builtin())


def choices_for(registry: ExecutableRegistry | None) -> list[StrategyChoice]:
    """``builtin_choices()`` for ``None``, else the eligible strategies of *registry*."""
    return builtin_choices() if registry is None else strategy_choices(registry)
