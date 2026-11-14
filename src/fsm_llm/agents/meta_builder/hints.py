"""Keyword hints of the meta-builder agent: pure functions of one message (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

Word-boundary matching over a normalized message: whether it asks for the build (``_is_build_trigger``),
may change the artifact type (``_is_type_switch``), and which type a keyword names
(``_detect_type_fallback``, used when the type classification gives no type); ``_turn_hints`` bundles them
into the driver-written context of one turn. ``HintsMixin`` holds the four methods; the patterns, the
alias table and ``_normalize_message`` are module-level here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from fsm_llm.agents.constants import (
    META_BUILD_NEGATION_FILLERS,
    META_BUILD_NEGATIONS,
    META_BUILD_PHRASES,
    META_BUILD_TRIGGERS,
    META_SWITCH_WORDS,
)
from fsm_llm.agents.definitions import ArtifactType

from .names import _K
from .surface import _MetaBuilderSurface

_APOSTROPHES = str.maketrans({"‘": "'", "’": "'"})


def _normalize_message(
    text: str, table: Mapping[int, str | int | None] = _APOSTROPHES
) -> str:
    """``text`` lower-cased, whitespace-collapsed, characters mapped by ``table``.

    The default table writes typographic apostrophes as '.
    """
    return " ".join(text.translate(table).split()).lower()


def _words_pattern(words: Iterable[str], *, plural: bool = False) -> re.Pattern[str]:
    """A pattern matching any of ``words`` as whole words (longest first)."""
    alternatives = "|".join(
        re.escape(w) for w in sorted(set(words), key=len, reverse=True)
    )
    suffix = "s?" if plural else ""
    return re.compile(rf"(?<!\w)(?:{alternatives}){suffix}(?!\w)")


_SWITCH_PATTERN = _words_pattern(META_SWITCH_WORDS)
_BUILD_PHRASE_PATTERN = _words_pattern(META_BUILD_PHRASES)
# A negation that governs what follows it: only filler words between it and
# the end of the text before the phrase; "why not" is not a negation.
_GOVERNING_NEGATION = re.compile(
    rf"(?<!\bwhy ){_words_pattern(META_BUILD_NEGATIONS).pattern}"
    rf"(?: {_words_pattern(META_BUILD_NEGATION_FILLERS).pattern})* $"
)
_CLAUSE_BREAK = re.compile(r"[.,;:!?\n]+")
# Closing punctuation a whole-message trigger ignores: "Done." is "done".
_TRAILING_PUNCTUATION = ".,;:!? "

# The artifact type a keyword names, longest alias first so "finite state
# machine" wins over "state machine" and "data pipeline" over "pipeline";
# each alias matches whole words with an optional plural "s". Used when the
# type classification gives no type (provider failure or no intent).
_TYPE_ALIASES: dict[str, str] = dict(
    sorted(
        {
            "finite state machine": "fsm",
            "fsm": "fsm",
            "state machine": "fsm",
            "state_machine": "fsm",
            "conversational": "fsm",
            "conversation": "fsm",
            "help desk": "fsm",
            "onboarding": "fsm",
            "interview": "fsm",
            "chat bot": "fsm",
            "helpdesk": "fsm",
            "chatbot": "fsm",
            "dialogue": "fsm",
            "dialog": "fsm",
            "survey": "fsm",
            "quiz": "fsm",
            "faq": "fsm",
            "bot": "fsm",
            "data pipeline": "workflow",
            "processing": "workflow",
            "workflow": "workflow",
            "automation": "workflow",
            "pipeline": "workflow",
            "sequence": "workflow",
            "process": "workflow",
            "steps": "workflow",
            "batch": "workflow",
            "flow": "workflow",
            "etl": "workflow",
            "agentic": "agent",
            "research": "agent",
            "navigate": "agent",
            "browse": "agent",
            "search": "agent",
            "react": "agent",
            "agent": "agent",
            "tools": "agent",
            "tool": "agent",
        }.items(),
        key=lambda kv: -len(kv[0]),
    )
)
_TYPE_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_words_pattern([alias], plural=True), type_str)
    for alias, type_str in _TYPE_ALIASES.items()
)


class HintsMixin(_MetaBuilderSurface):
    """The keyword hints: pure functions of one message, and the driver-written turn context."""

    @staticmethod
    def _is_build_trigger(normalized: str) -> bool:
        """True when the normalized message asks for the build.

        A whole-message trigger ("ok", "build it"), or a build phrase that no
        negation governs ("don't build it yet" and "do not ever build it" are
        not one; "it's not perfect but build it" and "why not build it" are).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-044: a negation blocks a
        # build phrase only when it governs it (right before it, filler words
        # allowed). Do NOT go back to "any negation earlier in the clause":
        # "don't forget to build it", "not a problem build it" and "it's not
        # perfect but build it" stopped building (they built at e1f63a9).
        if normalized.strip().rstrip(_TRAILING_PUNCTUATION) in META_BUILD_TRIGGERS:
            return True
        for clause in _CLAUSE_BREAK.split(normalized):
            for match in _BUILD_PHRASE_PATTERN.finditer(clause):
                if not _GOVERNING_NEGATION.search(clause[: match.start()]):
                    return True
        return False

    @staticmethod
    def _is_type_switch(normalized: str) -> bool:
        """True when the normalized message may change the type."""
        return _SWITCH_PATTERN.search(normalized) is not None

    @staticmethod
    def _detect_type_fallback(text: str) -> ArtifactType:
        """The artifact type named by a keyword alias in ``text``, else FSM.

        Used when the type classification gives no type (provider failure or
        no intent); see ``_TYPE_ALIASES``.
        """
        normalized = _normalize_message(text)
        for pattern, type_str in _TYPE_ALIAS_PATTERNS:
            if pattern.search(normalized):
                return ArtifactType(type_str)
        return ArtifactType.FSM

    def _turn_hints(
        self, message: str, requirements: list[str], *, build_requested: bool | None
    ) -> dict[str, Any]:
        """Driver-written context for the next turn on ``message``.

        ``build_requested`` overrides the keyword trigger (``run`` always
        builds, ``start`` never does); ``None`` reads it from the message.
        """
        normalized = _normalize_message(message)
        if build_requested is None:
            build_requested = self._is_build_trigger(normalized)
        return {
            _K.REQUIREMENTS: requirements,
            _K.LATEST_REQUEST: message,
            _K.BUILD_REQUESTED: build_requested,
            _K.TYPE_SWITCH: self._is_type_switch(normalized),
            _K.KEYWORD_TYPE: self._detect_type_fallback(message).value,
        }
