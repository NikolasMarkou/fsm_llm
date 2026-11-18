"""``FSMDefinition`` and the conversation, context and instance models of a running FSM."""

from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator

from fsm_llm.constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    MAX_PERSONA_LENGTH,
    MESSAGE_TRUNCATION_SUFFIX,
    has_internal_prefix,
)
from fsm_llm.core.state_kinds import SYNTHESIZE_SAMPLES, definition_problem, kind_of
from fsm_llm.logging import logger

from .classification import ClassificationResult
from .limits import CONVERSATION_SUMMARY_MAX_CHARS
from .pass_models import DataExtractionResponse, ResponseGenerationResponse
from .schema import SCHEMA_VERSION_DEFAULT, newer_fields
from .state import State
from .transition import logic_referenced_keys


def result_key_names(definition: FSMDefinition) -> frozenset[str]:
    """The ``result_key`` of every completion and synthesize state: the keys no
    extraction channel may write, classification included (only the state's own
    completion call writes its key, D-021, and only the runner writes a
    synthesize key, D-002 of plan 2cefeb61).

    Read from the kind registry (``KindSpec.result_key``), never from a branch on
    the kind configs. Takes a loaded definition, returns a new frozenset (empty
    for a definition with no such state), never raises.
    """
    return frozenset(
        key
        for state in definition.states.values()
        if (key := kind_of(state).result_key(state)) is not None
    )


def handler_only_names(definition: FSMDefinition) -> frozenset[str]:
    """Keys no field or bulk extraction may write (D-033): the FSM author's
    ``handler_only_keys`` plus ``result_key_names``.

    The one answer shared by ``MessagePipeline._handler_only_keys`` and
    ``flow_analysis``. Classification is a separate channel: it may write a
    LISTED key (D-033 trade-off (c), kept by D-002 of plan 52ce5464) and never a
    result key. Takes a loaded definition, returns a new frozenset (empty for a
    definition with none of the three), never raises.
    """
    return frozenset(definition.handler_only_keys) | result_key_names(definition)


def reachable_from(
    adjacency: Mapping[str, Iterable[str]], starts: Iterable[str]
) -> set[str]:
    """Every node reachable from ``starts`` along ``adjacency``, starts included.

    The one graph walk of the FSM tooling (forward reachability for the
    definition, the validator and the artifact builders; pass a reversed
    adjacency for "can reach"). Contract: ``adjacency`` maps a node to the
    nodes it points at; a node missing from it, or a target that is not a key,
    is reached but has no successors (no ``KeyError``); an empty or ``None``
    target is the caller's to drop. Returns a new set. Breadth-first, O(V + E).
    """
    reached = set(starts)
    queue = deque(reached)
    while queue:
        for target in adjacency.get(queue.popleft(), ()):
            if target not in reached:
                reached.add(target)
                queue.append(target)
    return reached


def unknown_initial_state_message(initial_state: str) -> str:
    """The structural defect "``initial_state`` names no state"."""
    return f"Initial state '{initial_state}' not found in states"


def dangling_transition_message(state_id: str, target: str) -> str:
    """The structural defect "a transition points at a state that is not defined"."""
    return f"Invalid transition from '{state_id}' to non-existent state '{target}'"


def orphaned_states_message(orphans: Iterable[str]) -> str:
    """The structural defect "states no path from the initial state reaches"."""
    return f"Orphaned states detected: {sorted(orphans)}"


# The three messages above are the structure defects of `FSMDefinition` that
# `FSMValidator` also finds on the raw dicts; one wording lets the validator
# report each defect once (it skips a message a `Schema:` error already says).


# The sample results a synthesize state's edges are run on (D-035) are the data
# of the synthesize kind (`state_kinds.SYNTHESIZE_SAMPLES`); the name stays for
# `test_codes`, which pins their strings against `codes.Status` and `Reason`.
_SYNTHESIZE_SAMPLES = SYNTHESIZE_SAMPLES


# Definition objects whose schema-version findings were already logged
# (D-003), keyed by identity. The value is weak: the entry vanishes when the
# object is collected, so a recycled id can never suppress a new definition.
_REPORTED_DEFINITIONS: weakref.WeakValueDictionary[int, FSMDefinition] = (
    weakref.WeakValueDictionary()
)


def _first_report(definition: FSMDefinition) -> bool:
    """True the first time `definition` is seen, False on every later call.

    Private to `FSMDefinition.validate_fsm_structure`; never raises. The flag
    lives outside the model so equality, `model_dump` and `fsm_id` are untouched.
    """
    key = id(definition)
    if _REPORTED_DEFINITIONS.get(key) is definition:
        return False
    _REPORTED_DEFINITIONS[key] = definition
    return True


class FSMDefinition(BaseModel):
    """
    Complete FSM definition for the 2-pass architecture.

    Enhanced with separate extraction and response capabilities.
    """

    name: str = Field(
        ..., description="FSM name identifier", min_length=1, max_length=100
    )

    description: str = Field(
        ...,
        description="FSM purpose and functionality description",
        min_length=1,
        max_length=1000,
    )

    states: dict[str, State] = Field(
        ..., description="All states in the FSM", min_length=1
    )

    initial_state: str = Field(
        ..., description="Starting state identifier", min_length=1
    )

    # DECISION plan-2026-11-12T023759-9013b0e3/D-004: the default is the
    # registry constant (4.2). `fsm_id` and the library digest hash
    # `model_dump(exclude_defaults=True)`, so the default is not hashed and an
    # omitted version keeps its id; an explicit older version ("4.1") is hashed
    # and moves the id, by design. Do NOT special-case `version` in the hash
    # (exclude it, or treat "4.1" as the default): that moves the ids of
    # explicit "3.0" and "4.0" definitions or adds a rule beside D-030.
    version: str = Field(
        default=SCHEMA_VERSION_DEFAULT,
        description="FSM definition version",
        min_length=1,
        max_length=20,
    )

    persona: str | None = Field(
        None,
        description="Conversation persona for response generation",
        max_length=MAX_PERSONA_LENGTH,
    )

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-033: opt-in and FSM-wide. The
    # default stays empty because a default-closed rule (auto-minting every
    # key a transition reads) would freeze 36 of 49 shipped example FSMs. The
    # pipeline consults it at three filters (bulk return, per-field configs,
    # post-transition configs). A listed key that is also a
    # classification_extractions field is NOT covered (classification writes
    # are a separate channel, in Pass 1 and after a transition: D-002 of plan
    # 52ce5464); only a framework result key is closed to classification.
    handler_only_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Context keys only handlers, update_context and initial_context may "
            "write; never extracted from user text (opt-in, default empty)"
        ),
    )

    @model_validator(mode="after")
    def validate_fsm_structure(self) -> FSMDefinition:
        """Comprehensive FSM validation for the 2-pass architecture."""
        logger.debug(f"Validating FSM: {self.name}")

        # Basic structure validation
        if self.initial_state not in self.states:
            raise ValueError(unknown_initial_state_message(self.initial_state))

        # Validate state.id matches dict key
        for state_id, state in self.states.items():
            if state.id != state_id:
                raise ValueError(
                    f"State id '{state.id}' does not match dict key '{state_id}'"
                )

        # Validate all transitions
        for state_id, state in self.states.items():
            for transition in state.transitions:
                if transition.target_state not in self.states:
                    raise ValueError(
                        dangling_transition_message(state_id, transition.target_state)
                    )

        # Terminal state validation
        terminal_states = {
            state_id for state_id, state in self.states.items() if not state.transitions
        }

        if not terminal_states:
            raise ValueError("FSM must have at least one terminal state")

        # Reachability validation
        reachable_states = reachable_from(
            {
                state_id: [t.target_state for t in state.transitions]
                for state_id, state in self.states.items()
            },
            [self.initial_state],
        )
        orphaned_states = set(self.states.keys()) - reachable_states

        if orphaned_states:
            raise ValueError(orphaned_states_message(orphaned_states))

        # Validate terminal state reachability
        reachable_terminals = terminal_states.intersection(reachable_states)
        if not reachable_terminals:
            raise ValueError("No terminal states are reachable from initial state")

        self._check_synthesize_states()
        # DECISION plan-2026-11-12T023759-9013b0e3/D-003: a declared version
        # older than the fields used is a WARNING, once per definition object,
        # and nothing else. Do NOT raise or add an error here: the loader and
        # `FSMValidator` must never be stricter than they were (the
        # never-stricter pins), and a file declaring "4.1" with a `synthesize`
        # state has always loaded. Library logging is off by default, so the
        # visible channel is the `FSMValidator` report (`_validate_schema_versions`).
        # Once per OBJECT, not per construction: pydantic runs this validator
        # again each time the instance is validated as a field of another model
        # (`FSMStackFrame` in `API.start_conversation` and `push_fsm`), which
        # logged one WARNING per conversation. The object is remembered in
        # `_REPORTED_DEFINITIONS` (by identity, dropped when it is collected).
        # DECISION plan-2026-11-12T023759-9013b0e3/D-016: the once-per-object
        # rule above is this decision's. A definition changed after its first
        # report is not re-reported (accepted).
        # Do NOT use a `PrivateAttr` or any field for the flag (a private attr
        # makes pydantic synthesize a `model_post_init` that reaches the public
        # API snapshot, and a field would reach `model_dump` and move `fsm_id`
        # and the library digest) and do NOT use a content-keyed or
        # once-per-process cache (unrelated definitions and tests would
        # suppress each other).
        if _first_report(self):
            newer = newer_fields(self)
            if newer:
                logger.warning(
                    f"FSM '{self.name}': "
                    + "; ".join(finding.message for finding in newer)
                )
        logger.debug(f"FSM '{self.name}' validation successful")
        return self

    # The edge rules of a synthesize state stay in the LOADER (they need the whole
    # FSM and `FSMValidator` may never be stricter than the loader: do NOT move
    # this call to core/verify/). They are the `check_definition` hook of the
    # synthesize kind in state_kinds.py; its D-034/D-035 anchor says what the
    # rules must keep. The real evaluator and `logic_referenced_keys` are handed
    # in because that leaf module cannot import them.
    def _check_synthesize_states(self) -> None:
        """Cross-state rules of the state kinds (a synthesize state's edges and
        host keys), each asked of the kind registry; see
        ``state_kinds.definition_problem``."""
        from fsm_llm.core.transition_evaluator import (
            TransitionEvaluator,  # circular at import
        )

        evaluator = TransitionEvaluator()
        for state in self.states.values():
            problem = definition_problem(self, state, evaluator, logic_referenced_keys)
            if problem is not None:
                raise ValueError(f"State '{state.id}': {problem}")


def exchange_preview_lines(exchanges: list[dict[str, str]], width: int) -> list[str]:
    """One ``"role: message"`` line per message, each message cut to ``width`` + ``...``."""
    return [
        f"{role}: {message[:width]}{'...' if len(message) > width else ''}"
        for exchange in exchanges
        for role, message in exchange.items()
    ]


class Conversation(BaseModel):
    """Enhanced conversation management for the 2-pass architecture."""

    exchanges: list[dict[str, str]] = Field(
        default_factory=list, description="Conversation history in chronological order"
    )

    max_history_size: int = Field(
        default=DEFAULT_MAX_HISTORY_SIZE,
        description="Maximum conversation exchanges to retain",
        ge=0,
        le=1000,
    )

    max_message_length: int = Field(
        default=DEFAULT_MAX_MESSAGE_LENGTH,
        description="Maximum message length in characters",
        ge=1,
        le=50000,
    )

    summary: str | None = Field(
        default=None,
        description=(
            "Compressed summary of older exchanges that were trimmed from history. "
            "Populated automatically when history exceeds max_history_size, or "
            "explicitly via ContextCompactor.summarize()."
        ),
    )

    def _clipped(self, message: str) -> str:
        """``message`` cut to ``max_message_length``, ending in the truncation suffix.

        Contract: a message within the limit is returned unchanged; a longer
        one ends in ``MESSAGE_TRUNCATION_SUFFIX`` and is exactly
        ``max_message_length`` long, or is a bare cut when the limit is not
        longer than the suffix. The one truncation rule of both ``add_*``.
        """
        limit = self.max_message_length
        if len(message) <= limit:
            return message
        suffix = MESSAGE_TRUNCATION_SUFFIX
        if limit <= len(suffix):
            return message[:limit]
        return message[: limit - len(suffix)] + suffix

    def add_user_message(self, message: str) -> None:
        """Add user message with automatic truncation."""
        self.exchanges.append({"user": self._clipped(message)})
        # Trimming is deferred to add_system_message (turn completion) so a
        # rolled-back user turn isn't prematurely compressed into the summary.

    def add_system_message(self, message: str) -> None:
        """Add system message with automatic truncation."""
        self.exchanges.append({"system": self._clipped(message)})
        self._maintain_history_size()

    def get_recent(self, n: int | None = None) -> list[dict[str, str]]:
        """Get recent conversation messages.

        Args:
            n: Number of exchange pairs (each pair = 1 user + 1 system message)
               to return. Returns the last ``n * 2`` individual messages (a
               window of messages, not of pairs: a silent state adds no reply
               and a step adds no user message, so the window can hold fewer
               than ``n`` turns). If fewer than ``n * 2`` messages exist, all
               are returned. Defaults to max_history_size.
        """
        if n is None:
            n = self.max_history_size

        if n <= 0:
            return []

        # Each exchange is assumed to be a user+system pair (2 messages).
        # If the conversation has an odd number of messages (e.g., user sent
        # but system hasn't replied yet), this may return a partial pair.
        return self.exchanges[-n * 2 :]

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search conversation history for exchanges matching a query.

        Performs case-insensitive substring matching on all exchange
        messages (both user and system). It searches ``exchanges`` only: the
        ``summary`` (the digest of the exchanges trimmed from history) is not
        searched, so text that was trimmed away is not found.

        Args:
            query: Search string.
            limit: Maximum number of matching exchanges to return.

        Returns:
            List of matching exchange dicts, most recent first.
        """
        if not query:
            return []

        query_lower = query.lower()
        matches: list[dict[str, str]] = []

        # Search exchanges in reverse (most recent first)
        for exchange in reversed(self.exchanges):
            if len(matches) >= limit:
                break
            for value in exchange.values():
                if query_lower in value.lower():
                    matches.append(exchange)
                    break

        return matches

    def get_summary_and_recent(
        self, n: int | None = None
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Get conversation summary (if any) and recent exchanges.

        Convenience method for prompt builders that want to include
        both the compressed summary of older history and the recent
        exchange window.

        Args:
            n: Number of recent exchanges. Defaults to max_history_size.

        Returns:
            Tuple of (summary_text_or_None, recent_exchanges).
        """
        return self.summary, self.get_recent(n)

    def _maintain_history_size(self) -> None:
        """Trim history to max_history_size exchanges (each = 2 messages).

        When trimming, captures a simple text summary of the removed
        exchanges to preserve context that would otherwise be lost.
        """
        limit = self.max_history_size * 2
        if limit == 0:
            self._append_to_summary(self.exchanges)
            self.exchanges.clear()
            return
        if len(self.exchanges) > limit:
            trimmed = self.exchanges[:-limit]
            self._append_to_summary(trimmed)
            self.exchanges = self.exchanges[-limit:]

    def _append_to_summary(self, trimmed_exchanges: list[dict[str, str]]) -> None:
        """Append trimmed exchanges to the conversation summary.

        Produces a compact text representation of the trimmed exchanges
        and appends it to the existing summary (if any). Caps total
        summary length at ``CONVERSATION_SUMMARY_MAX_CHARS`` characters.
        """
        if not trimmed_exchanges:
            return

        # Compact: first 100 chars of each message
        new_text = " | ".join(exchange_preview_lines(trimmed_exchanges, 100))
        combined = f"{self.summary} | {new_text}" if self.summary else new_text
        self.summary = combined[:CONVERSATION_SUMMARY_MAX_CHARS]


class FSMContext(BaseModel):
    """Enhanced context management for the 2-pass architecture.

    Supports an optional ``working_memory`` for structured buffer-based
    context management. When set, ``get_user_visible_data()`` includes
    data from all non-hidden working memory buffers (flattened). The flat
    ``data`` dict remains the primary storage.

    Prompt reach: non-hidden buffer data sits under ``data`` (``data`` wins
    on collision) in ``get_merged_data()``, which feeds the transition
    evaluator's working context and the Pass-2 ``<current_context>`` block at
    all three sites (``_execute_response_generation_pass``,
    ``_stream_response_generation_pass``, ``generate_initial_response``),
    scoped by ``read_keys``. ``get_user_visible_data()`` (the same merge
    without internal keys) feeds the Pass-1 per-field extraction prompt (a
    field whose ``context_keys`` is left at its default) and both classifier
    call sites (through ``MessagePipeline._build_classifier_context``). The
    bulk extraction pass sees ``data`` only. ``update_context`` and handlers
    write ``data`` only. See
    ``fsm_llm.memory.WorkingMemory`` for the full statement.
    """

    data: dict[str, Any] = Field(
        default_factory=dict, description="Conversation context data"
    )

    conversation: Conversation = Field(
        default_factory=Conversation, description="Conversation history management"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="System metadata and operational data"
    )

    # Runtime: WorkingMemory | None (typed Any to avoid a circular import).
    # `exclude=True` means model_dump()/model_dump_json() omit this field
    # ENTIRELY (no key, not even None); model_copy(deep=True) and copy.deepcopy
    # do carry it. Persist working memory via WorkingMemory.to_dict(), as
    # save_session does -- not via a context dump.
    working_memory: Any = Field(
        default=None,
        description=(
            "Optional WorkingMemory instance for structured buffer-based "
            "context management. Non-hidden buffer data is merged under "
            "the flat data dict (data wins) for the transition evaluator, "
            "the Pass-2 prompt, the per-field extraction prompt and the "
            "classifiers; hidden buffers are never merged. Import from "
            "fsm_llm.memory."
        ),
        exclude=True,
    )

    def __init__(self, **data):
        """Initialize with optional conversation configuration."""
        if "conversation" not in data:
            data = dict(data)
            max_history = data.pop("max_history_size", DEFAULT_MAX_HISTORY_SIZE)
            max_message_length = data.pop(
                "max_message_length", DEFAULT_MAX_MESSAGE_LENGTH
            )

            data["conversation"] = Conversation(
                max_history_size=max_history, max_message_length=max_message_length
            )

        super().__init__(**data)

    def update(self, new_data: dict[str, Any]) -> None:
        """Update context with new data.

        Writes exactly the keys of ``new_data``, internal-prefix keys included
        (no warning: see the DECISION below).
        """
        # DECISION plan-2026-11-06T093029-0068b055/D-026
        # No WARNING for an internal-prefix key here. Every caller that reaches
        # this method writes the key BY NAME through a public write path
        # (start_conversation seeding, push_fsm inheritance, API.update_context,
        # the pop-merge diff): agents seed `_max_iterations` etc. on every run,
        # so the line was noise on the documented path. The accidental paths
        # never get here: an extraction result is stripped by
        # `clean_context_keys` first, and a handler delta is merged straight
        # into `context.data` by `merge_delta` (reserved keys refused there
        # with their own WARNING). Do NOT re-add a warning in this method, and
        # do NOT re-inline a prefix check to bring it back; guard a new
        # accidental-write path at that path. The debug line below names the
        # keys.
        if new_data:
            logger.debug(f"Updating context with keys: {list(new_data.keys())}")
            self.data.update(new_data)

    def get_merged_data(self) -> dict[str, Any]:
        """Non-hidden WorkingMemory buffers under the flat ``data`` dict.

        Returns a new dict: ``working_memory.get_all_data()`` (hidden buffers
        never included) overlaid by ``data`` (``data`` wins on collision).
        Internal-prefix keys are NOT stripped, so the result is a drop-in
        for ``data`` wherever a consumer applies its own filter (the
        transition evaluator, the Pass-2 prompt builder). Without working
        memory, or with empty buffers, it equals ``dict(data)`` in content
        and key order. Never raises.
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-008: the one WM-under-data
        # merge for every decision point. Do NOT let a buffer value override
        # context.data (existing FSMs that never write WM must stay
        # byte-identical) and do NOT merge hidden buffers (get_all_data skips
        # them; they carry orchestration metadata that must never reach an LLM).
        if self.working_memory is not None and hasattr(
            self.working_memory, "get_all_data"
        ):
            merged: dict[str, Any] = self.working_memory.get_all_data()
            merged.update(self.data)
            return merged
        return dict(self.data)

    def get_user_visible_data(self) -> dict[str, Any]:
        """``get_merged_data()`` without internal-prefix keys."""
        # DECISION plan-2026-11-06T093029-0068b055/D-003: key policy lives in
        # security/ (has_internal_prefix). Do NOT re-add a literal key such
        # as "system" here: a user key named ``system`` is an ordinary key,
        # shown to extraction prompts as it already is in Pass 2 and get_data.
        return {
            key: value
            for key, value in self.get_merged_data().items()
            if not has_internal_prefix(key)
        }


class FSMInstance(BaseModel):
    """Enhanced FSM instance for the 2-pass architecture."""

    fsm_id: str = Field(
        ...,
        description="FSM definition identifier",
        min_length=1,
    )

    current_state: str = Field(
        ..., description="Current state identifier", min_length=1, max_length=100
    )

    context: FSMContext = Field(
        default_factory=FSMContext, description="Conversation context and history"
    )

    persona: str | None = Field(
        default=None,
        description="Conversation persona (None: the definition's persona)",
        max_length=MAX_PERSONA_LENGTH,
    )

    last_extraction_response: DataExtractionResponse | None = Field(
        None, description="Last data extraction response for debugging"
    )

    last_transition_decision: ClassificationResult | None = Field(
        None, description="Last classification result for transition debugging"
    )

    last_response_generation: ResponseGenerationResponse | None = Field(
        None, description="Last response generation for debugging"
    )
