"""Structure, key and kind checks of ``FSMValidator`` (``StructureMixin``).

Layer ``core`` (plan d454fd62, step 75): split out of the old ``fsm_llm.validator``;
the method bodies are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from pydantic import BaseModel

from fsm_llm.constants import (
    DEFAULT_COMPLETION_RESULT_KEY,
    TRANSITION_CLASSIFICATION_THRESHOLD_KEY,
)
from fsm_llm.core.definitions import (
    DEFAULT_CALL_RESULT_KEY,
    DEFAULT_SYNTHESIS_RESULT_KEY,
    ClassificationExtractionConfig,
    CompletionResponse,
    FieldExtractionConfig,
    FSMDefinition,
    IntentDefinition,
    State,
    Transition,
    TransitionCondition,
    dangling_transition_message,
    logic_referenced_keys,
    orphaned_states_message,
    unknown_initial_state_message,
)
from fsm_llm.core.definitions.schema import newer_fields
from fsm_llm.core.state_kinds import RawLintServices, raw_kinds

if TYPE_CHECKING:
    from .result import FSMValidationResult

# The result kinds a completion state can produce, read off the one model
# that defines them: every kind with tools, all but ``calls`` without.
_RESULT_KINDS: tuple[str, ...] = get_args(
    CompletionResponse.model_fields["kind"].annotation
)


# Who writes the result key of a kind, for the warning of an extraction channel
# that names it (D-002 of plan 52ce5464: every channel is closed to it).
_COMPLETION_WRITER = "a completion call"
_RESULT_WRITERS = {
    "completion": _COMPLETION_WRITER,
    "synthesize": "the synthesize executor",
    "call": "the call executor",
}


def _entries(value: Any) -> list[Any]:
    """``value`` when it is a list, else an empty list (a raw shape the loader
    refuses is stage 0's to report, never this pass's)."""
    return value if isinstance(value, list) else []


def _string_literals(node: Any) -> set[str]:
    """Every ``str`` leaf of a JsonLogic expression (keys excluded).

    Walked with an explicit stack: a deeply nested data list must not raise ``RecursionError``
    (a validator never raises on a mapping).
    """
    # DECISION plan-2026-11-28T050702-8525c797/D-007: do NOT make this recursive again (a
    # condition's data list is not depth-checked, so the nesting is the caller's to choose).
    found: set[str] = set()
    seen: set[int] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, (dict, list)) and id(item) not in seen:
            seen.add(id(item))
            stack.extend(item.values() if isinstance(item, dict) else item)
    return found


def _condition_keys(state: dict[str, Any]) -> set[str]:
    """Every context key a state's transition conditions gate on or read.

    Contract: ``state`` is a raw state dict (already known well-typed); returns
    the union of each condition's ``requires_context_keys`` and the keys its
    ``logic`` reads (``definitions.logic_referenced_keys``). D-011 of plan
    21cd7f8e: every optional list is read with ``or []``, never ``.get(k, [])``
    (a dumped FSM holds an explicit ``None`` for an absent one).
    """
    keys: set[str] = set()
    for transition in state.get("transitions") or []:
        for condition in transition.get("conditions") or []:
            keys.update(condition.get("requires_context_keys") or [])
            keys |= logic_referenced_keys(condition.get("logic"))
    return keys


class StructureMixin:
    """The structure and key checks of ``FSMValidator`` (stages 1 to 3 of ``validate``)."""

    if TYPE_CHECKING:
        # Provided by ``FSMValidator`` and ``GraphMixin`` (declared for mypy only).
        fsm_data: dict[str, Any]
        fsm_name: Any
        states: Any
        initial_state: Any
        result: FSMValidationResult
        _definition: FSMDefinition | None

        def _get_reachable_states(self) -> set[str]: ...

        def _get_terminal_states(self) -> set[str]: ...

    def _structure_is_well_typed(self) -> bool:
        """True when the raw dicts have the container types every later stage
        reads: ``initial_state`` a str, ``states`` a dict of dicts, each
        ``transitions`` (if present) a list of dicts whose ``target_state`` is
        a str or absent/None."""
        if not isinstance(self.initial_state, str) or not isinstance(self.states, dict):
            return False
        for state in self.states.values():
            if not isinstance(state, dict):
                return False
            transitions = state.get("transitions", [])
            if not isinstance(transitions, list):
                return False
            for transition in transitions:
                if not isinstance(transition, dict):
                    return False
                if not isinstance(transition.get("target_state"), str | None):
                    return False
        return True

    def _validate_fsm_structure(self):
        """
        Validate the basic structure of the FSM.

        Checks:
        - Initial state exists and is defined
        - States dictionary exists
        - All transition target states exist
        - No orphaned states (unreachable from initial state)
        """
        # DECISION plan-2026-11-05T185903-5980fbb7/D-003: stage 0 already
        # reported what `FSMDefinition`'s own structure validator raised (its
        # first defect, with this very wording). Report a structural defect
        # ONCE: skip a message a `Schema:` error already carries, keep the
        # defects stage 0 did not reach (it raises at the first). Do NOT drop
        # this stage when stage 0 failed: field errors stop pydantic's `after`
        # validator, and the model validator stops at its first defect.
        if not self.initial_state:
            if not any(
                e.startswith("Schema: initial_state:") for e in self.result.errors
            ):
                self.result.add_error("No initial state defined")
            return
        elif self.initial_state not in self.states:
            self._add_structure_error(unknown_initial_state_message(self.initial_state))
            return

        # NOTE (F-23): there is deliberately no "No states defined" branch here.
        # It was unreachable -- an empty `self.states` can never contain a truthy
        # `self.initial_state`, so the `initial_state not in self.states` arm
        # above always fires and returns first. The message users get for an
        # empty FSM is "Initial state 'X' not found in states", which is correct.

        # Check all target states exist
        for state_id, state in self.states.items():
            transitions = state.get("transitions", [])
            for transition in transitions:
                target = transition.get("target_state", "")
                if not target:
                    self.result.add_error(
                        f"Missing target_state in transition from '{state_id}'"
                    )
                elif target not in self.states:
                    self._add_structure_error(
                        dangling_transition_message(state_id, target)
                    )

        # Check for orphaned states (not reachable from initial state)
        reachable_states = self._get_reachable_states()
        orphaned_states = set(self.states.keys()) - reachable_states
        if orphaned_states:
            self._add_structure_error(orphaned_states_message(orphaned_states))

    def _add_structure_error(self, message: str) -> None:
        """Add ``message`` as an error unless an earlier error already says it
        (stage 0 carries the model validator's text after ``Value error, ``)."""
        if not any(error.endswith(message) for error in self.result.errors):
            self.result.add_error(message)

    def _validate_terminal_states(self):
        """
        Validate terminal states (states with no outgoing transitions).

        Checks:
        - At least one terminal state exists
        - At least one terminal state is reachable from the initial state
        """
        terminal_states = self._get_terminal_states()

        # Check if at least one terminal state exists
        if not terminal_states:
            self.result.add_error(
                "No terminal states found. At least one state must have no outgoing transitions."
            )
            return

        self.result.add_info(f"Terminal states: {', '.join(terminal_states)}")

        # Check if at least one terminal state is reachable
        reachable_states = self._get_reachable_states()
        reachable_terminal_states = terminal_states.intersection(reachable_states)

        if not reachable_terminal_states:
            self.result.add_error(
                "No terminal states are reachable from the initial state."
            )
        else:
            self.result.add_info(
                f"Reachable terminal states: {', '.join(reachable_terminal_states)}"
            )

    def _validate_required_context_keys(self):
        """
        Validate that required context keys are properly handled.

        Checks:
        - Each declared required_context_key is actually named by some transition
          condition, either in its requires_context_keys or read by its logic
          (``definitions.logic_referenced_keys``: ``var`` including the first
          segment of a dotted path, ``missing``, ``missing_some``)
        - Warns (never errors) if a declared key is gated on by nothing

        F-15: this used to ask only "does ANY transition have ANY non-empty
        `requires_context_keys`?", so a state declaring `["email"]` whose only
        gate referenced `["phone"]` passed silently -- the exact defect the check
        exists to catch. The comparison is now per-key.

        WARNING tier by design: an ungated required key is a design smell, not an
        FSM the loader rejects, and promoting it would make `fsm-llm-validate`
        stricter than `API.from_file`.
        """
        for state_id, state in self.states.items():
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-011
            # Do NOT revert this (and the sibling `conditions` /
            # `requires_context_keys` reads below) to `.get(k, [])`: pydantic
            # `model_dump()` writes an explicit `None` for an absent Optional
            # list, `.get(k, [])` then returns `None`, and the validator
            # reports a dumped FSM INVALID that `API.from_file` loads fine.
            # See decisions.md D-011.
            required_keys = state.get("required_context_keys") or []

            if not required_keys:
                continue  # No required keys to validate

            # Every key name any transition condition actually gates on
            gated_keys = _condition_keys(state)

            ungated = [key for key in required_keys if key not in gated_keys]
            if not ungated:
                continue

            if not gated_keys:
                self.result.add_warning(
                    f"State '{state_id}' has required_context_keys {required_keys} "
                    "but no transitions with conditions requiring these keys"
                )
            else:
                self.result.add_warning(
                    f"State '{state_id}' declares required_context_keys {ungated} "
                    f"that no transition condition gates on "
                    f"(conditions require {sorted(gated_keys)})"
                )

    def _validate_handler_only_keys(self):
        """D-051: report a ``handler_only_keys`` entry that may protect nothing.

        A security control whose failure mode is silence: a typo in the list
        is accepted and guards no key. Never an error (the validator stays no
        stricter than ``API.from_file``). A listed key that is a
        ``classification_extractions`` field name is a WARNING (the
        classification channel is not covered by the list). A listed key no
        state references is an INFO line: that is what a key written only by
        a handler looks like, and the validator cannot see handlers.

        A condition's ``logic`` references the keys
        ``definitions.logic_referenced_keys`` returns (``var`` names including
        the first segment of a dotted path, ``missing``/``missing_some``
        names); a key that only appears as a compared literal is not read.
        It only runs for a non-empty list.
        """
        listed = self.fsm_data.get("handler_only_keys") or []
        if not listed:
            return
        referenced: set[str] = set()
        classified: set[str] = set()
        for state in self.states.values():
            referenced.update(state.get("required_context_keys") or [])
            for fe in state.get("field_extractions") or []:
                referenced.add(fe.get("field_name"))
            for ce in state.get("classification_extractions") or []:
                classified.add(ce.get("field_name"))
            referenced |= _condition_keys(state)
        referenced |= classified
        for key in listed:
            if key in classified:
                self.result.add_warning(
                    f"handler_only_keys lists '{key}', a classification_extractions "
                    "field name: classification writes are not covered by the list"
                )
            elif key not in referenced:
                # INFO, not WARNING: a key only handlers write is the list's
                # normal use (every agent FSM lists its framework keys), and
                # the validator cannot see handlers.
                self.result.add_info(
                    f"handler_only_keys lists '{key}' and no state references it "
                    "(expected for a key only a handler writes; otherwise check "
                    "for a typo)"
                )

    def _validate_kind_states(self) -> None:
        """Warn on kind-specific designs that fail silently at run time.

        The raw-dict twin of the state kinds (``state_kinds``): this validator
        never builds a ``State``, so it asks each kind's ``lint_raw`` and
        ``raw_result_key`` for the rules over the raw dicts. Two WARNINGs (never
        errors: the loader accepts both shapes):

        - Each kind's own lint (today: a completion state whose transitions do
          not cover a result kind it can produce).
        - An extraction channel (``field_extractions``,
          ``classification_extractions``, ``required_context_keys``) naming a
          handler-owned key: a kind's ``raw_result_key`` (closed to every
          channel, classification included: D-002 of plan 52ce5464), or a
          ``handler_only_keys`` entry (classification may write one, so that
          channel is reported by ``_validate_handler_only_keys`` instead). The
          pipeline never extracts such a key, so the declaration is dead.
        """
        services = RawLintServices(
            result_kinds=_RESULT_KINDS,
            default_result_key=DEFAULT_COMPLETION_RESULT_KEY,
            string_literals=_string_literals,
            default_synthesis_result_key=DEFAULT_SYNTHESIS_RESULT_KEY,
            default_call_result_key=DEFAULT_CALL_RESULT_KEY,
        )
        result_keys: dict[str, str] = {}
        for state_id, state in self.states.items():
            for spec in raw_kinds(state):
                key = spec.raw_result_key(state, services)
                if key is not None:
                    result_keys[key] = _RESULT_WRITERS.get(
                        spec.name, _COMPLETION_WRITER
                    )
                for message in spec.lint_raw(state_id, state, self.fsm_data, services):
                    self.result.add_warning(message)
        handler_only = set(self.fsm_data.get("handler_only_keys") or [])
        for state_id, state in self.states.items():
            declared = [
                ("field_extractions", fe.get("field_name"))
                for fe in state.get("field_extractions") or []
            ]
            declared += [
                ("classification_extractions", ce.get("field_name"))
                for ce in state.get("classification_extractions") or []
            ]
            declared += [
                ("required_context_keys", key)
                for key in state.get("required_context_keys") or []
            ]
            for channel, key in declared:
                owned = key in result_keys or (
                    key in handler_only and channel != "classification_extractions"
                )
                if owned:
                    writer = result_keys.get(key, _COMPLETION_WRITER)
                    self.result.add_warning(
                        f"State '{state_id}' {channel} names '{key}', a key only "
                        f"handlers or {writer} write: it is never extracted"
                    )

    def _validate_unknown_keys(self):
        """Warn (never error) on FSM-JSON keys that no model field declares.

        Walks the fixed-schema levels only -- FSM -> states -> transitions ->
        conditions, and states -> field_extractions, classification_extractions
        -> intents -- comparing each raw dict's keys against the corresponding
        Pydantic model's `model_fields`. A misspelled OPTIONAL key (e.g.
        `requred_context_keys`, `confidence_treshold`) is silently dropped by the
        loader (`extra="ignore"`), so surfacing it here as a WARNING is the only
        signal an author gets that the key had no effect. The extraction levels
        are typed models too (B4 of plan 52ce5464, a plain extension of this
        rule, WARNING only, never `extra="forbid"`).

        # DECISION plan-2026-07-21T045419-9925aa3a/D-008
        # WARNING tier ONLY -- never `add_error`. The loader accepts extra keys,
        # so promoting them to ERROR would make `fsm-llm-validate` stricter than
        # `API.from_file` and break `test_constraint_sweep_validator_agrees_with_
        # loader`. Do NOT descend into the free-form dict fields (`logic`,
        # `metadata`, `schema`, `transition_classification`, `validation_rules`):
        # their keys are DATA, not model field names, so warning on them floods
        # the report with false positives (Pre-Mortem 3). This pass reads only
        # the STRUCTURAL child fields (`transitions`, `conditions`), never the
        # data fields. See decisions.md D-008.
        """

        def _warn_unknown(node: Any, model: type[BaseModel], where: str) -> None:
            if not isinstance(node, dict):
                return
            allowed = set(model.model_fields.keys())
            for key in node:
                if key not in allowed:
                    self.result.add_warning(
                        f"Unknown key '{key}' in {where} (not a recognized "
                        f"{model.__name__} field); it will be ignored on load"
                    )

        _warn_unknown(self.fsm_data, FSMDefinition, "FSM definition")

        if not isinstance(self.states, dict):
            return
        for state_id, state in self.states.items():
            _warn_unknown(state, State, f"state '{state_id}'")
            if not isinstance(state, dict):
                continue
            for f_idx, field in enumerate(_entries(state.get("field_extractions"))):
                where = f"state '{state_id}' field_extractions #{f_idx}"
                _warn_unknown(field, FieldExtractionConfig, where)
            classified = _entries(state.get("classification_extractions"))
            for c_idx, entry in enumerate(classified):
                where = f"state '{state_id}' classification_extractions #{c_idx}"
                _warn_unknown(entry, ClassificationExtractionConfig, where)
                intents = _entries(
                    entry.get("intents") if isinstance(entry, dict) else None
                )
                for i_idx, intent in enumerate(intents):
                    _warn_unknown(intent, IntentDefinition, f"{where} intent #{i_idx}")
            transitions = state.get("transitions", [])
            if not isinstance(transitions, list):
                continue
            for t_idx, transition in enumerate(transitions):
                _warn_unknown(
                    transition, Transition, f"state '{state_id}' transition #{t_idx}"
                )
                if not isinstance(transition, dict):
                    continue
                conditions = transition.get("conditions") or []
                if not isinstance(conditions, list):
                    continue
                for c_idx, condition in enumerate(conditions):
                    _warn_unknown(
                        condition,
                        TransitionCondition,
                        f"state '{state_id}' transition #{t_idx} condition #{c_idx}",
                    )

    def _validate_schema_versions(self) -> None:
        """Warn (never error) when the declared version predates a field used.

        # DECISION plan-2026-11-12T023759-9013b0e3/D-003
        # WARNING tier ONLY -- never `add_error`. The loader accepts a
        # definition declaring "4.1" that uses a 4.2 field, so an error would
        # make this validator stricter than `API.from_file`. It reads the
        # definition stage 0 loaded (nothing when that load failed) and the
        # registry in `schema_registry`, one place for the field versions.
        """
        if self._definition is None:
            return
        for finding in newer_fields(self._definition):
            self.result.add_warning(f"Version: {finding.message}")

    def _validate_transition_classification_keys(self) -> None:
        """Warn on a ``transition_classification`` key that names no target.

        Every key but the reserved threshold key names a target state of one
        of the state's own transitions; the pipeline reads the config for the
        tied options only (``config.get(opt.target_state, {})``), so a key
        naming anything else (a typo, a state of another transition list) is
        never used. WARNING tier only (D-008 of plan 9925aa3a, see
        ``_validate_unknown_keys``): the loader accepts the shape, so an error
        would make this validator stricter than ``API.from_file``. Shape errors
        are the loader's (stage 0).
        """
        for state_id, state in self.states.items():
            config = state.get("transition_classification")
            if not isinstance(config, dict):
                continue
            targets = {t.get("target_state") for t in state.get("transitions") or []}
            for key in config:
                if (
                    key != TRANSITION_CLASSIFICATION_THRESHOLD_KEY
                    and key not in targets
                ):
                    self.result.add_warning(
                        f"State '{state_id}' transition_classification key "
                        f"'{key}' names no target of its transitions; it is "
                        "never used"
                    )
