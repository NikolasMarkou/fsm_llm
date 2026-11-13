"""Transition models: options, conditions, transitions and the JsonLogic walk helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, model_validator

from fsm_llm.constants import (
    ALLOWED_JSONLOGIC_OPERATIONS,
    JSONLOGIC_RAW_ARGUMENT_OPERATIONS,
    MAX_JSONLOGIC_DEPTH,
)

from .limits import ASCII_IDENTIFIER_PATTERN, MAX_PRIORITY


class TransitionOption(BaseModel):
    """
    A possible transition option for LLM evaluation.

    Contains minimal information needed for transition decisions
    without exposing internal FSM structure.
    """

    target_state: str = Field(
        ...,
        description="Target state identifier",
        min_length=1,
        max_length=100,
        pattern=ASCII_IDENTIFIER_PATTERN,
    )

    description: str = Field(
        ...,
        description="Human-readable description of when this transition applies",
        min_length=1,
        max_length=500,
    )

    priority: int = Field(
        default=100,
        description="Priority for this transition (lower = higher priority)",
        ge=0,
        le=MAX_PRIORITY,
    )


def _walk_logic_nodes(node: Any, _depth: int = 0) -> Iterator[tuple[str, Any]]:
    """Yield every `(operator, arguments)` pair in a JsonLogic `node`, visiting
    exactly the positions `evaluate_logic` evaluates as logic.

    A JsonLogic object is a dict whose single key is an operator mapping to its
    argument value(s): a list/tuple of arguments, or one bare argument. Each
    argument is evaluated as logic one level deeper. A list or tuple met AS an
    argument is DATA (`evaluate_logic` returns it unevaluated), so its elements,
    dicts included, are never walked; nor are the raw arguments of the
    `JSONLOGIC_RAW_ARGUMENT_OPERATIONS` (`var`, `missing`, `missing_some`).

    # DECISION plan-2026-09-21T203800-8a03483a/D-014
    # This supersedes D-007's "recurse into list/tuple ELEMENTS" clause below
    # (B9): walking data-list elements rejected `{"in":[{"var":"x"},[{"foo":1}]]}`,
    # which evaluates fine. Do NOT recurse into data lists or raw-argument
    # operators again, and do NOT drop the depth or single-key checks (B5):
    # without them over-deep or multi-key logic loads clean and is silently
    # False at every turn (`evaluate_logic` raises, the evaluator swallows it).
    # The depth bound is `evaluate_logic`'s own (every evaluated position at
    # depth > MAX_JSONLOGIC_DEPTH raises there), so load and runtime agree.
    # D-007's empty-dict rejection is kept. See decisions.md D-014.

    # DECISION plan-2026-07-21T045419-9925aa3a/D-007
    # An EMPTY dict is malformed JsonLogic (an operator object with zero operator
    # keys), so this RAISES ValueError — that is how H6 rejects both a top-level
    # `logic: {}` and a nested `{"and": [{}]}`. Do NOT "soften" this to silently
    # skip empty dicts: `logic: {}` previously meant "always fail" while
    # `logic: null` means "always pass", a silent divergence this guard closes.
    # Do NOT recurse into a dict's KEYS (they are operators, already yielded) —
    # only into its VALUES and into list/tuple ELEMENTS [superseded by D-014:
    # only an operator's ARGUMENT list is walked, never a data list]. Primitives,
    # including the literal strings in a data-list arg such as
    # `{"in":[{"var":"x"},["a","b"]]}`, are DATA not operators and are ignored
    # (false-reject trap). This
    # helper lives here, NOT in expressions.py, because expressions.py imports
    # FROM definitions.py — the reverse edge would be a circular import.
    # See decisions.md D-007.

    Contract:
        node: a JsonLogic position that `evaluate_logic` evaluates (the root
            logic dict, or one operator argument).
        _depth: the `evaluate_logic` depth of `node` (0 for the root).
    Yields: each `(operator key, raw argument value)` pair encountered, in
        traversal order. Callers: `TransitionCondition._validate_logic` (load-time operator
        allow-list) and `logic_referenced_keys` (validator gating checks).
    Raises: ValueError if an evaluated position is deeper than
        MAX_JSONLOGIC_DEPTH, or an operator dict has zero keys (D-007) or more
        than one key.
    """
    if _depth > MAX_JSONLOGIC_DEPTH:
        raise ValueError(
            f"JsonLogic nesting exceeds the maximum depth ({MAX_JSONLOGIC_DEPTH})"
        )
    if not isinstance(node, dict):
        # Primitives and lists at an evaluated position are data.
        return
    if not node:
        raise ValueError("Empty JsonLogic object '{}' is not a valid condition")
    if len(node) != 1:
        raise ValueError(
            f"JsonLogic object has keys {sorted(map(str, node))}; each operator "
            "object must have exactly one key"
        )
    operator, arguments = next(iter(node.items()))
    yield operator, arguments
    if operator in JSONLOGIC_RAW_ARGUMENT_OPERATIONS:
        return
    if not isinstance(arguments, (list, tuple)):
        arguments = [arguments]
    for argument in arguments:
        yield from _walk_logic_nodes(argument, _depth + 1)


def _logic_read_names(logic: Any) -> Iterator[Any]:
    """Yield the raw name value of every context read in `logic`: the first
    argument of `var` (None when it has none), every name of `missing` and
    `missing_some`, and the argument of the one-argument `has_context`.

    The one place that says which positions of a JsonLogic tree name a context
    key; `logic_referenced_keys` and `logic_has_unresolvable_reads` read it, so
    the two cannot disagree. A yielded value is whatever the definition holds
    (a string, a computed `{"cat": ...}` dict, a number); the callers judge it.
    Raises ValueError exactly when `_walk_logic_nodes` does.
    """
    for operator, arguments in _walk_logic_nodes(logic):
        args = list(arguments) if isinstance(arguments, (list, tuple)) else [arguments]
        if operator == "var":
            yield args[0] if args else None
        elif operator == "missing":
            for arg in args:
                yield from arg if isinstance(arg, (list, tuple)) else [arg]
        elif operator == "missing_some" and len(args) == 2:
            names = args[1]
            yield from names if isinstance(names, (list, tuple)) else [names]
        elif operator == "has_context" and len(args) == 1:
            yield args[0]


def logic_referenced_keys(logic: Any) -> set[str]:
    """Return the top-level context keys a JsonLogic expression reads.

    A key counts when it is named literally by `var` (a dotted path counts as
    its first segment: `profile.email` reads `profile`), by `missing` /
    `missing_some`, or by the one-argument `has_context` shorthand. A value
    that only appears as a compared literal is not a reference, and a computed
    name (`{"var": {"cat": ...}}`) cannot be resolved statically and is skipped
    (`logic_has_unresolvable_reads` reports it).

    Contract:
        logic: a `TransitionCondition.logic` value (dict, or None/falsy).
    Returns: the set of referenced first-segment key names (empty for None).
    Raises: ValueError exactly when `_walk_logic_nodes` does (malformed logic,
        which the loader already rejects).
    """
    if not logic:
        return set()
    return {
        name.split(".", 1)[0]
        for name in _logic_read_names(logic)
        if isinstance(name, str) and name
    }


def logic_has_unresolvable_reads(logic: Any) -> bool:
    """Say whether `logic` reads a context key whose name cannot be resolved
    statically: a computed name (`{"var": {"cat": ...}}`), an empty name, a
    non-string name, a name with a leading dot (`.a`), or a `var` with no
    argument. Fails closed: such a read may touch any key, so an analysis that
    needs the exact key set must treat the whole expression as unknown.

    Only the positions `logic_referenced_keys` reads are judged: the raw
    arguments of `var`, `missing` and `missing_some` (a default value, say) and
    data lists are not logic, so a dict inside them never counts.

    Contract:
        logic: a `TransitionCondition.logic` value (dict, or None/falsy).
    Returns: True when any read name is unresolvable; False for None/falsy.
    Raises: ValueError exactly when `_walk_logic_nodes` does.
    """
    if not logic:
        return False
    return any(
        not isinstance(name, str) or not name or name.startswith(".")
        for name in _logic_read_names(logic)
    )


class TransitionCondition(BaseModel):
    """
    Enhanced transition condition with evaluation capabilities.

    Supports both simple key-based and complex JsonLogic conditions.
    """

    description: str = Field(
        ...,
        description="Human-readable description of this condition",
        min_length=1,
        max_length=500,
    )

    requires_context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys required for evaluation; a key that is absent, None "
            "or an empty string is missing and fails the condition"
        ),
    )

    logic: dict[str, Any] | None = Field(
        default=None, description="JsonLogic expression for complex evaluation"
    )

    evaluation_priority: int = Field(
        default=100,
        description="Priority for condition evaluation (lower = earlier)",
        ge=0,
        le=MAX_PRIORITY,
    )

    @model_validator(mode="after")
    def _validate_logic(self) -> TransitionCondition:
        """Reject an empty/nested-empty (H6), non-allow-listed-operator (H8),
        multi-key or over-deep (B5, D-014) `logic` dict at LOAD time, so `API.from_file` and `fsm-llm-validate`
        both fail on a malformed condition instead of blowing up mid-conversation.

        # DECISION plan-2026-07-21T045419-9925aa3a/D-007
        # Raise a plain ValueError (surfaces as pydantic error type `value_error`,
        # already promoted to ERROR by validator.py's allow-list) so the
        # validator/loader agreement IFF is preserved. Do NOT switch this to a
        # different exception type or add `extra="forbid"` — either would break
        # `test_constraint_sweep_validator_agrees_with_loader`. `logic is None`
        # (absent) stays valid: it means "pass if required keys present".
        # See decisions.md D-007.
        """
        if self.logic is None:
            return self
        unknown = sorted(
            {
                op
                for op, _arguments in _walk_logic_nodes(self.logic)
                if op not in ALLOWED_JSONLOGIC_OPERATIONS
            }
        )
        if unknown:
            raise ValueError(
                f"Unsupported JsonLogic operator(s): {unknown}. "
                f"Allowed: {sorted(ALLOWED_JSONLOGIC_OPERATIONS)}"
            )
        return self


class Transition(BaseModel):
    """
    Enhanced transition definition with evaluation metadata.

    Supports priority-based and condition-based transition logic.
    """

    target_state: str = Field(
        ...,
        description="Target state identifier",
        min_length=1,
        max_length=100,
        pattern=ASCII_IDENTIFIER_PATTERN,
    )

    description: str = Field(
        ...,
        description="When this transition should occur",
        min_length=1,
        max_length=500,
    )

    conditions: list[TransitionCondition] | None = Field(
        default=None, description="Conditions that must be satisfied"
    )

    priority: int = Field(
        default=100,
        description="Priority for transition selection",
        ge=0,
        le=MAX_PRIORITY,
    )

    llm_description: str | None = Field(
        None,
        description="Description for LLM when choosing between transitions",
        max_length=300,
    )
