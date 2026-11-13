"""Per-field extraction configuration and the ``typed_field_extraction`` helper."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, Field, model_validator

from fsm_llm.constants import (
    EXTRACTION_ENVELOPE_KEYS,
    FIELD_PROMPT_CONTEXT_LABEL,
    has_internal_prefix,
)

#: The field-type vocabulary of a per-field extraction (config and request).
FieldType = Literal["str", "int", "float", "bool", "list", "dict", "any"]


def _is_finite_number(value: object) -> bool:
    """True for an int/float (not a bool) or a string that ``float()`` reads,
    and whose ``float()`` is finite: what ``min_value``/``max_value`` accept."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, OverflowError):
        return False


class FieldExtractionConfig(BaseModel):
    """Configuration for targeted extraction of a single field.

    Declared on a :class:`State` via ``field_extractions`` to run focused
    extraction after the bulk ``extract_data`` pass completes.
    """

    field_name: str = Field(
        ...,
        description="Context key to extract into",
        min_length=1,
        max_length=100,
    )

    field_type: FieldType = Field(
        default="str",
        description="Expected type: str, int, float, bool, list, dict, any",
    )

    extraction_instructions: str = Field(
        ...,
        description="Focused instructions for extracting this specific field",
        min_length=1,
        max_length=5000,
    )

    context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys to include as dynamic context in the extraction prompt. "
            "If None, all user-visible context is passed."
        ),
    )

    validation_rules: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Validation rules: allowed_values, min_length, max_length, "
            "min_value, max_value, pattern (regex)"
        ),
    )

    required: bool = Field(
        default=True,
        description="Whether extraction failure should be treated as an error",
    )

    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for the extraction to be accepted",
    )

    _ALLOWED_VALIDATION_RULE_KEYS: ClassVar[set[str]] = {
        "allowed_values",
        "min_length",
        "max_length",
        "min_value",
        "max_value",
        "pattern",
    }

    @model_validator(mode="after")
    def validate_validation_rule_keys(self) -> FieldExtractionConfig:
        """Reject unknown keys in validation_rules to catch typos early."""
        if self.validation_rules:
            unknown = set(self.validation_rules) - self._ALLOWED_VALIDATION_RULE_KEYS
            if unknown:
                raise ValueError(
                    f"Unknown validation_rules keys: {sorted(unknown)}. "
                    f"Allowed: {sorted(self._ALLOWED_VALIDATION_RULE_KEYS)}"
                )
            self._check_validation_rule_types(self.validation_rules)
        return self

    @staticmethod
    def _check_validation_rule_types(rules: dict[str, Any]) -> None:
        """Type-check the rules the extractor compares against at runtime.

        Raises ``ValueError`` naming the offending key, so a bad rule fails at
        load instead of as a ``TypeError`` on the first extraction turn (D-023).
        ``min_value``/``max_value`` stay ``float()``-coerced at use, so they
        must be a finite number or a string ``float()`` reads as one (not a
        bool, ``None`` or a container): the extractor skips a numeric check
        when the VALUE is not a number, which is right, but it would also skip
        it for a non-numeric RULE and silently disable the check.
        """
        for key in ("min_value", "max_value"):
            if key in rules and not _is_finite_number(rules[key]):
                raise ValueError(
                    f"validation_rules {key!r} must be a finite number, "
                    f"got {rules[key]!r}"
                )
        for key in ("min_length", "max_length"):
            v = rules.get(key)
            if key in rules and (not isinstance(v, int) or isinstance(v, bool)):
                raise ValueError(f"validation_rules {key!r} must be an int, got {v!r}")
        allowed = rules.get("allowed_values")
        if "allowed_values" in rules and not isinstance(allowed, (list, tuple, set)):
            raise ValueError(
                f"validation_rules 'allowed_values' must be a list, got {allowed!r}"
            )
        if "pattern" in rules:
            try:
                re.compile(rules["pattern"])
            except (re.error, TypeError) as e:
                raise ValueError(
                    f"validation_rules 'pattern' must be a valid regex string: {e}"
                ) from e


def checked_key_names(keys: object, *, argument: str) -> tuple[str, ...]:
    """The names in ``keys``, checked to be a collection of ``str`` names.

    Contract (callers: :func:`typed_field_extraction`, ``handlers``
    ``clear_keys_delta``/``clear_keys_on_entry``, ``ContextCompactor``, ``API`` bounded runs'
    ``seconds_exempt_states``): ``keys`` must be an iterable of ``str`` that
    is not itself a ``str`` or ``bytes`` (whose characters would be taken as
    names). Returns the names in order, duplicates dropped (``()`` for an
    empty iterable; emptiness is the caller's rule). Raises ``ValueError``
    naming ``argument`` for ``None``, a ``str``/``bytes``, a non-iterable, or
    a non-``str`` member.
    """
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Iterable):
        raise ValueError(
            f"{argument} must be a collection of names (str), not {keys!r}"
        )
    names = tuple(keys)
    if not all(isinstance(name, str) for name in names):
        raise ValueError(f"{argument} must hold names (str), got {names!r}")
    return tuple(dict.fromkeys(names))


#: Field types :func:`typed_field_extraction` declares: ``str``, ``float``,
#: ``list``, ``bool``, or ``any`` for a whole generated artifact (core coerces
#: and rejects mismatches). A subset of ``FieldExtractionConfig.field_type``.
TypedFieldType = Literal["str", "float", "list", "bool", "any"]


def typed_field_extraction(
    field_name: str,
    field_type: TypedFieldType,
    instructions: str,
    *,
    context_keys: Sequence[str],
    required: bool = True,
) -> dict[str, Any]:
    """Build one typed ``field_extractions`` entry with a narrowed prompt context.

    Contract (callers: the agents FSM builders, the reasoning FSMs):
        - Returns a raw dict for ``State(field_extractions=[...])`` that
          :class:`FieldExtractionConfig` accepts: ``field_name`` declared as
          ``field_type``, ``context_keys`` as given (order kept, duplicates
          dropped), ``required`` as given (a null required field costs one
          retry call), and instructions that tell the model to read the value
          from the task and the listed keys in the per-field prompt's
          ``FIELD_PROMPT_CONTEXT_LABEL`` section, followed by
          ``instructions``.
        - ``context_keys`` must be non-empty: ``None`` (all visible context)
          is the unnarrowed prompt this builder exists to avoid (D-017 of plan
          06a5ec0a). An internal-prefixed key is refused (it names framework
          state, never a value the model should read).
        - Raises ``ValueError`` for a ``field_type`` outside
          :data:`TypedFieldType`, a ``field_name`` in
          ``constants.EXTRACTION_ENVELOPE_KEYS`` (D-035 of plan 06a5ec0a), a
          ``context_keys`` that is ``None``, a bare ``str`` or not a
          collection of ``str`` (:func:`checked_key_names`), an empty
          ``context_keys``, or an internal-prefixed context key.

    Pair it with an empty state-level ``extraction_instructions`` (D-009 of
    plan 06a5ec0a): core then runs only these per-field calls, each on its own
    narrowed context, and makes no bulk call for the state.
    """
    if field_type not in get_args(TypedFieldType):
        raise ValueError(f"unsupported typed field type: {field_type!r}")
    if field_name in EXTRACTION_ENVELOPE_KEYS:
        raise ValueError(
            f"typed field name {field_name!r} is an extraction envelope key; "
            "pick another name"
        )
    keys = list(checked_key_names(context_keys, argument="context_keys"))
    if not keys:
        raise ValueError("typed_field_extraction needs at least one context key")
    internal = [key for key in keys if has_internal_prefix(key)]
    if internal:
        raise ValueError(f"context keys not allowed in a field prompt: {internal}")
    return {
        "field_name": field_name,
        "field_type": field_type,
        "extraction_instructions": (
            f"Extract the '{field_name}' field ({field_type}) from the task and "
            f"the {', '.join(repr(k) for k in keys)} values in the "
            f"'{FIELD_PROMPT_CONTEXT_LABEL}' context. {instructions}"
        ),
        "context_keys": keys,
        "required": required,
    }
