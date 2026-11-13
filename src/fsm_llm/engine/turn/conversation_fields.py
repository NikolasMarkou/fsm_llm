"""Field-based extraction of the conversation kind (``ConversationFieldsMixin``): the
field configs of a state, the per-field extraction calls, the validation and the type
coercers.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from fsm_llm.context import record_cut_off, was_cut_off
from fsm_llm.core.coerce import coerce_float, coerce_int, parse_bool_word
from fsm_llm.core.definitions import (
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMInstance,
    State,
)
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.logging import logger

from .surface import _PipelineSurface


# --- Type coercion dispatch for field extraction validation ---
# DECISION plan-2026-09-19T175721-21cd7f8e/D-002
# `_coerce_str`/`_coerce_bool` DO raise `TypeError` on a dict or list. This supersedes
# the last clause of plan-2026-07-18T051819-80b0bd4d/D-018 (below), which said they
# "deliberately do NOT raise". Do NOT restore total `str(v)`/`bool(v)` on containers:
# live (LV-01) qwen3.5:9b returned `{"blue": "blue"}` for a str field, `str()` turned
# it into the "valid" value `"{'blue': 'blue'}"`, and `bool({...})` is True for any
# non-empty object, so junk was stored and drove gated transitions. D-018's premise
# ("coercion failed is not a reachable outcome") held for scalars only. The raise
# reuses the already-wired `except (ValueError, TypeError, json.JSONDecodeError)` in
# `_validate_field_extraction` (no sentinel, no new exception type). See decisions.md D-002.
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        # DECISION plan-2026-11-05T134105-5ed71cf2/D-008
        # Closed yes/no vocabulary via the shared `parse_bool_word`; an unknown
        # string RAISES (same except tuple as D-002) instead of becoming a valid
        # `False`. Do NOT restore `v.lower() in (...)` with an else-False.
        parsed = parse_bool_word(v)
        if parsed is None:
            raise ValueError(f"expected a yes/no word, got {v!r}")
        return parsed
    if isinstance(v, (dict, list)):
        raise TypeError(f"expected bool, got {type(v).__name__}")
    return bool(v)


def _coerce_str(v: Any) -> str:
    if isinstance(v, (dict, list)):
        raise TypeError(f"expected str, got {type(v).__name__}")
    return v if isinstance(v, str) else str(v)


# DECISION plan-2026-07-18T051819-80b0bd4d/D-018 [STALE]
# The trailing `raise TypeError` in each of the two coercers below is LOAD-BEARING.
# Do NOT "restore" a `return v` passthrough for wrong-typed input. Both functions used
# to end in a bare `return v`, so an int or a dict handed to a `field_type="list"` field
# sailed through unconverted, `_validate_field_extraction`'s
# `except (ValueError, TypeError, json.JSONDecodeError)` (see the sole call site, further
# down this file) never fired, and the field was recorded `is_valid=True` with a
# wrong-typed value written straight into FSM context. Raising here reuses that
# ALREADY-WIRED error protocol — do not invent a new exception type, a sentinel return,
# or a second validation path. This makes `list`/`dict` behave like their 4 siblings.
# [SUPERSEDED by plan-2026-09-19T175721-21cd7f8e/D-002, above, for dict/list input:]
# `_coerce_str`/`_coerce_bool` deliberately do NOT raise: `str()`/`bool()` are TOTAL, so
# "coercion failed" is not a reachable outcome for them, not a missing guard.
# `None` never reaches any coercer (the call site returns early on a None value).
# See decisions.md D-018.
def _coerce_list(v: Any) -> Any:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise TypeError("not a list")
        return parsed
    raise TypeError(f"expected list, got {type(v).__name__}")


def _coerce_dict(v: Any) -> Any:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, dict):
            raise TypeError("not a dict")
        return parsed
    raise TypeError(f"expected dict, got {type(v).__name__}")


def _number_for_range(value: Any) -> Any:
    """``value`` as what a ``min_value``/``max_value`` rule compares.

    Interface contract: ``float(value)`` as before (``TypeError`` or
    ``ValueError`` for a value that is no number, which the range checks read
    as "not numeric, skip"), except an integer too large for a float
    (``10**400``, whose ``float()`` raises ``OverflowError``) is returned as
    it is. Python compares an ``int`` with a ``float`` exactly and without
    overflow, so a huge value is simply above a max or below a min, and one
    inside an enormous bound still passes. Never raises ``OverflowError``.
    """
    try:
        return float(value)
    except OverflowError:
        return value


_TYPE_COERCERS: dict[str, Callable[[Any], Any]] = {
    "int": coerce_int,
    "float": coerce_float,
    "bool": _coerce_bool,
    "str": _coerce_str,
    "list": _coerce_list,
    "dict": _coerce_dict,
    # "any" — no coercion, not in dispatch dict
}


class ConversationFieldsMixin(_PipelineSurface):
    """Per-field extraction of the conversation kind."""

    @staticmethod
    def _build_field_configs_from_state(state: State) -> list[FieldExtractionConfig]:
        """Convert a state's key-based fields to a FieldExtractionConfig list.

        Translates ``required_context_keys`` + ``extraction_instructions``
        into per-field configs so the pipeline can use the unified
        ``extract_field`` primitive for all extraction.  Explicit
        ``field_extractions`` on the state are appended after the
        auto-generated ones. A completion state gets none: its Pass 1 is its
        completion call, and the keys its transitions read (its own
        ``result_key`` among them) are never minted into extractions. A
        synthesize state gets none either: a runner writes its result key.
        """
        # The minting rule (which keys, in which order, minus classification-owned
        # keys, explicit field_extractions winning) lives in ONE place:
        # `KindSpec.minted_keys` in src/fsm_llm/core/state_kinds.py, which also carries the
        # D-017 (item 7) anchor. The flow analyses read the same hook. Do NOT
        # re-spell the rule here.
        names = kind_of(state).minted_keys(state)
        if not names:
            return []  # a kind that mints nothing gets no config, explicit or not
        explicit = list(state.field_extractions or [])
        explicit_names = {fc.field_name for fc in explicit}
        instructions = (
            state.extraction_instructions
            or "Extract the value of this field from the user's input."
        )
        # One config per auto-minted key; an explicit config of the same name
        # overrides it (the explicit ones are appended after, user-defined, and
        # take priority).
        configs: list[FieldExtractionConfig] = [
            FieldExtractionConfig(
                field_name=key,
                field_type="any",
                extraction_instructions=f"Extract the '{key}' field. {instructions}",
                context_keys=None,  # all context
                required=True,
                confidence_threshold=state.extraction_confidence_threshold,
            )
            for key in names
            if key not in explicit_names
        ]
        configs.extend(explicit)

        return configs

    def _execute_field_extractions(
        self,
        instance: FSMInstance,
        user_message: str | None,
        field_configs: list[FieldExtractionConfig],
        conversation_id: str,
        memo: dict[tuple[str, str], FieldExtractionResponse] | None = None,
    ) -> list[FieldExtractionResponse]:
        """Execute targeted field extractions for a list of configs.

        Runs one LLM call per field.  Each config specifies its own
        instructions, dynamic context selection, and validation rules.

        Previously extracted values are added to the dynamic context
        for subsequent extractions, enabling dependent field extraction
        (e.g., tool_input can see that tool_name was already extracted).

        ``user_message`` of ``None`` (no user message) gives the context-only
        prompt and stays ``None`` on the request, never ``""`` (D-044).

        ``memo`` (D-035, ``None`` = disabled) maps ``(field_name, built prompt
        + "|" + message)`` to a previous NULL result; it is only ever handed an
        Ollama interface's dict by ``_execute_data_extraction``.
        """
        log = logger.bind(conversation_id=conversation_id)
        results: list[FieldExtractionResponse] = []
        # Accumulate extracted values so later fields can see earlier ones
        extracted_so_far: dict[str, Any] = {}

        for field_config in field_configs:
            log.debug(
                f"Extracting field '{field_config.field_name}' "
                f"(type={field_config.field_type})"
            )

            # Build dynamic context from config.context_keys
            if field_config.context_keys is not None:
                dynamic_context = {
                    k: v
                    for k, v in instance.context.data.items()
                    if k in field_config.context_keys
                }
            else:
                # Apply state-level context_scope as default filter
                current_state = self.get_state(instance, conversation_id)
                dynamic_context = self._apply_context_scope(
                    instance.context.get_user_visible_data(),
                    current_state,
                    conversation_id,
                )

            # Include previously extracted fields so the LLM can use them
            if extracted_so_far:
                dynamic_context.update(extracted_so_far)

            # Build prompt
            system_prompt = (
                self.field_extraction_prompt_builder.build_field_extraction_prompt(
                    instance=instance,
                    field_config=field_config,
                    user_message=user_message,
                    dynamic_context=dynamic_context,
                )
            )

            # Build request. None (no user message) stays None on the request:
            # the LLM layer fills the provider turn (07ad3f8c/D-044).
            request = FieldExtractionRequest.checked(
                prompt_context=dynamic_context,
                system_prompt=system_prompt,
                user_message=user_message,
                field_name=field_config.field_name,
                field_type=field_config.field_type,
                context=dynamic_context,
                validation_rules=field_config.validation_rules,
            )

            # Call LLM (or reuse an identical null, D-035)
            memo_key = (
                field_config.field_name,
                f"{system_prompt}|{user_message or ''}",
            )
            try:
                if memo is not None and memo_key in memo:
                    response = memo[memo_key]
                else:
                    response = self.llm_interface.extract_field(request)
                    if memo is not None and response.value is None:
                        memo[memo_key] = response
            except Exception as e:
                log.warning(
                    f"Field extraction failed for '{field_config.field_name}': {e}"
                )
                response = FieldExtractionResponse(
                    field_name=field_config.field_name,
                    value=None,
                    confidence=0.0,
                    is_valid=False,
                    validation_error=f"LLM call failed: {e}",
                )

            # Validate and coerce
            response = self._validate_field_extraction(response, field_config)
            if was_cut_off(response):
                record_cut_off(instance, field=field_config.field_name)

            log.debug(
                f"Field '{field_config.field_name}': "
                f"value={response.value!r}, confidence={response.confidence:.2f}, "
                f"valid={response.is_valid}"
                + (
                    f", rejected: {response.validation_error}"
                    if response.validation_error
                    else ""
                )
            )
            results.append(response)

            # Feed successful extractions into context for subsequent fields
            if response.is_valid and response.value is not None:
                extracted_so_far[field_config.field_name] = response.value

        return results

    @staticmethod
    def _validate_field_extraction(
        response: FieldExtractionResponse,
        config: FieldExtractionConfig,
    ) -> FieldExtractionResponse:
        """Validate and type-coerce a field extraction response."""
        # Skip validation if already failed
        if not response.is_valid or response.value is None:
            return response

        def rejected(error: str, **changes: Any) -> FieldExtractionResponse:
            """``response`` marked invalid with ``error`` (and ``changes``)."""
            return response.model_copy(
                update={"is_valid": False, "validation_error": error, **changes}
            )

        # Reject values that are obviously the field name echoed back —
        # small models sometimes confuse the JSON template keys with values.
        if isinstance(response.value, str) and response.value.strip().lower() in (
            config.field_name.lower(),
            "field_name",
            "value",
        ):
            return rejected(
                "Extracted value matches field name (model confusion)",
                value=None,
                confidence=0.0,
                reasoning="Model echoed field name instead of extracting a value",
            )

        # A self-reported confidence of exactly 0.0 means the model could not
        # ground the value (LV-01: dict-wrapped junk came back with low/zero
        # confidence); treat it as "not extracted" regardless of the configured
        # threshold (which defaults can leave at 0.0). decisions.md D-002.
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-016
        # Do NOT make this threshold-driven (`confidence <= threshold`): with
        # the default threshold 0.0 it is the same rule, and an author-set
        # threshold above 0 already rejects 0.0 below. Do NOT delete it either:
        # it is what rejects LV-01 dict junk that comes back at confidence 0.
        # Known cliff: a correct value the model scored exactly 0 is dropped
        # while 0.05 is kept. See decisions.md D-016.
        if response.confidence == 0.0:
            return rejected("Model reported zero confidence in the value")

        # Confidence threshold check
        if (
            config.confidence_threshold > 0.0
            and response.confidence < config.confidence_threshold
        ):
            return rejected(
                f"Confidence {response.confidence:.2f} below threshold "
                f"{config.confidence_threshold:.2f}"
            )

        # Type coercion via dispatch
        value = response.value
        try:
            coercer = _TYPE_COERCERS.get(config.field_type)
            if coercer is not None:
                value = coercer(value)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return rejected(f"Type coercion to {config.field_type} failed: {e}")

        # Validation rules. The pattern is compiled at load
        # (`FieldExtractionConfig._check_validation_rule_types`), so a bad regex
        # never reaches here.
        rules = config.validation_rules or {}
        if "allowed_values" in rules and value not in rules["allowed_values"]:
            return rejected(
                f"Value {value!r} not in allowed values: {rules['allowed_values']}",
                value=value,
            )

        if isinstance(value, str):
            if "min_length" in rules and len(value) < rules["min_length"]:
                return rejected(
                    f"Value length {len(value)} below minimum {rules['min_length']}",
                    value=value,
                )
            if "max_length" in rules and len(value) > rules["max_length"]:
                return rejected(
                    f"Value length {len(value)} exceeds maximum {rules['max_length']}",
                    value=value,
                )

        # Non-numeric values skip the numeric range checks.
        if "min_value" in rules:
            try:
                if _number_for_range(value) < _number_for_range(rules["min_value"]):
                    return rejected(
                        f"Value {value!r} is below minimum {rules['min_value']}",
                        value=value,
                    )
            except (TypeError, ValueError):
                pass

        if "max_value" in rules:
            try:
                if _number_for_range(value) > _number_for_range(rules["max_value"]):
                    return rejected(
                        f"Value {value!r} exceeds maximum {rules['max_value']}",
                        value=value,
                    )
            except (TypeError, ValueError):
                pass

        if (
            "pattern" in rules
            and isinstance(value, str)
            and not re.match(rules["pattern"], value)
        ):
            return rejected(
                f"Value does not match pattern: {rules['pattern']}", value=value
            )

        # All checks passed — return with coerced value
        return response.model_copy(update={"value": value, "validation_error": None})
