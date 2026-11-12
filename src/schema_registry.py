"""The field-introduction registry: which definition version added each field.

A leaf core module: it imports nothing from ``fsm_llm`` (so ``definitions``
can import `SCHEMA_VERSION_DEFAULT` from it), never logs and never raises on a
loaded definition. `newer_fields` returns frozen findings; the loader and the
validator decide how to report them.

The tiers are honest but coarse. A field of the 4.0 schema is "4.0": the 2-pass
package at commit 586f412f (version default "4.0") already declares
`Transition.llm_description` and `TransitionCondition.evaluation_priority`
beside the common name/state/transition fields. The fields first seen at the
"4.1" default (commit 4fb5a8b6: `State.extraction_instructions` and
`response_instructions`) and every field shipped since (each added as "v4.1
unchanged", so one tier is all the history supports) are "4.1";
`State.synthesize` and every `SynthesizeConfig` field are "4.2". A test walks the annotations
reachable from `FSMDefinition`, so a field or nested model missing from
`FIELD_VERSIONS` fails the build.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

__all__ = [
    "FIELD_VERSIONS",
    "SCHEMA_VERSION_DEFAULT",
    "SchemaFinding",
    "newer_fields",
    "parse_version",
]

# The version an omitted `FSMDefinition.version` takes.
SCHEMA_VERSION_DEFAULT = "4.2"

# model -> {introduced version -> its field names, space separated}
_TABLE: dict[str, dict[str, str]] = {
    "FSMDefinition": {
        "4.0": "name description states initial_state version persona",
        "4.1": "handler_only_keys",
    },
    "State": {
        "4.0": "id description purpose transitions required_context_keys",
        "4.1": (
            "extraction_instructions response_instructions extraction_retries "
            "extraction_confidence_threshold transition_classification "
            "field_extractions classification_extractions context_scope completion"
        ),
        "4.2": "synthesize",
    },
    "Transition": {
        "4.0": "target_state description conditions priority llm_description",
    },
    "TransitionCondition": {
        "4.0": "description requires_context_keys logic evaluation_priority",
    },
    "FieldExtractionConfig": {
        "4.1": (
            "field_name field_type extraction_instructions context_keys "
            "validation_rules required confidence_threshold"
        ),
    },
    "ClassificationExtractionConfig": {
        "4.1": (
            "intents fallback_intent confidence_threshold field_name required "
            "model prompt_config context_keys"
        ),
    },
    "IntentDefinition": {"4.1": "name description"},
    "ContextScope": {"4.1": "read_keys write_keys"},
    "CompletionStateConfig": {
        "4.1": (
            "tools tool_choice response_format_key instructions messages_key result_key"
        ),
    },
    "SynthesizeConfig": {
        "4.2": "goal inputs return_keys on_failure result_key min_lifecycle",
    },
}

# "Model.field" -> the definition version that introduced it.
FIELD_VERSIONS: dict[str, str] = {
    f"{model}.{name}": version
    for model, tiers in _TABLE.items()
    for version, names in tiers.items()
    for name in names.split()
}

_VERSION_RE = re.compile(r"([0-9]+)\.([0-9]+)", re.ASCII)


def parse_version(text: object) -> tuple[int, int] | None:
    """Parse an exact ``N.M`` version, or return None.

    Contract:
        text: any value; only a `str` of ASCII digits, one dot, ASCII digits
            parses ("4.1" -> (4, 1)).
    Returns: `(major, minor)`, or None for anything else ("4", "4.1.1",
        "v4.1", "", " 4.1", a non-string). Never raises.
    """
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.fullmatch(text)
    return (int(match[1]), int(match[2])) if match else None


@dataclass(frozen=True)
class SchemaFinding:
    """A field a definition uses although its declared version predates it."""

    model: str
    field: str
    introduced: str
    declared: str
    state_id: str | None = None

    @property
    def message(self) -> str:
        where = f" in state '{self.state_id}'" if self.state_id else ""
        return (
            f"{self.model}.{self.field}{where} needs definition version "
            f"{self.introduced}, but the definition declares {self.declared}"
        )


def newer_fields(definition: BaseModel) -> tuple[SchemaFinding, ...]:
    """List the fields `definition` uses that its declared version predates.

    Contract:
        definition: a loaded `FSMDefinition` (any pydantic model with a string
            `version` works). Not mutated.
    Returns: findings in walk order (field declaration order, states in
        definition order), one per (model, field, state id). Empty when the
        declared version is unparseable or below 4.0 (the shape of an older
        file is unknown) or at least every field used.
    A field counts as used when the author supplied it with a value that is
    neither None nor equal to the field default. Fields of a model held by a
    newer field are not listed again: the container already says it. Never
    raises on a loaded definition and never logs.
    """
    declared_text = getattr(definition, "version", None)
    declared = parse_version(declared_text)
    if declared is None or declared < (4, 0):
        return ()
    findings: dict[tuple[str, str, str | None], SchemaFinding] = {}
    _walk(definition, str(declared_text), declared, None, findings)
    return tuple(findings.values())


def _walk(
    model: BaseModel,
    declared_text: str,
    declared: tuple[int, int],
    state_id: str | None,
    findings: dict[tuple[str, str, str | None], SchemaFinding],
) -> None:
    name = type(model).__name__
    if name == "State":
        state_id = str(getattr(model, "id", state_id))
    for field_name, field in type(model).model_fields.items():
        if field_name not in model.model_fields_set:
            continue
        value = getattr(model, field_name)
        if value is None or (
            not field.is_required()
            and value == field.get_default(call_default_factory=True)
        ):
            continue
        introduced = FIELD_VERSIONS.get(f"{name}.{field_name}")
        parsed = parse_version(introduced)
        if parsed is not None and parsed > declared:
            key = (name, field_name, state_id)
            findings.setdefault(
                key,
                SchemaFinding(
                    name, field_name, str(introduced), declared_text, state_id
                ),
            )
            continue
        for child in _models_in(value):
            _walk(child, declared_text, declared, state_id, findings)


def _models_in(value: Any) -> list[BaseModel]:
    """The pydantic models held directly by a value, a list or a dict."""
    if isinstance(value, BaseModel):
        return [value]
    if isinstance(value, (list, tuple)):
        return [m for item in value for m in _models_in(item)]
    if isinstance(value, dict):
        return [m for item in value.values() for m in _models_in(item)]
    return []
