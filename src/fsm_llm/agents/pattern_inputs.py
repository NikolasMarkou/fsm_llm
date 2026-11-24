"""Turn a pattern envelope and an agent's state into one instantiated FSM (plan a555dcfe D-015).

``PatternAgent`` and the builder wrappers (``build_react_fsm``) share this one function, so the
FSM a run uses and the FSM a builder returns cannot drift. The envelope is a template: the
agent supplies the VARIANTS (``approval`` from its HITL predicate, never from an option), the
caller's OPTIONS (a boolean option of the same name as a variant selects it) and the DERIVED
inputs, which are the agent's own and which a caller may not give.

Interface contract: ``pattern_fsm(envelope, *, tools, task_description="", approval=False,
options=None, output_schema=None, context_keys=()) -> dict``: the instantiated FSM as a fresh
dict (key order is the template's). ``tools`` is the run's ``ToolRegistry`` (``None`` for a
pattern with no tools), ``task_description`` is only the retrieval query of a semantic registry,
``approval`` selects the ``approval`` variant, ``options`` are the caller's options,
``output_schema`` the config's output model, ``context_keys`` the caller's prompt keys (names
only, ``caller_prompt_keys``). Derived, filled for the names the envelope declares: options
``context_keys``, ``schema_field_names``, ``tool_intents`` (with the ``classification`` variant)
and the text variables ``think_instructions`` and ``context_names`` (the key names, quoted, as the
field prompts print them). Raises ``AgentError`` for an option named
``approval`` or like a derived input; ``ValueError`` for a context key that is ``agent_trace`` or
internal-prefixed (as the builder always did); ``InstantiationError`` for what the envelope
refuses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.fsm_definitions._loop import (
    _schema_field_names,
    loop_field_context_keys,
)
from fsm_llm.agents.prompts import build_think_extraction_instructions
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.core.definitions.executable_instance import instantiate_body
from fsm_llm.tools import ToolRegistry

__all__ = [
    "APPROVAL_VARIANT",
    "DERIVED_OPTIONS",
    "pattern_fsm",
    "refuse_reserved_options",
]

#: The one variant the agent selects from its HITL predicate; no option may carry its name.
APPROVAL_VARIANT = "approval"

#: The options the agent fills itself (a caller cannot give them).
DERIVED_OPTIONS = frozenset({"context_keys", "schema_field_names", "tool_intents"})

_THINK_TEXT = "think_instructions"
_KEYS_TEXT = "context_names"


def refuse_reserved_options(
    envelope: PatternEnvelope, given: Mapping[str, Any]
) -> None:
    """Raise ``AgentError`` for an option named ``approval`` (declared or given) or like a derived input."""
    declared = envelope.body.options_schema.properties
    if APPROVAL_VARIANT in declared or APPROVAL_VARIANT in given:
        raise AgentError(
            f"an option named {APPROVAL_VARIANT!r} is refused: the approval variant is "
            "selected by the agent's HITL predicate, never by an option"
        )
    if owned := sorted(DERIVED_OPTIONS & set(given)):
        raise AgentError(f"options {owned} are filled by the agent, not given")


def pattern_fsm(
    envelope: PatternEnvelope,
    *,
    tools: ToolRegistry | None,
    task_description: str = "",
    approval: bool = False,
    options: Mapping[str, Any] | None = None,
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """The instantiated FSM of *envelope* for one run (see the module docstring)."""
    given = dict(options or {})
    refuse_reserved_options(envelope, given)
    body = envelope.body
    declared = body.options_schema.properties
    variants = [APPROVAL_VARIANT] if approval else []
    variants += [
        name
        for name in body.variants
        if name != APPROVAL_VARIANT and name in declared and given.get(name) is True
    ]
    variables: dict[str, str] = {}
    if _THINK_TEXT in envelope.variables:
        # first, as the builder did: an unusable registry fails before a bad key does
        variables[_THINK_TEXT] = build_think_extraction_instructions(
            tools,  # type: ignore[arg-type]
            task_description=task_description,
        )
    values = dict(given)
    if "context_keys" in declared:
        keys = loop_field_context_keys((ContextKeys.AGENT_FEEDBACK, *context_keys))
        values["context_keys"] = keys
        if (
            _KEYS_TEXT in envelope.variables
        ):  # the field prompts name the keys they read
            variables[_KEYS_TEXT] = ", ".join(repr(key) for key in keys)
    if "schema_field_names" in declared:
        values["schema_field_names"] = _schema_field_names(output_schema)
    if "tool_intents" in declared and "classification" in variants:
        if tools is None:
            raise AgentError("the classification variant needs the run's tools")
        values["tool_intents"] = tools.to_classification_schema()["intents"]
    return instantiate_body(envelope, values, variants=variants, variables=variables)
