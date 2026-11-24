"""Turn a pattern envelope and an agent's state into one instantiated FSM (plan a555dcfe D-015).

``PatternAgent`` and the builder wrappers (``build_react_fsm``) share this one function, so the
FSM a run uses and the FSM a builder returns cannot drift. The envelope is a template: the
agent supplies the VARIANTS (``approval`` from its HITL predicate, never from an option), the
caller's OPTIONS (a boolean option of the same name as a variant selects it) and the DERIVED
inputs, which are the agent's own and which a caller may not give.

Interface contract: ``pattern_fsm(envelope, *, tools, task_description="", approval=False,
options=None, output_schema=None, context_keys=(), variables=None) -> dict``: the instantiated FSM as a fresh
dict (key order is the template's). The variant ``without_tools`` is derived too: selected, when the file declares
it, for a run whose registry is ``None`` or empty (step 15). ``tools`` is the run's ``ToolRegistry`` (``None`` for a
pattern with no tools), ``task_description`` is only the retrieval query of a semantic registry,
``approval`` selects the ``approval`` variant, ``options`` are the caller's options,
``output_schema`` the config's output model, ``context_keys`` the caller's prompt keys (names
only, ``caller_prompt_keys``). Derived, filled for the names the envelope declares: options
``context_keys`` (the default of that option is the file's lead keys, the caller's follow), ``schema_field_names``, ``tool_intents`` (with the ``classification`` variant)
and the text variables ``think_instructions``, ``plan_instructions``, ``step_instructions``,
``step_result_instructions``, ``plan_steps_instructions`` and ``replan_steps_instructions`` (the tool sections of the
ReAct, REWOO and Plan-and-Execute prompts) and ``context_names`` (the key names, quoted, as the field prompts print them); ``variables`` are the agent's own extra text variables (a pattern's system
message, D-021), merged after those. Raises ``AgentError`` for an option named
``approval`` or like a derived input; ``ValueError`` for a context key that is ``agent_trace`` or
internal-prefixed (as the builder always did); ``InstantiationError`` for what the envelope
refuses. A generator body (``prompt_chain``, D-019) is written by ``GENERATORS`` from the options.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from functools import partial
from typing import Any, cast

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.fsm_definitions._loop import (
    _schema_field_names,
    loop_field_context_keys,
)
from fsm_llm.agents.prompts import (
    build_execute_step_instructions,
    build_plan_steps_instructions,
    build_rewoo_plan_field_instructions,
    build_think_extraction_instructions,
)
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.core.definitions.executable_instance import instantiate_body
from fsm_llm.core.definitions.executable_pattern import PatternBodyError, check_option
from fsm_llm.engine.executables.namespaces import GENERATORS
from fsm_llm.tools import ToolRegistry

__all__ = [
    "APPROVAL_VARIANT",
    "DERIVED_OPTIONS",
    "TOOLLESS_VARIANT",
    "pattern_fsm",
    "refuse_reserved_options",
    "resolved_options",
]

#: The one variant the agent selects from its HITL predicate; no option may carry its name.
APPROVAL_VARIANT = "approval"

#: The options the agent fills itself (a caller cannot give them).
DERIVED_OPTIONS = frozenset({"context_keys", "schema_field_names", "tool_intents"})

#: The variant a run with no tools (``None`` or an empty registry) selects when the file declares it.
TOOLLESS_VARIANT = "without_tools"

#: The text variables built from the run's tools (the tool section of a prompt), by variable name,
#: in the order the builders called them.
_TOOL_TEXTS: dict[str, Callable[..., str]] = {
    "think_instructions": build_think_extraction_instructions,
    "plan_instructions": build_rewoo_plan_field_instructions,
    "step_instructions": build_execute_step_instructions,
    "step_result_instructions": partial(
        build_execute_step_instructions, step_result=True
    ),
    "plan_steps_instructions": build_plan_steps_instructions,
    "replan_steps_instructions": partial(build_plan_steps_instructions, replan=True),
}
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


def resolved_options(
    envelope: PatternEnvelope,
    given: Mapping[str, Any],
    *,
    config_named: Collection[str] = (),
) -> dict[str, Any]:
    """The options of ONE agent: *given* checked against the schema, then the file's default of every
    other declared option that has one (D-035).

    This is the one mapping the handler context, the hooks and ``pattern_fsm`` read, so a default
    written in a file or a preset is the value everywhere, not only in the FSM holes. The derived
    options (``DERIVED_OPTIONS``) are left out (``pattern_fsm`` fills them per run) and so are the
    defaults of the *config_named* options: their value lives in the agent's config, where the host's
    explicit field beats the file's default (D-033), so a handler falls back to the config. A given
    option is always kept. An option with neither a value nor a default is simply absent (the
    instantiation names it at run start). Raises ``AgentError`` naming the option, never its value,
    for a name the file does not declare or a value its schema refuses.
    """
    properties = envelope.body.options_schema.properties
    if unknown := sorted(n for n in given if n not in properties):
        raise AgentError(
            f"pattern {envelope.id!r} declares no option(s) {[str(n)[:60] for n in unknown[:5]]}"
        )
    out: dict[str, Any] = {}
    try:
        for name, value in given.items():
            out[name] = check_option(name, properties[name], value)
        for name, default in envelope.body.options.items():
            if (
                name in properties
                and name not in out
                and name not in DERIVED_OPTIONS
                and name not in config_named
            ):
                out[name] = check_option(name, properties[name], default)
    except PatternBodyError as error:
        raise AgentError(f"pattern {envelope.id!r}: {error}") from None
    return out


def pattern_fsm(
    envelope: PatternEnvelope,
    *,
    tools: ToolRegistry | None,
    task_description: str = "",
    approval: bool = False,
    options: Mapping[str, Any] | None = None,
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
    variables: Mapping[str, str] | None = None,
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
    if TOOLLESS_VARIANT in body.variants and not (tools is not None and len(tools) > 0):
        variants.append(TOOLLESS_VARIANT)
    texts: dict[str, str] = {}
    for name, build in _TOOL_TEXTS.items():
        if name in envelope.variables:
            # first, as the builders did: an unusable registry fails before a bad key does
            texts[name] = build(tools, task_description=task_description)
    values = dict(given)
    if "context_keys" in declared:
        # the file's default of this option holds its lead keys (react: agent_feedback;
        # reflexion adds episodic_memory), the caller's keys follow
        lead = cast(
            "Sequence[str]",
            body.options.get("context_keys", (ContextKeys.AGENT_FEEDBACK,)),
        )
        keys = loop_field_context_keys((*lead, *context_keys))
        values["context_keys"] = keys
        if (
            _KEYS_TEXT in envelope.variables
        ):  # the field prompts name the keys they read
            texts[_KEYS_TEXT] = ", ".join(repr(key) for key in keys)
    if "schema_field_names" in declared:
        values["schema_field_names"] = _schema_field_names(output_schema)
    if "tool_intents" in declared and "classification" in variants:
        if tools is None:
            raise AgentError("the classification variant needs the run's tools")
        values["tool_intents"] = tools.to_classification_schema()["intents"]
    return instantiate_body(
        envelope,
        values,
        variants=variants,
        variables={**texts, **(variables or {})},
        generators=GENERATORS.get,
    )
