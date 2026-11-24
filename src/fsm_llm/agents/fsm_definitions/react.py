"""ReAct FSM (``build_react_fsm``): the shipped file ``patterns/react.json``, instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.tools import ToolRegistry


def build_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
    use_classification: bool = False,
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build a ReAct FSM definition from a tool registry.

    The FSM implements the Observe-Think-Act loop:
    - think: LLM reasons about the task and selects a tool
    - act: tool is executed (via handler), observation is recorded
    - conclude: LLM produces final answer when should_terminate is true

    Optionally includes an await_approval state for HITL patterns.

    When *use_classification* is True, tool selection in the think state
    uses a ``classification_extractions`` config (backed by the core
    ``Classifier``) instead of relying solely on extraction instructions.
    This can improve tool selection accuracy for large tool registries.

    ``think`` extracts only typed per-field values; ``context_keys`` adds
    caller keys to those prompts (``caller_prompt_keys``).

    ``task_description`` is only the retrieval query of a semantic tool
    registry (the tool section of the prompts); the definition's name and
    description are static, so the task never enters definition text
    (D-002 and D-008 of plan a555dcfe).

    The FSM is the pattern file ``fsm_llm/agents/patterns/react.json`` with the
    ``approval`` and ``classification`` variants selected by the two flags and
    the tool text, the intents, the output schema's field names and the caller
    keys filled by ``pattern_inputs.pattern_fsm`` (plan a555dcfe step 9): the
    same dict, in the same key order, as the builder this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop`, so a module-level import would
    # make this package and that module import each other
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("react"),
        tools=registry,
        task_description=task_description,
        approval=bool(include_approval_state),
        options={"classification": bool(use_classification)},
        output_schema=output_schema,
        context_keys=context_keys,
    )
