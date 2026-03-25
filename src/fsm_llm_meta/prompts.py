from __future__ import annotations

"""
Prompt builders for each state of the meta-agent FSM.
"""

from .constants import Actions


def build_welcome_response_instructions() -> str:
    return (
        "Welcome the user and ask what type of artifact they want to build. "
        "Explain that you can help them build:\n"
        "1. An FSM (Finite State Machine) - for stateful conversations\n"
        "2. A Workflow - for multi-step async processes\n"
        "3. An Agent - for tool-using AI agents\n\n"
        "Ask them which one they'd like to create."
    )


def build_classify_extraction_instructions() -> str:
    return (
        "Extract the type of artifact the user wants to build.\n\n"
        "You MUST respond with valid JSON containing:\n"
        '{"artifact_type": "<type>"}\n\n'
        "Where <type> is one of: fsm, workflow, agent\n\n"
        "If the user hasn't clearly specified, infer from context:\n"
        '- "conversation", "chatbot", "states" → fsm\n'
        '- "pipeline", "process", "steps", "automation" → workflow\n'
        '- "tools", "search", "react", "actions" → agent'
    )


def build_classify_response_instructions() -> str:
    return (
        "Confirm what type of artifact you'll help build based on their choice. "
        "Then ask them for a name, description, and (for FSMs) an optional persona. "
        "Be encouraging and conversational."
    )


def build_gather_overview_extraction_instructions() -> str:
    return (
        "Extract overview information from the user's message.\n\n"
        "You MUST respond with valid JSON. Include only the fields the user provided:\n"
        "{\n"
        '  "artifact_name": "<name>",\n'
        '  "artifact_description": "<description>",\n'
        '  "artifact_persona": "<persona or null>"\n'
        "}\n\n"
        "Only include fields that the user explicitly mentioned."
    )


def build_gather_overview_response_instructions(builder_summary: str) -> str:
    return (
        "Review what the user has provided so far and ask for any missing overview details.\n\n"
        f"Current builder state:\n{builder_summary}\n\n"
        "If name and description are both set, confirm them and explain that "
        "you'll now help design the structure (states/steps/tools). "
        "Ask them to describe their first component."
    )


def build_design_structure_extraction_instructions(
    artifact_type: str,
    builder_summary: str,
) -> str:
    if artifact_type == "fsm":
        return _build_fsm_structure_extraction(builder_summary)
    elif artifact_type == "workflow":
        return _build_workflow_structure_extraction(builder_summary)
    else:
        return _build_agent_structure_extraction(builder_summary)


def _build_fsm_structure_extraction(builder_summary: str) -> str:
    return (
        "You are helping build an FSM. Extract the user's intent as a structured action.\n\n"
        f"Current FSM state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON containing an action:\n\n"
        f'To add a state: {{"action": "{Actions.ADD_STATE}", "action_params": {{'
        '"state_id": "<id>", "description": "<desc>", "purpose": "<purpose>", '
        '"extraction_instructions": "<optional>", "response_instructions": "<optional>"}}\n\n'
        f'To remove a state: {{"action": "{Actions.REMOVE_STATE}", "action_params": {{"state_id": "<id>"}}}}\n\n'
        f'To update a state: {{"action": "{Actions.UPDATE_STATE}", "action_params": {{"state_id": "<id>", ...fields}}}}\n\n'
        f'When the user is done adding states: {{"action": "{Actions.DONE}"}}\n\n'
        "If the user describes something that sounds like a state, extract it as an add_state action. "
        "If they say they're done or want to move on to transitions, use the done action."
    )


def _build_workflow_structure_extraction(builder_summary: str) -> str:
    return (
        "You are helping build a Workflow. Extract the user's intent as a structured action.\n\n"
        f"Current Workflow state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON containing an action:\n\n"
        f'To add a step: {{"action": "{Actions.ADD_STEP}", "action_params": {{'
        '"step_id": "<id>", "step_type": "<type>", "name": "<name>", "description": "<desc>"}}\n\n'
        "Valid step types: auto_transition, api_call, condition, llm_processing, "
        "wait_for_event, timer, parallel, conversation\n\n"
        f'To remove a step: {{"action": "{Actions.REMOVE_STEP}", "action_params": {{"step_id": "<id>"}}}}\n\n'
        f'When the user is done: {{"action": "{Actions.DONE}"}}'
    )


def _build_agent_structure_extraction(builder_summary: str) -> str:
    return (
        "You are helping build an Agent. Extract the user's intent as a structured action.\n\n"
        f"Current Agent state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON containing an action:\n\n"
        f'To set agent type: {{"action": "{Actions.SET_AGENT_TYPE}", "action_params": {{"agent_type": "<type>"}}}}\n'
        "Valid types: react, plan_execute, reflexion, rewoo, evaluator_optimizer, "
        "maker_checker, prompt_chain, self_consistency, debate, orchestrator, adapt\n\n"
        f'To add a tool: {{"action": "{Actions.ADD_TOOL}", "action_params": {{'
        '"name": "<name>", "description": "<desc>", '
        '"parameter_schema": {{"type": "object", "properties": {{...}}}}}}}}\n\n'
        f'To set config: {{"action": "{Actions.SET_CONFIG}", "action_params": {{'
        '"model": "<model>", "max_iterations": <n>}}}}\n\n'
        f'When done: {{"action": "{Actions.DONE}"}}'
    )


def build_design_structure_response_instructions(
    artifact_type: str,
    builder_summary: str,
) -> str:
    type_label = {
        "fsm": "states",
        "workflow": "steps",
        "agent": "tools and configuration",
    }.get(artifact_type, "components")

    return (
        f"You are helping the user design {type_label} for their {artifact_type}.\n\n"
        f"Current state:\n{builder_summary}\n\n"
        "After each action:\n"
        "1. Confirm what was just added/modified\n"
        "2. Show a brief summary of what exists so far\n"
        "3. Ask about the next component, or if they're done\n\n"
        "Be helpful and suggest common patterns when appropriate. "
        "If the user seems unsure, offer suggestions based on their description."
    )


def build_define_connections_extraction_instructions(
    artifact_type: str,
    builder_summary: str,
) -> str:
    if artifact_type == "fsm":
        return _build_fsm_connections_extraction(builder_summary)
    elif artifact_type == "workflow":
        return _build_workflow_connections_extraction(builder_summary)
    else:
        # Agents don't have explicit connections
        return (
            "The agent configuration is complete. "
            f'Respond with: {{"action": "{Actions.DONE}"}}'
        )


def _build_fsm_connections_extraction(builder_summary: str) -> str:
    return (
        "You are helping define transitions between FSM states.\n\n"
        f"Current FSM state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON containing an action:\n\n"
        f'To add a transition: {{"action": "{Actions.ADD_TRANSITION}", "action_params": {{'
        '"from_state": "<id>", "target_state": "<id>", "description": "<when to transition>", '
        '"priority": 100, "conditions": [{{"description": "<condition>", "logic": {{...}}}}]}}\n\n'
        "Conditions are optional. JsonLogic examples:\n"
        '  {"==": [{"var": "key"}, "value"]}  — check equality\n'
        '  {"has_context": "key"}  — check key exists\n'
        '  {"and": [...conditions]}  — combine conditions\n\n'
        f'To remove: {{"action": "{Actions.REMOVE_TRANSITION}", "action_params": {{"from_state": "<id>", "target_state": "<id>"}}}}\n\n'
        f'To set initial state: {{"action": "{Actions.SET_INITIAL_STATE}", "action_params": {{"state_id": "<id>"}}}}\n\n'
        f'When done: {{"action": "{Actions.DONE}"}}'
    )


def _build_workflow_connections_extraction(builder_summary: str) -> str:
    return (
        "You are helping define connections between workflow steps.\n\n"
        f"Current Workflow state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON containing an action:\n\n"
        f'To connect steps: {{"action": "{Actions.SET_STEP_TRANSITION}", "action_params": {{'
        '"from_step": "<id>", "to_step": "<id>", "condition": "<optional condition>"}}}}\n\n'
        f'To set initial step: {{"action": "{Actions.SET_INITIAL_STEP}", "action_params": {{"step_id": "<id>"}}}}\n\n'
        f'When done: {{"action": "{Actions.DONE}"}}'
    )


def build_define_connections_response_instructions(
    artifact_type: str,
    builder_summary: str,
) -> str:
    conn_label = "transitions" if artifact_type == "fsm" else "connections"

    return (
        f"You are helping define {conn_label} between components.\n\n"
        f"Current state:\n{builder_summary}\n\n"
        "After each connection:\n"
        "1. Confirm what was connected\n"
        "2. Show remaining components that might need connections\n"
        "3. Suggest logical connections if the user seems unsure\n\n"
        "Make sure all non-terminal states/steps have at least one outgoing connection."
    )


def build_review_extraction_instructions(builder_summary: str) -> str:
    return (
        "The user is reviewing their artifact. Extract their decision.\n\n"
        f"Current state:\n{builder_summary}\n\n"
        "You MUST respond with valid JSON:\n"
        '{"user_decision": "approve"} — if they confirm it looks good\n'
        '{"user_decision": "revise"} — if they want to make changes'
    )


def build_review_response_instructions(
    builder_summary: str,
    validation_errors: list[str],
    validation_warnings: list[str],
) -> str:
    parts = [
        "Present a complete summary of the artifact for the user to review.\n",
        f"Current state:\n{builder_summary}\n",
    ]

    if validation_errors:
        parts.append("\nValidation ERRORS (must be fixed):")
        for e in validation_errors:
            parts.append(f"  - {e}")
        parts.append("\nTell the user about these errors and suggest fixes.")
    elif validation_warnings:
        parts.append("\nValidation warnings (optional to fix):")
        for w in validation_warnings:
            parts.append(f"  - {w}")

    if not validation_errors:
        parts.append(
            "\nAsk the user if they approve the artifact or want to make changes."
        )

    return "\n".join(parts)


def build_output_response_instructions() -> str:
    return (
        "The artifact has been built successfully! "
        "Present the final JSON to the user and let them know it's ready. "
        "Mention that they can save it to a file and use it with fsm-llm."
    )
