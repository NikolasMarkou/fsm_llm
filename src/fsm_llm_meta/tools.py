from __future__ import annotations

"""
Builder tool factories for the meta-agent.

Each factory creates a ``ToolRegistry`` whose tools are closures over a
concrete builder instance.  The ReactAgent uses this registry during the
build phase to autonomously construct the artifact.
"""

from typing import Any

from fsm_llm_agents.tools import ToolRegistry, tool

from .builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from .definitions import ArtifactType
from .exceptions import BuilderError

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fmt(msg: str, warnings: list[str]) -> str:
    """Format a result message, appending warnings if any."""
    if warnings:
        msg += f" (warnings: {'; '.join(warnings)})"
    return msg


def _safe(fn, *args: Any, **kwargs: Any) -> str:
    """Call *fn* and return its result string, catching BuilderError."""
    try:
        return fn(*args, **kwargs)
    except BuilderError as e:
        return f"Error: {e}"


# ------------------------------------------------------------------
# FSM tools
# ------------------------------------------------------------------


def create_fsm_tools(builder: FSMBuilder) -> ToolRegistry:
    """Create tools for building an FSM definition."""
    registry = ToolRegistry()

    @tool
    def set_overview(name: str, description: str, persona: str = "") -> str:
        """Set the FSM name, description, and optional persona. Call this first."""
        warnings = builder.set_overview(
            name=name,
            description=description,
            persona=persona or None,
        )
        return _fmt(f"Overview set: name='{name}'", warnings)

    @tool
    def add_state(
        state_id: str,
        description: str,
        purpose: str,
        extraction_instructions: str = "",
        response_instructions: str = "",
    ) -> str:
        """Add a state to the FSM. The first state added automatically becomes the initial state."""
        return _safe(
            lambda: _fmt(
                f"Added state '{state_id}'",
                builder.add_state(
                    state_id=state_id,
                    description=description,
                    purpose=purpose,
                    extraction_instructions=extraction_instructions or None,
                    response_instructions=response_instructions or None,
                ),
            )
        )

    @tool
    def update_state(
        state_id: str,
        description: str = "",
        purpose: str = "",
        extraction_instructions: str = "",
        response_instructions: str = "",
    ) -> str:
        """Update fields on an existing state."""
        fields = {
            k: v
            for k, v in {
                "description": description,
                "purpose": purpose,
                "extraction_instructions": extraction_instructions,
                "response_instructions": response_instructions,
            }.items()
            if v
        }
        return _safe(
            lambda: _fmt(
                f"Updated state '{state_id}'",
                builder.update_state(state_id, **fields),
            )
        )

    @tool
    def remove_state(state_id: str) -> str:
        """Remove a state and all its transitions."""
        removed = builder.remove_state(state_id)
        return (
            f"Removed state '{state_id}'"
            if removed
            else f"State '{state_id}' not found"
        )

    @tool
    def add_transition(
        from_state: str,
        target_state: str,
        description: str,
        priority: int = 100,
    ) -> str:
        """Add a transition between two existing states."""
        return _safe(
            lambda: _fmt(
                f"Added transition '{from_state}' -> '{target_state}'",
                builder.add_transition(
                    from_state=from_state,
                    target_state=target_state,
                    description=description,
                    priority=priority,
                ),
            )
        )

    @tool
    def remove_transition(from_state: str, target_state: str) -> str:
        """Remove a transition between two states."""
        removed = builder.remove_transition(from_state, target_state)
        return "Removed transition" if removed else "Transition not found"

    @tool
    def set_initial_state(state_id: str) -> str:
        """Set which state the FSM starts in."""
        return _safe(
            lambda: _fmt(
                f"Initial state set to '{state_id}'",
                builder.set_initial_state(state_id),
            )
        )

    @tool
    def validate() -> str:
        """Validate the artifact. Returns errors and warnings. Call before concluding."""
        errors = builder.validate_complete()
        warnings = builder.validate_partial()
        if errors:
            return f"ERRORS: {'; '.join(errors)}"
        if warnings:
            return f"Valid (warnings: {'; '.join(warnings)})"
        return "Valid: no errors or warnings"

    @tool
    def get_summary() -> str:
        """Get the current builder state as a human-readable summary."""
        return builder.get_summary(detail_level="full")

    for fn in [
        set_overview,
        add_state,
        update_state,
        remove_state,
        add_transition,
        remove_transition,
        set_initial_state,
        validate,
        get_summary,
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Workflow tools
# ------------------------------------------------------------------


def create_workflow_tools(builder: WorkflowBuilder) -> ToolRegistry:
    """Create tools for building a workflow definition."""
    registry = ToolRegistry()

    @tool
    def set_overview(workflow_id: str, name: str, description: str) -> str:
        """Set the workflow ID, name, and description. Call this first."""
        warnings = builder.set_overview(
            workflow_id=workflow_id,
            name=name,
            description=description,
        )
        return _fmt(f"Overview set: name='{name}'", warnings)

    @tool
    def add_step(
        step_id: str,
        step_type: str,
        name: str,
        description: str = "",
    ) -> str:
        """Add a workflow step. Valid step types: auto_transition, api_call, condition, llm_processing, wait_for_event, timer, parallel, conversation. The first step added automatically becomes the initial step."""
        return _safe(
            lambda: _fmt(
                f"Added step '{step_id}' ({step_type})",
                builder.add_step(
                    step_id=step_id,
                    step_type=step_type,
                    name=name,
                    description=description,
                ),
            )
        )

    @tool
    def remove_step(step_id: str) -> str:
        """Remove a workflow step."""
        removed = builder.remove_step(step_id)
        return f"Removed step '{step_id}'" if removed else f"Step '{step_id}' not found"

    @tool
    def set_step_transition(
        from_step: str,
        to_step: str,
        condition: str = "",
    ) -> str:
        """Connect two workflow steps with an optional condition."""
        return _safe(
            lambda: _fmt(
                f"Connected '{from_step}' -> '{to_step}'",
                builder.set_step_transition(
                    from_step=from_step,
                    to_step=to_step,
                    condition=condition or None,
                ),
            )
        )

    @tool
    def set_initial_step(step_id: str) -> str:
        """Set which step the workflow starts at."""
        return _safe(
            lambda: _fmt(
                f"Initial step set to '{step_id}'",
                builder.set_initial_step(step_id),
            )
        )

    @tool
    def validate() -> str:
        """Validate the artifact. Returns errors and warnings. Call before concluding."""
        errors = builder.validate_complete()
        warnings = builder.validate_partial()
        if errors:
            return f"ERRORS: {'; '.join(errors)}"
        if warnings:
            return f"Valid (warnings: {'; '.join(warnings)})"
        return "Valid: no errors or warnings"

    @tool
    def get_summary() -> str:
        """Get the current builder state as a human-readable summary."""
        return builder.get_summary(detail_level="full")

    for fn in [
        set_overview,
        add_step,
        remove_step,
        set_step_transition,
        set_initial_step,
        validate,
        get_summary,
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Agent tools
# ------------------------------------------------------------------


def create_agent_tools(builder: AgentBuilder) -> ToolRegistry:
    """Create tools for building an agent configuration."""
    registry = ToolRegistry()

    @tool
    def set_overview(name: str, description: str) -> str:
        """Set the agent name and description. Call this first."""
        warnings = builder.set_overview(name=name, description=description)
        return _fmt(f"Overview set: name='{name}'", warnings)

    @tool
    def set_agent_type(agent_type: str) -> str:
        """Set the agent pattern type. Valid types: react, plan_execute, reflexion, rewoo, evaluator_optimizer, maker_checker, prompt_chain, self_consistency, debate, orchestrator, adapt."""
        return _safe(
            lambda: _fmt(
                f"Agent type set to '{agent_type}'",
                builder.set_agent_type(agent_type),
            )
        )

    @tool
    def add_tool(name: str, description: str) -> str:
        """Add a tool definition to the agent."""
        return _safe(
            lambda: _fmt(
                f"Added tool '{name}'",
                builder.add_tool(name=name, description=description),
            )
        )

    @tool
    def remove_tool(name: str) -> str:
        """Remove a tool from the agent."""
        removed = builder.remove_tool(name)
        return f"Removed tool '{name}'" if removed else f"Tool '{name}' not found"

    @tool
    def set_config(
        model: str = "",
        max_iterations: int = 0,
        timeout_seconds: float = 0.0,
        temperature: float = -1.0,
        max_tokens: int = 0,
    ) -> str:
        """Update agent configuration fields. Only non-default values are applied."""
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if max_iterations > 0:
            kwargs["max_iterations"] = max_iterations
        if timeout_seconds > 0:
            kwargs["timeout_seconds"] = timeout_seconds
        if temperature >= 0:
            kwargs["temperature"] = temperature
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        if not kwargs:
            return "No config fields to update"
        warnings = builder.set_config(**kwargs)
        return _fmt("Config updated", warnings)

    @tool
    def validate() -> str:
        """Validate the artifact. Returns errors and warnings. Call before concluding."""
        errors = builder.validate_complete()
        warnings = builder.validate_partial()
        if errors:
            return f"ERRORS: {'; '.join(errors)}"
        if warnings:
            return f"Valid (warnings: {'; '.join(warnings)})"
        return "Valid: no errors or warnings"

    @tool
    def get_summary() -> str:
        """Get the current builder state as a human-readable summary."""
        return builder.get_summary(detail_level="full")

    for fn in [
        set_overview,
        set_agent_type,
        add_tool,
        remove_tool,
        set_config,
        validate,
        get_summary,
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Factory dispatch
# ------------------------------------------------------------------


def create_builder_tools(
    builder: FSMBuilder | WorkflowBuilder | AgentBuilder,
    artifact_type: ArtifactType,
) -> ToolRegistry:
    """Create the appropriate tool registry for a builder instance."""
    if artifact_type == ArtifactType.FSM:
        if not isinstance(builder, FSMBuilder):
            raise TypeError(f"Expected FSMBuilder, got {type(builder).__name__}")
        return create_fsm_tools(builder)
    if artifact_type == ArtifactType.WORKFLOW:
        if not isinstance(builder, WorkflowBuilder):
            raise TypeError(f"Expected WorkflowBuilder, got {type(builder).__name__}")
        return create_workflow_tools(builder)
    if artifact_type == ArtifactType.AGENT:
        if not isinstance(builder, AgentBuilder):
            raise TypeError(f"Expected AgentBuilder, got {type(builder).__name__}")
        return create_agent_tools(builder)
    raise ValueError(f"Unknown artifact type: {artifact_type}")
