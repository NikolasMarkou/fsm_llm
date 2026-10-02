"""
Builder tool factories for the meta-agent.

Each factory creates a ``ToolRegistry`` whose tools are closures over a
concrete builder instance.  These registries are part of the public API
for programmatic artifact construction outside of MetaBuilderAgent.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

from .definitions import ArtifactType
from .exceptions import BuilderError
from .meta_builders import (
    AgentArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from .tools import ToolRegistry, tool

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fmt(msg: str, warnings: list[str]) -> str:
    """Format a result message, appending warnings if any."""
    if warnings:
        msg += f" (warnings: {'; '.join(warnings)})"
    return msg


def _safe(fn: Any, *args: Any, **kwargs: Any) -> str:
    """Call *fn* and return its result string, catching BuilderError."""
    try:
        result: str = fn(*args, **kwargs)
        return result
    except BuilderError as e:
        return f"Error: {e}"


_AnyBuilder = FSMArtifactBuilder | WorkflowArtifactBuilder | AgentArtifactBuilder

# DECISION plan-2026-10-02T052921-89b03f61/D-013
# Every tool call runs under one re-entrant lock per builder, and clears stale
# warnings inside it before the mutator runs. Do NOT remove the lock or the
# pre-call clear and go back to bare ``mutate(...).take_warnings()``:
# ParallelReact runs tools concurrently on one registry (one builder), so
# mutate-then-take pairs interleave and replies steal or carry foreign
# warnings; warnings left by direct programmatic mutator calls also leaked
# into the next reply. Do NOT move the lock into ``ArtifactBuilder.__init__``:
# an ``RLock`` attribute breaks ``copy.deepcopy`` and ``pickle`` of builders,
# which work today, and subclasses would all have to call ``super().__init__()``.
_builder_locks: weakref.WeakKeyDictionary[Any, threading.RLock] = (
    weakref.WeakKeyDictionary()
)
_builder_locks_guard = threading.Lock()


def _call(builder: _AnyBuilder, body: Any, *, safe: bool = True) -> str:
    """Run one tool body on *builder* alone, with no stale warnings.

    Parameters: the bound *builder*; *body*, a zero-argument callable returning
    the reply string (it mutates, then reads ``builder.take_warnings()``);
    *safe*, True to turn a ``BuilderError`` into ``"Error: ..."`` (the tools
    that never caught it pass False, so their refusals are unchanged).
    Returns the body's reply. Any other exception propagates; the lock is
    released either way.
    """
    with _builder_locks_guard:
        lock = _builder_locks.setdefault(builder, threading.RLock())
    with lock:
        builder.take_warnings()
        return _safe(body) if safe else body()


def _make_validate_tool(builder: _AnyBuilder) -> Any:
    """Return the ``validate`` tool shared by all three factories, bound to *builder*."""

    @tool
    def validate() -> str:
        """Validate the artifact. Returns errors and warnings. Call before concluding."""

        def body() -> str:
            errors = builder.validate_complete()
            warnings = builder.validate_partial()
            if errors:
                return f"ERRORS: {'; '.join(errors)}"
            if warnings:
                return f"Valid (warnings: {'; '.join(warnings)})"
            return "Valid: no errors or warnings"

        return _call(builder, body, safe=False)

    return validate


def _make_summary_tool(builder: _AnyBuilder) -> Any:
    """Return the ``get_summary`` tool shared by all three factories, bound to *builder*."""

    @tool
    def get_summary() -> str:
        """Get the current builder state as a human-readable summary."""
        return _call(
            builder, lambda: builder.get_summary(detail_level="full"), safe=False
        )

    return get_summary


# ------------------------------------------------------------------
# FSM tools
# ------------------------------------------------------------------


def create_fsm_tools(builder: FSMArtifactBuilder) -> ToolRegistry:
    """Create tools for building an FSM definition."""
    registry = ToolRegistry()

    @tool
    def set_overview(name: str, description: str, persona: str = "") -> str:
        """Set the FSM name, description, and optional persona. Call this first."""

        def body() -> str:
            builder.set_overview(
                name=name,
                description=description,
                persona=persona or None,
            )
            return _fmt(f"Overview set: name='{name}'", builder.take_warnings())

        return _call(builder, body, safe=False)

    @tool
    def add_state(
        state_id: str,
        description: str,
        purpose: str,
        extraction_instructions: str = "",
        response_instructions: str = "",
    ) -> str:
        """Add a state to the FSM. The first state added automatically becomes the initial state."""
        return _call(
            builder,
            lambda: _fmt(
                f"Added state '{state_id}'",
                builder.add_state(
                    state_id=state_id,
                    description=description,
                    purpose=purpose,
                    extraction_instructions=extraction_instructions or None,
                    response_instructions=response_instructions or None,
                ).take_warnings(),
            ),
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
        return _call(
            builder,
            lambda: _fmt(
                f"Updated state '{state_id}'",
                builder.update_state(state_id, **fields).take_warnings(),
            ),
        )

    @tool
    def remove_state(state_id: str) -> str:
        """Remove a state and all its transitions."""

        def body() -> str:
            if state_id not in builder.states:
                return f"State '{state_id}' not found"
            builder.remove_state(state_id)
            return f"Removed state '{state_id}'"

        return _call(builder, body, safe=False)

    @tool
    def add_transition(
        from_state: str,
        target_state: str,
        description: str,
        priority: int = 100,
    ) -> str:
        """Add a transition between two existing states."""
        return _call(
            builder,
            lambda: _fmt(
                f"Added transition '{from_state}' -> '{target_state}'",
                builder.add_transition(
                    from_state=from_state,
                    target_state=target_state,
                    description=description,
                    priority=priority,
                ).take_warnings(),
            ),
        )

    @tool
    def remove_transition(from_state: str, target_state: str) -> str:
        """Remove a transition between two states."""

        def body() -> str:
            if not any(
                t["target_state"] == target_state
                for t in builder.states.get(from_state, {}).get("transitions", [])
            ):
                return "Transition not found"
            builder.remove_transition(from_state, target_state)
            return "Removed transition"

        return _call(builder, body, safe=False)

    @tool
    def set_initial_state(state_id: str) -> str:
        """Set which state the FSM starts in."""
        return _call(
            builder,
            lambda: _fmt(
                f"Initial state set to '{state_id}'",
                builder.set_initial_state(state_id).take_warnings(),
            ),
        )

    for fn in [
        set_overview,
        add_state,
        update_state,
        remove_state,
        add_transition,
        remove_transition,
        set_initial_state,
        _make_validate_tool(builder),
        _make_summary_tool(builder),
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Workflow tools
# ------------------------------------------------------------------


def create_workflow_tools(builder: WorkflowArtifactBuilder) -> ToolRegistry:
    """Create tools for building a workflow definition."""
    registry = ToolRegistry()

    @tool
    def set_overview(workflow_id: str, name: str, description: str) -> str:
        """Set the workflow ID, name, and description. Call this first."""

        def body() -> str:
            builder.set_overview(
                workflow_id=workflow_id,
                name=name,
                description=description,
            )
            return _fmt(f"Overview set: name='{name}'", builder.take_warnings())

        return _call(builder, body, safe=False)

    @tool
    def add_step(
        step_id: str,
        step_type: str,
        name: str,
        description: str = "",
    ) -> str:
        """Add a workflow step. Valid step types: auto_transition, api_call, condition, llm_processing, wait_for_event, timer, parallel, conversation. The first step added automatically becomes the initial step."""
        return _call(
            builder,
            lambda: _fmt(
                f"Added step '{step_id}' ({step_type})",
                builder.add_step(
                    step_id=step_id,
                    step_type=step_type,
                    name=name,
                    description=description,
                ).take_warnings(),
            ),
        )

    @tool
    def remove_step(step_id: str) -> str:
        """Remove a workflow step."""

        def body() -> str:
            if step_id not in builder.steps:
                return f"Step '{step_id}' not found"
            builder.remove_step(step_id)
            return f"Removed step '{step_id}'"

        return _call(builder, body, safe=False)

    @tool
    def set_step_transition(
        from_step: str,
        to_step: str,
        condition: str = "",
    ) -> str:
        """Connect two workflow steps with an optional condition."""
        return _call(
            builder,
            lambda: _fmt(
                f"Connected '{from_step}' -> '{to_step}'",
                builder.set_step_transition(
                    from_step=from_step,
                    to_step=to_step,
                    condition=condition or None,
                ).take_warnings(),
            ),
        )

    @tool
    def set_initial_step(step_id: str) -> str:
        """Set which step the workflow starts at."""
        return _call(
            builder,
            lambda: _fmt(
                f"Initial step set to '{step_id}'",
                builder.set_initial_step(step_id).take_warnings(),
            ),
        )

    for fn in [
        set_overview,
        add_step,
        remove_step,
        set_step_transition,
        set_initial_step,
        _make_validate_tool(builder),
        _make_summary_tool(builder),
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Agent tools
# ------------------------------------------------------------------


def create_agent_tools(builder: AgentArtifactBuilder) -> ToolRegistry:
    """Create tools for building an agent configuration."""
    registry = ToolRegistry()

    @tool
    def set_overview(name: str, description: str) -> str:
        """Set the agent name and description. Call this first."""

        def body() -> str:
            builder.set_overview(name=name, description=description)
            return _fmt(f"Overview set: name='{name}'", builder.take_warnings())

        return _call(builder, body, safe=False)

    @tool
    def set_agent_type(agent_type: str) -> str:
        """Set the agent pattern type. Valid types: react, plan_execute, reflexion, rewoo, evaluator_optimizer, maker_checker, prompt_chain, self_consistency, debate, orchestrator, adapt."""
        return _call(
            builder,
            lambda: _fmt(
                f"Agent type set to '{agent_type}'",
                builder.set_agent_type(agent_type).take_warnings(),
            ),
        )

    @tool
    def add_tool(name: str, description: str) -> str:
        """Add a tool definition to the agent."""
        return _call(
            builder,
            lambda: _fmt(
                f"Added tool '{name}'",
                builder.add_tool(name=name, description=description).take_warnings(),
            ),
        )

    @tool
    def remove_tool(name: str) -> str:
        """Remove a tool from the agent."""

        def body() -> str:
            if not any(t["name"] == name for t in builder.tools):
                return f"Tool '{name}' not found"
            builder.remove_tool(name)
            return f"Removed tool '{name}'"

        return _call(builder, body, safe=False)

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

        def body() -> str:
            builder.set_config(**kwargs)
            return _fmt("Config updated", builder.take_warnings())

        return _call(builder, body, safe=False)

    for fn in [
        set_overview,
        set_agent_type,
        add_tool,
        remove_tool,
        set_config,
        _make_validate_tool(builder),
        _make_summary_tool(builder),
    ]:
        registry.register(fn._tool_definition)

    return registry


# ------------------------------------------------------------------
# Factory dispatch
# ------------------------------------------------------------------


def create_builder_tools(
    builder: FSMArtifactBuilder | WorkflowArtifactBuilder | AgentArtifactBuilder,
    artifact_type: ArtifactType,
) -> ToolRegistry:
    """Create the appropriate tool registry for a builder instance."""
    if artifact_type == ArtifactType.FSM:
        if not isinstance(builder, FSMArtifactBuilder):
            raise TypeError(
                f"Expected FSMArtifactBuilder, got {type(builder).__name__}"
            )
        return create_fsm_tools(builder)
    if artifact_type == ArtifactType.WORKFLOW:
        if not isinstance(builder, WorkflowArtifactBuilder):
            raise TypeError(
                f"Expected WorkflowArtifactBuilder, got {type(builder).__name__}"
            )
        return create_workflow_tools(builder)
    if artifact_type == ArtifactType.AGENT:
        if not isinstance(builder, AgentArtifactBuilder):
            raise TypeError(
                f"Expected AgentArtifactBuilder, got {type(builder).__name__}"
            )
        return create_agent_tools(builder)
    raise ValueError(f"Unknown artifact type: {artifact_type}")
