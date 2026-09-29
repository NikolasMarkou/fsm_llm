"""
FSM-LLM Agents
==============

Agentic patterns (ReAct, Human-in-the-Loop) built on top of
FSM-LLM's core, classification, and workflow packages.

Basic Usage:
    from fsm_llm.agents import ReactAgent, ToolRegistry

    registry = ToolRegistry()
    registry.register_function(my_tool, name="search", description="Search")

    agent = ReactAgent(tools=registry)
    result = agent.run("What is the population of France?")
    print(result.answer)

With HITL:
    from fsm_llm.agents import ReactAgent, ToolRegistry, HumanInTheLoop

    hitl = HumanInTheLoop(
        approval_policy=lambda call, ctx: call.tool_name == "send_email",
        approval_callback=my_approval_handler,
    )
    agent = ReactAgent(tools=registry, hitl=hitl)
"""

from __future__ import annotations

import warnings

from .__version__ import __version__
from .adapt import ADaPTAgent
from .agent_graph import AgentGraph, AgentGraphBuilder
from .auto_memory import (
    AutoMemoryReactAgent,
    augment_task_with_memories,
    remember_interaction,
    with_auto_memory,
)
from .base import BaseAgent, accepts_tools
from .composition import default_llm_judge, react_worker_factory
from .constants import StopReason
from .debate import DebateAgent
from .definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
    ApprovalRequest,
    ArtifactType,
    BuildProgress,
    ChainStep,
    DebateRound,
    DecompositionResult,
    EvaluationResult,
    MetaBuilderConfig,
    MetaBuilderResult,
    PlanStep,
    ReflexionMemory,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .evaluator_optimizer import EvaluatorOptimizerAgent
from .exceptions import (
    AgentError,
    AgentTimeoutError,
    ApprovalDeniedError,
    BudgetExhaustedError,
    BuilderError,
    DecompositionError,
    EvaluationError,
    MetaBuilderError,
    MetaValidationError,
    OutputError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from .hitl import (
    ApprovalCallback,
    ApprovalPolicy,
    EscalationCallback,
    HumanInTheLoop,
)
from .maker_checker import MakerCheckerAgent
from .mcp import MCPToolProvider
from .memory_persistence import (
    MemorySessionStore,
    load_working_memory,
    save_working_memory,
)
from .memory_tools import create_memory_tools
from .meta_builder import MetaBuilderAgent
from .meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    WorkflowBuilder,
)
from .meta_output import format_artifact_json, format_summary, save_artifact
from .meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)
from .native_fc import NativeFunctionCallingReactAgent
from .orchestrator import OrchestratorAgent
from .parallel_react import ParallelReactAgent, build_parallel_react_fsm
from .plan_execute import PlanExecuteAgent
from .prompt_chain import PromptChainAgent
from .react import ReactAgent
from .reasoning_react import ReasoningReactAgent
from .reflexion import ReflexionAgent
from .remote import AgentServer, RemoteAgentTool
from .rewoo import REWOOAgent
from .self_consistency import SelfConsistencyAgent
from .semantic_memory import (
    MemoryEntry,
    SemanticMemoryStore,
    create_semantic_memory_tools,
)
from .semantic_tools import SemanticToolRegistry
from .skills import SkillDefinition, SkillLoader
from .sop import SOPDefinition, SOPRegistry, load_builtin_sops
from .summarization import make_observation_summarizer
from .swarm import SwarmAgent
from .tool_registries import CachingToolRegistry, RetryingToolRegistry
from .tools import ToolRegistry, tool
from .verified_react import VerifiedReactAgent

# Pattern name -> agent class for create_agent. AgentGraph is intentionally
# absent: it is built only via AgentGraphBuilder (a node/edge graph, not the
# flat tools/config/**kwargs shape create_agent forwards).
_PATTERNS: dict[str, type] = {
    "react": ReactAgent,
    "rewoo": REWOOAgent,
    "debate": DebateAgent,
    "plan_execute": PlanExecuteAgent,
    "prompt_chain": PromptChainAgent,
    "self_consistency": SelfConsistencyAgent,
    "orchestrator": OrchestratorAgent,
    "adapt": ADaPTAgent,
    "evaluator_optimizer": EvaluatorOptimizerAgent,
    "maker_checker": MakerCheckerAgent,
    "reflexion": ReflexionAgent,
    "meta_builder": MetaBuilderAgent,
    "swarm": SwarmAgent,
    "parallel_react": ParallelReactAgent,
    "native_fc": NativeFunctionCallingReactAgent,
    "verified_react": VerifiedReactAgent,
    "auto_memory": AutoMemoryReactAgent,
    "reasoning_react": ReasoningReactAgent,
}

# Patterns whose own prompts never see AgentConfig.instructions: Swarm hands
# the task to member agents, and MetaBuilderAgent is not an FSM agent.
_NO_INSTRUCTIONS_PATTERNS = frozenset({"swarm", "meta_builder"})

# A legacy first positional (the old system_prompt) is told apart from a
# mistyped pattern name by shape: pattern names are short single words.
_LEGACY_PROMPT_MIN_LENGTH = 33


def _is_legacy_system_prompt(value: object) -> bool:
    """Whether a first positional that names no pattern is a legacy prompt."""
    return isinstance(value, str) and (
        any(ch.isspace() for ch in value) or len(value) >= _LEGACY_PROMPT_MIN_LENGTH
    )


def create_agent(
    pattern: str = "react",
    tools: list | ToolRegistry | None = None,
    *,
    config: AgentConfig | None = None,
    system_prompt: str | None = None,
    **kwargs,
):
    """Create an agent in one line.

    Args:
        pattern: Agent pattern: "react" (default), "debate", "rewoo", ...
            (see ``_PATTERNS``).
        tools: List of @tool-decorated functions or a ToolRegistry.
        config: ``AgentConfig`` for the agent; omitted, the pattern's default.
        system_prompt: Standing instructions for the agent, stored as
            ``AgentConfig.instructions`` (FSM patterns put them in their
            prompts; ``native_fc`` uses them as its ``system_policy``).
        **kwargs: Passed to the agent constructor (hitl, llm_interface, ...).
            A kwarg the pattern cannot use (``hitl``, ``tools``, ``model``,
            ...) raises ``TypeError``; set the model on ``AgentConfig``.

    Raises:
        ValueError: Unknown pattern (the message lists the valid ones);
            ``system_prompt`` for ``swarm``/``meta_builder``; or a
            ``system_prompt`` that conflicts with ``config.instructions``.
        TypeError: ``tools`` given to a pattern whose constructor takes none.

    Deprecated:
        The legacy call ``create_agent("You are ...", tools)`` still works:
        a first argument that names no pattern and contains whitespace or is
        longer than 32 characters is taken as ``system_prompt`` with a
        ``DeprecationWarning``, and the pattern is "react". A short unknown
        name raises ``ValueError``.

    Returns:
        A configured agent instance with ``__call__`` support.

    Example::

        from fsm_llm.agents import create_agent, tool

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return "results"

        agent = create_agent("react", [search], system_prompt="Cite sources.")
        result = agent("What is the capital of France?")
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-012: the pattern comes first.
    # Do NOT treat every unknown first argument as a legacy prompt (a typo like
    # "debat" must raise), and do NOT drop the shim: before this a positional
    # prompt was silently ignored, so old callers must keep running, warned.
    if pattern not in _PATTERNS and _is_legacy_system_prompt(pattern):
        warnings.warn(
            "create_agent(system_prompt, tools) is deprecated: the first "
            "argument is now the pattern. Use create_agent(pattern, tools, "
            "system_prompt=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        if system_prompt is not None:
            raise ValueError(
                "create_agent got a legacy positional system prompt and "
                "system_prompt=; pass it once"
            )
        system_prompt, pattern = pattern, "react"

    cls = _PATTERNS.get(pattern)
    if cls is None:
        raise ValueError(f"Unknown pattern {pattern!r}. Available: {sorted(_PATTERNS)}")

    if system_prompt is not None:
        if pattern in _NO_INSTRUCTIONS_PATTERNS:
            raise ValueError(
                f"Pattern {pattern!r} does not use system_prompt; set "
                "AgentConfig.instructions on the agents it runs"
            )
        if config is not None and config.instructions not in (None, system_prompt):
            raise ValueError(
                "create_agent got both system_prompt= and config.instructions; "
                "pass the instructions once"
            )
        base = config if config is not None else AgentConfig()
        fields = {name: getattr(base, name) for name in type(base).model_fields}
        config = type(base)(**{**fields, "instructions": system_prompt})
    if config is not None:
        kwargs["config"] = config

    # Build ToolRegistry from list of @tool-decorated functions
    registry = None
    if isinstance(tools, ToolRegistry):
        registry = tools
    elif tools is not None:
        registry = ToolRegistry()
        for fn in tools:
            if hasattr(fn, "_tool_definition"):
                registry.register(fn._tool_definition)
            else:
                registry.register_function(fn)

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-003: route tools by constructor
    # signature. Do NOT inject `tools` into every pattern: a tool-less pattern
    # (debate, self_consistency, ...) forwarded it to litellm (PAT-12).
    if registry is not None and "tools" not in kwargs:
        if not accepts_tools(cls):
            raise TypeError(
                f"Pattern '{pattern}' ({cls.__name__}) does not take tools; "
                f"remove the tools argument"
            )
        kwargs["tools"] = registry

    return cls(**kwargs)


__all__ = [
    # Main classes
    "BaseAgent",
    "ReactAgent",
    "REWOOAgent",
    "EvaluatorOptimizerAgent",
    "MakerCheckerAgent",
    "ReflexionAgent",
    "PlanExecuteAgent",
    "PromptChainAgent",
    "SelfConsistencyAgent",
    "DebateAgent",
    "OrchestratorAgent",
    "ADaPTAgent",
    "MetaBuilderAgent",
    "SwarmAgent",
    "ParallelReactAgent",
    "build_parallel_react_fsm",
    "VerifiedReactAgent",
    "NativeFunctionCallingReactAgent",
    "make_observation_summarizer",
    "react_worker_factory",
    "default_llm_judge",
    "ToolRegistry",
    "SemanticToolRegistry",
    "CachingToolRegistry",
    "RetryingToolRegistry",
    "HumanInTheLoop",
    # Phase 2: Graph, MCP, SOP, Remote
    "AgentGraph",
    "AgentGraphBuilder",
    "MCPToolProvider",
    "SOPDefinition",
    "SOPRegistry",
    "load_builtin_sops",
    "AgentServer",
    "RemoteAgentTool",
    "ReasoningReactAgent",
    # Decorator + factory + skill loading
    "tool",
    "create_agent",
    "create_memory_tools",
    "MemorySessionStore",
    "save_working_memory",
    "load_working_memory",
    "SemanticMemoryStore",
    "MemoryEntry",
    "create_semantic_memory_tools",
    "AutoMemoryReactAgent",
    "augment_task_with_memories",
    "remember_interaction",
    "with_auto_memory",
    "SkillDefinition",
    "SkillLoader",
    # Meta-builder
    "ArtifactBuilder",
    "FSMBuilder",
    "WorkflowBuilder",
    "AgentBuilder",
    "create_builder_tools",
    "create_fsm_tools",
    "create_workflow_tools",
    "create_agent_tools",
    "format_artifact_json",
    "format_summary",
    "save_artifact",
    # Models
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "AgentStep",
    "AgentTrace",
    "AgentConfig",
    "AgentResult",
    "StopReason",
    "ArtifactType",
    "BuildProgress",
    "MetaBuilderConfig",
    "MetaBuilderResult",
    "ApprovalRequest",
    "ChainStep",
    "DebateRound",
    "DecompositionResult",
    "EvaluationResult",
    "PlanStep",
    "ReflexionMemory",
    # Type aliases
    "ApprovalCallback",
    "ApprovalPolicy",
    "EscalationCallback",
    # Exceptions
    "AgentError",
    "MetaBuilderError",
    "BuilderError",
    "MetaValidationError",
    "OutputError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "BudgetExhaustedError",
    "ApprovalDeniedError",
    "AgentTimeoutError",
    "DecompositionError",
    "EvaluationError",
    # Version
    "__version__",
]
