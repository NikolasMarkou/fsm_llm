"""
FSM-LLM Agents
==============

Agent patterns (ReAct and 17 others, see ``create_agent``), tools,
Human-in-the-Loop approval, memory and MCP/A2A integration, built on the
FSM-LLM core.

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

from fsm_llm.agents.constants import StopReason
from fsm_llm.agents.fsm_definitions.parallel_react import build_parallel_react_fsm
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.engine.executables import ExecutableNotFoundError, ExecutableRegistry
from fsm_llm.engine.executables.namespaces import (
    EXECUTABLE_KINDS,
    EXECUTABLE_SOURCES,
    PATTERN_AGENTS,
)
from fsm_llm.policy import (
    ApprovalCallback,
    ApprovalDeniedError,
    ApprovalPolicy,
    ApprovalRequest,
    EscalationCallback,
    HumanInTheLoop,
)
from fsm_llm.tools import (
    ToolAnnotations,
    ToolCall,
    ToolDefinition,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    registry_from,
    tool,
)

from .__version__ import __version__
from .adapt import ADaPTAgent
from .agent_graph import AgentGraph, AgentGraphBuilder
from .agent_skills import create_skill_tools
from .auto_memory import (
    AutoMemoryReactAgent,
    augment_task_with_memories,
    remember_interaction,
)
from .base import BaseAgent, accepts_tools
from .composition import default_llm_judge, react_worker_factory
from .debate import DebateAgent
from .definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
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
    config_with,
    default_agent_config,
)
from .evaluator_optimizer import EvaluatorOptimizerAgent
from .exceptions import (
    AgentTimeoutError,
    BudgetExhaustedError,
    BuilderError,
    EvaluationError,
    MetaBuilderError,
    MetaValidationError,
    OutputError,
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
    AgentArtifactBuilder,
    ArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from .meta_output import format_artifact_json, format_summary, save_artifact
from .meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)
from .native_fc import NativeFunctionCallingReactAgent
from .okf import (
    OKFMemoryBackend,
    create_okf_tools,
    export_agent_result,
    export_semantic_memory,
)
from .orchestrator import OrchestratorAgent
from .parallel_react import ParallelReactAgent
from .pattern_agent import PatternAgent
from .plan_execute import PlanExecuteAgent
from .prompt_chain import PromptChainAgent
from .python_tools import (
    PythonToolDefinition,
    PythonToolLoader,
    PythonToolLoadReport,
)
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
from .summarization import make_observation_summarizer
from .swarm import SwarmAgent
from .tool_registries import CachingToolRegistry, RetryingToolRegistry
from .verified_react import VerifiedReactAgent

# Pattern name -> agent class for create_agent: the one class table is the `pattern_agents`
# namespace (plan a555dcfe D-030), filled here in this order. AgentGraph is intentionally absent:
# it is built only via AgentGraphBuilder (a node/edge graph, not the flat tools/config/**kwargs
# shape create_agent forwards).
for _name, _class in (
    ("react", ReactAgent),
    ("rewoo", REWOOAgent),
    ("debate", DebateAgent),
    ("plan_execute", PlanExecuteAgent),
    ("prompt_chain", PromptChainAgent),
    ("self_consistency", SelfConsistencyAgent),
    ("orchestrator", OrchestratorAgent),
    ("adapt", ADaPTAgent),
    ("evaluator_optimizer", EvaluatorOptimizerAgent),
    ("maker_checker", MakerCheckerAgent),
    ("reflexion", ReflexionAgent),
    ("meta_builder", MetaBuilderAgent),
    ("swarm", SwarmAgent),
    ("parallel_react", ParallelReactAgent),
    ("native_fc", NativeFunctionCallingReactAgent),
    ("verified_react", VerifiedReactAgent),
    ("auto_memory", AutoMemoryReactAgent),
    ("reasoning_react", ReasoningReactAgent),
):
    PATTERN_AGENTS.register(
        _name, _class, override=True
    )  # override: a reload re-registers

# Patterns whose own prompts never see AgentConfig.instructions: Swarm hands
# the task to member agents, and MetaBuilderAgent is not an FSM agent.
_NO_INSTRUCTIONS_PATTERNS = frozenset({"swarm", "meta_builder"})


def create_agent(
    pattern: str = "react",
    tools: list | ToolRegistry | None = None,
    *,
    registry: ExecutableRegistry | None = None,
    config: AgentConfig | None = None,
    system_prompt: str | None = None,
    **kwargs,
):
    """Create an agent in one line.

    Args:
        pattern: Agent pattern: "react" (default), "debate", "rewoo", ... (the ids of the
            ``pattern_agents`` namespace), or the id of a pattern file held by ``registry``.
        tools: A list of ``@tool`` functions, plain functions or
            ``ToolDefinition`` objects, or a ToolRegistry.
        registry: An ``ExecutableRegistry`` holding user pattern files. A built-in id never
            consults it (a registry without ``react`` still builds ``react``, and a user file
            named like a built-in does not replace it); any other id must name a pattern
            executable of it and builds a ``PatternAgent`` over its file, which refuses an
            unapprovable flagged tool at creation like ``ReactAgent`` does.
        config: ``AgentConfig`` for the agent; omitted, the pattern's default.
        system_prompt: Standing instructions for the agent, stored as
            ``AgentConfig.instructions`` (FSM patterns put them in their
            prompts; ``native_fc`` uses them as its ``system_policy``).
        **kwargs: Passed to the agent constructor (hitl, llm_interface, ...).
            A kwarg the pattern cannot use (``hitl``, ``tools``, ``model``,
            ...) raises ``TypeError``; set the model on ``AgentConfig``.

    Raises:
        ValueError: Unknown pattern (the message lists the valid ones; built-in names
            are matched after ``strip().lower()``, so ``"Debate "`` is the
            debate pattern);
            ``system_prompt`` for ``swarm``/``meta_builder``; or a
            ``system_prompt`` that conflicts with ``config.instructions``.
        TypeError: ``tools`` given to a pattern whose constructor takes none.

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
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-041 [supersedes
    # plan-2026-09-29T103145-06a5ec0a/D-012]: the first argument is a pattern
    # name and nothing else. Do NOT guess that a long or spaced first argument
    # is a system prompt (the removed shim): it raises below, and the caller
    # passes ``system_prompt=``. Keep the normalisation: "debate " and " React"
    # name a pattern. See decisions.md D-041.
    if isinstance(pattern, str) and pattern.strip().lower() in PATTERN_AGENTS:
        pattern = pattern.strip().lower()

    # DECISION plan-2026-11-24T043150-a555dcfe/D-030
    # One class table: `PATTERN_AGENTS`. A built-in id builds its class and the registry is NOT
    # consulted; any other id is a pattern FILE of `registry`. Do NOT keep a second dict of ids
    # beside the namespace, do NOT let a registry entry replace a built-in id (override is a
    # collision, never a feature), do NOT add a rule here for a user file: its refusals are those
    # of `PatternAgent` (flagged tool with no gate, a gate with nobody to approve) at creation.
    cls = PATTERN_AGENTS.get(pattern) if isinstance(pattern, str) else None
    envelope = None
    if cls is None:
        envelope = _pattern_file(pattern, registry)
        if envelope is None:
            known = {*PATTERN_AGENTS.list()}
            if registry is not None:
                known.update(e.id for e in registry.list(kind="pattern"))
            raise ValueError(f"Unknown pattern {pattern!r}. Available: {sorted(known)}")
        cls = PatternAgent

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
        config = config_with(
            config
            if config is not None
            else default_agent_config(kwargs.get("llm_interface")),
            instructions=system_prompt,
        )
    if config is not None:
        kwargs["config"] = config

    tool_registry = None
    if isinstance(tools, ToolRegistry):
        tool_registry = tools
    elif tools is not None:
        tool_registry = registry_from(tools)

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-003: route tools by constructor
    # signature. Do NOT inject `tools` into every pattern: a tool-less pattern
    # (debate, self_consistency, ...) forwarded it to litellm (PAT-12).
    if tool_registry is not None and "tools" not in kwargs:
        takes = accepts_tools(cls) and (
            envelope is None or envelope.requires.tools != "none"
        )
        if takes:
            kwargs["tools"] = tool_registry
        elif len(tool_registry):  # an empty list is no tools given (F-AP-14)
            raise TypeError(
                f"Pattern '{pattern}' ({cls.__name__}) does not take tools; "
                f"remove the tools argument"
            )

    return cls(envelope, **kwargs) if envelope is not None else cls(**kwargs)


def _pattern_file(
    name: object, registry: ExecutableRegistry | None
) -> PatternEnvelope | None:
    """The envelope of the pattern file *name* held by *registry*, or ``None`` (no registry, no
    such id, or an executable of another kind or with no file behind it)."""
    if registry is None or not isinstance(name, str):
        return None
    try:
        found = registry.get(name)
    except ExecutableNotFoundError:
        return None
    envelope = getattr(found, "envelope", None)
    return envelope if isinstance(envelope, PatternEnvelope) else None


from .builders import ConfiguredAgentBuilder


def _builtin_executables() -> list:
    """The ``agents`` source of ``executable_sources``: one executable per pattern.

    Lazy on purpose: ``agents.executables`` is imported on the first call, so importing this
    package loads the ``fsm_llm.engine.executables`` package (its namespaces live there; about ten
    modules come with it) but not ``agents.executables`` itself (plan ace7a8c8 D-061, D-070).
    """
    from .executables import builtin_executables

    return builtin_executables()


def _builtin_presets() -> list:
    """The ``agent_presets`` source: the shipped presets that no agent class stands behind
    (``code-review``, ``summarize``, ``data-extraction``; plan a555dcfe D-033). Lazy like the
    ``agents`` source."""
    from .executables import preset_executables

    return preset_executables()


def _pattern_adapter(envelope, **options):
    """The ``pattern`` kind of ``executable_kinds``: lazy like the source, so importing this
    package never imports ``agents.executables`` (D4) and a registry that loads a pattern file
    gets its executable (plan a555dcfe step 9, D-015)."""
    from .executables import pattern_adapter

    return pattern_adapter(envelope, **options)


# `override=True`: re-importing the package (a reload) replaces the factory, never collides.
EXECUTABLE_SOURCES.register("agents", _builtin_executables, override=True)
EXECUTABLE_SOURCES.register("agent_presets", _builtin_presets, override=True)
EXECUTABLE_KINDS.register("pattern", _pattern_adapter, override=True)

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
    # Phase 2: Graph, MCP, Remote
    "AgentGraph",
    "AgentGraphBuilder",
    "ConfiguredAgentBuilder",
    "MCPToolProvider",
    "AgentServer",
    "RemoteAgentTool",
    "ReasoningReactAgent",
    # Decorator + factory + Python tool loading
    "tool",
    "PythonToolDefinition",
    "PythonToolLoader",
    "PythonToolLoadReport",
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
    # OKF knowledge bundles
    "OKFMemoryBackend",
    "create_okf_tools",
    "export_agent_result",
    "export_semantic_memory",
    # Agent Skills folders
    "create_skill_tools",
    # Meta-builder
    "ArtifactBuilder",
    "FSMArtifactBuilder",
    "WorkflowArtifactBuilder",
    "AgentArtifactBuilder",
    "create_builder_tools",
    "create_fsm_tools",
    "create_workflow_tools",
    "create_agent_tools",
    "format_artifact_json",
    "format_summary",
    "save_artifact",
    # Models
    "ToolDefinition",
    "ToolAnnotations",
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
    "BudgetExhaustedError",
    "ApprovalDeniedError",
    "AgentTimeoutError",
    "EvaluationError",
    # Version
    "__version__",
]
