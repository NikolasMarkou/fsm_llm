from __future__ import annotations

"""
FSM-LLM Agents
==============

Agentic patterns (ReAct, Human-in-the-Loop) built on top of
FSM-LLM's core, classification, and workflow packages.

Basic Usage:
    from fsm_llm_agents import ReactAgent, ToolRegistry

    registry = ToolRegistry()
    registry.register_function(my_tool, name="search", description="Search")

    agent = ReactAgent(tools=registry)
    result = agent.run("What is the population of France?")
    print(result.answer)

With HITL:
    from fsm_llm_agents import ReactAgent, ToolRegistry, HumanInTheLoop

    hitl = HumanInTheLoop(
        approval_policy=lambda call, ctx: call.tool_name == "send_email",
        approval_callback=my_approval_handler,
    )
    agent = ReactAgent(tools=registry, hitl=hitl)
"""

from .__version__ import __version__
from .adapt import ADaPTAgent
from .agent_graph import AgentGraph, AgentGraphBuilder
from .base import BaseAgent
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
from .memory_tools import create_memory_tools
from .meta_builder import MetaBuilderAgent
from .meta_builders import AgentBuilder, ArtifactBuilder, FSMBuilder, WorkflowBuilder
from .meta_output import format_artifact_json, format_summary, save_artifact
from .meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)
from .mcp import MCPToolProvider
from .orchestrator import OrchestratorAgent
from .plan_execute import PlanExecuteAgent
from .prompt_chain import PromptChainAgent
from .react import ReactAgent
from .reflexion import ReflexionAgent
from .remote import AgentServer, RemoteAgentTool
from .rewoo import REWOOAgent
from .self_consistency import SelfConsistencyAgent
from .semantic_tools import SemanticToolRegistry
from .skills import SkillDefinition, SkillLoader
from .sop import SOPDefinition, SOPRegistry, load_builtin_sops
from .swarm import SwarmAgent
from .tools import ToolRegistry, tool


def create_agent(
    system_prompt: str = "You are a helpful assistant.",
    tools: list | ToolRegistry | None = None,
    pattern: str = "react",
    **kwargs,
):
    """Create an agent in one line.

    Args:
        system_prompt: System prompt (used in tool descriptions, not FSM persona).
        tools: List of @tool-decorated functions or a ToolRegistry.
        pattern: Agent pattern — "react" (default), "debate", "rewoo", etc.
        **kwargs: Passed to the agent constructor (config, hitl, etc.).

    Returns:
        A configured agent instance with ``__call__`` support.

    Example::

        from fsm_llm_agents import create_agent, tool

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return "results"

        agent = create_agent(tools=[search])
        result = agent("What is the capital of France?")
    """
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

    _PATTERNS = {
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
    }

    cls = _PATTERNS.get(pattern)
    if cls is None:
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {sorted(_PATTERNS)}")

    # Tool-using agents need a registry
    if registry is not None and "tools" not in kwargs:
        kwargs["tools"] = registry

    return cls(**kwargs)


_has_reasoning_react = False
try:
    from .reasoning_react import ReasoningReactAgent  # noqa: F401

    _has_reasoning_react = True
except ImportError:
    pass

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
    "ToolRegistry",
    "SemanticToolRegistry",
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
    # Conditionally available (requires fsm_llm_reasoning)
    *((["ReasoningReactAgent"]) if _has_reasoning_react else []),
    # Decorator + factory + skill loading
    "tool",
    "create_agent",
    "create_memory_tools",
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
