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
from .base import BaseAgent
from .debate import DebateAgent
from .definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
    ApprovalRequest,
    ChainStep,
    DebateRound,
    DecompositionResult,
    EvaluationResult,
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
    DecompositionError,
    EvaluationError,
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
from .orchestrator import OrchestratorAgent
from .plan_execute import PlanExecuteAgent
from .prompt_chain import PromptChainAgent
from .react import ReactAgent
from .reflexion import ReflexionAgent
from .rewoo import REWOOAgent
from .self_consistency import SelfConsistencyAgent
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
    }

    cls = _PATTERNS.get(pattern)
    if cls is None:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Available: {sorted(_PATTERNS)}"
        )

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
    "ToolRegistry",
    "HumanInTheLoop",
    # Conditionally available (requires fsm_llm_reasoning)
    *((["ReasoningReactAgent"]) if _has_reasoning_react else []),
    # Decorator + factory
    "tool",
    "create_agent",
    # Models
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "AgentStep",
    "AgentTrace",
    "AgentConfig",
    "AgentResult",
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
