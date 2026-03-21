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

try:
    from .reasoning_react import ReasoningReactAgent
except ImportError:
    pass

__all__ = [
    # Main classes
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
    "ReasoningReactAgent",
    "ToolRegistry",
    "HumanInTheLoop",
    # Decorator
    "tool",
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
