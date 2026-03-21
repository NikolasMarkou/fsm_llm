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
from .definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
    ApprovalRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .exceptions import (
    AgentError,
    AgentTimeoutError,
    ApprovalDeniedError,
    BudgetExhaustedError,
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
from .react import ReactAgent
from .tools import ToolRegistry, tool

__all__ = [
    # Main classes
    "ReactAgent",
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
    # Version
    "__version__",
]
