"""Backward-compat alias. Real implementation: fsm_llm.stdlib.agents.

Module identity is preserved across both names via sys.modules aliasing,
so `fsm_llm_agents.base is fsm_llm.stdlib.agents.base` is True (and the
same for the other 36 submodules). This shim is silent — no
DeprecationWarning is raised.

The shim does NOT alias `cli` (agents ships no `cli` submodule); imports
like `import fsm_llm_agents.cli` continue to raise ModuleNotFoundError.
"""

from __future__ import annotations

import sys as _sys

# Import all 37 real submodules from the canonical home.
from fsm_llm.stdlib.agents import (
    adapt,
    agent_graph,
    base,
    constants,
    debate,
    definitions,
    evaluator_optimizer,
    exceptions,
    fsm_definitions,
    handlers,
    hitl,
    maker_checker,
    mcp,
    memory_tools,
    meta_builder,
    meta_builders,
    meta_cli,
    meta_fsm,
    meta_output,
    meta_prompts,
    meta_tools,
    orchestrator,
    plan_execute,
    prompt_chain,
    prompts,
    react,
    reasoning_react,
    reflexion,
    remote,
    rewoo,
    self_consistency,
    semantic_tools,
    skills,
    sop,
    swarm,
    tools,
    truncation,
)

# Register sys.modules aliases BEFORE re-exporting public symbols so that
# `from fsm_llm_agents.base import BaseAgent` and
# `from fsm_llm_agents import base` resolve to the same module object.
# Critical for the 3 `patch()` sites: semantic_tools.logger,
# semantic_tools.SemanticToolRegistry._get_embedding, base.API.from_definition.
_sys.modules["fsm_llm_agents.adapt"] = adapt
_sys.modules["fsm_llm_agents.agent_graph"] = agent_graph
_sys.modules["fsm_llm_agents.base"] = base
_sys.modules["fsm_llm_agents.constants"] = constants
_sys.modules["fsm_llm_agents.debate"] = debate
_sys.modules["fsm_llm_agents.definitions"] = definitions
_sys.modules["fsm_llm_agents.evaluator_optimizer"] = evaluator_optimizer
_sys.modules["fsm_llm_agents.exceptions"] = exceptions
_sys.modules["fsm_llm_agents.fsm_definitions"] = fsm_definitions
_sys.modules["fsm_llm_agents.handlers"] = handlers
_sys.modules["fsm_llm_agents.hitl"] = hitl
_sys.modules["fsm_llm_agents.maker_checker"] = maker_checker
_sys.modules["fsm_llm_agents.mcp"] = mcp
_sys.modules["fsm_llm_agents.memory_tools"] = memory_tools
_sys.modules["fsm_llm_agents.meta_builder"] = meta_builder
_sys.modules["fsm_llm_agents.meta_builders"] = meta_builders
_sys.modules["fsm_llm_agents.meta_cli"] = meta_cli
_sys.modules["fsm_llm_agents.meta_fsm"] = meta_fsm
_sys.modules["fsm_llm_agents.meta_output"] = meta_output
_sys.modules["fsm_llm_agents.meta_prompts"] = meta_prompts
_sys.modules["fsm_llm_agents.meta_tools"] = meta_tools
_sys.modules["fsm_llm_agents.orchestrator"] = orchestrator
_sys.modules["fsm_llm_agents.plan_execute"] = plan_execute
_sys.modules["fsm_llm_agents.prompt_chain"] = prompt_chain
_sys.modules["fsm_llm_agents.prompts"] = prompts
_sys.modules["fsm_llm_agents.react"] = react
_sys.modules["fsm_llm_agents.reasoning_react"] = reasoning_react
_sys.modules["fsm_llm_agents.reflexion"] = reflexion
_sys.modules["fsm_llm_agents.remote"] = remote
_sys.modules["fsm_llm_agents.rewoo"] = rewoo
_sys.modules["fsm_llm_agents.self_consistency"] = self_consistency
_sys.modules["fsm_llm_agents.semantic_tools"] = semantic_tools
_sys.modules["fsm_llm_agents.skills"] = skills
_sys.modules["fsm_llm_agents.sop"] = sop
_sys.modules["fsm_llm_agents.swarm"] = swarm
_sys.modules["fsm_llm_agents.tools"] = tools
_sys.modules["fsm_llm_agents.truncation"] = truncation

# Mirror the public __all__ from the canonical home, byte-for-byte.
from fsm_llm.stdlib.agents.__version__ import __version__
from fsm_llm.stdlib.agents import create_agent
from fsm_llm.stdlib.agents.adapt import ADaPTAgent
from fsm_llm.stdlib.agents.agent_graph import AgentGraph, AgentGraphBuilder
from fsm_llm.stdlib.agents.base import BaseAgent
from fsm_llm.stdlib.agents.debate import DebateAgent
from fsm_llm.stdlib.agents.definitions import (
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
from fsm_llm.stdlib.agents.evaluator_optimizer import EvaluatorOptimizerAgent
from fsm_llm.stdlib.agents.exceptions import (
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
from fsm_llm.stdlib.agents.hitl import (
    ApprovalCallback,
    ApprovalPolicy,
    EscalationCallback,
    HumanInTheLoop,
)
from fsm_llm.stdlib.agents.maker_checker import MakerCheckerAgent
from fsm_llm.stdlib.agents.mcp import MCPToolProvider
from fsm_llm.stdlib.agents.memory_tools import create_memory_tools
from fsm_llm.stdlib.agents.meta_builder import MetaBuilderAgent
from fsm_llm.stdlib.agents.meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    WorkflowBuilder,
)
from fsm_llm.stdlib.agents.meta_output import (
    format_artifact_json,
    format_summary,
    save_artifact,
)
from fsm_llm.stdlib.agents.meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)
from fsm_llm.stdlib.agents.orchestrator import OrchestratorAgent
from fsm_llm.stdlib.agents.plan_execute import PlanExecuteAgent
from fsm_llm.stdlib.agents.prompt_chain import PromptChainAgent
from fsm_llm.stdlib.agents.react import ReactAgent
from fsm_llm.stdlib.agents.reflexion import ReflexionAgent
from fsm_llm.stdlib.agents.remote import AgentServer, RemoteAgentTool
from fsm_llm.stdlib.agents.rewoo import REWOOAgent
from fsm_llm.stdlib.agents.self_consistency import SelfConsistencyAgent
from fsm_llm.stdlib.agents.semantic_tools import SemanticToolRegistry
from fsm_llm.stdlib.agents.skills import SkillDefinition, SkillLoader
from fsm_llm.stdlib.agents.sop import SOPDefinition, SOPRegistry, load_builtin_sops
from fsm_llm.stdlib.agents.swarm import SwarmAgent
from fsm_llm.stdlib.agents.tools import ToolRegistry, tool

# Preserve the conditional reasoning_react handling so the legacy import
# surface matches canonical under both [agents] and [reasoning,agents] install
# profiles.
_has_reasoning_react = False
try:
    from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent  # noqa: F401

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
