from __future__ import annotations

"""
FSM-LLM Meta-Agent
==================

Interactively builds FSMs, Workflows, and Agents using a
ReactAgent-powered build phase that autonomously constructs
artifacts from user requirements.

Basic Usage::

    from fsm_llm_meta import MetaAgent

    agent = MetaAgent()
    result = agent.run_interactive()
    print(result.artifact_json)

Turn-by-turn Usage::

    from fsm_llm_meta import MetaAgent

    agent = MetaAgent()
    response = agent.start()

    while not agent.is_complete():
        user_input = input("> ")
        response = agent.send(user_input)
        print(response)

    result = agent.get_result()
"""

from .__version__ import __version__
from .agent import MetaAgent
from .builders import AgentBuilder, ArtifactBuilder, FSMBuilder, WorkflowBuilder
from .definitions import (
    ArtifactType,
    BuildProgress,
    MetaAgentConfig,
    MetaAgentResult,
)
from .exceptions import (
    BuilderError,
    MetaAgentError,
    MetaValidationError,
    OutputError,
)
from .output import format_artifact_json, format_summary, save_artifact
from .tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)

__all__ = [
    # Main class
    "MetaAgent",
    # Builders
    "ArtifactBuilder",
    "FSMBuilder",
    "WorkflowBuilder",
    "AgentBuilder",
    # Tool factories
    "create_builder_tools",
    "create_fsm_tools",
    "create_workflow_tools",
    "create_agent_tools",
    # Models
    "ArtifactType",
    "BuildProgress",
    "MetaAgentConfig",
    "MetaAgentResult",
    # Exceptions
    "MetaAgentError",
    "BuilderError",
    "MetaValidationError",
    "OutputError",
    # Output utilities
    "format_artifact_json",
    "format_summary",
    "save_artifact",
    # Version
    "__version__",
]
