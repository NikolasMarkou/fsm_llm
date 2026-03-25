from __future__ import annotations

"""
FSM-LLM Meta-Agent
==================

Interactively builds FSMs, Workflows, and Agents through
adaptive conversation. Asks the user questions until the
artifact is fully specified, validated, and ready to use.

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

__all__ = [
    # Main class
    "MetaAgent",
    # Builders
    "ArtifactBuilder",
    "FSMBuilder",
    "WorkflowBuilder",
    "AgentBuilder",
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
