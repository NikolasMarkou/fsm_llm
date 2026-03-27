from __future__ import annotations

"""
FSM-LLM Meta-Agent (Compatibility Shim)
========================================

This package has been moved to ``fsm_llm_agents``. All imports are
re-exported from their new location for backward compatibility.

New code should import from ``fsm_llm_agents`` directly::

    from fsm_llm_agents import MetaBuilderAgent, MetaBuilderConfig

Legacy imports still work::

    from fsm_llm_meta import MetaAgent, MetaAgentConfig
"""

import warnings as _warnings

_warnings.warn(
    "fsm_llm_meta is deprecated. Use fsm_llm_agents instead. "
    "Example: from fsm_llm_agents import MetaBuilderAgent",
    DeprecationWarning,
    stacklevel=2,
)

from .__version__ import __version__

# Re-export everything from new location
from fsm_llm_agents.definitions import (
    ArtifactType,
    BuildProgress,
    MetaBuilderConfig as MetaAgentConfig,
    MetaBuilderResult as MetaAgentResult,
)
from fsm_llm_agents.exceptions import (
    BuilderError,
    MetaBuilderError as MetaAgentError,
    MetaValidationError,
    OutputError,
)
from fsm_llm_agents.meta_builder import MetaBuilderAgent as MetaAgent
from fsm_llm_agents.meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    WorkflowBuilder,
)
from fsm_llm_agents.meta_output import format_artifact_json, format_summary, save_artifact
from fsm_llm_agents.meta_tools import (
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
