"""Compatibility shim — use fsm_llm_agents.definitions instead."""
from fsm_llm_agents.definitions import (  # noqa: F401
    ArtifactType,
    BuildProgress,
    MetaBuilderConfig as MetaAgentConfig,
    MetaBuilderResult as MetaAgentResult,
)
