"""Compatibility shim — use fsm_llm_agents.exceptions instead."""
from fsm_llm_agents.exceptions import (  # noqa: F401
    BuilderError,
    MetaBuilderError as MetaAgentError,
    MetaValidationError,
    OutputError,
)
