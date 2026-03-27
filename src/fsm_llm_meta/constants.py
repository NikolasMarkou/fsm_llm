"""Compatibility shim — use fsm_llm_agents.constants instead."""
from fsm_llm_agents.constants import (  # noqa: F401
    DecisionWords,
    MetaBuilderStates as MetaPhases,
    MetaDefaults as Defaults,
    MetaErrorMessages as ErrorMessages,
    MetaLogMessages as LogMessages,
)
