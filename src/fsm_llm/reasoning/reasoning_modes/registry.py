"""The registry of every reasoning FSM, keyed by name."""

from __future__ import annotations

from typing import Any

from .abductive import abductive_fsm
from .analogical import analogical_fsm
from .analytical import analytical_fsm
from .classifier import classifier_fsm
from .creative import creative_fsm
from .critical import critical_fsm
from .deductive import deductive_fsm
from .hybrid import hybrid_fsm
from .inductive import inductive_fsm
from .orchestrator import orchestrator_fsm
from .simple_calculator import simple_calculator_fsm

ALL_REASONING_FSMS: dict[str, dict[str, Any]] = {
    "orchestrator": orchestrator_fsm,
    "classifier": classifier_fsm,
    "simple_calculator": simple_calculator_fsm,
    "analytical": analytical_fsm,
    "deductive": deductive_fsm,
    "inductive": inductive_fsm,
    "creative": creative_fsm,
    "critical": critical_fsm,
    "abductive": abductive_fsm,
    "analogical": analogical_fsm,
    "hybrid": hybrid_fsm,
}

# Core requires a state's "id" to equal its key in "states": written here once
# instead of beside every state.
for _fsm in ALL_REASONING_FSMS.values():
    _fsm["states"] = {
        state_id: {"id": state_id, **state}
        for state_id, state in _fsm["states"].items()
    }
