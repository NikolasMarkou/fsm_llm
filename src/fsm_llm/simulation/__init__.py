"""The simulation layer: the generic simulator under G6b, repair replay and eval cases.

Imports core, llm, runtime, engine and the kernel only, never a consumer package.
"""

from __future__ import annotations

from .driver import Trial, play
from .exceptions import CODE_JUDGE_NOT_DISTINCT, SimulationError
from .roles import Roles
from .scenarios import write_scenarios
from .simulator import Budget, Simulator
from .stats import below_percent, pass_rate, wilson_ci

__all__ = [
    "CODE_JUDGE_NOT_DISTINCT",
    "Budget",
    "Roles",
    "SimulationError",
    "Simulator",
    "Trial",
    "below_percent",
    "pass_rate",
    "play",
    "wilson_ci",
    "write_scenarios",
]
