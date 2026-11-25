"""The simulation layer: the generic simulator under G6b, repair replay and eval cases.

Imports core, llm, runtime, engine and the kernel only, never a consumer package.
"""

from __future__ import annotations

from .stats import below_percent, fisher_exact_two_sided, pass_rate, wilson_ci

__all__ = ["below_percent", "fisher_exact_two_sided", "pass_rate", "wilson_ci"]
