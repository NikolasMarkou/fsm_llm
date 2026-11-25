"""
Binomial statistics for evaluation results (stdlib only, no scipy).

``wilson_ci`` was moved verbatim from the bench helper module
``scripts/bench_common.py`` (deleted with the synthesis bench, plan baa37f40
D-012; see git history). ``fisher_exact_two_sided`` lived here until plan
c720601d step 17 deleted it: it had no caller in ``src``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95% default); math-only, scipy is not in venv."""
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"impossible count: k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p, zz = k / n, z * z
    denom = 1 + zz / n
    center = (p + zz / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pass_rate(k: int, n: int) -> dict[str, Any]:
    """Summarise ``k`` passes out of ``n`` trials as a JSON-ready dict.

    Interface contract (callers: conversation-case reports, any new scorer):
        - Returns ``{"k": k, "n": n, "rate": k / n, "wilson_ci": [lo, hi]}``
          with the 95% Wilson interval. ``n == 0`` gives ``rate`` 0.0 and the
          vacuous interval ``[0.0, 1.0]``, so an empty result never reads as
          a measured zero.
        - Raises ``ValueError`` on impossible counts (``k < 0``, ``n < 0``,
          ``k > n``), same as ``wilson_ci``.
    """
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "wilson_ci": [lo, hi]}


def below_percent(k: int, n: int, percent: float) -> bool:
    """Whether ``k`` out of ``n`` is strictly below ``percent`` (0-100), exactly.

    Interface contract (callers: both CLI ``--fail-under`` checks):
        - Compares ``100 * k < percent * n`` in rationals, with ``percent``
          taken as the decimal it prints as, so 57/100 is not below 57 and
          116/200 is not below 58 (float division gives 56.99999... there).
        - ``n == 0`` counts as a 0% rate: below any positive ``percent``.
    """
    return 100 * k < Fraction(repr(percent)) * n if n else percent > 0
