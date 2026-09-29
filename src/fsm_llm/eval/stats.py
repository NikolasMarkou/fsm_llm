"""
Binomial statistics for evaluation results (stdlib only, no scipy).

``wilson_ci`` and ``fisher_exact_two_sided`` were moved verbatim from
``scripts/harness_bench.py``, which keeps its own stdlib copies so it stays
offline (D-008 of plan 581c2634); ``tests/test_fsm_llm_eval/test_bench_parity.py``
keeps the copies equal.
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


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Fisher exact p for [[k1,n1-k1],[k2,n2-k2]]: sum of pmf <= observed pmf."""
    for k, n in ((k1, n1), (k2, n2)):
        if n <= 0 or k < 0 or k > n:
            raise ValueError(f"impossible arm: k={k}, n={n}")
    r1, denom = k1 + k2, math.comb(n1 + n2, k1 + k2)

    def pmf(a: int) -> float:
        return math.comb(n1, a) * math.comb(n2, r1 - a) / denom

    p_obs = pmf(k1)
    span = range(max(0, r1 - n2), min(r1, n1) + 1)
    return min(1.0, sum(pmf(a) for a in span if pmf(a) <= p_obs * (1 + 1e-9)))


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
