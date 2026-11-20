"""`scripts/bench_common.py` keeps stdlib copies of two statistics so it stays offline.

D-008 of plan 581c2634: importing `fsm_llm` pulls litellm, which opens a socket, so the bench does
not delegate to `fsm_llm.eval.stats`. This row keeps the two copies from drifting.
"""

from __future__ import annotations

import importlib.util

from fsm_llm.eval import stats
from tests.test_fsm_llm_regression.support import REPO


def _bench_common():
    spec = importlib.util.spec_from_file_location(
        "_bench_common_parity", REPO / "scripts" / "bench_common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_two_statistics_equal_their_eval_twins() -> None:
    bc = _bench_common()
    for n in (0, 1, 5, 40):
        for k in range(n + 1):
            assert bc.wilson_ci(k, n) == stats.wilson_ci(k, n)
            assert bc.wilson_ci(k, n, 2.576) == stats.wilson_ci(k, n, 2.576)
    for table in [(0, 1, 0, 1), (1, 5, 3, 5), (3, 10, 7, 10), (13, 17, 2, 9)]:
        assert bc.fisher_exact_two_sided(*table) == stats.fisher_exact_two_sided(*table)
