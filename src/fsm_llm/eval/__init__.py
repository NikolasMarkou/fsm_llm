"""
FSM-LLM Eval
============

Evaluation tooling for FSM-LLM: binomial statistics (Wilson intervals, Fisher
exact test), append-only result rows, and collision-safe run directories,
shared by the example-run evaluator and the harness live bench.

Import it as ``from fsm_llm import eval as fsm_eval`` or import names directly
(``from fsm_llm.eval import wilson_ci``) to avoid shadowing the builtin.

Install:
    pip install fsm-llm[eval]
"""

from __future__ import annotations

# Version info — imported via __version__.py to stay in sync
from .__version__ import __version__
from .exceptions import EvalConfigError, EvalDatasetError, EvalError
from .records import (
    append_row,
    git_commit,
    git_short_hash,
    make_run_dir,
    model_slug,
    read_rows,
    utc_now,
    write_json,
)
from .stats import fisher_exact_two_sided, pass_rate, wilson_ci

__all__ = [
    "__version__",
    "EvalConfigError",
    "EvalDatasetError",
    "EvalError",
    "append_row",
    "fisher_exact_two_sided",
    "git_commit",
    "git_short_hash",
    "make_run_dir",
    "model_slug",
    "pass_rate",
    "read_rows",
    "utc_now",
    "wilson_ci",
    "write_json",
]
