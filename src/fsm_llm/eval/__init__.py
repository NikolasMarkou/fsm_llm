"""
FSM-LLM Eval
============

Evaluation tooling for FSM-LLM: the examples evaluator (run every example
script, score it 0-4, write a scorecard; CLI ``fsm-llm-eval examples``), a
layered ``EvalConfig``, binomial statistics (Wilson intervals, Fisher exact
test), append-only result rows, and collision-safe run directories shared with
the harness live bench.

Import it as ``from fsm_llm import eval as fsm_eval`` or import names directly
(``from fsm_llm.eval import wilson_ci``) to avoid shadowing the builtin.

Install:
    pip install fsm-llm[eval]
"""

from __future__ import annotations

# Version info — imported via __version__.py to stay in sync
from .__version__ import __version__
from .config import EvalConfig, load_config, merge_config, resolve_model
from .examples import (
    ExampleReport,
    ExampleResult,
    ExampleTarget,
    discover_examples,
    get_timeout,
    run_example,
    run_examples,
    write_example_log,
    write_scorecard,
)
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
from .scoring import classify_result
from .stats import fisher_exact_two_sided, pass_rate, wilson_ci

__all__ = [
    "__version__",
    "EvalConfig",
    "EvalConfigError",
    "EvalDatasetError",
    "EvalError",
    "ExampleReport",
    "ExampleResult",
    "ExampleTarget",
    "append_row",
    "classify_result",
    "discover_examples",
    "fisher_exact_two_sided",
    "get_timeout",
    "git_commit",
    "git_short_hash",
    "load_config",
    "make_run_dir",
    "merge_config",
    "model_slug",
    "pass_rate",
    "read_rows",
    "resolve_model",
    "run_example",
    "run_examples",
    "utc_now",
    "wilson_ci",
    "write_example_log",
    "write_json",
    "write_scorecard",
]
