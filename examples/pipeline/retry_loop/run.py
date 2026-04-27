"""
Retry Loop — bounded retry via stdlib.workflows.retry_term
=============================================================

Demonstrates ``fsm_llm.stdlib.workflows.retry_term``: a fix-wrapped
loop that calls a host-callable body, checks a host-callable success
predicate, and either returns or recurses. Theorem-2: 0 oracle calls
(no Leaf nodes). The success predicate closes over an attempt counter
that succeeds on attempt 2 — demonstrating the bounded retry behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.stdlib.workflows import retry_term

MAX_ATTEMPTS = 3
SUCCESS_ON_ATTEMPT = 2  # body returns "ok" on this attempt; "fail" before

# Mutable attempt counter — closed over by `body_fn` and `success_fn`.
_ATTEMPTS = {"count": 0}


def body_fn(seed: Any) -> dict[str, Any]:
    _ATTEMPTS["count"] += 1
    n = _ATTEMPTS["count"]
    if n >= SUCCESS_ON_ATTEMPT:
        return {"status": "ok", "attempt": n, "seed": seed}
    return {"status": "fail", "attempt": n, "seed": seed}


def success_fn(attempt_result: Any) -> str:
    if isinstance(attempt_result, dict) and attempt_result.get("status") == "ok":
        return "true"
    return "false"


def build_term():
    return retry_term(
        body_var="retry_body",
        success_var="retry_success",
        input_var="seed",
        max_attempts=MAX_ATTEMPTS,
    )


def checks(result, error, oracle_calls):
    is_ok = isinstance(result, dict) and result.get("status") == "ok"
    final_attempt = result.get("attempt") if isinstance(result, dict) else None
    return {
        "result_status_ok": bool(is_ok),
        "succeeded_on_expected_attempt": final_attempt == SUCCESS_ON_ATTEMPT,
        "zero_oracle_calls": oracle_calls == 0,
        "no_error": error is None,
    }


def main() -> int:
    # Reset counter on each run.
    _ATTEMPTS["count"] = 0
    return run_pipeline(
        build_term(),
        env={
            "seed": "deploy-attempt",
            "retry_body": body_fn,
            "retry_success": success_fn,
        },
        checks_fn=checks,
        title="Retry Loop (stdlib.workflows.retry_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
