"""Orchestrator — λ-DSL twin (S4 memory/orchestrator shape).

λ-twin: 2-leaf chain — context_leaf (recall/classify/decompose) →
answer_leaf. 2 oracle calls. Original at examples/agents/orchestrator/ untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.runtime import leaf, let

SCHEMA_CTX = "examples.pipeline.orchestrator.schemas.Context"
SCHEMA_ANS = "examples.pipeline.orchestrator.schemas.Answer"

TASK = (
    "Route this complex task: analyze a customer support ticket and recommend action."
)


def build_term():
    ctx = leaf(
        template=(
            "Build relevant context for the task. Provide a summary and "
            "2-3 key points as a single string.\nTask: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_CTX,
    )
    ans = leaf(
        template=(
            "Provide the final answer using the context.\n"
            "Task: {task}\nContext: {context}"
        ),
        input_vars=("task", "context"),
        schema_ref=SCHEMA_ANS,
    )
    return let("context", ctx, ans)


def checks(result, error, oracle_calls):
    ok = isinstance(result, dict) and len(str(result.get("answer", ""))) > 1
    return {
        "answer_present": ok,
        "iterations_ok": oracle_calls >= 1,
        "pipeline_completed": error is None and oracle_calls >= 2,
    }


def main():
    return run_pipeline(
        build_term(), {"task": TASK}, checks_fn=checks, title="Orchestrator (λ-DSL)"
    )


if __name__ == "__main__":
    main()
