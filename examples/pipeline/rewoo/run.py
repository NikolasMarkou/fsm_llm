"""Rewoo — λ-DSL twin (S2 REWOO shape).

λ-twin: plan_leaf → plan_executor → synth_leaf. 2 oracle calls.
Original at examples/agents/rewoo/ untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import make_plan_executor, run_pipeline
from fsm_llm.runtime import app, leaf, let, var

SCHEMA_PLAN = "examples.pipeline.rewoo.schemas.Plan"
SCHEMA_FINAL = "examples.pipeline.rewoo.schemas.FinalAnswer"

TASK = "What is the population of France divided by 2?"


def search(params: dict) -> str:
    return f"Search result for {params.get('query', params.get('args'))!r}: data found."


def calculate(params: dict) -> str:
    expr = str(params.get("expression", params.get("args", params.get("query", ""))))
    try:
        return f"{expr} = {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"calc error: {e}"


def lookup(params: dict) -> str:
    return f"Lookup {params.get('topic', params.get('query', params.get('args')))!r}: facts found."


TOOLS = {"search": search, "calculate": calculate, "lookup": lookup}


def _normalize_plan(plan):
    """Ensure plan steps carry a ``query`` key so the dispatcher can route them."""
    if not isinstance(plan, dict):
        return plan
    steps = plan.get("steps") or []
    norm = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        s2 = dict(s)
        if "args" in s2 and "query" not in s2:
            s2["query"] = s2["args"]
        norm.append(s2)
    return {"steps": norm}


def build_term():
    plan_l = leaf(
        template=(
            "Produce a 2-3 step plan. Each step has tool_name (one of: "
            "search, calculate, lookup), args (one string), and label (#E1, #E2, ...).\n"
            "Task: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_PLAN,
    )
    synth = leaf(
        template=(
            "Synthesize the final answer using the plan and evidence.\n"
            "Task: {task}\nPlan: {plan}\nEvidence: {evidence}"
        ),
        input_vars=("task", "plan", "evidence"),
        schema_ref=SCHEMA_FINAL,
    )
    return let(
        "plan",
        plan_l,
        let("evidence", app(var("plan_exec"), var("plan")), synth),
    )


def checks(result, error, oracle_calls):
    ok = isinstance(result, dict) and len(str(result.get("answer", ""))) > 1
    return {
        "answer_present": ok,
        "iterations_ok": oracle_calls >= 1,
        "plan_executed": error is None and oracle_calls >= 2,
    }


def main():
    plan_exec = make_plan_executor(TOOLS)

    def plan_exec_wrapped(plan):
        return plan_exec(_normalize_plan(plan))

    env = {"task": TASK, "plan_exec": plan_exec_wrapped}
    return run_pipeline(build_term(), env, checks_fn=checks, title="Rewoo (λ-DSL)")


if __name__ == "__main__":
    main()
