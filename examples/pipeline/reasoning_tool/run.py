"""Reasoning Tool — λ-DSL twin (S1 ReAct shape).

λ-twin: decide_leaf → tool_dispatch → synth_leaf. 2 oracle calls.
Original at examples/agents/reasoning_tool/ untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import make_tool_dispatcher, run_pipeline
from fsm_llm.lam import app, leaf, let_, var

SCHEMA_DECISION = "examples.pipeline.reasoning_tool.schemas.ToolDecision"
SCHEMA_FINAL = "examples.pipeline.reasoning_tool.schemas.FinalAnswer"

TASK = 'Solve: if x + 5 = 12, what is x?'


def search(params: dict) -> str:
    return f"Search results for {params.get('query')!r}: ok."


def calculate(params: dict) -> str:
    expr = str(params.get("expression", params.get("query", "")))
    try:
        return f"{expr} = {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"calc error: {e}"


def lookup(params: dict) -> str:
    return f"Looked up {params.get('query')!r}: found."


TOOLS = {"search": search, "calculate": calculate, "lookup": lookup}


def build_term():
    decide = leaf(
        template=(
            "Pick one tool. tool_name in: calculate, search, lookup. "
            "query is the argument.\nTask: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_DECISION,
    )
    synth = leaf(
        template=(
            "Provide the final answer.\nTask: {task}\nDecision: {decision}\n"
            "Observation: {observation}"
        ),
        input_vars=("task", "decision", "observation"),
        schema_ref=SCHEMA_FINAL,
    )
    return let_(
        "decision", decide,
        let_("observation", app(var("tool_dispatch"), var("decision")), synth),
    )


def checks(result, error, oracle_calls):
    ok = isinstance(result, dict) and len(str(result.get("answer", ""))) > 1
    return {
        "answer_present": ok,
        "iterations_ok": oracle_calls >= 1,
        "tools_called": error is None and oracle_calls >= 2,
    }


def main():
    env = {"task": TASK, "tool_dispatch": make_tool_dispatcher(TOOLS)}
    return run_pipeline(build_term(), env, checks_fn=checks, title='Reasoning Tool (λ-DSL)')


if __name__ == "__main__":
    main()
