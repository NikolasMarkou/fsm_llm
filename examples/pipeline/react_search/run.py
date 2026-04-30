"""
ReAct Search — λ-DSL twin
==========================

Single tool-call ReAct flatten: decide → tool_dispatch → synthesize.
Original: examples/agents/react_search uses ``ReactAgent`` with up to 8
iterations. λ-twin runs depth-1 (1 tool call) — same UX bar as M4
maker_checker. 2 oracle calls per run.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/react_search/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import make_tool_dispatcher, run_pipeline
from fsm_llm.runtime import app, leaf, let, var

SCHEMA_DECISION = "examples.pipeline.react_search.schemas.ToolDecision"
SCHEMA_FINAL = "examples.pipeline.react_search.schemas.FinalAnswer"

TASK = "What is the population of France?"


def search(params: dict) -> str:
    q = str(params.get("query", "")).lower()
    if "france" in q and "population" in q:
        return "France has a population of approximately 68.4 million (2024)."
    if "japan" in q:
        return "Japan: capital Tokyo; pop ~125M."
    return f"Search results for {params.get('query')!r}: nothing specific."


def calculate(params: dict) -> str:
    expr = str(params.get("expression", params.get("query", "")))
    try:
        return f"{expr} = {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"calc error: {e}"


def lookup(params: dict) -> str:
    topic = str(params.get("topic", params.get("query", ""))).lower()
    facts = {"france": "France: pop 68M, capital Paris, EUR currency."}
    for k, v in facts.items():
        if k in topic:
            return v
    return f"No fact for {topic!r}."


TOOLS = {"search": search, "calculate": calculate, "lookup": lookup}


def build_term():
    decide = leaf(
        template=(
            "Pick one tool for the task. tool_name must be one of: "
            "search, calculate, lookup. query is the argument string.\n"
            "Task: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_DECISION,
    )
    synth = leaf(
        template=(
            "Write the final answer using the tool result.\n"
            "Task: {task}\n"
            "Decision: {decision}\n"
            "Observation: {observation}"
        ),
        input_vars=("task", "decision", "observation"),
        schema_ref=SCHEMA_FINAL,
    )
    return let(
        "decision",
        decide,
        let("observation", app(var("tool_dispatch"), var("decision")), synth),
    )


def checks(result, error, oracle_calls):
    ok = isinstance(result, dict) and len(str(result.get("answer", ""))) > 5
    return {
        "answer_present": ok,
        "iterations_ok": oracle_calls >= 1,
        "tools_called": error is None and oracle_calls >= 2,
    }


def main():
    env = {"task": TASK, "tool_dispatch": make_tool_dispatcher(TOOLS)}
    return run_pipeline(
        build_term(), env, checks_fn=checks, title="ReAct Search (λ-DSL)"
    )


if __name__ == "__main__":
    main()
