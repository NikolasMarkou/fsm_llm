"""
Tool Decorator -- λ-DSL twin
=============================

The original ``examples/agents/tool_decorator`` demos the ``@tool``
decorator and ``Annotated[T, "desc"]`` schema inference. That novelty
lives at the *tool-registration* layer; at the λ layer it surfaces
only as the dispatcher routing a flat ``query`` string. We preserve
the original task (Fahrenheit → Celsius conversion) and return a
structured ``ConversionResult``.

Canonical S1 ReAct flatten (decide → tool_dispatch → synth), 2 oracle
calls per run.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/tool_decorator/run.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import make_tool_dispatcher, run_pipeline
from fsm_llm.lam import app, leaf, let_, var

SCHEMA_DECISION = "examples.pipeline.tool_decorator.schemas.ToolDecision"
SCHEMA_FINAL = "examples.pipeline.tool_decorator.schemas.ConversionResult"

TASK = "Convert 100 degrees Fahrenheit to Celsius."

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_num(s: str, default: float = 0.0) -> float:
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else default


def convert_temperature(params: dict) -> str:
    """Convert between Celsius/Fahrenheit/Kelvin from a flat string spec."""
    q = str(params.get("query", "")).lower()
    val = _first_num(q)
    if "fahrenheit" in q or " f " in f" {q} ":
        celsius = (val - 32) * 5 / 9
        if "kelvin" in q:
            return f"{val}F = {celsius + 273.15:.2f}K"
        return f"{val}F = {celsius:.2f}C"
    if "kelvin" in q:
        celsius = val - 273.15
        if "fahrenheit" in q:
            return f"{val}K = {celsius * 9 / 5 + 32:.2f}F"
        return f"{val}K = {celsius:.2f}C"
    # default: celsius -> ?
    if "fahrenheit" in q or "to f" in q:
        return f"{val}C = {val * 9 / 5 + 32:.2f}F"
    if "kelvin" in q or "to k" in q:
        return f"{val}C = {val + 273.15:.2f}K"
    return f"{val}C = {val:.2f}C"


def convert_distance(params: dict) -> str:
    """Convert between km/miles/m/ft from a flat string spec."""
    q = str(params.get("query", "")).lower()
    val = _first_num(q)
    table = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
    }
    for (src, dst), factor in table.items():
        if src in q and dst in q:
            return f"{val} {src} = {val * factor:.4f} {dst}"
    return f"Distance conversion not parsed from {q!r}"


TOOLS = {
    "convert_temperature": convert_temperature,
    "convert_distance": convert_distance,
}


def build_term():
    decide = leaf(
        template=(
            "Pick a tool for the task. tool_name must be one of: "
            "convert_temperature, convert_distance. "
            "query is a flat string describing the conversion "
            "(e.g. '100 fahrenheit to celsius').\n"
            "Task: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_DECISION,
    )
    synth = leaf(
        template=(
            "Return a structured conversion result. "
            "Fill value (number), unit (target unit symbol like C, F, K, "
            "km, miles), and a one-line explanation.\n"
            "Task: {task}\n"
            "Decision: {decision}\n"
            "Tool result: {observation}"
        ),
        input_vars=("task", "decision", "observation"),
        schema_ref=SCHEMA_FINAL,
    )
    return let_(
        "decision", decide,
        let_("observation", app(var("tool_dispatch"), var("decision")), synth),
    )


def checks(result, error, oracle_calls):
    is_dict = isinstance(result, dict)
    explanation_ok = is_dict and len(str(result.get("explanation", ""))) > 5
    unit_ok = is_dict and len(str(result.get("unit", ""))) > 0
    return {
        "answer_present": explanation_ok,
        "iterations_ok": oracle_calls >= 1,
        "tools_called": error is None and oracle_calls >= 2 and unit_ok,
    }


def main():
    env = {"task": TASK, "tool_dispatch": make_tool_dispatcher(TOOLS)}
    return run_pipeline(
        build_term(), env, checks_fn=checks, title="Tool Decorator (λ-DSL)"
    )


if __name__ == "__main__":
    main()
