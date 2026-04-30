"""
Structured Output -- λ-DSL twin
================================

Single tool-call ReAct flatten: decide → tool_dispatch → synthesize,
where the synth_leaf enforces a Pydantic ``MovieRecommendation`` schema
(the original example's whole point: structured-output enforcement).

Original: ``examples/agents/structured_output`` uses ``ReactAgent`` with
``output_schema=MovieRecommendation`` and ``max_iterations=6``. λ-twin
runs depth-1 (1 tool call); the structured contract is enforced at the
synth_leaf via ``schema_ref``. 2 oracle calls per run.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/structured_output/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import make_tool_dispatcher, run_pipeline
from fsm_llm.runtime import app, leaf, let_, var

SCHEMA_DECISION = "examples.pipeline.structured_output.schemas.ToolDecision"
SCHEMA_FINAL = "examples.pipeline.structured_output.schemas.MovieRecommendation"

TASK = (
    "Recommend a great sci-fi movie. "
    "Return your answer as a movie recommendation with title, year, genre, "
    "reason, and rating fields."
)


def search_movies(params: dict) -> str:
    q = str(params.get("query", "")).lower()
    movies = {
        "sci-fi": (
            '{"title": "Blade Runner 2049", "year": 2017, "genre": "Sci-Fi", '
            '"reason": "Stunning visuals and deep themes about humanity", '
            '"rating": 8.5}'
        ),
        "action": (
            '{"title": "Mad Max: Fury Road", "year": 2015, "genre": "Action", '
            '"reason": "Non-stop thrilling action with great world-building", '
            '"rating": 8.8}'
        ),
        "drama": (
            '{"title": "Parasite", "year": 2019, "genre": "Drama", '
            '"reason": "Masterful social commentary with unexpected twists", '
            '"rating": 9.0}'
        ),
        "comedy": (
            '{"title": "The Grand Budapest Hotel", "year": 2014, '
            '"genre": "Comedy", '
            '"reason": "Witty humor with beautiful cinematography", '
            '"rating": 8.1}'
        ),
        "horror": (
            '{"title": "Get Out", "year": 2017, "genre": "Horror", '
            '"reason": "Intelligent social horror that redefines the genre", '
            '"rating": 8.2}'
        ),
    }
    for key, value in movies.items():
        if key in q:
            return value
    return (
        '{"title": "Inception", "year": 2010, "genre": "Sci-Fi", '
        '"reason": "Mind-bending thriller with layered storytelling", '
        '"rating": 8.7}'
    )


TOOLS = {"search_movies": search_movies}


def build_term():
    decide = leaf(
        template=(
            "Pick a tool to find a movie matching the task. "
            "tool_name must be 'search_movies'. "
            "query is a genre keyword (e.g. sci-fi, action, drama).\n"
            "Task: {task}"
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_DECISION,
    )
    synth = leaf(
        template=(
            "Return a structured movie recommendation based on the search "
            "result. Fill all fields: title, year, genre, reason, rating "
            "(0-10).\n"
            "Task: {task}\n"
            "Decision: {decision}\n"
            "Search result: {observation}"
        ),
        input_vars=("task", "decision", "observation"),
        schema_ref=SCHEMA_FINAL,
    )
    return let_(
        "decision",
        decide,
        let_("observation", app(var("tool_dispatch"), var("decision")), synth),
    )


def checks(result, error, oracle_calls):
    is_dict = isinstance(result, dict)
    title_ok = is_dict and len(str(result.get("title", ""))) > 0
    rating_val = result.get("rating") if is_dict else None
    rating_ok = isinstance(rating_val, int | float) and 0.0 <= float(rating_val) <= 10.0
    return {
        "answer_present": title_ok,
        "iterations_ok": oracle_calls >= 1,
        "tools_called": error is None and oracle_calls >= 2 and rating_ok,
    }


def main():
    env = {"task": TASK, "tool_dispatch": make_tool_dispatcher(TOOLS)}
    return run_pipeline(
        build_term(), env, checks_fn=checks, title="Structured Output (λ-DSL)"
    )


if __name__ == "__main__":
    main()
