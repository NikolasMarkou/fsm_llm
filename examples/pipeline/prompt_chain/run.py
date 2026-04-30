"""
Prompt Chain Pipeline Example — λ-DSL twin of ``examples/agents/prompt_chain``
================================================================================

Demonstrates docs/lambda.md M4: a linear λ-term that threads three structured
LLM leaves (research → draft → polish), each schema-validated via a Pydantic
model. No ``fsm_llm_agents`` Agent classes — the term is built inline from
``fsm_llm.lam`` primitives.

Pipeline shape (sugar form):

    let_("research_out", leaf_research,
      let_("draft_out",  leaf_draft,
        leaf_polish))

Oracle-call equivalence: 3 leaf invocations, matching
``examples/agents/prompt_chain/run.py``'s 3-step chain.

Run::

    export OPENAI_API_KEY=your-key-here
    python examples/pipeline/prompt_chain/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/prompt_chain/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make the project root importable so dotted ``schema_ref`` strings resolve
# regardless of cwd. See plan decision D-006.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, leaf, let_

SCHEMA_RES = "examples.pipeline.prompt_chain.schemas.ResearchOut"
SCHEMA_DRA = "examples.pipeline.prompt_chain.schemas.DraftOut"
SCHEMA_POL = "examples.pipeline.prompt_chain.schemas.PolishOut"

TASK = (
    "Write a short essay (200-300 words) arguing why learning to code "
    "is valuable even for people who don't plan to become programmers."
)

RESEARCH_TEMPLATE = (
    "You are a research assistant.\n"
    "Topic: {topic}\n\n"
    "Research the topic thoroughly. Identify the most important facts, "
    "arguments, and perspectives. Return a JSON object with these fields:\n"
    "- key_points: list of 3-5 key facts or arguments\n"
    "- sources_mentioned: list of any sources or evidence cited "
    "(may be empty)\n"
    "- main_thesis: one-sentence summary of the core argument\n"
)

DRAFT_TEMPLATE = (
    "You are a drafting assistant.\n"
    "Topic: {topic}\n"
    "Research brief (JSON): {research_out}\n\n"
    "Using the research above, write a well-structured draft (intro, "
    "body, conclusion) that uses the key points and thesis from the "
    "research. Return a JSON object with:\n"
    "- outline_sections: list of section titles\n"
    "- word_count_estimate: integer estimated word count\n"
    "- draft_text: the full draft text as a single string\n"
)

POLISH_TEMPLATE = (
    "You are a copy-editor.\n"
    "Topic: {topic}\n"
    "Research brief (JSON): {research_out}\n"
    "Draft (JSON): {draft_out}\n\n"
    "Polish the draft. Improve clarity, flow, and persuasiveness. Fix "
    "awkward phrasing. Return a JSON object with:\n"
    "- final_text: the polished final text as a single string\n"
    "- improvements_made: list of improvements over the draft\n"
    "- confidence: float in 0..1 reflecting quality of the final text\n"
)


def build_term() -> Any:
    """Build the chain term: let research → let draft → polish."""
    leaf_research = leaf(
        template=RESEARCH_TEMPLATE,
        input_vars=("topic",),
        schema_ref=SCHEMA_RES,
    )
    leaf_draft = leaf(
        template=DRAFT_TEMPLATE,
        input_vars=("topic", "research_out"),
        schema_ref=SCHEMA_DRA,
    )
    leaf_polish = leaf(
        template=POLISH_TEMPLATE,
        input_vars=("topic", "research_out", "draft_out"),
        schema_ref=SCHEMA_POL,
    )
    return let_(
        "research_out",
        leaf_research,
        let_("draft_out", leaf_draft, leaf_polish),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Task: {TASK}")
    print(f"Model: {model}")
    print("Steps: Research -> Draft -> Polish")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    term = build_term()
    env = {"topic": TASK}

    error: Exception | None = None
    final: dict[str, Any] | None = None
    try:
        final = ex.run(term, env)
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if final is not None:
        final_text = final.get("final_text", "")
        print(f"\nFinal output:\n{final_text}")
        print(f"\nOracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    final_text = (final or {}).get("final_text", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (
            error is None
            and final is not None
            and isinstance(final_text, str)
            and len(final_text) > 10
        ),
        "iterations_ok": ex.oracle_calls >= 1,
        "steps_completed": ex.oracle_calls >= 3,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} "
        f"({100 * extracted // len(checks)}%)"
    )


if __name__ == "__main__":
    main()
