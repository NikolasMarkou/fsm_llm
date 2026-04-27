"""
Linear Chain Factory — first standalone demo of stdlib.workflows.linear_term
==============================================================================

Demonstrates ``fsm_llm.stdlib.workflows.linear_term`` as a stand-alone
pipeline. The factory folds (name, term) pairs into a let-chain of
3 leaves: facts → outline → summary. Theorem-2 strict: 3 oracle calls.

Existing pipeline examples build let-chains inline; this one shows the
named factory equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path for dotted schema_ref resolution.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.lam import leaf
from fsm_llm.stdlib.workflows import linear_term

SCHEMA_FACTS = "examples.pipeline.linear_chain_factory.schemas.FactsOut"
SCHEMA_OUTLINE = "examples.pipeline.linear_chain_factory.schemas.OutlineOut"
SCHEMA_SUMMARY = "examples.pipeline.linear_chain_factory.schemas.SummaryOut"

TOPIC = (
    "Why distributed version control changed software collaboration "
    "compared to centralised systems."
)

FACTS_TEMPLATE = (
    "You are a research assistant.\n"
    "Topic: {topic}\n\n"
    "Identify the 3-5 most important facts about the topic. "
    "Return JSON with:\n"
    "- facts: list of fact strings\n"
    "- domain: one-word area\n"
)

OUTLINE_TEMPLATE = (
    "You are an outliner.\n"
    "Topic: {topic}\n"
    "Facts (JSON): {facts}\n\n"
    "Turn the facts into a 3-section outline. Return JSON with:\n"
    "- sections: list of section titles\n"
    "- thesis: one-sentence thesis\n"
)

SUMMARY_TEMPLATE = (
    "You are a writer.\n"
    "Topic: {topic}\n"
    "Facts (JSON): {facts}\n"
    "Outline (JSON): {outline}\n\n"
    "Produce a 3-4 sentence summary. Return JSON with:\n"
    "- summary: the summary text\n"
    "- confidence: 0..1 quality score\n"
)


def build_term():
    """Build via linear_term using (name, term) pairs uniformly.

    The last pair's term becomes the innermost body of the let-chain;
    its bound name is unused outside the term.
    """
    facts_leaf = leaf(
        template=FACTS_TEMPLATE,
        input_vars=("topic",),
        schema_ref=SCHEMA_FACTS,
    )
    outline_leaf = leaf(
        template=OUTLINE_TEMPLATE,
        input_vars=("topic", "facts"),
        schema_ref=SCHEMA_OUTLINE,
    )
    summary_leaf = leaf(
        template=SUMMARY_TEMPLATE,
        input_vars=("topic", "facts", "outline"),
        schema_ref=SCHEMA_SUMMARY,
    )
    # linear_term takes (name, term) pairs; last pair's term is the
    # innermost body of the let-chain.
    return linear_term(
        ("facts", facts_leaf),
        ("outline", outline_leaf),
        ("summary", summary_leaf),
    )


def checks(result, error, oracle_calls):
    has_summary_key = isinstance(result, dict) and "summary" in result
    return {
        "result_is_dict": isinstance(result, dict),
        "summary_key_present": has_summary_key,
        "three_oracle_calls": oracle_calls == 3,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"topic": TOPIC},
        checks_fn=checks,
        title="Linear Chain Factory (stdlib.workflows.linear_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
