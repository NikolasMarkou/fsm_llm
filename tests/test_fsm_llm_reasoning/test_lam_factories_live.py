"""Live (real-LLM) smoke tests for ``fsm_llm.stdlib.reasoning.lam_factories``.

Opt-in via ``TEST_REAL_LLM=1``; default-skipped in CI. Runs on
``ollama_chat/qwen3.5:4b`` (override with ``LLM_MODEL``).

Each test verifies Theorem-2 strictly: ``Executor.run(term, env).oracle_calls
== leaf_count``. Only 5 representative cells (per D-S2-004); the bench script
``scripts/bench_reasoning_factories.py`` covers all 10.
"""

from __future__ import annotations

import os

import pytest

from fsm_llm.stdlib.reasoning.lam_factories import (
    analytical_term,
    classifier_term,
    creative_term,
    deductive_term,
    hybrid_term,
)


def _real_llm_env() -> tuple[str, str]:
    model = os.environ.get("LLM_MODEL", "ollama_chat/qwen3.5:4b")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return model, api_key


def _make_oracle_executor():
    from fsm_llm.runtime import Executor, LiteLLMOracle
    from fsm_llm.runtime._litellm import LiteLLMInterface

    model, _ = _real_llm_env()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    return Executor(oracle=oracle)


PROBLEM = "How does compound interest amplify long-term wealth growth?"


@pytest.mark.real_llm
@pytest.mark.slow
class TestSmokeRuns:
    def test_analytical_term_smoke(self) -> None:
        ex = _make_oracle_executor()
        term = analytical_term(
            decomposition_prompt=(
                "Decompose the problem into 2-3 sub-problems.\nProblem: {problem}"
            ),
            analysis_prompt=(
                "Analyze each sub-problem briefly.\n"
                "Problem: {problem}\nDecomposition: {decomposition}"
            ),
            integration_prompt=(
                "Integrate the analyses into a single answer.\n"
                "Problem: {problem}\nAnalysis: {analysis}"
            ),
        )
        result = ex.run(term, {"problem": PROBLEM})
        assert ex.oracle_calls == 3, (
            f"analytical: expected 3 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_deductive_term_smoke(self) -> None:
        ex = _make_oracle_executor()
        term = deductive_term(
            premises_prompt="State 2-3 governing premises.\nProblem: {problem}",
            inference_prompt=(
                "Derive a key inference from the premises.\n"
                "Problem: {problem}\nPremises: {premises}"
            ),
            conclusion_prompt=(
                "Conclude with a one-sentence answer.\n"
                "Problem: {problem}\nInference: {inference}"
            ),
        )
        result = ex.run(term, {"problem": PROBLEM})
        assert ex.oracle_calls == 3, (
            f"deductive: expected 3 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_creative_term_smoke(self) -> None:
        ex = _make_oracle_executor()
        term = creative_term(
            divergence_prompt=(
                "Diverge — list 3 unconventional angles.\nProblem: {problem}"
            ),
            combination_prompt=(
                "Combine the most promising angles.\n"
                "Problem: {problem}\nDivergence: {divergence}"
            ),
            refinement_prompt=(
                "Refine into a final creative answer.\n"
                "Problem: {problem}\nCombination: {combination}"
            ),
        )
        result = ex.run(term, {"problem": PROBLEM})
        assert ex.oracle_calls == 3, (
            f"creative: expected 3 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_hybrid_term_smoke(self) -> None:
        ex = _make_oracle_executor()
        term = hybrid_term(
            facets_prompt="Identify 2-3 distinct facets.\nProblem: {problem}",
            strategies_prompt=(
                "Pick a reasoning strategy per facet.\n"
                "Problem: {problem}\nFacets: {facets}"
            ),
            execution_prompt=(
                "Execute the strategies briefly.\n"
                "Problem: {problem}\nStrategies: {strategies}"
            ),
            integration_prompt=(
                "Integrate the executions into a single answer.\n"
                "Problem: {problem}\nExecution: {execution}"
            ),
        )
        result = ex.run(term, {"problem": PROBLEM})
        assert ex.oracle_calls == 4, (
            f"hybrid: expected 4 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_classifier_term_smoke(self) -> None:
        ex = _make_oracle_executor()
        term = classifier_term(
            domain_prompt=(
                "Identify the problem domain in one phrase.\nProblem: {problem}"
            ),
            structure_prompt=(
                "Describe the problem structure briefly.\n"
                "Problem: {problem}\nDomain: {domain}"
            ),
            needs_prompt=(
                "List the reasoning needs in 1-2 lines.\n"
                "Problem: {problem}\nStructure: {structure}"
            ),
            recommendation_prompt=(
                "Recommend ONE strategy from "
                "[analytical, deductive, inductive, abductive, analogical, "
                "creative, critical, hybrid, calculator]. Output the name only.\n"
                "Problem: {problem}\nNeeds: {needs}"
            ),
        )
        result = ex.run(term, {"problem": PROBLEM})
        assert ex.oracle_calls == 4, (
            f"classifier: expected 4 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None
