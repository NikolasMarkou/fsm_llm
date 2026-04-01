"""
ReAct → Structured Output → PromptChain Pipeline
==================================================

Demonstrates composing three agent patterns in sequence:
  1. ReAct agent gathers data via tools
  2. Structured output validates the data into a Pydantic model
  3. PromptChain agent transforms the structured data into a report

This shows how to build complex multi-agent pipelines where each
stage has a different reasoning pattern optimized for its purpose.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/react_structured_pipeline/run.py
"""

import json
import os

from pydantic import BaseModel, Field

from fsm_llm_agents import (
    AgentConfig,
    ChainStep,
    PromptChainAgent,
    ReactAgent,
    ToolRegistry,
    tool,
)

# ── Stage 1: Data gathering tools ──


@tool
def get_city_stats(city: str) -> str:
    """Get demographic and economic statistics for a city."""
    cities = {
        "austin": json.dumps(
            {
                "city": "Austin",
                "state": "Texas",
                "population": 1028225,
                "median_income": 75752,
                "cost_of_living_index": 103.2,
                "unemployment_rate": 3.1,
                "top_industries": [
                    "Technology",
                    "Government",
                    "Education",
                    "Healthcare",
                ],
                "tech_companies": 5500,
            }
        ),
        "denver": json.dumps(
            {
                "city": "Denver",
                "state": "Colorado",
                "population": 713252,
                "median_income": 72661,
                "cost_of_living_index": 107.8,
                "unemployment_rate": 3.4,
                "top_industries": ["Technology", "Energy", "Aerospace", "Healthcare"],
                "tech_companies": 4200,
            }
        ),
    }
    for key, value in cities.items():
        if key in city.lower():
            return value
    return json.dumps({"city": city, "population": 500000, "median_income": 60000})


@tool
def get_quality_of_life(city: str) -> str:
    """Get quality of life metrics for a city."""
    data = {
        "austin": "Austin QoL: climate score 7/10, outdoor recreation 9/10, food scene 9/10, traffic 4/10, air quality 7/10.",
        "denver": "Denver QoL: climate score 8/10, outdoor recreation 10/10, food scene 8/10, traffic 6/10, air quality 6/10.",
    }
    for key, value in data.items():
        if key in city.lower():
            return value
    return f"{city} QoL: average scores across categories."


# ── Stage 2: Structured output schema ──


class CityComparison(BaseModel):
    """Structured comparison of two cities."""

    city_a: str = Field(description="First city name")
    city_b: str = Field(description="Second city name")
    population_winner: str = Field(description="City with larger population")
    income_winner: str = Field(description="City with higher median income")
    cost_winner: str = Field(description="City with lower cost of living")
    overall_recommendation: str = Field(description="Which city is recommended and why")
    confidence: float = Field(
        description="Confidence in recommendation 0-1", ge=0, le=1
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("ReAct → Structured → PromptChain Pipeline")
    print("=" * 60)
    print(f"Model: {model}")
    print("Pipeline: Data Gathering → Validation → Report Generation")
    print("-" * 60)

    # ── Stage 1: ReAct Data Gathering ──
    print("\n[Stage 1] ReAct — Gathering city data...")
    registry = ToolRegistry()
    registry.register(get_city_stats._tool_definition)
    registry.register(get_quality_of_life._tool_definition)

    react_config = AgentConfig(model=model, max_iterations=5, temperature=0.5)
    react_agent = ReactAgent(tools=registry, config=react_config)

    gather_task = (
        "Compare Austin and Denver as cities to live in. "
        "Use get_city_stats for demographics and get_quality_of_life for livability. "
        "Gather data for both cities and summarize the comparison."
    )

    try:
        gather_result = react_agent.run(gather_task)
        raw_data = gather_result.answer
        print(f"  Data gathered: {gather_result.success}")
        print(f"  Tools used: {gather_result.tools_used}")
        print(f"  Raw data preview: {raw_data[:150]}...")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # ── Stage 2: Structured Validation ──
    print("\n[Stage 2] Structured Output — Validating data...")

    @tool
    def validate_data(text: str) -> str:
        """Validate and summarize city comparison data."""
        return f"Data validated: {text[:200]}"

    struct_registry = ToolRegistry()
    struct_registry.register(validate_data._tool_definition)

    struct_config = AgentConfig(
        model=model,
        max_iterations=3,
        temperature=0.3,
        output_schema=CityComparison,
    )
    struct_agent = ReactAgent(tools=struct_registry, config=struct_config)

    struct_task = (
        f"Based on this city comparison data, create a structured comparison:\n\n"
        f"{raw_data[:1500]}\n\n"
        f"Return JSON with fields: city_a, city_b, population_winner, "
        f"income_winner, cost_winner, overall_recommendation, confidence (0-1)."
    )

    try:
        struct_result = struct_agent.run(struct_task)
        print(f"  Validation: {struct_result.success}")

        if struct_result.structured_output:
            comparison = struct_result.structured_output
            print(f"  Population winner: {comparison.population_winner}")
            print(f"  Income winner: {comparison.income_winner}")
            print(f"  Recommendation: {comparison.overall_recommendation[:80]}")
            structured_summary = (
                f"City comparison: {comparison.city_a} vs {comparison.city_b}. "
                f"Population winner: {comparison.population_winner}. "
                f"Income winner: {comparison.income_winner}. "
                f"Cost winner: {comparison.cost_winner}. "
                f"Recommendation: {comparison.overall_recommendation}"
            )
        else:
            structured_summary = struct_result.answer
            print(f"  Fallback to raw: {structured_summary[:100]}")
    except Exception as e:
        print(f"  Error: {e}")
        structured_summary = raw_data

    # ── Stage 3: PromptChain Report ──
    print("\n[Stage 3] PromptChain — Generating polished report...")
    chain = [
        ChainStep(
            step_id="expand",
            name="Expand Analysis",
            extraction_instructions=(
                "Extract as JSON:\n"
                '- "detailed_analysis": expanded analysis text\n'
                '- "pros_cons": dict with city names as keys, each having pros and cons lists'
            ),
            response_instructions=(
                "Expand the structured comparison into a detailed analysis. "
                "For each city, list specific pros and cons based on the data. "
                "Include practical considerations for someone relocating."
            ),
        ),
        ChainStep(
            step_id="report",
            name="Final Report",
            extraction_instructions=(
                "Extract as JSON:\n"
                '- "report": the final polished report text\n'
                '- "executive_summary": 2-3 sentence summary'
            ),
            response_instructions=(
                "Write a polished relocation guide based on the analysis. "
                "Include an executive summary, detailed comparison, and "
                "clear recommendation with actionable next steps."
            ),
        ),
    ]

    chain_config = AgentConfig(model=model, max_iterations=8, temperature=0.5)
    chain_agent = PromptChainAgent(chain=chain, config=chain_config)

    report_task = (
        f"Create a relocation guide based on this analysis:\n\n{structured_summary}"
    )

    try:
        report_result = chain_agent.run(report_task)

        print(f"\n{'=' * 60}")
        print("FINAL RELOCATION GUIDE")
        print("=" * 60)
        print(report_result.answer)
        print(
            f"\nPipeline success: {all([gather_result.success, struct_result.success, report_result.success])}"
        )
        total_iterations = (
            gather_result.iterations_used
            + struct_result.iterations_used
            + report_result.iterations_used
        )
        print(f"Total iterations across 3 stages: {total_iterations}")
    except Exception as e:
        print(f"  Error: {e}")
        report_result = None

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "gather_answer_present": gather_result.answer is not None
        and len(str(gather_result.answer)) > 10,
        "gather_tools_called": len(gather_result.tools_used) > 0,
        "struct_answer_present": struct_result.success,
        "report_answer_present": report_result is not None
        and report_result.answer is not None
        and len(str(report_result.answer)) > 10,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
