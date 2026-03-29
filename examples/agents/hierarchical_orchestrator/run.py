"""
Hierarchical Orchestrator — Nested Multi-Level Delegation
==========================================================

Demonstrates a hierarchical orchestrator pattern where the top-level
orchestrator delegates to mid-level orchestrators, which in turn
delegate to specialized workers. This creates a tree of agents.

Architecture:
  Top Orchestrator
  ├── Research Lead (orchestrator)
  │   ├── Market researcher (worker)
  │   └── Technical researcher (worker)
  └── Strategy Lead (orchestrator)
      ├── SWOT analyst (worker)
      └── Recommendation writer (worker)

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/hierarchical_orchestrator/run.py
"""

import os

from fsm_llm_agents import AgentConfig, AgentResult, OrchestratorAgent


def market_researcher(subtask: str) -> AgentResult:
    """Market research specialist."""
    s = subtask.lower()
    if any(kw in s for kw in ["market", "size", "growth", "trend"]):
        return AgentResult(
            answer=(
                "Market analysis: The global electric vehicle market reached $500B in 2024, "
                "growing at 25% CAGR. China leads with 60% of sales. Battery costs dropped "
                "to $139/kWh. Key trend: solid-state batteries expected by 2027."
            ),
            success=True,
        )
    if any(kw in s for kw in ["competitor", "player", "company"]):
        return AgentResult(
            answer=(
                "Competitive landscape: Tesla (18% market share), BYD (16%), "
                "Volkswagen Group (8%), Hyundai-Kia (7%), GM (5%). "
                "New entrants: Rivian, Lucid, Chinese brands NIO and XPeng expanding globally."
            ),
            success=True,
        )
    return AgentResult(answer=f"Market research on: {subtask[:100]}", success=True)


def tech_researcher(subtask: str) -> AgentResult:
    """Technology research specialist."""
    s = subtask.lower()
    if any(kw in s for kw in ["battery", "technology", "tech"]):
        return AgentResult(
            answer=(
                "Technology assessment: Current lithium-ion: 250 Wh/kg. "
                "Solid-state promises 500+ Wh/kg. Charging: 350kW fast charging standard. "
                "Autonomous driving: Level 3 available, Level 4 in testing. "
                "V2G (vehicle-to-grid) becoming a revenue opportunity."
            ),
            success=True,
        )
    if any(kw in s for kw in ["infrastructure", "charging"]):
        return AgentResult(
            answer=(
                "Infrastructure: 2.7M public chargers globally. Need 14M by 2030. "
                "Gap: rural areas underserved. Ultra-fast (350kW+) only 5% of network. "
                "Home charging dominates at 80% of daily charging events."
            ),
            success=True,
        )
    return AgentResult(answer=f"Technical research on: {subtask[:100]}", success=True)


def swot_analyst(subtask: str) -> AgentResult:
    """SWOT analysis specialist."""
    return AgentResult(
        answer=(
            "SWOT Analysis:\n"
            "STRENGTHS: Zero emissions, lower operating costs, government incentives, "
            "technology improving rapidly.\n"
            "WEAKNESSES: High upfront cost, range anxiety, charging infrastructure gaps, "
            "battery degradation concerns.\n"
            "OPPORTUNITIES: Fleet electrification, energy storage services, "
            "autonomous EV taxis, developing market expansion.\n"
            "THREATS: Raw material supply constraints (lithium, cobalt), "
            "grid capacity limits, hydrogen fuel cell competition, policy changes."
        ),
        success=True,
    )


def recommendation_writer(subtask: str) -> AgentResult:
    """Strategy recommendation specialist."""
    return AgentResult(
        answer=(
            "Strategic recommendations:\n"
            "1. INVEST in solid-state battery partnerships (2-3 year horizon)\n"
            "2. BUILD charging network in underserved regions (first-mover advantage)\n"
            "3. TARGET fleet operators for B2B sales (higher volume, predictable demand)\n"
            "4. DEVELOP V2G capabilities as differentiator\n"
            "5. PARTNER with Chinese manufacturers for cost-competitive models"
        ),
        success=True,
    )


def create_research_lead(model: str) -> OrchestratorAgent:
    """Create the research lead orchestrator."""

    def research_worker(subtask: str) -> AgentResult:
        s = subtask.lower()
        if any(kw in s for kw in ["market", "competitor", "sales", "growth"]):
            return market_researcher(subtask)
        return tech_researcher(subtask)

    config = AgentConfig(model=model, max_iterations=10, temperature=0.5)
    return OrchestratorAgent(
        worker_factory=research_worker,
        config=config,
        max_workers=2,
    )


def create_strategy_lead(model: str) -> OrchestratorAgent:
    """Create the strategy lead orchestrator."""

    def strategy_worker(subtask: str) -> AgentResult:
        s = subtask.lower()
        if any(
            kw in s for kw in ["swot", "strength", "weakness", "opportunity", "threat"]
        ):
            return swot_analyst(subtask)
        return recommendation_writer(subtask)

    config = AgentConfig(model=model, max_iterations=10, temperature=0.5)
    return OrchestratorAgent(
        worker_factory=strategy_worker,
        config=config,
        max_workers=2,
    )


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Create mid-level orchestrators
    research_lead = create_research_lead(model)
    strategy_lead = create_strategy_lead(model)

    # Top-level orchestrator delegates to mid-level leads
    def top_worker(subtask: str) -> AgentResult:
        s = subtask.lower()
        if any(
            kw in s for kw in ["research", "market", "technology", "competitor", "data"]
        ):
            return research_lead.run(subtask)
        elif any(
            kw in s for kw in ["strategy", "swot", "recommend", "plan", "analysis"]
        ):
            return strategy_lead.run(subtask)
        else:
            return AgentResult(answer=f"Delegated: {subtask[:100]}", success=True)

    top_config = AgentConfig(model=model, max_iterations=15, temperature=0.7)
    top_orchestrator = OrchestratorAgent(
        worker_factory=top_worker,
        config=top_config,
        max_workers=2,
    )

    task = (
        "Create a comprehensive strategic analysis of the electric vehicle industry. "
        "Phase 1: Research market trends, competitive landscape, and technology status. "
        "Phase 2: Perform SWOT analysis and develop strategic recommendations."
    )

    print("=" * 60)
    print("Hierarchical Orchestrator — Nested Delegation")
    print("=" * 60)
    print(f"Model: {model}")
    print("Architecture: Top -> Research Lead + Strategy Lead -> Workers")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = top_orchestrator.run(task)

        print(f"\nFinal Analysis:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
