"""
Orchestrator with Specialist Agents
====================================

Demonstrates the Orchestrator pattern with real specialist agents (not
mock workers). The orchestrator decomposes a task and delegates subtasks
to specialized ReactAgents, each with domain-specific tools.

Architecture:
  - Orchestrator: Decomposes and synthesizes
  - Financial Analyst: Has market data tools
  - Technical Analyst: Has tech comparison tools
  - Risk Assessor: Has risk evaluation tools

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/orchestrator_specialist/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    AgentResult,
    OrchestratorAgent,
    ReactAgent,
    ToolRegistry,
    tool,
)

# ── Financial Analyst Tools ──────────────────


@tool
def market_data(sector: str) -> str:
    """Get market data for a business sector."""
    sectors = {
        "saas": "SaaS market: $197B (2024), growing 18% CAGR. Key metrics: ARR, churn rate, LTV/CAC ratio.",
        "ecommerce": "E-commerce: $6.3T globally (2024). Mobile commerce is 73% of total. Average conversion rate 2.5%.",
        "fintech": "Fintech: $305B market (2024). Digital payments dominate at 48%. Regulatory compliance costs rising 15%/year.",
        "healthtech": "HealthTech: $280B (2024). Telemedicine grew 38x since 2020. AI diagnostics accuracy at 94%.",
        "ai": "AI market: $214B (2024), 35% CAGR. Enterprise adoption at 55%. GPU costs dominating infrastructure spend.",
    }
    for key, value in sectors.items():
        if key in sector.lower():
            return value
    return f"Market data for {sector}: emerging sector with growth potential."


@tool
def competitor_analysis(company_type: str) -> str:
    """Analyze competitive landscape for a business type."""
    landscapes = {
        "saas": "Top SaaS: Salesforce ($31B), Microsoft 365 ($60B), ServiceNow ($7B). Barriers: switching costs, data lock-in.",
        "ai platform": "AI platforms: OpenAI (leading), Google DeepMind, Anthropic, Meta AI. Barriers: compute costs, talent, data moats.",
        "marketplace": "Marketplaces: network effects create winner-take-all. Average take rate 15-25%. Critical mass is key barrier.",
    }
    for key, value in landscapes.items():
        if key in company_type.lower():
            return value
    return f"Competitive analysis for {company_type}: moderate competition, differentiation opportunities exist."


# ── Technical Analyst Tools ──────────────────


@tool
def tech_stack_analysis(technology: str) -> str:
    """Evaluate technology stack choices."""
    stacks = {
        "cloud": "Cloud infrastructure: AWS (32%), Azure (23%), GCP (11%). Multi-cloud adoption at 89% of enterprises.",
        "database": "Database trends: PostgreSQL leading open-source. NoSQL for scale. NewSQL for distributed ACID.",
        "ml ops": "MLOps: experiment tracking (MLflow/W&B), model serving (TF Serving/Triton), monitoring (Evidently/Whylabs).",
        "security": "Security stack: zero-trust architecture, SAST/DAST in CI/CD, SOC2 compliance essential for enterprise sales.",
    }
    for key, value in stacks.items():
        if key in technology.lower():
            return value
    return f"Tech analysis for {technology}: mature ecosystem with multiple viable options."


@tool
def scalability_assessment(architecture: str) -> str:
    """Assess scalability of an architecture pattern."""
    patterns = {
        "microservices": "Microservices: scales independently, but adds complexity. Best above 10 engineers. Service mesh recommended.",
        "monolith": "Monolith: simpler to start, harder to scale. Modular monolith is the pragmatic middle ground for startups.",
        "serverless": "Serverless: auto-scales, pay-per-use. Cold starts can be 1-5s. Best for event-driven workloads.",
        "event-driven": "Event-driven: excellent for async processing. Kafka/SQS for reliability. Complexity in debugging.",
    }
    for key, value in patterns.items():
        if key in architecture.lower():
            return value
    return (
        f"Scalability for {architecture}: depends on workload patterns and team size."
    )


# ── Risk Assessor Tools ──────────────────


@tool
def risk_evaluation(risk_type: str) -> str:
    """Evaluate specific business risks."""
    risks = {
        "market": "Market risks: customer concentration, market timing, regulatory changes. Mitigation: diversify revenue streams.",
        "technical": "Technical risks: tech debt accumulation, single points of failure, vendor lock-in. Mitigation: architecture reviews.",
        "financial": "Financial risks: burn rate, runway, unit economics. Startups need 18+ months runway. CAC payback < 12 months.",
        "regulatory": "Regulatory risks: data privacy (GDPR/CCPA), AI regulation (EU AI Act), financial compliance (PCI-DSS/SOX).",
        "team": "Team risks: key person dependency, hiring in competitive market, culture dilution during scale. Mitigation: documentation.",
    }
    for key, value in risks.items():
        if key in risk_type.lower():
            return value
    return f"Risk assessment for {risk_type}: moderate risk level, standard mitigations apply."


# ── Specialist Agent Factory ──────────────────


def create_specialist_worker(model: str):
    """Create a worker factory that routes to specialist agents."""
    # Pre-build specialist agents
    fin_registry = ToolRegistry()
    fin_registry.register(market_data._tool_definition)
    fin_registry.register(competitor_analysis._tool_definition)

    tech_registry = ToolRegistry()
    tech_registry.register(tech_stack_analysis._tool_definition)
    tech_registry.register(scalability_assessment._tool_definition)

    risk_registry = ToolRegistry()
    risk_registry.register(risk_evaluation._tool_definition)

    config = AgentConfig(model=model, max_iterations=6, temperature=0.5)

    specialists = {
        "financial": ReactAgent(tools=fin_registry, config=config),
        "technical": ReactAgent(tools=tech_registry, config=config),
        "risk": ReactAgent(tools=risk_registry, config=config),
    }

    def worker(subtask: str) -> AgentResult:
        s = subtask.lower()
        # Route to appropriate specialist
        if any(
            kw in s for kw in ["market", "revenue", "financ", "competitor", "pricing"]
        ):
            agent = specialists["financial"]
        elif any(
            kw in s for kw in ["tech", "architect", "stack", "scal", "infrastructure"]
        ):
            agent = specialists["technical"]
        elif any(kw in s for kw in ["risk", "regulat", "compliance", "threat"]):
            agent = specialists["risk"]
        else:
            # Default to financial analysis
            agent = specialists["financial"]

        try:
            return agent.run(subtask)
        except Exception as e:
            return AgentResult(answer=f"Analysis inconclusive: {e}", success=False)

    return worker


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(model=model, max_iterations=15, temperature=0.7)
    worker = create_specialist_worker(model)

    agent = OrchestratorAgent(
        worker_factory=worker,
        config=config,
        max_workers=4,
    )

    task = (
        "Evaluate the viability of launching an AI-powered SaaS platform for "
        "automated code review. Cover: market opportunity and competitive landscape, "
        "technical architecture recommendations, financial projections, and key risks."
    )

    print("=" * 60)
    print("Orchestrator with Specialist Agents")
    print("=" * 60)
    print(f"Model: {model}")
    print("Specialists: Financial Analyst, Technical Analyst, Risk Assessor")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)
        print(f"\nSynthesized Analysis:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
