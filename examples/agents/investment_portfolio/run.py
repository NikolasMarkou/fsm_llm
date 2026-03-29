"""
Investment Portfolio Rebalancing -- PlanExecuteAgent
====================================================

Demonstrates a PlanExecuteAgent that analyzes market conditions,
evaluates risk, researches sectors, optimizes allocations, and
verifies regulatory compliance for a portfolio rebalancing request.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/investment_portfolio/run.py
"""

import os
from typing import Annotated

from fsm_llm_agents import AgentConfig, PlanExecuteAgent, ToolRegistry, tool


@tool
def market_data(
    asset: Annotated[str, "Asset ticker or class (e.g. SPY, BTC, bonds, gold)"],
    period: Annotated[str, "Time period for data (e.g. 1y, 6m, ytd)"],
) -> str:
    """Fetch current market data and historical performance for an asset."""
    a = asset.lower()

    data = {
        "spy": (
            "S&P 500 (SPY): Current 5,280. 1Y return +22.4%. Volatility (annualized): 14.2%. "
            "P/E ratio: 24.8. Dividend yield: 1.3%. 52-week range: 4,103-5,320. "
            "RSI: 68 (approaching overbought). Sector leaders: Tech (+31%), Healthcare (+18%)."
        ),
        "bond": (
            "US Aggregate Bond Index: YTD return +3.1%. 10Y Treasury yield: 4.35%. "
            "Duration risk: moderate (6.2 years). Credit spreads: IG 95bps, HY 340bps. "
            "Fed funds rate: 4.50-4.75%. Market expects 2 rate cuts in 2026."
        ),
        "gold": (
            "Gold (XAU): $2,680/oz. 1Y return +28.5%. All-time high: $2,790. "
            "Central bank buying: 1,037 tonnes (2025). Real yields inversely correlated. "
            "Geopolitical risk premium estimated at $200-250/oz."
        ),
        "btc": (
            "Bitcoin (BTC): $97,400. 1Y return +145%. Volatility (30d): 52%. "
            "Market dominance: 58%. ETF inflows: $42B cumulative since launch. "
            "Halving effect (April 2024) historically bullish for 12-18 months."
        ),
        "real estate": (
            "US REITs (VNQ): 1Y return +8.2%. Dividend yield: 3.8%. "
            "Occupancy rates: Office 78%, Residential 95%, Industrial 97%. "
            "Cap rates compressing in industrial/logistics. Office sector remains distressed."
        ),
        "emerging": (
            "Emerging Markets (EEM): 1Y return +11.3%. P/E ratio: 13.2. "
            "Currency risk: moderate. Top allocations: China 28%, India 18%, Taiwan 16%. "
            "India growth outlook strongest at 6.5% GDP."
        ),
    }

    for key, value in data.items():
        if key in a:
            return value
    return f"Market data for {asset} ({period}): Moderate performance, average volatility."


@tool
def risk_analysis(
    portfolio: Annotated[str, "Current portfolio allocation description"],
    risk_tolerance: Annotated[str, "Risk tolerance level: conservative, moderate, aggressive"],
) -> str:
    """Analyze portfolio risk metrics and stress-test scenarios."""
    r = risk_tolerance.lower()

    base = (
        "RISK ANALYSIS REPORT\n"
        "Portfolio VaR (95%, 1-day): "
    )

    if "aggressive" in r:
        return (
            base + "$47,200 (4.72% of $1M)\n"
            "Max Drawdown (historical): -32.5%\n"
            "Sharpe Ratio: 1.42\n"
            "Beta to S&P 500: 1.28\n"
            "Concentration risk: HIGH (>40% in single sector)\n"
            "Stress scenarios:\n"
            "  - 2008-style crisis: -38% estimated loss\n"
            "  - Rate shock (+200bps): -12% estimated loss\n"
            "  - Tech correction (-25%): -22% estimated loss\n"
            "Recommendation: Reduce concentration, add uncorrelated assets"
        )
    elif "conservative" in r:
        return (
            base + "$12,800 (1.28% of $1M)\n"
            "Max Drawdown (historical): -11.2%\n"
            "Sharpe Ratio: 0.89\n"
            "Beta to S&P 500: 0.45\n"
            "Concentration risk: LOW (well-diversified)\n"
            "Stress scenarios:\n"
            "  - 2008-style crisis: -14% estimated loss\n"
            "  - Rate shock (+200bps): -8% estimated loss\n"
            "  - Inflation surge: +2% (inflation hedges working)\n"
            "Recommendation: Portfolio aligned with risk tolerance"
        )
    return (
        base + "$28,500 (2.85% of $1M)\n"
        "Max Drawdown (historical): -22.1%\n"
        "Sharpe Ratio: 1.18\n"
        "Beta to S&P 500: 0.85\n"
        "Concentration risk: MODERATE\n"
        "Stress scenarios:\n"
        "  - 2008-style crisis: -26% estimated loss\n"
        "  - Rate shock (+200bps): -10% estimated loss\n"
        "  - Stagflation: -15% estimated loss\n"
        "Recommendation: Consider rebalancing toward target allocation"
    )


@tool
def sector_research(
    sector: Annotated[str, "Sector to research (e.g. technology, healthcare, energy)"],
) -> str:
    """Research sector outlook, trends, and key drivers."""
    s = sector.lower()

    sectors = {
        "tech": (
            "TECHNOLOGY SECTOR OUTLOOK: OVERWEIGHT\n"
            "AI/ML spending: +45% YoY, $320B enterprise market by 2027.\n"
            "Semiconductor cycle: Recovery underway, TSMC capex +25%.\n"
            "Cloud revenue: AWS +17%, Azure +29%, GCP +26%.\n"
            "Risks: Antitrust regulation, AI bubble concerns, high valuations (35x P/E).\n"
            "Top picks: Semiconductor equipment, cloud infrastructure, cybersecurity."
        ),
        "health": (
            "HEALTHCARE SECTOR OUTLOOK: MARKET WEIGHT\n"
            "GLP-1 drug market: $80B by 2030 (Novo Nordisk, Eli Lilly leading).\n"
            "M&A activity: $180B in 2025, large-cap pharma acquiring biotech.\n"
            "Medicare negotiation: 10 drugs in 2026, margin pressure on pharma.\n"
            "Risks: Drug pricing regulation, patent cliffs ($120B revenue at risk).\n"
            "Top picks: Managed care, medical devices, GLP-1 beneficiaries."
        ),
        "energy": (
            "ENERGY SECTOR OUTLOOK: MARKET WEIGHT\n"
            "Oil: $72/bbl, OPEC+ cuts extended. US production record 13.4M bpd.\n"
            "Renewables: Solar installations +32%, battery storage +85%.\n"
            "Natural gas: $3.40/MMBtu, LNG export capacity expanding.\n"
            "Risks: Demand destruction from EVs, geopolitical supply risk.\n"
            "Top picks: Integrated majors, LNG exporters, utility-scale solar."
        ),
        "financ": (
            "FINANCIAL SECTOR OUTLOOK: OVERWEIGHT\n"
            "Net interest margins: Stabilizing at 3.2% average.\n"
            "Capital markets: IPO pipeline recovering, M&A advisory +22%.\n"
            "Credit quality: Delinquencies rising but manageable (1.8% avg).\n"
            "Risks: Commercial real estate exposure, rate cut impact on NIM.\n"
            "Top picks: Large-cap banks, insurance, fintech platforms."
        ),
    }

    for key, value in sectors.items():
        if key in s:
            return value
    return f"Sector research for {sector}: Neutral outlook, monitor for catalysts."


@tool
def portfolio_optimizer(
    current_allocation: Annotated[str, "Current allocation breakdown"],
    target_return: Annotated[str, "Target annual return percentage"],
    constraints: Annotated[str, "Investment constraints and preferences"],
) -> str:
    """Optimize portfolio allocation using mean-variance optimization."""
    return (
        "OPTIMIZED ALLOCATION (Mean-Variance, target Sharpe maximization):\n"
        "  US Large-Cap Equity:    35% (was 45%) -- reduce concentration\n"
        "  International Equity:   15% (was 10%) -- increase diversification\n"
        "  Emerging Markets:        8% (was  5%) -- capture growth premium\n"
        "  US Aggregate Bonds:     20% (was 25%) -- reduce duration\n"
        "  TIPS (Inflation):        5% (was  0%) -- add inflation hedge\n"
        "  Gold / Commodities:      7% (was  5%) -- geopolitical hedge\n"
        "  REITs:                   5% (was  5%) -- maintain income\n"
        "  Cash / Short-term:       5% (was  5%) -- liquidity buffer\n\n"
        "Expected return: 9.8% (vs 8.2% current)\n"
        "Expected volatility: 12.4% (vs 14.8% current)\n"
        "Sharpe ratio: 0.79 -> 0.95 (improvement)\n"
        "Turnover required: 22% of portfolio ($220K in trades)\n"
        "Estimated tax impact: $8,400 in short-term capital gains"
    )


@tool
def compliance_check(
    strategy: Annotated[str, "Investment strategy description to check"],
    account_type: Annotated[str, "Account type (e.g. IRA, 401k, taxable, trust)"],
) -> str:
    """Verify investment strategy compliance with regulations and account rules."""
    a = account_type.lower()

    checks = []
    if "ira" in a or "401k" in a:
        checks.extend([
            "PASS: No prohibited transactions (IRC Sec. 4975)",
            "PASS: No collectibles (except US gold/silver coins)",
            "PASS: Diversification meets ERISA guidelines",
            "NOTE: RMD considerations -- account holder age may affect strategy",
            "NOTE: Crypto ETFs now permitted in IRAs (IRS guidance 2024)",
        ])
    elif "trust" in a:
        checks.extend([
            "PASS: Prudent investor standard met (Uniform Prudent Investor Act)",
            "PASS: Diversification adequate per trust instrument",
            "NOTE: Review trust document for specific investment restrictions",
            "NOTE: Consider income vs principal beneficiary interests",
        ])
    else:
        checks.extend([
            "PASS: No wash-sale violations detected in proposed trades",
            "PASS: Position sizes within concentration limits",
            "NOTE: Tax-loss harvesting opportunities in international equity",
            "NOTE: Qualified dividend eligibility maintained (>60-day holding)",
        ])

    checks.append("PASS: All proposed ETFs meet liquidity thresholds (>$10M daily volume)")
    return "COMPLIANCE CHECK:\n" + "\n".join(f"  - {c}" for c in checks)


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Register tools
    registry = ToolRegistry()
    registry.register(market_data._tool_definition)
    registry.register(risk_analysis._tool_definition)
    registry.register(sector_research._tool_definition)
    registry.register(portfolio_optimizer._tool_definition)
    registry.register(compliance_check._tool_definition)

    config = AgentConfig(model=model, max_iterations=20, temperature=0.5)
    agent = PlanExecuteAgent(
        tools=registry,
        config=config,
        max_replans=2,
    )

    task = (
        "I need to rebalance a $1 million investment portfolio for a moderate-risk "
        "client (age 52, retirement target 2039, current annual income $185,000). "
        "The client has a 13-year investment horizon with no anticipated liquidity "
        "needs before retirement. The current allocation is heavily concentrated in "
        "US large-cap equities (45%) with the remainder split across bonds (25%), "
        "international developed markets (10%), gold (5%), REITs (5%), emerging "
        "markets (5%), and cash (5%). The account is a taxable brokerage account "
        "at a major custodian with no restrictions on asset classes. The client has "
        "expressed concerns about a potential tech bubble, rising geopolitical risks, "
        "and the impact of AI disruption on traditional sectors.\n\n"
        "Please execute the following rebalancing analysis:\n\n"
        "1. MARKET ASSESSMENT: Pull current market data for the S&P 500 (SPY), "
        "US aggregate bonds, gold, and emerging markets. Assess whether current "
        "valuations (particularly the elevated P/E on US equities) support the "
        "existing overweight or suggest tactical rotation. Evaluate the bond market "
        "given the current rate environment and expected Fed policy path.\n\n"
        "2. RISK EVALUATION: Analyze the current portfolio risk profile given "
        "moderate risk tolerance. Calculate Value-at-Risk and maximum drawdown "
        "estimates. Stress-test against three scenarios: (a) a 2008-style financial "
        "crisis with correlations spiking to 0.9, (b) a rate shock of +200bps over "
        "6 months, and (c) a stagflation environment with 6% inflation and 0% growth. "
        "Compare the current portfolio's resilience against these scenarios.\n\n"
        "3. SECTOR RESEARCH: Research the technology and healthcare sectors for "
        "thematic opportunities that align with a 13-year investment horizon. "
        "Specifically evaluate: AI infrastructure spending trends, semiconductor "
        "cycle positioning, GLP-1 drug market trajectory, and managed care "
        "opportunities. Assess whether current sector valuations offer attractive "
        "entry points or suggest waiting for a pullback.\n\n"
        "4. OPTIMIZATION: Generate an optimized allocation using mean-variance "
        "optimization targeting 9-10% annualized returns with portfolio volatility "
        "under 13%. Constraints: no single asset class above 40%, minimum 15% "
        "fixed income, maximum 10% in alternatives (gold + commodities), maintain "
        "at least 5% cash buffer, and include inflation protection (TIPS) given "
        "the client's concern about purchasing power erosion over 13 years.\n\n"
        "5. COMPLIANCE: Verify the proposed rebalancing strategy complies with "
        "all applicable regulations for a taxable brokerage account. Check for "
        "wash-sale rule violations on any positions being reduced, verify position "
        "concentration limits, assess tax-loss harvesting opportunities in "
        "underperforming positions, and confirm qualified dividend eligibility "
        "is maintained for income-generating holdings.\n\n"
        "Deliver a consolidated rebalancing recommendation with specific trade "
        "instructions, expected impact on risk-adjusted returns (Sharpe ratio "
        "improvement), and projected tax consequences for the current tax year."
    )

    print("=" * 60)
    print("Investment Portfolio Rebalancing -- PlanExecuteAgent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Max replans: 2")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nRebalancing Plan:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        steps = result.final_context.get("plan_steps", [])
        if steps:
            print(f"\nPlan execution ({len(steps)} steps):")
            for s in steps:
                if isinstance(s, dict):
                    status = s.get("status", "?")
                    desc = s.get("description", "N/A")
                    print(f"  [{status}] {desc[:80]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
