"""
Supply Chain Optimizer with Specialist Workers
================================================

Demonstrates the Orchestrator pattern with three specialist ReactAgent
workers, each equipped with domain-specific tools. The orchestrator
decomposes a supply chain optimization problem and delegates to
logistics, procurement, and demand forecasting specialists.

Architecture:
  - Orchestrator: Decomposes and synthesizes supply chain strategy
  - Logistics Worker: Route optimization and warehouse tools
  - Procurement Worker: Supplier evaluation and cost analysis tools
  - Demand Worker: Demand forecasting and inventory planning tools

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/supply_chain_optimizer/run.py
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

# -- Logistics Worker Tools --------------------------------------------------


@tool
def route_optimization(origin: str, destination: str) -> str:
    """Optimize shipping routes between locations."""
    routes = {
        ("shanghai", "los angeles"): (
            "Trans-Pacific route via Shanghai-LA corridor: 14 days by sea, "
            "$2,400/TEU. Alternative air freight: 2 days, $8,500/ton. "
            "Rail via Eurasia land bridge: 18 days, $4,200/TEU."
        ),
        ("hamburg", "new york"): (
            "Trans-Atlantic route: 9 days by sea, $1,800/TEU. "
            "Express service available: 7 days, $2,600/TEU. "
            "Air freight: 1 day, $7,200/ton."
        ),
        ("mumbai", "rotterdam"): (
            "Suez Canal route: 18 days by sea, $2,100/TEU. "
            "Cape of Good Hope alternative: 25 days, $1,900/TEU (avoids Suez fees). "
            "Air freight: 2 days, $9,000/ton."
        ),
    }
    key = (origin.lower().strip(), destination.lower().strip())
    for (o, d), info in routes.items():
        if o in key[0] or key[0] in o or d in key[1] or key[1] in d:
            return info
    return (
        f"Route {origin} -> {destination}: standard sea freight 10-21 days, "
        f"$1,500-3,000/TEU. Air freight 1-3 days at 3-4x cost."
    )


@tool
def warehouse_capacity(region: str) -> str:
    """Check warehouse capacity and utilization for a region."""
    warehouses = {
        "north america": (
            "3 distribution centers: LA (85% utilized, 50K sqft), "
            "Chicago (72% utilized, 35K sqft), Newark (91% utilized, 40K sqft). "
            "Newark approaching capacity — expansion or overflow needed by Q3. "
            "Average pick-pack-ship time: 4.2 hours."
        ),
        "europe": (
            "2 distribution centers: Rotterdam (68% utilized, 45K sqft), "
            "Munich (77% utilized, 30K sqft). Seasonal peak expected Q4 (+25%). "
            "Rotterdam has cold-chain capability for perishables. "
            "Average pick-pack-ship time: 3.8 hours."
        ),
        "asia": (
            "2 distribution centers: Shenzhen (93% utilized, 60K sqft), "
            "Mumbai (55% utilized, 25K sqft). Shenzhen critically over-utilized — "
            "quality control delays averaging 6 hours. Mumbai has expansion room. "
            "Average pick-pack-ship time: 5.1 hours."
        ),
    }
    for key, info in warehouses.items():
        if key in region.lower():
            return info
    return f"Warehouse data for {region}: moderate utilization, standard capacity."


# -- Procurement Worker Tools -------------------------------------------------


@tool
def supplier_evaluation(material: str) -> str:
    """Evaluate suppliers for a given material or component."""
    suppliers = {
        "semiconductor": (
            "3 qualified suppliers: TaiwanSemi (lead time 12 weeks, $4.20/unit, "
            "99.2% quality), KoreaTech (14 weeks, $3.80/unit, 98.5% quality), "
            "USChip (8 weeks, $5.60/unit, 99.7% quality). Geopolitical risk: "
            "TaiwanSemi HIGH, others MODERATE. Recommend dual-sourcing strategy."
        ),
        "battery": (
            "4 qualified suppliers: ChinaCell (10 weeks, $12/unit, 97.8% quality), "
            "JapanPower (8 weeks, $18/unit, 99.5% quality), EuroBatt (12 weeks, "
            "$15/unit, 99.1% quality), IndiaEnergy (14 weeks, $9/unit, 96.2% quality). "
            "ChinaCell offers best cost but quality variance is higher."
        ),
        "steel": (
            "5 qualified suppliers across 3 regions. Spot price $780/ton, contract "
            "price $720/ton (12-month). Lead time 4-6 weeks. Quality grades: "
            "A-grade from ArcelorMittal and Nippon Steel. Tariff impact: +25% on "
            "imports, favoring domestic sourcing for North American operations."
        ),
    }
    for key, info in suppliers.items():
        if key in material.lower():
            return info
    return (
        f"Supplier data for {material}: 2-3 qualified suppliers available. "
        f"Average lead time 8-12 weeks. Standard quality compliance."
    )


@tool
def cost_analysis(category: str) -> str:
    """Analyze cost breakdown for a supply chain category."""
    costs = {
        "transportation": (
            "Total logistics spend: $14.2M annually. Breakdown: ocean freight 42%, "
            "trucking 28%, air freight 15%, rail 8%, last-mile 7%. Year-over-year "
            "increase: +8.3% driven by fuel surcharges and port congestion. "
            "Consolidation opportunity: combining LTL shipments could save $1.1M."
        ),
        "inventory": (
            "Total inventory carrying cost: $8.7M annually. Breakdown: warehousing "
            "35%, capital cost 28%, insurance 12%, obsolescence 15%, handling 10%. "
            "Current inventory turns: 6.2x (industry avg 8.5x). Safety stock levels "
            "are 40% above optimal — reducing to 20% buffer saves $1.8M."
        ),
        "procurement": (
            "Total procurement spend: $52M annually. Top 3 categories: raw materials "
            "55%, components 30%, packaging 15%. Maverick spending at 12% (target <5%). "
            "Contract compliance at 78% (target >90%). Renegotiation of top 10 "
            "contracts projected to save $3.2M."
        ),
    }
    for key, info in costs.items():
        if key in category.lower():
            return info
    return f"Cost analysis for {category}: data available upon detailed review."


# -- Demand Worker Tools ------------------------------------------------------


@tool
def demand_forecast(product_line: str) -> str:
    """Generate demand forecast for a product line."""
    forecasts = {
        "electronics": (
            "Q1: 45K units (confidence 88%), Q2: 52K units (82%), Q3: 48K units "
            "(79%), Q4: 71K units (91% — holiday peak). YoY growth: +12%. "
            "Seasonal decomposition shows 35% variance between peak/trough. "
            "Forecast model: SARIMA with external regressors (GDP, consumer confidence). "
            "MAPE on 12-month rolling: 7.2%."
        ),
        "automotive": (
            "Q1: 12K units (confidence 85%), Q2: 14K units (80%), Q3: 13K units "
            "(77%), Q4: 15K units (84%). YoY growth: +6%. EV component demand "
            "growing 28% YoY. Supply-constrained forecast: actual deliveries may "
            "be 8-10% below demand due to semiconductor shortages."
        ),
        "consumer goods": (
            "Q1: 180K units (confidence 90%), Q2: 165K units (87%), Q3: 155K units "
            "(83%), Q4: 210K units (92% — promotional period). YoY growth: +4%. "
            "E-commerce channel growing 22% while retail flat. Direct-to-consumer "
            "margin 15% higher than wholesale."
        ),
    }
    for key, info in forecasts.items():
        if key in product_line.lower():
            return info
    return (
        f"Demand forecast for {product_line}: moderate growth expected, "
        f"seasonal patterns apply. Detailed forecast requires historical data."
    )


@tool
def inventory_optimization(strategy: str) -> str:
    """Recommend inventory optimization strategies."""
    strategies = {
        "just-in-time": (
            "JIT feasibility assessment: current supplier lead time variance is "
            "2.3 weeks (target <1 week for JIT). Recommended hybrid approach: "
            "JIT for A-class items (top 20% by value) with safety stock for "
            "B/C-class. Expected inventory reduction: 30-35%. Risk: stockout "
            "probability increases from 2% to 5% during transition."
        ),
        "safety stock": (
            "Current safety stock: 6.2 weeks of supply (industry benchmark: 4 weeks). "
            "ABC analysis: A-items over-stocked by 45%, C-items under-stocked by 20%. "
            "Recommended rebalancing: reduce A-item buffer to 3 weeks, increase C-item "
            "to 5 weeks. Net working capital freed: $2.4M."
        ),
        "vendor managed": (
            "VMI program assessment: 3 of top 10 suppliers are VMI-capable. "
            "Pilot program with ChinaCell showed 15% reduction in stockouts and "
            "22% reduction in expediting costs. Full VMI rollout to top 5 suppliers "
            "projected to save $1.6M annually. Implementation timeline: 6 months."
        ),
    }
    for key, info in strategies.items():
        if key in strategy.lower():
            return info
    return (
        f"Inventory strategy '{strategy}': analysis available. Key metrics to "
        f"evaluate: carrying cost, service level, and lead time variability."
    )


# -- Specialist Worker Factory ------------------------------------------------


def create_supply_chain_workers(model: str):
    """Create a worker factory that routes to supply chain specialist agents."""
    logistics_registry = ToolRegistry()
    logistics_registry.register(route_optimization._tool_definition)
    logistics_registry.register(warehouse_capacity._tool_definition)

    procurement_registry = ToolRegistry()
    procurement_registry.register(supplier_evaluation._tool_definition)
    procurement_registry.register(cost_analysis._tool_definition)

    demand_registry = ToolRegistry()
    demand_registry.register(demand_forecast._tool_definition)
    demand_registry.register(inventory_optimization._tool_definition)

    config = AgentConfig(model=model, max_iterations=6, temperature=0.5)

    specialists = {
        "logistics": ReactAgent(tools=logistics_registry, config=config),
        "procurement": ReactAgent(tools=procurement_registry, config=config),
        "demand": ReactAgent(tools=demand_registry, config=config),
    }

    def worker(subtask: str) -> AgentResult:
        s = subtask.lower()
        if any(
            kw in s
            for kw in [
                "route",
                "shipping",
                "warehouse",
                "logistics",
                "transport",
                "freight",
                "distribution",
            ]
        ):
            agent = specialists["logistics"]
        elif any(
            kw in s
            for kw in [
                "supplier",
                "procure",
                "cost",
                "sourcing",
                "vendor",
                "contract",
                "material",
            ]
        ):
            agent = specialists["procurement"]
        elif any(
            kw in s
            for kw in [
                "demand",
                "forecast",
                "inventory",
                "stock",
                "seasonal",
                "sales",
                "planning",
            ]
        ):
            agent = specialists["demand"]
        else:
            agent = specialists["logistics"]

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
    worker = create_supply_chain_workers(model)

    agent = OrchestratorAgent(
        worker_factory=worker,
        config=config,
        max_workers=4,
    )

    task = (
        "Develop a comprehensive supply chain optimization strategy for NovaTech "
        "Industries, a mid-size electronics manufacturer producing consumer devices "
        "(smartphones, tablets, IoT sensors) with annual revenue of $380M. The company "
        "operates manufacturing in Shenzhen (China) and assembly in Guadalajara (Mexico), "
        "with distribution centers in North America (LA, Chicago, Newark), Europe "
        "(Rotterdam, Munich), and Asia (Shenzhen, Mumbai). The workforce includes 2,400 "
        "employees across 8 countries with a supply chain team of 85 professionals. "
        "Current pain points include: (1) semiconductor lead times have increased from "
        "8 to 14 weeks, causing production delays and $12M in missed revenue last "
        "quarter — the primary chipset supplier TaiwanSemi is sole-source for 3 critical "
        "components; (2) ocean freight costs rose 35% year-over-year with the Shanghai-LA "
        "corridor experiencing 3-5 day port congestion delays, and the Suez Canal route "
        "to Rotterdam facing intermittent disruptions adding 7-10 days when rerouted via "
        "Cape of Good Hope; (3) inventory carrying costs are $8.7M annually with turns "
        "at 6.2x versus industry average of 8.5x, driven by excessive safety stock on "
        "A-class items (45% above optimal) while C-class items are understocked by 20%; "
        "(4) the Shenzhen warehouse is at 93% utilization with quality control backlogs "
        "averaging 6 hours per shipment and the Newark facility approaching 91% capacity; "
        "(5) single-source dependency on TaiwanSemi for critical chipsets creates "
        "geopolitical risk, while the KoreaTech alternative requires 6-month qualification. "
        "The CEO wants a strategy covering: logistics route optimization across all three "
        "major corridors and warehouse capacity planning including potential expansion, "
        "procurement diversification to reduce single-supplier risk with a dual-sourcing "
        "roadmap, demand forecasting improvements leveraging the new SARIMA model to "
        "reduce safety stock while maintaining 98% service levels, and a total cost "
        "reduction target of 12-15% ($11-13M) over 18 months. Consider seasonal demand "
        "patterns (Q4 holiday peak drives 35% variance with electronics forecast at 71K "
        "units), the accelerating shift toward direct-to-consumer channels (currently 22% "
        "of sales, margin 15% higher than wholesale), and upcoming tariff changes "
        "affecting cross-border component flows between China, Mexico, and the US."
    )

    print("=" * 60)
    print("Supply Chain Optimizer with Specialist Workers")
    print("=" * 60)
    print(f"Model: {model}")
    print("Specialists: Logistics, Procurement, Demand Forecasting")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)
        print(f"\nSynthesized Strategy:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.iterations_used < config.max_iterations,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
