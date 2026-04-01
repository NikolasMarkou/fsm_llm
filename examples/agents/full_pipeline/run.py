"""
Method 7: Full Pipeline — All 4 Modules
==========================================

Combines all four FSM-LLM modules in a single pipeline:
  1. Classification: Classify the task type
  2. Intent Router: Dispatch to the right agent
  3. Agent: ReactAgent with tools
  4. Reasoning: ReasoningEngine as a tool for complex analysis

This is the most comprehensive integration example, showing how
classification, agents, and reasoning work together.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/full_pipeline/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/full_pipeline/run.py
"""

import os

from fsm_llm import (
    ClassificationSchema,
    Classifier,
    IntentDefinition,
    IntentRouter,
)
from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry

# Try to import reasoning
try:
    from fsm_llm_reasoning import ReasoningEngine

    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False


# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Search for information."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "renewable energy": "Solar and wind account for 12% of global electricity (growing 20%/year).",
        "climate": "Global temperatures have risen ~1.1C. Paris Agreement targets 1.5C.",
        "ai ethics": "Key concerns: bias, privacy, job displacement, autonomous weapons.",
        "quantum": "Quantum advantage demonstrated for specific problems. Still early for general use.",
        "space exploration": "SpaceX Starship, Artemis program, Mars colonization plans.",
    }

    for key, value in results.items():
        if key in q:
            return value
    return f"No results for '{query}'."


def calculate(params: dict) -> str:
    """Evaluate a math expression safely using AST parsing."""
    import ast
    import operator

    expression = params.get("expression", "")
    if not expression or not expression.strip():
        return "Error: Empty expression"

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error: {e}"


def lookup(params: dict) -> str:
    """Look up factual data."""
    topic = params.get("topic", "").lower()

    facts = {
        "solar": "Solar panel efficiency: 20-25% for commercial. Cost: ~$0.20/watt.",
        "wind": "Offshore wind capacity factor: 40-50%. Onshore: 25-35%.",
        "battery": "Lithium-ion energy density: 250-300 Wh/kg. Cost declining ~10%/year.",
    }

    for key, value in facts.items():
        if key in topic:
            return value
    return f"No data for '{topic}'."


def _build_reason_tool(model: str):
    """Build a reasoning tool wrapping ReasoningEngine."""
    engine = ReasoningEngine(model=model)

    def reason(params: dict) -> str:
        """Use structured reasoning for complex analysis."""
        problem = params.get("problem", "")
        if not problem:
            return "Error: No problem statement."
        try:
            solution, _trace_info = engine.solve_problem(problem)
            return f"[Reasoning] {solution}"
        except Exception as e:
            return f"Reasoning failed: {e}"

    return reason


# ──────────────────────────────────────────────
# Agent builders
# ──────────────────────────────────────────────


def _build_research_agent(model: str, with_reasoning: bool = False) -> ReactAgent:
    """Build ReactAgent with optional reasoning tool."""
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search for information on a topic",
        parameter_schema={
            "properties": {"query": {"type": "string", "description": "Search query"}}
        },
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Evaluate a math expression",
        parameter_schema={
            "properties": {
                "expression": {"type": "string", "description": "Expression"}
            }
        },
    )
    registry.register_function(
        lookup,
        name="lookup",
        description="Look up factual data about a specific topic",
        parameter_schema={
            "properties": {"topic": {"type": "string", "description": "Topic"}}
        },
    )

    if with_reasoning and _HAS_REASONING:
        registry.register_function(
            _build_reason_tool(model),
            name="reason",
            description="Use structured reasoning for complex multi-step analysis",
            parameter_schema={
                "properties": {
                    "problem": {"type": "string", "description": "Problem to analyze"}
                }
            },
        )

    config = AgentConfig(model=model, max_iterations=10, temperature=0.7)
    return ReactAgent(tools=registry, config=config)


def _build_simple_agent(model: str) -> ReactAgent:
    """Build a simple ReactAgent for factual lookups."""
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search for information",
        parameter_schema={
            "properties": {"query": {"type": "string", "description": "Query"}}
        },
    )
    registry.register_function(
        lookup,
        name="lookup",
        description="Look up facts",
        parameter_schema={
            "properties": {"topic": {"type": "string", "description": "Topic"}}
        },
    )
    config = AgentConfig(model=model, max_iterations=6, temperature=0.5)
    return ReactAgent(tools=registry, config=config)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    # Step 1: Build task classifier
    schema = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="research",
                description="In-depth research requiring multiple sources and analysis",
            ),
            IntentDefinition(
                name="analytical",
                description="Complex analytical tasks requiring structured reasoning",
            ),
            IntentDefinition(
                name="factual",
                description="Simple factual questions with straightforward answers",
            ),
            IntentDefinition(
                name="general",
                description="General queries that don't fit other categories",
            ),
        ],
        fallback_intent="general",
        confidence_threshold=0.4,
    )

    classifier = Classifier(schema, model=model)

    # Step 2: Build intent router with agent dispatch
    def handle_research(message: str, entities: dict) -> str:
        print("  -> Dispatching to: Research Agent (with search + lookup)")
        agent = _build_research_agent(model, with_reasoning=False)
        result = agent.run(message)
        return result.answer

    def handle_analytical(message: str, entities: dict) -> str:
        if _HAS_REASONING:
            print("  -> Dispatching to: Research Agent + Reasoning Tool")
        else:
            print("  -> Dispatching to: Research Agent (reasoning not available)")
        agent = _build_research_agent(model, with_reasoning=True)
        result = agent.run(message)
        return result.answer

    def handle_factual(message: str, entities: dict) -> str:
        print("  -> Dispatching to: Simple Agent (fast lookup)")
        agent = _build_simple_agent(model)
        result = agent.run(message)
        return result.answer

    def handle_general(message: str, entities: dict) -> str:
        print("  -> Dispatching to: Research Agent (general)")
        agent = _build_research_agent(model, with_reasoning=False)
        result = agent.run(message)
        return result.answer

    router = IntentRouter(schema)
    router.register("research", handle_research)
    router.register("analytical", handle_analytical)
    router.register("factual", handle_factual)
    router.register("general", handle_general)

    print("=" * 60)
    print("Method 7: Full Pipeline (Classification + Agent + Reasoning)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Modules: classification=yes, reasoning={'yes' if _HAS_REASONING else 'no'}")
    print("Pipeline: Classify -> Route -> Agent [-> Reasoning]")
    print("Type a question or 'quit' to exit.\n")

    last_answer = None
    last_intent = None
    tasks_completed = 0

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print("\n[1] Classifying task type...")
        try:
            classification = classifier.classify(task)
            print(
                f"    Type: {classification.intent} "
                f"(confidence: {classification.confidence:.2f})"
            )

            print("\n[2] Routing to agent...")
            answer = router.route(task, classification)

            print("\n[3] Result:")
            print(f"    {answer}")
            last_answer = answer
            last_intent = classification.intent
            tasks_completed += 1
        except Exception as e:
            print(f"Error: {e}")
        print()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": last_answer is not None and len(str(last_answer)) > 10,
        "iterations_ok": tasks_completed >= 1,
        "completed": last_intent is not None,
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
