"""
Method 6: Classification Router for Multi-Agent Dispatch
==========================================================

Uses ``Classifier`` to classify task type (research, analysis,
creative, factual) and ``IntentRouter`` to dispatch to the most
appropriate agent pattern: ReactAgent, DebateAgent, or
PromptChainAgent.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/classified_dispatch/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/classified_dispatch/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    ChainStep,
    DebateAgent,
    PromptChainAgent,
    ReactAgent,
    ToolRegistry,
)

# Try to import classification
try:
    from fsm_llm_classification import (
        ClassificationSchema,
        Classifier,
        IntentDefinition,
        IntentRouter,
    )

    _HAS_CLASSIFICATION = True
except ImportError:
    _HAS_CLASSIFICATION = False


# ──────────────────────────────────────────────
# Tool Definitions (for ReactAgent)
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Search for information."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "renewable energy": "Solar, wind, hydro, and geothermal are the main renewable sources.",
        "climate change": "Global average temperature has risen ~1.1C since pre-industrial times.",
        "artificial intelligence": "AI encompasses ML, NLP, computer vision, and robotics.",
        "quantum computing": "Quantum computers use qubits that can exist in superposition states.",
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
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
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


# ──────────────────────────────────────────────
# Agent Builders
# ──────────────────────────────────────────────


def _build_research_agent(model: str) -> ReactAgent:
    """Build a ReactAgent for research tasks."""
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search for information",
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
    config = AgentConfig(model=model, max_iterations=8, temperature=0.7)
    return ReactAgent(tools=registry, config=config)


def _build_debate_agent(model: str) -> DebateAgent:
    """Build a DebateAgent for analytical tasks."""
    config = AgentConfig(model=model, max_iterations=20, temperature=0.7)
    return DebateAgent(
        config=config,
        num_rounds=2,
        proposer_persona="You are an optimistic analyst who sees opportunities.",
        critic_persona="You are a skeptical analyst who identifies risks.",
    )


def _build_creative_agent(model: str) -> PromptChainAgent:
    """Build a PromptChainAgent for creative tasks."""
    config = AgentConfig(model=model, max_iterations=15, temperature=0.8)
    chain = [
        ChainStep(
            step_id="brainstorm",
            name="Brainstorm",
            extraction_instructions=(
                "Generate 3-5 creative ideas related to the task.\n"
                "Extract as JSON:\n"
                '- "chain_step_result": a list of creative ideas'
            ),
            response_instructions="Present your creative ideas with brief explanations.",
        ),
        ChainStep(
            step_id="develop",
            name="Develop",
            extraction_instructions=(
                "Take the best ideas from the brainstorming phase and develop "
                "them into a coherent creative output.\n"
                "Extract as JSON:\n"
                '- "chain_step_result": the developed creative output'
            ),
            response_instructions="Present the developed creative work.",
        ),
        ChainStep(
            step_id="polish",
            name="Polish",
            extraction_instructions=(
                "Refine and polish the creative output for final presentation.\n"
                "Extract as JSON:\n"
                '- "chain_step_result": the polished final output'
            ),
            response_instructions="Present the final polished creative output.",
        ),
    ]
    return PromptChainAgent(config=config, chain=chain)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    if not _HAS_CLASSIFICATION:
        print("This example requires fsm_llm_classification.")
        print("Install with: pip install fsm-llm[classification]")
        return

    # Build task-type classifier
    schema = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="research",
                description="Tasks requiring information gathering, fact-finding, or answering factual questions",
            ),
            IntentDefinition(
                name="analysis",
                description="Tasks requiring deep analysis, comparison, evaluation, or debate of pros/cons",
            ),
            IntentDefinition(
                name="creative",
                description="Tasks requiring creative writing, brainstorming, or content generation",
            ),
            IntentDefinition(
                name="general",
                description="General tasks that don't fit other categories",
            ),
        ],
        fallback_intent="general",
        confidence_threshold=0.4,
    )

    classifier = Classifier(schema, model=model)

    # Build intent router
    def handle_research(message: str, entities: dict) -> str:
        agent = _build_research_agent(model)
        result = agent.run(message)
        return result.answer

    def handle_analysis(message: str, entities: dict) -> str:
        agent = _build_debate_agent(model)
        result = agent.run(message)
        return result.answer

    def handle_creative(message: str, entities: dict) -> str:
        agent = _build_creative_agent(model)
        result = agent.run(message)
        return result.answer

    def handle_general(message: str, entities: dict) -> str:
        agent = _build_research_agent(model)
        result = agent.run(message)
        return result.answer

    router = IntentRouter(schema)
    router.register("research", handle_research)
    router.register("analysis", handle_analysis)
    router.register("creative", handle_creative)
    router.register("general", handle_general)

    print("=" * 60)
    print("Method 6: Classification Router for Multi-Agent Dispatch")
    print("=" * 60)
    print(f"Model: {model}")
    print("Agent types: ReactAgent (research), DebateAgent (analysis),")
    print("             PromptChainAgent (creative)")
    print("Type a question or 'quit' to exit.\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print("\nClassifying task type...")
        try:
            result = classifier.classify(task)
            print(f"  Task type: {result.intent} (confidence: {result.confidence:.2f})")
            print(f"  Reasoning: {result.reasoning}")

            print(f"\nDispatching to {result.intent} agent...")
            print("-" * 40)

            answer = router.route(task, result)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
