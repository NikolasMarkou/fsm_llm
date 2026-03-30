"""
Method 2: Classification for Tool Selection
=============================================

Uses ``ToolRegistry.to_classification_schema()`` to build a
``Classifier`` from the tool schema. The classifier pre-selects
the best tool before the agent runs, passing the classification
result as initial context to improve tool selection accuracy.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/classified_tools/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/classified_tools/run.py
"""

import os

from fsm_llm import (
    ClassificationSchema,
    Classifier,
    IntentDefinition,
)
from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Search the web for information about a topic."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "population": "France: ~68.4 million. Germany: ~84.4 million. Japan: ~125 million.",
        "capital": "France: Paris. Germany: Berlin. Japan: Tokyo.",
        "weather": "Current weather conditions vary by location. Check local forecasts.",
    }

    for key, value in results.items():
        if key in q:
            return value
    return f"No results for '{query}'."


def calculate(params: dict) -> str:
    """Evaluate a mathematical expression safely using AST parsing."""
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


def translate(params: dict) -> str:
    """Translate text between languages (simulated)."""
    text = params.get("text", "")
    target = params.get("target_language", "Spanish")
    translations = {
        "hello": {"spanish": "hola", "french": "bonjour", "german": "hallo"},
        "goodbye": {
            "spanish": "adios",
            "french": "au revoir",
            "german": "auf wiedersehen",
        },
        "thank you": {"spanish": "gracias", "french": "merci", "german": "danke"},
    }

    for key, langs in translations.items():
        if key in text.lower():
            translated = langs.get(target.lower(), text)
            return f"'{text}' in {target}: '{translated}'"
    return f"Translation of '{text}' to {target}: [simulated translation]"


def summarize(params: dict) -> str:
    """Summarize a block of text."""
    text = params.get("text", "")
    if len(text) < 50:
        return f"Summary: {text}"
    return f"Summary: {text[:100]}... (key points extracted)"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    # Build tool registry
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search the web for factual information",
        parameter_schema={
            "properties": {"query": {"type": "string", "description": "Search query"}}
        },
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Evaluate a mathematical expression (arithmetic, percentages)",
        parameter_schema={
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            }
        },
    )
    registry.register_function(
        translate,
        name="translate",
        description="Translate text from English to another language",
        parameter_schema={
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language"},
            }
        },
    )
    registry.register_function(
        summarize,
        name="summarize",
        description="Summarize a block of text into key points",
        parameter_schema={
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"}
            }
        },
    )

    # Build classifier from tool registry schema
    schema_dict = registry.to_classification_schema()
    schema = ClassificationSchema(
        intents=[IntentDefinition(**i) for i in schema_dict["intents"]],
        fallback_intent=schema_dict["fallback_intent"],
        confidence_threshold=schema_dict["confidence_threshold"],
    )
    classifier = Classifier(schema, model=model)

    config = AgentConfig(model=model, max_iterations=5, temperature=0.7)

    print("=" * 60)
    print("Method 2: Classification for Tool Selection")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Type a question or 'quit' to exit.\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print("\nClassifying task...")

        # Classify first to suggest the best tool
        try:
            classification = classifier.classify(task)
            print(
                f"  Suggested tool: {classification.intent} "
                f"(confidence: {classification.confidence:.2f})"
            )
            print(f"  Reasoning: {classification.reasoning}")

            # Pass classification as initial context hint
            initial_context = {
                "suggested_tool": classification.intent,
                "tool_confidence": classification.confidence,
                "classification_reasoning": classification.reasoning,
            }
        except Exception as e:
            print(f"  Classification failed: {e}, running without hint")
            initial_context = {}

        print(f"\nAgent working on: {task}")
        print("-" * 40)

        try:
            agent = ReactAgent(tools=registry, config=config)
            result = agent.run(task, initial_context=initial_context)
            print(f"\nAnswer: {result.answer}")
            print(f"Tools used: {result.tools_used}")
            print(f"Iterations: {result.iterations_used}")
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
