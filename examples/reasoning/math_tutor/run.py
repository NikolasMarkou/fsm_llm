"""
Math Tutor Example — Reasoning Engine + Core FSM
==================================================

Demonstrates combining the reasoning engine with a conversational FSM.
The FSM manages the tutoring conversation flow (welcome, receive problem,
explain, clarify, practice) while the reasoning engine solves math problems
with structured step-by-step traces.

Combines:
    - fsm_llm: Core FSM for conversation management
    - fsm_llm_reasoning: ReasoningEngine for structured problem solving

Key Concepts:
    - ReasoningEngine.solve_problem() for structured solutions
    - Injecting reasoning results into FSM context
    - Conversation flow independent of reasoning logic
    - Reasoning trace displayed alongside tutoring

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os

from fsm_llm import API
from fsm_llm_reasoning import ReasoningEngine


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    # Initialize the tutoring FSM
    fsm = API.from_file(path=fsm_path, model=model, api_key=api_key, temperature=0.7)

    # Initialize the reasoning engine (shares the same model)
    reasoning = ReasoningEngine(model=model)

    print("Math Tutor with Structured Reasoning")
    print("=" * 50)

    initial_context = {
        "problems_solved": 0,
        "topics_covered": [],
    }

    conversation_id, response = fsm.start_conversation(initial_context=initial_context)
    print(f"\nTutor: {response}")

    while not fsm.has_conversation_ended(conversation_id):
        user_input = input("\nStudent: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        state = fsm.get_current_state(conversation_id)

        # When in receive_problem state, use the reasoning engine to solve
        # the problem before the FSM processes the turn
        if state == "receive_problem":
            print("\n  [Reasoning Engine working...]")
            try:
                solution, trace = reasoning.solve_problem(user_input)

                # Inject reasoning results into FSM context
                ctx = fsm.get_data(conversation_id)
                ctx["reasoning_solution"] = solution
                ctx["reasoning_steps"] = _format_trace(trace)
                ctx["problems_solved"] = ctx.get("problems_solved", 0) + 1

                topics = ctx.get("topics_covered", [])
                topic = trace.get("reasoning_type", "general")
                if topic not in topics:
                    topics.append(topic)
                ctx["topics_covered"] = topics

                print(
                    f"  [Solved using {trace.get('reasoning_type', 'unknown')} reasoning]"
                )

            except Exception as e:
                print(f"  [Reasoning engine encountered an issue: {e}]")
                # FSM will still process — it just won't have reasoning context

        # Process through the FSM for conversational response
        try:
            response = fsm.converse(user_input, conversation_id)
            print(f"\nTutor: {response}")
        except Exception as e:
            print(f"\n  [Error: {e}]")

    # Session summary
    ctx = fsm.get_data(conversation_id)
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"  Problems solved: {ctx.get('problems_solved', 0)}")
    topics = ctx.get("topics_covered", [])
    if topics:
        print(f"  Topics covered:  {', '.join(topics)}")

    fsm.end_conversation(conversation_id)


def _format_trace(trace: dict) -> str:
    """Format a reasoning trace into a readable summary for the FSM context."""
    parts = []
    reasoning_type = trace.get("reasoning_type", "unknown")
    parts.append(f"Strategy: {reasoning_type}")

    steps = trace.get("reasoning_trace", [])
    if steps:
        parts.append(f"Steps taken: {len(steps)}")
        for i, step in enumerate(steps[:5], 1):  # Show up to 5 steps
            if isinstance(step, dict):
                desc = step.get("description", step.get("step", str(step)))
            else:
                desc = str(step)
            parts.append(f"  {i}. {desc[:100]}")

    confidence = trace.get("confidence", "unknown")
    parts.append(f"Confidence: {confidence}")

    return "\n".join(parts)


if __name__ == "__main__":
    main()
