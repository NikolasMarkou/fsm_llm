"""
Meta-Agent Example: Build a Workflow Definition
================================================

Demonstrates using the MetaBuilderAgent to interactively build a
workflow definition. The meta-agent guides the user through designing
a multi-step workflow with conditional branching.

Unlike build_fsm which creates FSMs, this example targets workflow
artifacts with step definitions, transitions, and context mappings.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/meta/build_workflow/run.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = "gpt-4o-mini" if api_key else "ollama_chat/qwen3.5:4b"

    print(f"Using model: {model}")
    print("=" * 60)
    print("Meta-Agent: Build a Workflow Definition")
    print("=" * 60)

    try:
        from fsm_llm_agents import MetaBuilderAgent, MetaBuilderConfig
    except ImportError:
        print("Error: fsm_llm_agents not installed.")
        print("Install with: pip install -e '.[agents]'")
        return

    config = MetaBuilderConfig(
        model=model,
        temperature=0.7,
        max_tokens=2000,
        max_turns=50,
    )

    agent = MetaBuilderAgent(config=config)

    print(
        "\nThis will guide you through building a workflow definition.\n"
        "Tell the agent you want to build a 'workflow' for an order\n"
        "processing pipeline with validation, payment, and fulfillment steps.\n"
    )

    try:
        result = agent.run_interactive()
    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during build: {e}")
        return

    print("\n" + "=" * 60)
    print(f"Build {'succeeded' if result.is_valid else 'completed (with issues)'}!")
    print(f"Artifact type: {result.artifact_type.value}")
    print(f"Turns: {result.conversation_turns}")

    if result.is_valid:
        print("\nGenerated Workflow JSON:")
        print(result.artifact_json)
    else:
        print("\nPartial output or validation issues:")
        for e in result.validation_errors:
            print(f"  - {e}")
        if result.artifact_json:
            print(f"\nPartial JSON:\n{result.artifact_json[:500]}")


if __name__ == "__main__":
    main()
