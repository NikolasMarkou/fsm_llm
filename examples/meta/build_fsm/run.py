"""
Meta-Agent Example: Interactively Build an FSM
===============================================

This example starts the meta-agent in interactive mode, which will
guide you through building an FSM definition step by step.

The agent will:
1. Ask what type of artifact you want to build (FSM/Workflow/Agent)
2. Gather a name, description, and persona
3. Help you design states
4. Help you define transitions
5. Validate and output the final JSON

Usage:
    python examples/meta/build_fsm/run.py

Environment Variables:
    LLM_MODEL       - LLM model to use (default: gpt-4o-mini)
    OPENAI_API_KEY  - OpenAI API key (if using OpenAI models)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    # Try OpenAI first, fall back to Ollama
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            model = "gpt-4o-mini"
        else:
            model = "ollama_chat/qwen3.5:4b"

    print(f"Using model: {model}")
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

    try:
        result = agent.run_interactive()
    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during build: {e}")
        return

    print("\n" + "=" * 60)
    print(f"Build {'succeeded' if result.is_valid else 'failed'}!")
    print(f"Artifact type: {result.artifact_type.value}")
    print(f"Turns: {result.conversation_turns}")

    if result.is_valid:
        print("\nGenerated JSON:")
        print(result.artifact_json)

        # Optionally save to file
        output_path = os.environ.get("OUTPUT_PATH")
        if output_path:
            from fsm_llm_agents import save_artifact

            try:
                path = save_artifact(result.artifact, output_path)
                print(f"\nSaved to: {path}")
            except Exception as e:
                print(f"\nError saving artifact: {e}")
    else:
        print("\nValidation errors:")
        for e in result.validation_errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
