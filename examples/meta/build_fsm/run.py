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
        from fsm_llm.stdlib.agents import MetaBuilderAgent, MetaBuilderConfig
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
            from fsm_llm.stdlib.agents import save_artifact

            try:
                path = save_artifact(result.artifact, output_path)
                print(f"\nSaved to: {path}")
            except Exception as e:
                print(f"\nError saving artifact: {e}")
    else:
        print("\nValidation errors:")
        for e in result.validation_errors:
            print(f"  - {e}")

    # ── Verification ──
    artifact = result.artifact if hasattr(result, "artifact") else {}
    has_states = isinstance(artifact, dict) and bool(artifact.get("states"))
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "artifact_generated": artifact is not None and artifact != {},
        "artifact_valid": result.is_valid,
        "artifact_type": result.artifact_type.value
        if hasattr(result.artifact_type, "value")
        else str(result.artifact_type),
        "has_states": has_states,
        "has_initial_state": isinstance(artifact, dict)
        and bool(artifact.get("initial_state")),
        "conversation_turns": result.conversation_turns,
    }
    extracted = 0
    for key, value in checks.items():
        passed = value is not None and value not in (False, 0, "", "failed")
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
