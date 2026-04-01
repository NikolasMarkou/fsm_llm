"""
Meta-Agent Example: Build an Agent Configuration
=================================================

Demonstrates using the MetaBuilderAgent to interactively design an
agent configuration, including pattern selection, tool definitions,
and behavioral parameters.

The meta-agent helps users understand which agent pattern best fits
their use case and generates the configuration needed to instantiate it.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/meta/build_agent/run.py
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
    print("Meta-Agent: Build an Agent Configuration")
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
        "\nThis will guide you through building an agent configuration.\n"
        "Tell the agent you want to build an 'agent' — for example,\n"
        "a research assistant that can search the web, summarize articles,\n"
        "and generate reports using the ReAct pattern with tools.\n"
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
        print("\nGenerated Agent Configuration:")
        print(result.artifact_json)
    else:
        print("\nPartial output or validation issues:")
        for e in result.validation_errors:
            print(f"  - {e}")
        if result.artifact_json:
            print(f"\nPartial JSON:\n{result.artifact_json[:500]}")

    # ── Verification ──
    artifact = result.artifact if hasattr(result, "artifact") else {}
    has_tools = isinstance(artifact, dict) and bool(
        artifact.get("tools") or artifact.get("agent_type")
    )
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "artifact_generated": artifact is not None and artifact != {},
        "artifact_valid": result.is_valid,
        "artifact_type": result.artifact_type.value
        if hasattr(result.artifact_type, "value")
        else str(result.artifact_type),
        "has_tools_or_type": has_tools,
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
