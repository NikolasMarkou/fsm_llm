"""
Meta-Builder from Text Specification
======================================

Demonstrates the programmatic (non-interactive) MetaBuilder API.
Instead of interactive conversation, this passes a complete text
specification and gets back a validated artifact.

This is useful for automated pipelines where FSM/workflow/agent
definitions need to be generated from structured requirements.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/meta/meta_from_spec/run.py
"""

import os

from dotenv import load_dotenv

load_dotenv()


def main():
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = "gpt-4o-mini" if api_key else "ollama_chat/qwen3.5:4b"

    print(f"Using model: {model}")
    print("=" * 60)
    print("Meta-Builder: Programmatic FSM from Spec")
    print("=" * 60)

    try:
        from fsm_llm_agents import MetaBuilderAgent
        from fsm_llm_agents.definitions import MetaBuilderConfig
    except ImportError:
        print("Error: fsm_llm_agents not installed.")
        print("Install with: pip install -e '.[agents]'")
        return

    config = MetaBuilderConfig(model=model, max_iterations=20, temperature=0.5)

    # ── Build 1: FSM from specification ──
    print("\n[Build 1] FSM — Appointment booking bot")
    print("-" * 40)

    fsm_spec = (
        "Build an FSM for an appointment booking chatbot with states: "
        "greeting (welcome, ask what service), "
        "select_service (collect service type), "
        "select_date (ask for date and time), "
        "confirm (show summary, ask confirmation), "
        "booked (confirm booking, give reference number). "
        "Persona: A professional salon receptionist."
    )

    try:
        agent = MetaBuilderAgent(config=config)
        result = agent.run(fsm_spec)

        artifact = result.artifact if hasattr(result, "artifact") else {}
        if artifact and isinstance(artifact, dict) and artifact.get("states"):
            states = artifact.get("states", {})
            if isinstance(states, dict):
                print(f"  Generated: {artifact.get('name', 'unnamed')}")
                print(f"  States: {len(states)} ({', '.join(list(states.keys())[:5])})")
                print(f"  Initial: {artifact.get('initial_state', 'N/A')}")
                transitions = sum(
                    len(s.get("transitions", [])) for s in states.values()
                )
                print(f"  Transitions: {transitions}")
            else:
                print(f"  Generated: {artifact.get('name', 'unnamed')}")
                print(f"  States: {len(states)}")
        else:
            print(f"  Result: {str(result.answer)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Build 2: Workflow from specification ──
    print("\n[Build 2] Workflow — Document processing pipeline")
    print("-" * 40)

    workflow_spec = (
        "Build a workflow for document processing with steps: "
        "upload (receive and validate document), "
        "extract (extract text content), "
        "classify (classify document type), "
        "process (route to type-specific processing), "
        "store (save processed results), "
        "notify (send completion notification)."
    )

    try:
        agent = MetaBuilderAgent(config=config)
        result = agent.run(workflow_spec)

        artifact = result.artifact if hasattr(result, "artifact") else {}
        if (
            artifact
            and isinstance(artifact, dict)
            and (artifact.get("steps") or artifact.get("states"))
        ):
            steps = artifact.get("steps", artifact.get("states", {}))
            print(f"  Generated: {artifact.get('name', 'unnamed')}")
            if isinstance(steps, dict):
                print(f"  Steps: {len(steps)} ({', '.join(list(steps.keys())[:5])})")
            elif isinstance(steps, list):
                print(f"  Steps: {len(steps)}")
        else:
            print(f"  Result: {str(result.answer)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Build 3: Agent from specification ──
    print("\n[Build 3] Agent — Customer support assistant")
    print("-" * 40)

    agent_spec = (
        "Build an agent for customer support using ReAct pattern with tools: "
        "search_kb (search knowledge base), create_ticket (create support ticket), "
        "check_order (check order status), escalate (escalate to human). "
        "Max iterations: 8. Temperature: 0.3."
    )

    try:
        agent = MetaBuilderAgent(config=config)
        result = agent.run(agent_spec)

        artifact = result.artifact if hasattr(result, "artifact") else {}
        if (
            artifact
            and isinstance(artifact, dict)
            and (artifact.get("tools") or artifact.get("agent_type"))
        ):
            print(f"  Generated: {artifact.get('name', 'unnamed')}")
            print(
                f"  Pattern: {artifact.get('agent_type', artifact.get('pattern', 'N/A'))}"
            )
            tools = artifact.get("tools", [])
            if isinstance(tools, list):
                print(f"  Tools: {len(tools)}")
                for t in tools[:4]:
                    name = t.get("name", "?") if isinstance(t, dict) else str(t)
                    print(f"    - {name}")
        else:
            print(f"  Result: {str(result.answer)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print("All builds complete.")


if __name__ == "__main__":
    main()
