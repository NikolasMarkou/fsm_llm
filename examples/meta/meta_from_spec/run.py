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
        from fsm_llm_agents import AgentBuilder, FSMBuilder, WorkflowBuilder
    except ImportError:
        print("Error: fsm_llm_agents not installed.")
        print("Install with: pip install -e '.[agents]'")
        return

    # ── Build 1: FSM from specification ──
    print("\n[Build 1] FSM — Appointment booking bot")
    print("-" * 40)

    fsm_spec = (
        "Create an appointment booking chatbot with these states:\n"
        "1. greeting: Welcome and ask what service they need\n"
        "2. select_service: Collect service type (haircut, coloring, styling)\n"
        "3. select_date: Ask for preferred date and time\n"
        "4. confirm: Show appointment summary and ask for confirmation\n"
        "5. booked: Confirm booking and provide reference number\n"
        "Transitions: greeting -> select_service (always), "
        "select_service -> select_date (when service selected), "
        "select_date -> confirm (when date selected), "
        "confirm -> booked (when confirmed), "
        "confirm -> select_date (when user wants to change).\n"
        "Persona: A professional salon receptionist."
    )

    try:
        fsm_builder = FSMBuilder(model=model)
        fsm_result = fsm_builder.build(fsm_spec)

        if fsm_result and isinstance(fsm_result, dict):
            states = fsm_result.get("states", {})
            print(f"  Generated: {fsm_result.get('name', 'unnamed')}")
            print(f"  States: {len(states)} ({', '.join(list(states.keys())[:5])})")
            print(f"  Initial: {fsm_result.get('initial_state', 'N/A')}")
            transitions = sum(len(s.get("transitions", [])) for s in states.values())
            print(f"  Transitions: {transitions}")
        else:
            print(f"  Result: {str(fsm_result)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Build 2: Workflow from specification ──
    print("\n[Build 2] Workflow — Document processing pipeline")
    print("-" * 40)

    workflow_spec = (
        "Create a document processing workflow:\n"
        "1. upload: Receive and validate document format (PDF, DOCX)\n"
        "2. extract: Extract text content from the document\n"
        "3. classify: Classify document type (invoice, contract, report)\n"
        "4. process: Route to type-specific processing\n"
        "5. store: Save processed results to database\n"
        "6. notify: Send completion notification\n"
        "Error handling: any step failure goes to 'error_handler' step.\n"
        "Processing should support parallel extraction of metadata."
    )

    try:
        workflow_builder = WorkflowBuilder(model=model)
        wf_result = workflow_builder.build(workflow_spec)

        if wf_result and isinstance(wf_result, dict):
            print(f"  Generated: {wf_result.get('name', 'unnamed')}")
            steps = wf_result.get("steps", wf_result.get("states", {}))
            if isinstance(steps, dict):
                print(f"  Steps: {len(steps)} ({', '.join(list(steps.keys())[:5])})")
            elif isinstance(steps, list):
                print(f"  Steps: {len(steps)}")
        else:
            print(f"  Result: {str(wf_result)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Build 3: Agent from specification ──
    print("\n[Build 3] Agent — Customer support assistant")
    print("-" * 40)

    agent_spec = (
        "Create a customer support agent configuration:\n"
        "- Pattern: ReAct (tool-using)\n"
        "- Tools: search_kb (search knowledge base), create_ticket (create support ticket), "
        "check_order (check order status), escalate (escalate to human)\n"
        "- Max iterations: 8\n"
        "- Should check order status before creating tickets\n"
        "- Should escalate if unable to resolve in 3 tool calls\n"
        "- Temperature: 0.3 for consistent responses"
    )

    try:
        agent_builder = AgentBuilder(model=model)
        agent_result = agent_builder.build(agent_spec)

        if agent_result and isinstance(agent_result, dict):
            print(f"  Generated: {agent_result.get('name', 'unnamed')}")
            print(f"  Pattern: {agent_result.get('pattern', 'N/A')}")
            tools = agent_result.get("tools", [])
            if isinstance(tools, list):
                print(f"  Tools: {len(tools)}")
                for t in tools[:4]:
                    name = t.get("name", "?") if isinstance(t, dict) else str(t)
                    print(f"    - {name}")
        else:
            print(f"  Result: {str(agent_result)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print("All builds complete.")


if __name__ == "__main__":
    main()
