"""
Human-in-the-Loop Approval Example
====================================

Demonstrates the HITL (Human-in-the-Loop) pattern where certain
agent actions require explicit human approval before execution.

The agent can search freely, but sending emails or deleting records
requires the user to approve each action.

Patterns demonstrated:
  - Approval policies (which tools need approval)
  - Approval callbacks (interactive user confirmation)
  - Confidence-based escalation
  - Denial handling (agent reconsiders on denial)

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/hitl_approval/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    ApprovalRequest,
    HumanInTheLoop,
    ReactAgent,
    ToolRegistry,
)

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search_contacts(params: dict) -> str:
    """Search the contact database."""
    query = params.get("query", "").lower()

    contacts = {
        "alice": "Alice Johnson <alice@example.com> — Engineering Manager",
        "bob": "Bob Smith <bob@example.com> — Product Lead",
        "carol": "Carol Davis <carol@example.com> — Designer",
        "team": "Team mailing list: team@example.com (Alice, Bob, Carol)",
    }

    for name, info in contacts.items():
        if name in query:
            return f"Found: {info}"

    return f"No contacts matching '{query}'. Available: Alice, Bob, Carol, Team."


def send_email(params: dict) -> str:
    """Send an email (simulated). This is a sensitive action."""
    to = params.get("to", "unknown")
    subject = params.get("subject", "No subject")
    body = params.get("body", "")
    return f"Email sent successfully to {to} with subject '{subject}'. Body: {body[:100]}..."


def check_calendar(params: dict) -> str:
    """Check calendar availability."""
    date = params.get("date", "today")
    return f"Calendar for {date}: 10am-11am Team Standup, 2pm-3pm Design Review, rest is free."


def delete_record(params: dict) -> str:
    """Delete a record from the database. This is a destructive action."""
    record_id = params.get("record_id", "unknown")
    return f"Record {record_id} has been deleted from the database."


# ──────────────────────────────────────────────
# HITL Callbacks
# ──────────────────────────────────────────────


def approval_policy(tool_call, context: dict) -> bool:
    """Decide which tool calls require human approval."""
    # Sensitive tools that need approval
    sensitive_tools = {"send_email", "delete_record"}
    return tool_call.tool_name in sensitive_tools


def approval_callback(request: ApprovalRequest) -> bool:
    """Ask the user for approval interactively."""
    print("\n" + "=" * 50)
    print("APPROVAL REQUIRED")
    print("=" * 50)
    print(f"  Tool:       {request.tool_name}")
    print(f"  Parameters: {request.parameters}")
    print(f"  Reasoning:  {request.reasoning}")
    print("=" * 50)

    while True:
        response = input("  Approve? (y/n): ").strip().lower()
        if response in ("y", "yes"):
            print("  -> APPROVED")
            return True
        if response in ("n", "no"):
            print("  -> DENIED")
            return False
        print("  Please enter 'y' or 'n'.")


def on_escalation(reason: str, context: dict) -> None:
    """Handle escalation to human."""
    print(f"\n[ESCALATION] Agent needs help: {reason}")
    print(f"  Current context keys: {list(context.keys())[:5]}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    # Build tool registry (mark sensitive tools)
    registry = ToolRegistry()
    registry.register_function(
        search_contacts,
        name="search_contacts",
        description="Search the contact database for people by name",
        parameter_schema={
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name or role to search for",
                },
            }
        },
    )
    registry.register_function(
        send_email,
        name="send_email",
        description="Send an email to a contact (requires approval)",
        parameter_schema={
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
            }
        },
        requires_approval=True,
    )
    registry.register_function(
        check_calendar,
        name="check_calendar",
        description="Check calendar availability for a given date",
        parameter_schema={
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to check (e.g., 'today', 'tomorrow')",
                },
            }
        },
    )
    registry.register_function(
        delete_record,
        name="delete_record",
        description="Delete a record from the database (requires approval)",
        parameter_schema={
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "ID of the record to delete",
                },
            }
        },
        requires_approval=True,
    )

    # Create HITL manager
    hitl = HumanInTheLoop(
        approval_policy=approval_policy,
        approval_callback=approval_callback,
        on_escalation=on_escalation,
        confidence_threshold=0.3,
    )

    # Create the agent with HITL
    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.7,
    )
    agent = ReactAgent(tools=registry, config=config, hitl=hitl)

    print("=" * 60)
    print("HITL Approval Agent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Sensitive tools (require approval): send_email, delete_record")
    print("Type a task or 'quit' to exit.\n")
    print("Try: 'Find Alice's email and send her a meeting invite for tomorrow'")
    print("Try: 'Check my calendar for today'\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print(f"\nAgent working on: {task}")
        print("-" * 40)

        try:
            result = agent.run(task)
            print(f"\nAnswer: {result.answer}")
            print(f"Tools used: {result.tools_used}")
            print(f"Iterations: {result.iterations_used}")
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
