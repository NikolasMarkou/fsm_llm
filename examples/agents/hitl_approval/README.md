# Human-in-the-Loop Approval Agent

Demonstrates the **HITL (Human-in-the-Loop)** pattern where the agent must get human approval before executing sensitive actions.

## What This Example Shows

- Defining an **approval policy** (which tools need approval)
- Implementing an **approval callback** (interactive user confirmation)
- Handling **denied approvals** (agent reconsiders its approach)
- Marking tools with `requires_approval=True`
- Configuring **confidence-based escalation**

## Tools

| Tool | Approval Required | Description |
|------|:-:|-------------|
| `search_contacts` | No | Search the contact database |
| `check_calendar` | No | Check calendar availability |
| `send_email` | **Yes** | Send an email to a contact |
| `delete_record` | **Yes** | Delete a record from the database |

## How to Run

```bash
export OPENAI_API_KEY=your-key-here
python examples/agents/hitl_approval/run.py
```

## Expected Output

```
Task: Find Alice's email and send her a meeting invite for tomorrow

Agent working on: Find Alice's email and send her a meeting invite for tomorrow
----------------------------------------
==================================================
APPROVAL REQUIRED
==================================================
  Tool:       send_email
  Parameters: {'to': 'alice@example.com', 'subject': 'Meeting Invite', ...}
  Reasoning:  Found Alice's email, now sending the meeting invite
==================================================
  Approve? (y/n): y
  -> APPROVED

Answer: I found Alice Johnson's email (alice@example.com) and sent her a meeting invite for tomorrow.
Tools used: ['search_contacts', 'send_email']
Iterations: 5
```

## Learning Points

- Safe tools (search, calendar) execute immediately with no interruption
- Sensitive tools (email, delete) pause for human approval before executing
- If the user denies an approval, the agent re-thinks its approach
- The `HumanInTheLoop` class is pluggable — replace the callback with a web API, Slack bot, etc.
