"""
Combined ReAct + HITL Example — Research Assistant
====================================================

A research assistant agent that combines ReAct tool use with
Human-in-the-Loop approval for a realistic agentic workflow.

The agent can:
  - Search for information (free)
  - Analyze data (free)
  - Summarize findings (free)
  - Publish reports (requires approval)
  - Archive old data (requires approval)

This demonstrates:
  - Multi-tool ReAct loop with 5 tools
  - HITL approval on publishing/archiving actions
  - @tool decorator for registration
  - Custom AgentConfig tuning
  - Escalation callback for low-confidence situations

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/react_hitl_combined/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    ApprovalRequest,
    HumanInTheLoop,
    ReactAgent,
    ToolRegistry,
    tool,
)

# ──────────────────────────────────────────────
# Tools (using @tool decorator)
# ──────────────────────────────────────────────


@tool(
    description="Search academic papers and articles on a topic",
    parameter_schema={
        "properties": {
            "query": {"type": "string", "description": "Research topic to search for"},
        }
    },
)
def search_papers(params: dict) -> str:
    """Search for academic papers."""
    query = params.get("query", "").lower()

    papers = {
        "climate": (
            "Found 3 papers:\n"
            "1. 'Global Temperature Trends 2020-2025' — Nature Climate (2024)\n"
            "2. 'Carbon Capture Technologies: A Review' — Science (2023)\n"
            "3. 'Ocean Acidification and Marine Ecosystems' — PNAS (2024)"
        ),
        "ai": (
            "Found 3 papers:\n"
            "1. 'Large Language Models: Capabilities and Limitations' — NeurIPS (2024)\n"
            "2. 'Reinforcement Learning from Human Feedback' — ICML (2023)\n"
            "3. 'Multimodal AI Systems' — CVPR (2024)"
        ),
        "health": (
            "Found 3 papers:\n"
            "1. 'mRNA Vaccine Platform Advances' — NEJM (2024)\n"
            "2. 'AI-Assisted Diagnostics in Radiology' — Lancet (2023)\n"
            "3. 'Mental Health and Remote Work' — BMJ (2024)"
        ),
    }

    for key, result in papers.items():
        if key in query:
            return result

    return f"Found 0 papers for '{query}'. Try topics like climate, AI, or health."


@tool(
    description="Analyze and extract key findings from research data",
    parameter_schema={
        "properties": {
            "data": {
                "type": "string",
                "description": "Research data or paper summaries to analyze",
            },
        }
    },
)
def analyze_data(params: dict) -> str:
    """Analyze research data for key findings."""
    data = params.get("data", "")
    word_count = len(data.split())
    return (
        f"Analysis complete ({word_count} words processed):\n"
        f"- Key themes: multiple emerging trends identified\n"
        f"- Confidence: high (based on recent publications)\n"
        f"- Recommendation: suitable for report synthesis"
    )


@tool(
    description="Generate a structured summary from research findings",
    parameter_schema={
        "properties": {
            "topic": {"type": "string", "description": "Topic of the summary"},
            "findings": {"type": "string", "description": "Key findings to summarize"},
        }
    },
)
def summarize(params: dict) -> str:
    """Generate a research summary."""
    topic = params.get("topic", "research")
    return (
        f"Research Summary: {topic}\n"
        f"===========================\n"
        f"This summary synthesizes recent findings across multiple peer-reviewed publications.\n"
        f"The analysis indicates significant progress in the field with several actionable insights.\n"
        f"Key takeaway: continued investment in research and cross-disciplinary collaboration is recommended."
    )


@tool(
    description="Publish a research report to the shared repository (requires approval)",
    parameter_schema={
        "properties": {
            "title": {"type": "string", "description": "Report title"},
            "content": {"type": "string", "description": "Report content"},
            "visibility": {"type": "string", "description": "public or private"},
        }
    },
    requires_approval=True,
)
def publish_report(params: dict) -> str:
    """Publish a report to the repository."""
    title = params.get("title", "Untitled")
    visibility = params.get("visibility", "private")
    return f"Report '{title}' published as {visibility}. URL: https://reports.example.com/r/12345"


@tool(
    description="Archive old research data (requires approval, irreversible)",
    parameter_schema={
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "ID of the dataset to archive",
            },
            "reason": {"type": "string", "description": "Reason for archiving"},
        }
    },
    requires_approval=True,
)
def archive_data(params: dict) -> str:
    """Archive a dataset (irreversible)."""
    dataset_id = params.get("dataset_id", "unknown")
    reason = params.get("reason", "no reason given")
    return f"Dataset {dataset_id} archived. Reason: {reason}. This action cannot be undone."


# ──────────────────────────────────────────────
# HITL Configuration
# ──────────────────────────────────────────────


def approval_policy(tool_call, context: dict) -> bool:
    """Approval policy: require approval for publish and archive actions."""
    return tool_call.tool_name in ("publish_report", "archive_data")


def approval_callback(request: ApprovalRequest) -> bool:
    """Interactive approval prompt."""
    print("\n" + "-" * 50)
    print(f"  APPROVAL NEEDED: {request.tool_name}")
    print(f"  Parameters: {request.parameters}")
    if request.reasoning:
        print(f"  Agent reasoning: {request.reasoning}")
    print("-" * 50)

    response = input("  Approve this action? (y/n): ").strip().lower()
    approved = response in ("y", "yes")
    print(f"  -> {'APPROVED' if approved else 'DENIED'}")
    return approved


def on_escalation(reason: str, context: dict) -> None:
    """Handle agent escalation."""
    print(f"\n[ESCALATION] {reason}")


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

    # Register tools from decorated functions
    registry = ToolRegistry()
    for fn in [search_papers, analyze_data, summarize, publish_report, archive_data]:
        registry.register(fn._tool_definition)

    # Create HITL manager
    hitl = HumanInTheLoop(
        approval_policy=approval_policy,
        approval_callback=approval_callback,
        on_escalation=on_escalation,
        confidence_threshold=0.3,
    )

    # Create the agent
    config = AgentConfig(
        model=model,
        max_iterations=6,
        temperature=0.7,
        timeout_seconds=120.0,
    )
    agent = ReactAgent(tools=registry, config=config, hitl=hitl)

    print("=" * 60)
    print("Research Assistant (ReAct + HITL)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Approval required for: publish_report, archive_data")
    print()
    print("Example tasks:")
    print("  'Research recent AI papers and publish a summary report'")
    print("  'Find climate change papers, analyze them, and summarize'")
    print("  'Search for health papers'")
    print()

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
            print("\nTrace:")
            print(f"  Tools used: {result.tools_used}")
            print(f"  Iterations: {result.iterations_used}")
            if result.final_context.get("observations"):
                obs = result.final_context["observations"]
                print(f"  Observations collected: {len(obs)}")
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
