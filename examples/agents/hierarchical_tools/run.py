"""
Method 5: HierarchicalClassifier for Large Tool Sets
=====================================================

Demonstrates using ``HierarchicalClassifier`` for agents with many
tools (12+). Two-stage classification: first classify the domain
(data, communication, analysis), then the specific tool within
that domain.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/hierarchical_tools/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/hierarchical_tools/run.py
"""

import os

from fsm_llm import (
    ClassificationSchema,
    HierarchicalClassifier,
    HierarchicalSchema,
    IntentDefinition,
)
from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry

# ──────────────────────────────────────────────
# Tool Definitions — 12 tools across 3 domains
# ──────────────────────────────────────────────

# --- Data domain ---


def query_database(params: dict) -> str:
    """Query a database for records."""
    table = params.get("table", "users")
    return f"Database query on '{table}': 42 records found."


def read_csv(params: dict) -> str:
    """Read and parse a CSV file."""
    filename = params.get("filename", "data.csv")
    return f"CSV '{filename}': 100 rows, 5 columns (id, name, value, date, status)."


def write_json(params: dict) -> str:
    """Write data to a JSON file."""
    filename = params.get("filename", "output.json")
    return f"Data written to '{filename}' successfully."


def fetch_api(params: dict) -> str:
    """Fetch data from an external API."""
    url = params.get("url", "api.example.com")
    return f'API response from \'{url}\': {{"status": "ok", "data": [...]}}.'


# --- Communication domain ---


def send_email(params: dict) -> str:
    """Send an email message."""
    to = params.get("to", "user@example.com")
    return f"Email sent to {to} successfully."


def send_slack(params: dict) -> str:
    """Send a Slack message."""
    channel = params.get("channel", "#general")
    return f"Slack message sent to {channel}."


def create_ticket(params: dict) -> str:
    """Create a support ticket."""
    title = params.get("title", "New ticket")
    return f"Ticket created: '{title}' (ID: TICKET-42)."


def schedule_meeting(params: dict) -> str:
    """Schedule a calendar meeting."""
    title = params.get("title", "Meeting")
    return f"Meeting '{title}' scheduled for tomorrow at 2pm."


# --- Analysis domain ---


def run_statistics(params: dict) -> str:
    """Run statistical analysis on data."""
    dataset = params.get("dataset", "sales")
    return f"Statistics for '{dataset}': mean=42.5, median=40, std=8.3."


def generate_chart(params: dict) -> str:
    """Generate a chart visualization."""
    chart_type = params.get("type", "bar")
    return f"{chart_type.capitalize()} chart generated with 5 data series."


def predict_trend(params: dict) -> str:
    """Predict future trends from historical data."""
    metric = params.get("metric", "revenue")
    return f"Trend prediction for '{metric}': +12% growth expected next quarter."


def compare_datasets(params: dict) -> str:
    """Compare two datasets for differences."""
    dataset_a = params.get("dataset_a", "current")
    dataset_b = params.get("dataset_b", "previous")
    return f"Comparison of '{dataset_a}' vs '{dataset_b}': 3 significant differences found."


# ──────────────────────────────────────────────
# Build Hierarchical Schema
# ──────────────────────────────────────────────


def _build_hierarchical_schema() -> HierarchicalSchema:
    """Build a two-stage classification schema for 12 tools across 3 domains."""
    # Stage 1: Domain classification
    domain_schema = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="data",
                description="Data operations: database queries, file reading/writing, API calls",
            ),
            IntentDefinition(
                name="communication",
                description="Communication: emails, Slack messages, tickets, meetings",
            ),
            IntentDefinition(
                name="analysis",
                description="Data analysis: statistics, charts, predictions, comparisons",
            ),
            IntentDefinition(
                name="general",
                description="General queries that don't fit a specific domain",
            ),
        ],
        fallback_intent="general",
        confidence_threshold=0.4,
    )

    # Stage 2: Intent schemas per domain
    data_schema = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="query_database", description="Query a database for records"
            ),
            IntentDefinition(name="read_csv", description="Read and parse a CSV file"),
            IntentDefinition(
                name="write_json", description="Write data to a JSON file"
            ),
            IntentDefinition(
                name="fetch_api", description="Fetch data from an external API"
            ),
            IntentDefinition(name="none", description="No specific data tool needed"),
        ],
        fallback_intent="none",
        confidence_threshold=0.4,
    )

    comm_schema = ClassificationSchema(
        intents=[
            IntentDefinition(name="send_email", description="Send an email message"),
            IntentDefinition(name="send_slack", description="Send a Slack message"),
            IntentDefinition(
                name="create_ticket", description="Create a support ticket"
            ),
            IntentDefinition(
                name="schedule_meeting", description="Schedule a calendar meeting"
            ),
            IntentDefinition(
                name="none", description="No specific communication tool needed"
            ),
        ],
        fallback_intent="none",
        confidence_threshold=0.4,
    )

    analysis_schema = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="run_statistics", description="Run statistical analysis"
            ),
            IntentDefinition(
                name="generate_chart", description="Generate a chart visualization"
            ),
            IntentDefinition(name="predict_trend", description="Predict future trends"),
            IntentDefinition(
                name="compare_datasets", description="Compare two datasets"
            ),
            IntentDefinition(
                name="none", description="No specific analysis tool needed"
            ),
        ],
        fallback_intent="none",
        confidence_threshold=0.4,
    )

    return HierarchicalSchema(
        domain_schema=domain_schema,
        intent_schemas={
            "data": data_schema,
            "communication": comm_schema,
            "analysis": analysis_schema,
        },
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    # Build tool registry with all 12 tools
    registry = ToolRegistry()
    tools = [
        (query_database, "query_database", "Query a database for records"),
        (read_csv, "read_csv", "Read and parse a CSV file"),
        (write_json, "write_json", "Write data to a JSON file"),
        (fetch_api, "fetch_api", "Fetch data from an external API"),
        (send_email, "send_email", "Send an email message"),
        (send_slack, "send_slack", "Send a Slack message"),
        (create_ticket, "create_ticket", "Create a support ticket"),
        (schedule_meeting, "schedule_meeting", "Schedule a calendar meeting"),
        (run_statistics, "run_statistics", "Run statistical analysis on data"),
        (generate_chart, "generate_chart", "Generate a chart visualization"),
        (predict_trend, "predict_trend", "Predict future trends from data"),
        (compare_datasets, "compare_datasets", "Compare two datasets"),
    ]
    for fn, name, desc in tools:
        registry.register_function(fn, name=name, description=desc)

    # Build hierarchical classifier
    h_schema = _build_hierarchical_schema()
    classifier = HierarchicalClassifier(h_schema, model=model)

    config = AgentConfig(model=model, max_iterations=8, temperature=0.7)

    print("=" * 60)
    print("Method 5: HierarchicalClassifier for Large Tool Sets")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools ({len(registry)}): {', '.join(registry.tool_names)}")
    print("Domains: data, communication, analysis")
    print("Type a question or 'quit' to exit.\n")

    last_result = None
    tasks_completed = 0

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print("\nHierarchical classification...")
        try:
            h_result = classifier.classify(task)
            print(
                f"  Domain: {h_result.domain.intent} (confidence: {h_result.domain.confidence:.2f})"
            )
            print(
                f"  Tool: {h_result.intent.intent} (confidence: {h_result.intent.confidence:.2f})"
            )

            initial_context = {
                "suggested_domain": h_result.domain.intent,
                "suggested_tool": h_result.intent.intent,
                "domain_confidence": h_result.domain.confidence,
                "tool_confidence": h_result.intent.confidence,
            }
        except Exception as e:
            print(f"  Classification failed: {e}")
            initial_context = {}

        print(f"\nAgent working on: {task}")
        print("-" * 40)

        try:
            agent = ReactAgent(tools=registry, config=config)
            result = agent.run(task, initial_context=initial_context)
            print(f"\nAnswer: {result.answer}")
            print(f"Tools used: {result.tools_used}")
            last_result = result
            tasks_completed += 1
        except Exception as e:
            print(f"Error: {e}")
        print()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": last_result is not None and len(str(last_result.answer)) > 10,
        "iterations_ok": last_result is not None and last_result.iterations_used >= 1,
        "tools_called": last_result is not None and len(last_result.tools_used) > 0,
        "completed": tasks_completed >= 1,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
