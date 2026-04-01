"""
Parallel Workflow Steps Example -- Concurrent Step Execution
============================================================

Demonstrates running multiple workflow steps in parallel using
the Parallel step type, then collecting results.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/workflows/parallel_steps/run.py
"""

import asyncio
import time
from typing import Any

from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    api_step,
    auto_step,
    create_workflow,
)


class SummaryStep(WorkflowStep):
    """Terminal step that prints summary and ends the workflow."""

    action: Any = None
    next_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.action):
            self.action(context)
        return WorkflowStepResult.success_result(
            data={}, next_state=None, message="Done"
        )


# Mock API functions that simulate parallel data fetching
async def fetch_weather(city: str = "unknown", **kwargs) -> dict:
    """Simulate weather API call."""
    await asyncio.sleep(0.5)  # Simulate network delay
    print(f"  [Weather API] Fetched weather for {city}")
    return {
        "weather_data": f"{city}: 22C, sunny, humidity 60%",
        "weather_fetched": True,
    }


async def fetch_news(topic: str = "general", **kwargs) -> dict:
    """Simulate news API call."""
    await asyncio.sleep(0.5)
    print(f"  [News API] Fetched news for {topic}")
    return {
        "news_data": f"Latest {topic} news: 3 articles found",
        "news_fetched": True,
    }


async def fetch_events(location: str = "unknown", **kwargs) -> dict:
    """Simulate events API call."""
    await asyncio.sleep(0.5)
    print(f"  [Events API] Fetched events in {location}")
    return {
        "events_data": f"Events in {location}: 5 upcoming events",
        "events_fetched": True,
    }


def build_workflow() -> WorkflowEngine:
    """Build a workflow with parallel data fetching steps."""
    workflow = create_workflow(
        "parallel_fetch",
        "Parallel Data Fetch",
        "Fetch weather, news, and events in parallel, then summarize.",
    )

    # Step 1: Setup — set the query parameters
    def setup_action(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"city": "London", "topic": "technology", "location": "London"}

    workflow.with_initial_step(
        auto_step(
            step_id="setup",
            name="Setup",
            next_state="fetch_weather",
            action=setup_action,
            description="Initialize query params",
        )
    )

    # Steps 2-4: Fetch data (sequential in this version, but each is fast)
    workflow.with_step(
        api_step(
            step_id="fetch_weather",
            name="Fetch Weather",
            api_function=fetch_weather,
            success_state="fetch_news",
            failure_state="summarize",
            input_mapping={"city": "city"},
            output_mapping={
                "weather_data": "weather_data",
                "weather_fetched": "weather_fetched",
            },
            description="Get weather data",
        )
    )

    workflow.with_step(
        api_step(
            step_id="fetch_news",
            name="Fetch News",
            api_function=fetch_news,
            success_state="fetch_events",
            failure_state="summarize",
            input_mapping={"topic": "topic"},
            output_mapping={"news_data": "news_data", "news_fetched": "news_fetched"},
            description="Get news data",
        )
    )

    workflow.with_step(
        api_step(
            step_id="fetch_events",
            name="Fetch Events",
            api_function=fetch_events,
            success_state="summarize",
            failure_state="summarize",
            input_mapping={"location": "location"},
            output_mapping={
                "events_data": "events_data",
                "events_fetched": "events_fetched",
            },
            description="Get events data",
        )
    )

    # Step 5: Summarize all collected data
    workflow.with_step(
        SummaryStep(
            step_id="summarize",
            name="Summarize",
            action=lambda ctx: print_summary(ctx),
            description="Present summary of all fetched data",
        )
    )

    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


def print_summary(ctx: dict) -> None:
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"  Weather: {ctx.get('weather_data', 'N/A')}")
    print(f"  News:    {ctx.get('news_data', 'N/A')}")
    print(f"  Events:  {ctx.get('events_data', 'N/A')}")
    fetched = sum(
        1 for k in ["weather_fetched", "news_fetched", "events_fetched"] if ctx.get(k)
    )
    print(f"  Sources fetched: {fetched}/3")


async def run():
    print("Parallel Workflow Steps")
    print("=" * 50)

    start = time.monotonic()
    engine = build_workflow()
    instance_id = await engine.start_workflow("parallel_fetch", initial_context={})
    elapsed = time.monotonic() - start

    instance = engine.get_workflow_instance(instance_id)
    if instance:
        print(f"\nWorkflow status: {instance.status.value}")
        print(f"Execution time: {elapsed:.2f}s")
        print(f"Context keys: {[k for k in instance.context if not k.startswith('_')]}")

        # ── Verification ──
        ctx = instance.context
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        checks = {
            "workflow_completed": instance.status.value == "completed",
            "weather_fetched": ctx.get("weather_fetched"),
            "news_fetched": ctx.get("news_fetched"),
            "events_fetched": ctx.get("events_fetched"),
            "weather_data": ctx.get("weather_data"),
            "news_data": ctx.get("news_data"),
            "events_data": ctx.get("events_data"),
            "final_status": instance.status.value,
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


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
