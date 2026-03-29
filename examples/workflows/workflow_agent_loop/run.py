"""
Workflow-Agent Loop — Quality-Gated Agent Execution
=====================================================

Demonstrates a workflow that executes an agent step with built-in
quality checking and retry logic. The agent step runs, evaluates
its own output quality, and retries internally if quality is below
threshold (up to a configurable retry limit).

Flow:
  Setup → Agent with Quality Gate → Format Output → Done

The quality gate is implemented inside the agent step, which
internally retries up to 2 times before passing output forward.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/workflows/workflow_agent_loop/run.py
"""

import asyncio
import os
from typing import Any

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool
from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
)

# ── Agent Tools ──


@tool
def search_topic(query: str) -> str:
    """Search for information about a topic."""
    q = query.lower()
    results = {
        "quantum computing": (
            "Quantum computing uses qubits that can be in superposition. "
            "Key concepts: entanglement, quantum gates, decoherence. "
            "Current leaders: IBM (1,121 qubits), Google (72 qubits). "
            "Applications: cryptography, drug discovery, optimization."
        ),
        "quantum advantage": (
            "Google claimed quantum supremacy in 2019 with Sycamore (53 qubits). "
            "IBM disputed the claim. Practical quantum advantage for real-world "
            "problems remains 5-10 years away for most applications."
        ),
        "quantum error": (
            "Error correction is the biggest challenge. Current error rates: ~0.1%. "
            "Need ~1000 physical qubits per logical qubit. Surface codes are "
            "the leading error correction approach."
        ),
    }
    for key, value in results.items():
        if key in q:
            return value
    return f"Results for '{query}': topic under active research and development."


@tool
def fact_check(claim: str) -> str:
    """Verify a factual claim."""
    c = claim.lower()
    if "ibm" in c and "qubit" in c:
        return "Verified: IBM Condor processor has 1,121 qubits (released Dec 2023)."
    if "google" in c and "supremacy" in c:
        return "Verified: Google claimed quantum supremacy in Oct 2019 with Sycamore."
    if "error" in c and "correction" in c:
        return (
            "Verified: ~1000:1 physical-to-logical qubit ratio is the current estimate."
        )
    return f"Claim '{claim[:50]}': unable to verify from available sources."


# ── Quality Checking ──


def check_quality(output: str) -> tuple[bool, float, str]:
    """Evaluate agent output quality. Returns (passed, score, feedback)."""
    checks = {}
    lower = output.lower()

    checks["has_content"] = len(output.strip()) > 50
    checks["mentions_qubits"] = "qubit" in lower
    checks["mentions_companies"] = any(
        c in lower for c in ["ibm", "google", "microsoft"]
    )
    checks["has_challenges"] = any(
        kw in lower for kw in ["challenge", "error", "difficult", "problem"]
    )
    checks["has_future"] = any(
        kw in lower for kw in ["future", "will", "expect", "outlook", "potential"]
    )

    score = sum(checks.values()) / len(checks) if checks else 0.0
    passed = score >= 0.6

    feedback_parts = []
    if not checks["has_content"]:
        feedback_parts.append("Output too short.")
    if not checks["mentions_qubits"]:
        feedback_parts.append("Should mention qubits.")
    if not checks["mentions_companies"]:
        feedback_parts.append("Mention key companies.")
    if not checks["has_challenges"]:
        feedback_parts.append("Cover challenges.")
    if not checks["has_future"]:
        feedback_parts.append("Include future outlook.")

    feedback = " ".join(feedback_parts) if feedback_parts else "Good quality."
    return passed, score, feedback


# ── Workflow Steps ──


class AgentWithQualityGate(WorkflowStep):
    """Agent step with internal quality checking and retry loop."""

    next_state: str = ""
    max_retries: int = 2

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        model = context.get("_model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        best_output = ""
        best_score = 0.0
        tools_used: list = []
        total_iterations = 0
        feedback = ""

        for attempt in range(self.max_retries + 1):
            print(f"  [Agent] Attempt {attempt + 1}/{self.max_retries + 1}...")

            registry = ToolRegistry()
            registry.register(search_topic._tool_definition)
            registry.register(fact_check._tool_definition)

            config = AgentConfig(model=model, max_iterations=8, temperature=0.5)
            agent = ReactAgent(tools=registry, config=config)

            task = (
                "Write a brief summary about quantum computing covering: "
                "overview, current state, challenges, and future outlook. "
                "Use the search_topic tool to gather facts and fact_check "
                "to verify key claims. Be specific with numbers and dates."
            )
            if feedback:
                task += f"\n\nPrevious feedback: {feedback}. Address these issues."

            try:
                result = agent.run(task)
                output = result.answer
                tools_used = result.tools_used
                total_iterations += result.iterations_used

                passed, score, feedback = check_quality(output)
                print(
                    f"  [Quality] Score: {score:.0%} ({'PASS' if passed else 'FAIL'})"
                )

                if score > best_score:
                    best_output = output
                    best_score = score

                if passed:
                    break
            except Exception as e:
                print(f"  [Agent] Error: {e}")
                feedback = str(e)

        return WorkflowStepResult.success_result(
            data={
                "agent_output": best_output,
                "quality_score": best_score,
                "tools_used": tools_used,
                "total_iterations": total_iterations,
                "attempts": min(attempt + 1, self.max_retries + 1),
            },
            next_state=self.next_state or None,
            message=f"Agent done (score: {best_score:.0%})",
        )


class TerminalStep(WorkflowStep):
    """Terminal step that formats and displays results."""

    action_fn: Any = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.action_fn):
            self.action_fn(context)
        return WorkflowStepResult.success_result(
            data={}, next_state=None, message="Done"
        )


def format_output(ctx: dict[str, Any]) -> None:
    """Format the final output."""
    output = ctx.get("agent_output", "No output generated")
    score = ctx.get("quality_score", 0)
    attempts = ctx.get("attempts", 1)

    print(f"\n{'=' * 50}")
    print("FINAL OUTPUT")
    print("=" * 50)
    print(f"\n{output}")
    print(f"\nQuality score: {score:.0%}")
    print(f"Attempts used: {attempts}")
    print(f"Tools used: {ctx.get('tools_used', [])}")
    print(f"Total iterations: {ctx.get('total_iterations', 0)}")


def build_workflow(model: str) -> WorkflowEngine:
    """Build the quality-gated agent workflow."""
    workflow = create_workflow(
        "agent_loop",
        "Quality-Gated Agent Loop",
        "Execute agent with quality checks and internal retry.",
    )

    # Step 1: Setup
    workflow.with_initial_step(
        auto_step(
            step_id="setup",
            name="Setup",
            next_state="agent_with_gate",
            action=lambda ctx: {"_model": ctx.get("_model", model)},
            description="Initialize workflow context",
        )
    )

    # Step 2: Agent with quality gate (retries internally)
    workflow.with_step(
        AgentWithQualityGate(
            step_id="agent_with_gate",
            name="Agent with Quality Gate",
            next_state="format_output",
            max_retries=2,
            description="Run agent with quality checking and retry",
        )
    )

    # Step 3: Format output (terminal)
    workflow.with_step(
        TerminalStep(
            step_id="format_output",
            name="Format Output",
            action_fn=format_output,
            description="Format and present final output",
        )
    )

    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


async def run():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not os.getenv("OPENAI_API_KEY") and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Workflow-Agent Loop — Quality-Gated Execution")
    print("=" * 60)
    print(f"Model: {model}")
    print("Flow: Setup → Agent (with quality gate + retry) → Output\n")

    engine = build_workflow(model)
    instance_id = await engine.start_workflow(
        "agent_loop", initial_context={"_model": model}
    )

    instance = engine.get_workflow_instance(instance_id)
    if instance:
        print(f"\nWorkflow status: {instance.status.value}")
        print(f"Quality score: {instance.context.get('quality_score', 'N/A')}")
        print(f"Attempts: {instance.context.get('attempts', 0)}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
