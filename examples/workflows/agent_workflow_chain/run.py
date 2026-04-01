"""
Agent-Workflow Chain Example -- Workflow Driving Multiple Agents
===============================================================

Demonstrates a workflow that orchestrates multiple agent steps
in sequence: research agent -> analysis agent -> summary,
passing context between them.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/workflows/agent_workflow_chain/run.py
"""

import asyncio
import os
from typing import Any

from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    conversation_step,
    create_workflow,
)


class TerminalStep(WorkflowStep):
    """Terminal step that ends the workflow."""

    action: Any = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.action):
            self.action(context)
        return WorkflowStepResult.success_result(
            data={}, next_state=None, message="Done"
        )


def build_research_fsm() -> dict:
    """FSM for the research phase."""
    return {
        "name": "Researcher",
        "description": "Research a topic",
        "initial_state": "research",
        "persona": "A thorough researcher who gathers key facts",
        "states": {
            "research": {
                "id": "research",
                "description": "Research the topic",
                "purpose": "Gather key facts and findings",
                "extraction_instructions": "Extract key_findings from the research discussion",
                "response_instructions": "Present 3-5 key findings about the topic being researched",
                "transitions": [],
            }
        },
    }


def build_analysis_fsm() -> dict:
    """FSM for the analysis phase."""
    return {
        "name": "Analyst",
        "description": "Analyze research findings",
        "initial_state": "analyze",
        "persona": "An analytical thinker who draws conclusions from data",
        "states": {
            "analyze": {
                "id": "analyze",
                "description": "Analyze findings",
                "purpose": "Draw conclusions and recommendations from research",
                "extraction_instructions": "Extract conclusions and recommendation from the analysis",
                "response_instructions": "Analyze the research findings provided and give 2-3 actionable conclusions",
                "transitions": [],
            }
        },
    }


def build_chained_workflow(model: str) -> WorkflowEngine:
    """Build a workflow that chains research -> analysis -> summary."""
    research_fsm = build_research_fsm()
    analysis_fsm = build_analysis_fsm()

    workflow = create_workflow(
        "agent_chain",
        "Research-Analysis Pipeline",
        "Chain research and analysis agents with workflow orchestration.",
    )

    # Step 1: Research phase
    workflow.with_initial_step(
        conversation_step(
            step_id="research",
            name="Research Phase",
            fsm_definition=research_fsm,
            model=model,
            auto_messages=[
                "Research the impact of artificial intelligence on healthcare. "
                "Cover diagnostics, drug discovery, and patient care."
            ],
            context_mapping={"key_findings": "final_answer"},
            success_state="analysis",
            error_state="summary",
            description="Gather research findings via FSM conversation",
        )
    )

    # Step 2: Analysis phase
    workflow.with_step(
        conversation_step(
            step_id="analysis",
            name="Analysis Phase",
            fsm_definition=analysis_fsm,
            model=model,
            auto_messages=[
                "Based on the research: AI is transforming healthcare through "
                "improved diagnostics (95% accuracy in some imaging tasks), "
                "accelerated drug discovery (cutting timelines by 40%), "
                "and personalized patient care. Analyze these findings."
            ],
            context_mapping={
                "conclusions": "final_answer",
            },
            success_state="summary",
            error_state="summary",
            description="Analyze research findings",
        )
    )

    # Step 3: Summary
    workflow.with_step(
        TerminalStep(
            step_id="summary",
            name="Pipeline Summary",
            action=lambda ctx: print_pipeline_summary(ctx),
            description="Present final pipeline results",
        )
    )

    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


def print_pipeline_summary(ctx: dict) -> None:
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Research findings: {ctx.get('key_findings', 'N/A')}")
    print(f"  Conclusions: {ctx.get('conclusions', 'N/A')}")


async def run():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not os.getenv("OPENAI_API_KEY") and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("Agent-Workflow Chain Pipeline")
    print("=" * 50)
    print(f"Model: {model}")
    print("Pipeline: Research -> Analysis -> Summary\n")

    try:
        engine = build_chained_workflow(model)
        instance_id = await engine.start_workflow("agent_chain", initial_context={})

        instance = engine.get_workflow_instance(instance_id)
        if instance:
            print(f"\nWorkflow status: {instance.status.value}")
            context_keys = [k for k in instance.context if not k.startswith("_")]
            print(f"Context keys: {context_keys}")

            # ── Verification ──
            ctx = instance.context
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            checks = {
                "workflow_completed": instance.status.value == "completed",
                "key_findings": ctx.get("key_findings"),
                "conclusions": ctx.get("conclusions"),
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
    except Exception as e:
        print(f"Error: {e}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
