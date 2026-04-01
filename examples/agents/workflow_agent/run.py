"""
Method 3: Workflow Orchestrating Agent FSMs
=============================================

Uses ``build_react_fsm()`` output with ``conversation_step()`` in
the workflow DSL. Demonstrates how a workflow can orchestrate an
agent FSM as a step in a larger pipeline.

Note: Tool execution handlers are NOT registered when using
ConversationStep (handlers are tied to ReactAgent, not the FSM
definition). This approach works best with non-tool agent patterns
(Debate, SelfConsistency) or when tool execution is handled
externally. For ReAct patterns, use ReactAgent directly.

This example uses a SelfConsistency-style FSM via ConversationStep.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/workflow_agent/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/workflow_agent/run.py
"""

import asyncio
import os

# Try to import workflows
try:
    from fsm_llm_workflows import (
        WorkflowEngine,
        auto_step,
        conversation_step,
        create_workflow,
    )

    _HAS_WORKFLOWS = True
except ImportError:
    _HAS_WORKFLOWS = False

# Try to import agents (for FSM definitions)
try:
    from fsm_llm_agents.fsm_definitions import build_self_consistency_fsm

    _HAS_AGENTS = True
except ImportError:
    _HAS_AGENTS = False


# ──────────────────────────────────────────────
# Workflow Step Functions
# ──────────────────────────────────────────────


def setup_task(context: dict) -> dict:
    """Prepare the task for the agent."""
    task = context.get("user_task", "What are the main benefits of renewable energy?")
    return {
        "task": task,
        "task_context": f"Analyze the following question using multiple perspectives: {task}",
        "status": "setup_complete",
    }


def post_process(context: dict) -> dict:
    """Post-process the agent's output."""
    answer = context.get("agent_answer", "No answer received")
    return {
        "final_output": f"Processed result: {answer}",
        "status": "complete",
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


async def run_workflow(model: str, task: str) -> dict:
    """Build and run the workflow."""
    # Build a self-consistency FSM (no tool execution needed)
    agent_fsm = build_self_consistency_fsm(task_description=task[:200])

    # Create workflow: setup -> agent conversation -> post-process
    wf_builder = create_workflow("agent_workflow", "Workflow with Agent FSM")

    wf_builder.with_initial_step(
        auto_step(
            "setup",
            "Task Setup",
            next_state="agent_step",
            action=setup_task,
            description="Prepare task context for the agent",
        )
    )

    wf_builder.with_step(
        conversation_step(
            "agent_step",
            "Agent Analysis",
            fsm_definition=agent_fsm,
            model=model,
            initial_context={"task": "task"},
            context_mapping={"agent_answer": "final_answer"},
            auto_messages=["Continue.", "Continue.", "Continue."],
            max_turns=10,
            success_state="post_process",
            error_state="post_process",
            description="Run agent FSM to analyze the task",
        )
    )

    wf_builder.with_step(
        auto_step(
            "post_process",
            "Post Processing",
            next_state="",
            action=post_process,
            description="Post-process the agent output",
        )
    )

    # Register and run the workflow
    engine = WorkflowEngine()
    engine.register_workflow(wf_builder)

    instance_id = await engine.start_workflow(
        "agent_workflow",
        initial_context={"user_task": task},
    )

    instance = engine.get_workflow_instance(instance_id)
    return instance.context if instance else {}


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    if not _HAS_WORKFLOWS:
        print("This example requires fsm_llm_workflows.")
        print("Install with: pip install fsm-llm[workflows]")
        return

    if not _HAS_AGENTS:
        print("This example requires fsm_llm_agents.")
        return

    print("=" * 60)
    print("Method 3: Workflow Orchestrating Agent FSMs")
    print("=" * 60)
    print(f"Model: {model}")
    print()
    print("Note: This uses ConversationStep to embed an agent FSM")
    print("inside a workflow. Tool execution handlers are NOT active")
    print("in ConversationStep, so this works best with non-tool patterns.")
    print()

    task = "What are the main benefits and drawbacks of remote work?"
    print(f"Task: {task}")
    print("-" * 40)

    try:
        result = asyncio.run(run_workflow(model, task))
        print(f"\nFinal output: {result.get('final_output', 'N/A')}")
        print(f"Status: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.get("final_output") is not None
        and len(str(result.get("final_output", ""))) > 10,
        "iterations_ok": result.get("status") is not None,
        "completed": result.get("status") == "complete",
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
