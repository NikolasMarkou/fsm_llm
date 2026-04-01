"""
Conditional Branching Workflow
===============================

Demonstrates a workflow with condition-based routing. The workflow
processes a support ticket through different paths based on the
ticket's priority and category.

Flow:
  Intake → Classify → [High priority?]
    Yes → Urgent Handler → Escalation → Notify
    No  → Standard Handler → Auto-Response → Notify

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/workflows/conditional_branching/run.py
"""

import asyncio
from typing import Any

from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    create_workflow,
)


class ProcessingStep(WorkflowStep):
    """A step that processes context and routes to next state."""

    handler: Any = None
    success_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        data = {}
        if callable(self.handler):
            data = self.handler(context) or {}
        return WorkflowStepResult.success_result(
            data=data,
            next_state=self.success_state,
            message=f"Processed: {self.step_id}",
        )


class ConditionStep(WorkflowStep):
    """A step that routes based on a condition."""

    condition: Any = None
    true_state: str = ""
    false_state: str = ""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.condition):
            result = self.condition(context)
        else:
            result = False

        next_state = self.true_state if result else self.false_state
        return WorkflowStepResult.success_result(
            data={"condition_result": result, "routed_to": next_state},
            next_state=next_state,
            message=f"Condition: {'true' if result else 'false'} → {next_state}",
        )


# ── Processing Handlers ──


def intake_handler(ctx: dict) -> dict:
    """Simulate ticket intake."""
    ticket = {
        "ticket_id": "TK-2024-001",
        "customer": "John Doe",
        "subject": "System outage affecting production",
        "category": "infrastructure",
        "description": "Production database is unresponsive since 2pm",
        "priority": "high",
        "created_at": "2024-03-15T14:30:00Z",
    }
    print(f"  [Intake] Received ticket: {ticket['ticket_id']} — {ticket['subject']}")
    return ticket


def classify_handler(ctx: dict) -> dict:
    """Classify the ticket based on keywords."""
    subject = ctx.get("subject", "").lower()
    description = ctx.get("description", "").lower()
    priority = ctx.get("priority", "normal").lower()

    # Auto-classify priority
    urgent_keywords = ["outage", "down", "critical", "production", "security breach"]
    is_urgent = priority == "high" or any(
        kw in subject + description for kw in urgent_keywords
    )

    # Classify category
    if any(
        kw in subject + description for kw in ["database", "server", "infrastructure"]
    ):
        team = "infrastructure"
    elif any(kw in subject + description for kw in ["billing", "payment", "invoice"]):
        team = "billing"
    else:
        team = "general_support"

    classification = {
        "is_urgent": is_urgent,
        "assigned_team": team,
        "classification_confidence": 0.92,
    }
    print(f"  [Classify] Urgent: {is_urgent}, Team: {team}")
    return classification


def urgent_handler(ctx: dict) -> dict:
    """Handle urgent tickets with immediate response."""
    print(f"  [URGENT] Ticket {ctx.get('ticket_id', '?')} flagged as urgent!")
    print(
        f"  [URGENT] Paging on-call engineer for {ctx.get('assigned_team', 'unknown')}"
    )
    return {
        "response_type": "urgent",
        "sla_hours": 1,
        "paged_oncall": True,
        "incident_id": "INC-2024-042",
    }


def standard_handler(ctx: dict) -> dict:
    """Handle standard tickets with normal SLA."""
    print(
        f"  [Standard] Ticket {ctx.get('ticket_id', '?')} queued for {ctx.get('assigned_team', 'unknown')}"
    )
    return {
        "response_type": "standard",
        "sla_hours": 24,
        "auto_response_sent": True,
        "queue_position": 3,
    }


def escalation_handler(ctx: dict) -> dict:
    """Escalate urgent tickets to management."""
    print(
        f"  [Escalation] Incident {ctx.get('incident_id', '?')} escalated to management"
    )
    return {
        "escalated": True,
        "escalation_level": "P1",
        "management_notified": True,
    }


def auto_response_handler(ctx: dict) -> dict:
    """Send automated response for standard tickets."""
    print(f"  [AutoResponse] Sent acknowledgment to {ctx.get('customer', 'customer')}")
    return {
        "auto_response": (
            f"Thank you for contacting support. Your ticket "
            f"{ctx.get('ticket_id', '')} has been received and assigned "
            f"to our {ctx.get('assigned_team', '')} team. "
            f"Expected response within {ctx.get('sla_hours', 24)} hours."
        ),
    }


def notify_handler(ctx: dict) -> dict:
    """Send final notification."""
    response_type = ctx.get("response_type", "standard")
    print(f"\n  [Notify] Ticket processing complete (type: {response_type})")
    print(f"  [Notify] SLA: {ctx.get('sla_hours', '?')} hours")
    if ctx.get("escalated"):
        print(f"  [Notify] Escalation level: {ctx.get('escalation_level', '?')}")
    return {"notification_sent": True}


def build_workflow() -> WorkflowEngine:
    """Build the conditional branching workflow."""
    workflow = create_workflow(
        "ticket_router",
        "Support Ticket Router",
        "Route support tickets through conditional processing paths.",
    )

    # Step 1: Intake
    workflow.with_initial_step(
        ProcessingStep(
            step_id="intake",
            name="Ticket Intake",
            handler=intake_handler,
            success_state="classify",
            description="Receive and parse support ticket",
        )
    )

    # Step 2: Classify
    workflow.with_step(
        ProcessingStep(
            step_id="classify",
            name="Classification",
            handler=classify_handler,
            success_state="route",
            description="Classify ticket priority and category",
        )
    )

    # Step 3: Conditional routing
    workflow.with_step(
        ConditionStep(
            step_id="route",
            name="Priority Router",
            condition=lambda ctx: ctx.get("is_urgent", False),
            true_state="urgent_handler",
            false_state="standard_handler",
            description="Route based on urgency",
        )
    )

    # Path A: Urgent
    workflow.with_step(
        ProcessingStep(
            step_id="urgent_handler",
            name="Urgent Handler",
            handler=urgent_handler,
            success_state="escalation",
            description="Handle urgent ticket with immediate response",
        )
    )

    workflow.with_step(
        ProcessingStep(
            step_id="escalation",
            name="Escalation",
            handler=escalation_handler,
            success_state="notify",
            description="Escalate to management",
        )
    )

    # Path B: Standard
    workflow.with_step(
        ProcessingStep(
            step_id="standard_handler",
            name="Standard Handler",
            handler=standard_handler,
            success_state="auto_response",
            description="Queue ticket with standard SLA",
        )
    )

    workflow.with_step(
        ProcessingStep(
            step_id="auto_response",
            name="Auto Response",
            handler=auto_response_handler,
            success_state="notify",
            description="Send automated acknowledgment",
        )
    )

    # Convergence: Both paths end at notify
    workflow.with_step(
        ProcessingStep(
            step_id="notify",
            name="Notification",
            handler=notify_handler,
            success_state=None,  # Terminal
            description="Send final notification",
        )
    )

    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


async def run():
    print("=" * 60)
    print("Conditional Branching Workflow — Ticket Router")
    print("=" * 60)
    print("Flow: Intake → Classify → [Urgent?] → Handle → Notify\n")

    engine = build_workflow()
    instance_id = await engine.start_workflow("ticket_router", initial_context={})

    instance = engine.get_workflow_instance(instance_id)
    if instance:
        print(f"\nWorkflow status: {instance.status.value}")
        context_keys = sorted(k for k in instance.context if not k.startswith("_"))
        print(f"Context keys: {context_keys}")

        # Show key results
        print("\nResults:")
        print(f"  Ticket: {instance.context.get('ticket_id', 'N/A')}")
        print(f"  Urgent: {instance.context.get('is_urgent', 'N/A')}")
        print(f"  Response type: {instance.context.get('response_type', 'N/A')}")
        print(f"  SLA: {instance.context.get('sla_hours', 'N/A')} hours")
        if instance.context.get("auto_response"):
            print(f"  Auto-response: {instance.context['auto_response'][:100]}...")

        # ── Verification ──
        ctx = instance.context
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        checks = {
            "workflow_completed": instance.status.value == "completed",
            "ticket_id": ctx.get("ticket_id"),
            "is_urgent": ctx.get("is_urgent"),
            "assigned_team": ctx.get("assigned_team"),
            "response_type": ctx.get("response_type"),
            "sla_hours": ctx.get("sla_hours"),
            "notification_sent": ctx.get("notification_sent"),
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
