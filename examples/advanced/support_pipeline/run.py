"""
Support Pipeline Example — Classification + Core FSM + Stacking + Handlers
============================================================================

Demonstrates a full integration of multiple FSM-LLM sub-packages in a single
application: classification for triage, core FSM for conversation management,
FSM stacking for specialist handoffs, and handlers for metrics tracking.

Combines:
    - fsm_llm_classification: Classifier for initial intent triage
    - fsm_llm: Core FSM + FSM stacking (push_fsm/pop_fsm)
    - fsm_llm handlers: HandlerBuilder for metrics and logging

Key Concepts:
    - Classification-driven FSM selection
    - FSM stacking: push specialist FSMs, pop back to main
    - Handlers at multiple timing points for analytics
    - ContextMergeStrategy for context flow between FSMs
    - Shared context keys across stacked FSMs

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os
from datetime import datetime, timezone
from typing import Any

from fsm_llm import API, ContextMergeStrategy
from fsm_llm.handlers import HandlerTiming
from fsm_llm import (
    Classifier,
    ClassificationSchema,
    ClassificationPromptConfig,
    IntentDefinition,
)


# ------------------------------------------------------------------
# Metrics collector (populated by handlers)
# ------------------------------------------------------------------

metrics: dict[str, Any] = {
    "conversation_start": None,
    "state_transitions": [],
    "specialist_sessions": [],
    "total_turns": 0,
}


# ------------------------------------------------------------------
# Classification schema
# ------------------------------------------------------------------

schema = ClassificationSchema(
    intents=[
        IntentDefinition(
            name="billing",
            description="Billing questions: charges, invoices, refunds, payment methods, subscription changes",
        ),
        IntentDefinition(
            name="technical",
            description="Technical issues: bugs, errors, performance problems, setup help, configuration",
        ),
        IntentDefinition(
            name="general",
            description="General questions, feedback, or anything else",
        ),
    ],
    fallback_intent="general",
    confidence_threshold=0.5,
)


# ------------------------------------------------------------------
# FSM definitions
# ------------------------------------------------------------------

def create_main_fsm() -> dict:
    """Main support FSM that triages and delegates."""
    return {
        "name": "support_main",
        "description": "Main support conversation that delegates to specialists",
        "version": "4.1",
        "persona": "You are a friendly support agent. Identify the customer's issue and help them or connect them with the right specialist.",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Welcome the customer",
                "purpose": "Greet and understand their issue",
                "extraction_instructions": "Extract 'customer_name' and 'issue_summary' from the message.",
                "response_instructions": "Welcome the customer. Acknowledge their issue based on 'classified_intent' in context. Let them know you'll help or connect them with a specialist.",
                "transitions": [
                    {"target_state": "delegate_specialist", "description": "Issue needs specialist help", "priority": 5},
                    {"target_state": "quick_answer", "description": "Simple question with quick answer", "priority": 10},
                ],
            },
            "delegate_specialist": {
                "id": "delegate_specialist",
                "description": "Hand off to specialist FSM",
                "purpose": "Prepare context for specialist handoff",
                "extraction_instructions": "No additional extraction needed.",
                "response_instructions": "Let the customer know you're connecting them with a specialist for their specific issue.",
                "transitions": [
                    {"target_state": "post_specialist", "description": "Return from specialist", "priority": 5},
                ],
            },
            "post_specialist": {
                "id": "post_specialist",
                "description": "Follow up after specialist session",
                "purpose": "Check if the issue was resolved",
                "extraction_instructions": "Extract 'issue_resolved' (true/false) and 'needs_more_help' (true/false).",
                "response_instructions": "Welcome the customer back. Ask if the specialist resolved their issue and if they need anything else.",
                "transitions": [
                    {"target_state": "farewell", "description": "Issue resolved", "priority": 5,
                     "conditions": [{"description": "Resolved", "logic": {"==": [{"var": "issue_resolved"}, True]}}]},
                    {"target_state": "delegate_specialist", "description": "Needs more help", "priority": 10},
                    {"target_state": "farewell", "description": "Done", "priority": 15},
                ],
            },
            "quick_answer": {
                "id": "quick_answer",
                "description": "Provide a quick answer for simple queries",
                "purpose": "Answer simple questions directly",
                "extraction_instructions": "Extract 'needs_more_help' (true/false).",
                "response_instructions": "Provide a helpful answer to the customer's general question. Ask if they need anything else.",
                "transitions": [
                    {"target_state": "farewell", "description": "Done", "priority": 5,
                     "conditions": [{"description": "No more help needed", "logic": {"==": [{"var": "needs_more_help"}, False]}}]},
                    {"target_state": "greeting", "description": "Has more questions", "priority": 10},
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "End the conversation",
                "purpose": "Thank and close",
                "extraction_instructions": "Extract 'satisfaction_rating' (1-5) if provided.",
                "response_instructions": "Thank the customer for contacting support. Summarize what was accomplished. Wish them well.",
                "transitions": [],
            },
        },
    }


def create_billing_specialist_fsm() -> dict:
    """Billing specialist FSM."""
    return {
        "name": "billing_specialist",
        "description": "Specialist FSM for billing and payment issues",
        "version": "4.1",
        "persona": "You are a billing specialist. You can look up charges, process refunds, and update payment methods. Be precise with financial details.",
        "initial_state": "assess_billing_issue",
        "states": {
            "assess_billing_issue": {
                "id": "assess_billing_issue",
                "description": "Understand the specific billing issue",
                "purpose": "Categorize the billing problem",
                "extraction_instructions": "Extract 'billing_issue_type' (charge_dispute, refund, payment_update, subscription) and 'account_id' if mentioned.",
                "response_instructions": "Introduce yourself as a billing specialist. Reference the issue summary from context. Ask for specific details about their billing concern.",
                "transitions": [
                    {"target_state": "resolve_billing", "description": "Issue understood", "priority": 5,
                     "conditions": [{"description": "Issue type known", "requires_context_keys": ["billing_issue_type"]}]},
                ],
            },
            "resolve_billing": {
                "id": "resolve_billing",
                "description": "Work on resolving the billing issue",
                "purpose": "Process the billing request",
                "extraction_instructions": "Extract 'resolution_details' describing what was done. Set 'specialist_resolved' to true if the issue is handled.",
                "response_instructions": "Based on the billing issue type, walk through the resolution. For disputes: explain the charge. For refunds: confirm the refund amount. For payment updates: guide through the process. Provide specific next steps.",
                "transitions": [
                    {"target_state": "billing_handoff", "description": "Issue resolved or needs escalation", "priority": 5},
                ],
            },
            "billing_handoff": {
                "id": "billing_handoff",
                "description": "Hand back to main support",
                "purpose": "Complete specialist session",
                "extraction_instructions": "No extraction needed.",
                "response_instructions": "Summarize what was accomplished. Let the customer know you're transferring them back to the main support team.",
                "transitions": [],
            },
        },
    }


def create_technical_specialist_fsm() -> dict:
    """Technical specialist FSM."""
    return {
        "name": "technical_specialist",
        "description": "Specialist FSM for technical support issues",
        "version": "4.1",
        "persona": "You are a technical support specialist. You're methodical and walk users through debugging steps. You can check system status and review logs.",
        "initial_state": "diagnose_issue",
        "states": {
            "diagnose_issue": {
                "id": "diagnose_issue",
                "description": "Diagnose the technical problem",
                "purpose": "Understand the technical issue in detail",
                "extraction_instructions": "Extract 'affected_system' (app, website, api, device), 'error_message' if any, and 'issue_severity' (low, medium, high).",
                "response_instructions": "Introduce yourself as a technical specialist. Ask about the specific error, when it started, and what they were doing when it occurred. Check if there's an error message.",
                "transitions": [
                    {"target_state": "troubleshoot", "description": "Issue diagnosed", "priority": 5,
                     "conditions": [{"description": "System identified", "requires_context_keys": ["affected_system"]}]},
                ],
            },
            "troubleshoot": {
                "id": "troubleshoot",
                "description": "Walk through troubleshooting steps",
                "purpose": "Systematically resolve the technical issue",
                "extraction_instructions": "Extract 'step_worked' (true/false) indicating if the troubleshooting step helped. Track 'steps_tried' count.",
                "response_instructions": "Provide the next troubleshooting step. Start simple (clear cache, restart) and escalate. Number each step. Ask the user to try it and report back.",
                "transitions": [
                    {"target_state": "tech_handoff", "description": "Issue resolved or exhausted steps", "priority": 5,
                     "conditions": [{"description": "Step worked", "logic": {"==": [{"var": "step_worked"}, True]}}]},
                    {"target_state": "troubleshoot", "description": "Continue troubleshooting", "priority": 10,
                     "conditions": [{"description": "Under step limit", "logic": {"<": [{"var": "steps_tried"}, 4]}}]},
                    {"target_state": "tech_handoff", "description": "Exhausted basic steps", "priority": 15},
                ],
            },
            "tech_handoff": {
                "id": "tech_handoff",
                "description": "Hand back to main support",
                "purpose": "Complete technical session",
                "extraction_instructions": "Set 'specialist_resolved' to true if the issue was fixed, false otherwise.",
                "response_instructions": "Summarize the troubleshooting session: what was tried, what worked or didn't. Transfer back to main support.",
                "transitions": [],
            },
        },
    }


SPECIALIST_FSMS = {
    "billing": create_billing_specialist_fsm,
    "technical": create_technical_specialist_fsm,
}

SPECIALIST_TERMINAL_STATES = {
    "billing": "billing_handoff",
    "technical": "tech_handoff",
}


# ------------------------------------------------------------------
# Handler helper functions
# ------------------------------------------------------------------

def _track_transition(ctx: dict) -> dict:
    """Track state transitions for analytics."""
    metrics["state_transitions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_turns": metrics["total_turns"],
    })
    return {}


def _count_turn(ctx: dict) -> dict:
    """Count conversation turns."""
    metrics["total_turns"] += 1
    return {}


# ------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------

def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    # Build classifier
    classifier = Classifier(
        schema,
        model=model,
        config=ClassificationPromptConfig(temperature=0.0),
    )

    # Build main FSM
    main_fsm_def = create_main_fsm()
    api = API.from_definition(main_fsm_def, model=model, temperature=0.7, max_tokens=500)

    # Register handlers
    # Handler 1: Track state transitions
    api.register_handler(
        api.create_handler("TransitionTracker")
        .at(HandlerTiming.POST_TRANSITION)
        .do(lambda ctx: _track_transition(ctx))
    )

    # Handler 2: Count conversation turns
    api.register_handler(
        api.create_handler("TurnCounter")
        .at(HandlerTiming.POST_PROCESSING)
        .do(lambda ctx: _count_turn(ctx))
    )

    print("Support Pipeline (Classification + FSM + Stacking + Handlers)")
    print("=" * 60)
    print("Describe your issue and the system will triage and route you.\n")

    # Step 1: Get initial message and classify
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() in ("quit", "exit"):
        return

    result = classifier.classify(user_input)
    intent = result.intent
    confidence = result.confidence

    print(f"\n  [Triage] Intent: {intent} (confidence: {confidence:.2f})")
    if result.entities:
        print(f"  [Triage] Entities: {result.entities}")

    # Step 2: Start main FSM with classification context
    metrics["conversation_start"] = datetime.now(timezone.utc).isoformat()

    initial_context = {
        "classified_intent": intent,
        "classification_confidence": confidence,
        "initial_message": user_input,
        "entities": result.entities or {},
    }

    conv_id, response = api.start_conversation(initial_context=initial_context)
    print(f"\nAgent: {response}")

    # Step 3: Send the initial message through the FSM
    response = api.converse(user_input, conv_id)
    print(f"\nAgent: {response}")

    # Step 4: Interactive conversation loop with stacking
    while not api.has_conversation_ended(conv_id):
        state = api.get_current_state(conv_id)

        # Check if we should push a specialist FSM
        if state == "delegate_specialist" and intent in SPECIALIST_FSMS:
            print(f"\n  [Routing to {intent} specialist...]")

            ctx = api.get_data(conv_id)
            specialist_fsm = SPECIALIST_FSMS[intent]()

            # Push specialist FSM
            specialist_response = api.push_fsm(
                conv_id,
                specialist_fsm,
                context_to_pass={
                    "customer_name": ctx.get("customer_name", "Customer"),
                    "issue_summary": ctx.get("issue_summary", user_input),
                    "classified_intent": intent,
                },
                shared_context_keys=["customer_name"],
                preserve_history=True,
                inherit_context=True,
            )

            metrics["specialist_sessions"].append({
                "type": intent,
                "started": datetime.now(timezone.utc).isoformat(),
            })

            print(f"\nSpecialist: {specialist_response}")

            # Run specialist conversation
            terminal = SPECIALIST_TERMINAL_STATES.get(intent)
            while not api.has_conversation_ended(conv_id):
                specialist_state = api.get_current_state(conv_id)

                if specialist_state == terminal:
                    # Pop back to main FSM
                    specialist_ctx = api.get_data(conv_id)
                    pop_response = api.pop_fsm(
                        conv_id,
                        context_to_return={
                            "specialist_resolved": specialist_ctx.get("specialist_resolved", False),
                            "resolution_details": specialist_ctx.get("resolution_details", ""),
                        },
                        merge_strategy=ContextMergeStrategy.UPDATE,
                    )
                    print("\n  [Back to main support]")
                    print(f"\nAgent: {pop_response}")
                    break

                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() in ("exit", "quit"):
                    break

                resp = api.converse(user_input, conv_id)
                print(f"\nSpecialist: {resp}")

            continue

        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        response = api.converse(user_input, conv_id)
        state = api.get_current_state(conv_id)
        print(f"  [{state}]")
        print(f"\nAgent: {response}")

    # Print metrics
    print("\n" + "=" * 60)
    print("SESSION METRICS (collected by handlers)")
    print("=" * 60)
    print(f"  Started:              {metrics['conversation_start']}")
    print(f"  Total turns:          {metrics['total_turns']}")
    print(f"  State transitions:    {len(metrics['state_transitions'])}")
    print(f"  Specialist sessions:  {len(metrics['specialist_sessions'])}")
    for s in metrics["specialist_sessions"]:
        print(f"    - {s['type']} specialist at {s['started']}")

    api.close()


if __name__ == "__main__":
    main()
