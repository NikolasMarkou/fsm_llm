"""
Employee Onboarding Flow -- Handler-Tracked Onboarding Pipeline
===============================================================

Demonstrates an employee onboarding process with handler hooks
tracking completion status, document verification, and orientation
scheduling across a detailed HR scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/employee_onboarding/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming


metrics: dict[str, Any] = {
    "steps_completed": [],
    "time_per_step": [],
    "context_snapshots": [],
}


def build_fsm() -> dict:
    return {
        "name": "EmployeeOnboardingBot",
        "description": "Guides new employees through the onboarding process",
        "initial_state": "welcome",
        "persona": (
            "You are an HR onboarding specialist at Meridian Global Consulting, a "
            "management consulting firm with 2,800 employees across 14 offices in "
            "North America, Europe, and Asia-Pacific. Be welcoming and thorough "
            "when guiding new employees through the onboarding process."
        ),
        "states": {
            "welcome": {
                "id": "welcome",
                "description": "Welcome new employee and collect basic info",
                "purpose": "Identify the new hire and their role",
                "extraction_instructions": "Extract 'employee_name' (full name) and 'department' (which department they're joining).",
                "response_instructions": "Welcome the new employee to Meridian Global Consulting, founded in 2005 and ranked among the top 50 consulting firms for 8 consecutive years. New hires go through a structured 5-day onboarding: IT setup (laptop, email, VPN), benefits enrollment (medical, dental, vision, 401k with 5% match, FSA, life insurance), compliance training (code of conduct, data privacy, anti-harassment, information security), team integration (buddy assignment, department lunch, project briefing), and goal setting. The company uses Workday for HR, Salesforce for CRM, Slack for communication, and Confluence for documentation. New hires receive a $2,000 home office stipend if remote, or ergonomic desk setup if in-office. Performance reviews are semi-annual (March and September). The culture emphasizes collaboration, continuous learning ($3,500 annual professional development budget), and work-life balance. Ask for their name and which department they're joining.",
                "transitions": [
                    {
                        "target_state": "it_setup",
                        "description": "Welcome complete",
                        "conditions": [
                            {
                                "description": "Name provided",
                                "requires_context_keys": ["employee_name"],
                                "logic": {"has_context": "employee_name"},
                            }
                        ],
                    }
                ],
            },
            "it_setup": {
                "id": "it_setup",
                "description": "IT equipment and access setup",
                "purpose": "Determine IT needs and work location",
                "extraction_instructions": "Extract 'work_location' (remote, office, or hybrid) and 'laptop_preference' (Mac or Windows).",
                "response_instructions": "Ask about their work arrangement (remote/office/hybrid) and laptop preference (Mac or Windows). Mention the $2,000 home office stipend for remote workers.",
                "transitions": [
                    {
                        "target_state": "benefits",
                        "description": "IT setup configured",
                        "conditions": [
                            {
                                "description": "Location and laptop set",
                                "requires_context_keys": ["work_location"],
                                "logic": {"has_context": "work_location"},
                            }
                        ],
                    }
                ],
            },
            "benefits": {
                "id": "benefits",
                "description": "Benefits enrollment",
                "purpose": "Guide through benefits selection",
                "extraction_instructions": "Extract 'benefits_tier' (individual, spouse, or family) and 'retirement_contribution' (401k percentage).",
                "response_instructions": "Walk through benefits options: medical/dental/vision tiers, 401k with 5% company match, FSA, and life insurance. Ask about their coverage needs.",
                "transitions": [
                    {
                        "target_state": "compliance",
                        "description": "Benefits selected",
                        "conditions": [
                            {
                                "description": "Benefits chosen",
                                "requires_context_keys": ["benefits_tier"],
                                "logic": {"has_context": "benefits_tier"},
                            }
                        ],
                    }
                ],
            },
            "compliance": {
                "id": "compliance",
                "description": "Compliance training acknowledgment",
                "purpose": "Confirm understanding of company policies",
                "extraction_instructions": "Extract 'compliance_acknowledged' (yes or no) and 'emergency_contact' (name of emergency contact person).",
                "response_instructions": "Review required compliance training: code of conduct, data privacy, anti-harassment, and information security. Ask them to acknowledge understanding and provide an emergency contact name.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Compliance complete",
                        "conditions": [
                            {
                                "description": "Acknowledged",
                                "requires_context_keys": ["compliance_acknowledged"],
                                "logic": {"has_context": "compliance_acknowledged"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Onboarding summary",
                "purpose": "Present complete onboarding status",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the onboarding: employee name, department, work location, laptop, benefits tier, 401k contribution, compliance status, and emergency contact. Welcome them to the Meridian team.",
                "transitions": [],
            },
        },
    }


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Employee Onboarding -- Handler-Tracked Pipeline")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="step_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["steps_completed"].append(
            {"state": ctx.get("_current_state", "?"), "time": time.strftime("%H:%M:%S")}
        ),
    )

    fsm.create_handler(
        name="context_snapshot",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["context_snapshots"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, I'm Alex Rivera. I'm joining the Strategy and Analytics department",
        "I'll be working hybrid, 3 days in office. I'd prefer a Mac laptop",
        "Family coverage for benefits please, and I'll contribute 6% to the 401k",
        "Yes, I acknowledge all compliance policies. My emergency contact is Maria Rivera",
    ]

    expected_keys = [
        "employee_name", "department", "work_location", "laptop_preference",
        "benefits_tier", "retirement_contribution", "compliance_acknowledged", "emergency_contact",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        metrics["time_per_step"].append(elapsed)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("ONBOARDING SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:30s}: {str(value)[:35]:35s} [{status}]")

    print(f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)")

    print("\n" + "=" * 60)
    print("HANDLER METRICS")
    print("=" * 60)
    print(f"  Steps completed: {len(metrics['steps_completed'])}")
    for step in metrics["steps_completed"]:
        print(f"    -> {step['state']} at {step['time']}")
    print(f"  Processing times: {[f'{t:.1f}s' for t in metrics['time_per_step']]}")
    print(f"  Context growth: {metrics['context_snapshots']}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
