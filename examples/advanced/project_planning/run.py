"""
Project Planning Intake -- Handler-Tracked Project Assessment
=============================================================

Demonstrates a project planning intake process with handler hooks
tracking scope assessment, resource estimation, and risk identification
across a detailed project management scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/project_planning/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "phases": [],
    "data_points_collected": [],
    "processing_times": [],
}


def build_fsm() -> dict:
    return {
        "name": "ProjectPlanningBot",
        "description": "Project intake with scope, resources, and risk assessment",
        "initial_state": "project_overview",
        "persona": (
            "You are a senior project manager at Apex Digital Solutions, a technology "
            "consulting firm specializing in digital transformation projects for "
            "Fortune 500 companies. Be thorough and strategic when scoping new projects."
        ),
        "states": {
            "project_overview": {
                "id": "project_overview",
                "description": "Collect high-level project information",
                "purpose": "Understand the project scope and objectives",
                "extraction_instructions": "Extract 'project_name' (name or title) and 'practice_area' (which practice area: cloud, data, app mod, security, devops, or ai/ml).",
                "response_instructions": "Welcome the stakeholder to Apex project intake. The firm employs 450 consultants across 6 practice areas: Cloud Migration, Data Analytics, Application Modernization, Cybersecurity, DevOps Transformation, and AI/ML Implementation. Project sizes range from $200K (3-month tactical) to $15M (18-month enterprise). The standardized framework has 5 phases: Discovery (2-4 weeks), Design (4-8 weeks), Build (8-20 weeks), Test & Deploy (4-8 weeks), and Optimize (ongoing). Resource allocation uses a pod model: 1 tech lead, 2-4 senior engineers, 1-2 junior engineers, 1 QA specialist per pod. Key metrics: on-time delivery (87%), client satisfaction (NPS 72), budget adherence (94% within 10%). Risk management follows RAID (Risks, Assumptions, Issues, Dependencies). PMO reviews all projects weekly. Ask about the project name and which practice area it falls under.",
                "transitions": [
                    {
                        "target_state": "scope_definition",
                        "description": "Overview captured",
                        "conditions": [
                            {
                                "description": "Project named",
                                "requires_context_keys": ["project_name"],
                                "logic": {"has_context": "project_name"},
                            }
                        ],
                    }
                ],
            },
            "scope_definition": {
                "id": "scope_definition",
                "description": "Define project scope and deliverables",
                "purpose": "Establish clear scope boundaries",
                "extraction_instructions": "Extract 'primary_deliverable' (main output) and 'estimated_duration' (expected timeline).",
                "response_instructions": "Ask about the primary deliverable and expected timeline. Reference the standard phases: Discovery, Design, Build, Test & Deploy, Optimize.",
                "transitions": [
                    {
                        "target_state": "resource_needs",
                        "description": "Scope defined",
                        "conditions": [
                            {
                                "description": "Deliverable known",
                                "requires_context_keys": ["primary_deliverable"],
                                "logic": {"has_context": "primary_deliverable"},
                            }
                        ],
                    }
                ],
            },
            "resource_needs": {
                "id": "resource_needs",
                "description": "Assess resource requirements",
                "purpose": "Determine team size and skills needed",
                "extraction_instructions": "Extract 'team_size' (number of people needed) and 'budget_range' (estimated budget).",
                "response_instructions": "Ask about expected team size and budget range. Mention the pod model (tech lead + seniors + juniors + QA). Provide context on typical project sizes.",
                "transitions": [
                    {
                        "target_state": "risk_assessment",
                        "description": "Resources assessed",
                        "conditions": [
                            {
                                "description": "Team size known",
                                "requires_context_keys": ["team_size"],
                                "logic": {"has_context": "team_size"},
                            }
                        ],
                    }
                ],
            },
            "risk_assessment": {
                "id": "risk_assessment",
                "description": "Identify key risks and dependencies",
                "purpose": "Capture risks using RAID framework",
                "extraction_instructions": "Extract 'top_risk' (biggest risk to the project) and 'key_dependency' (critical external dependency).",
                "response_instructions": "Ask about the biggest risk and key external dependencies. Explain the RAID framework approach to risk management.",
                "transitions": [
                    {
                        "target_state": "project_summary",
                        "description": "Risks captured",
                        "conditions": [
                            {
                                "description": "Risk identified",
                                "requires_context_keys": ["top_risk"],
                                "logic": {"has_context": "top_risk"},
                            }
                        ],
                    }
                ],
            },
            "project_summary": {
                "id": "project_summary",
                "description": "Present project assessment summary",
                "purpose": "Summarize the project intake",
                "extraction_instructions": "None",
                "response_instructions": "Summarize: project name, practice area, deliverable, timeline, team size, budget, top risk, and key dependency. Provide a preliminary project classification (tactical/strategic/enterprise) and recommend next steps.",
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
    print("Project Planning -- Handler-Tracked Assessment")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="phase_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["phases"].append(ctx.get("_current_state", "?")),
    )

    fsm.create_handler(
        name="data_counter",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["data_points_collected"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "We're calling it Project Phoenix. It's a cloud migration initiative",
        "The main deliverable is migrating our on-premises data warehouse to AWS. We're targeting 6 months",
        "We'll need a team of about 8 people. Budget is around $1.2 million",
        "The biggest risk is data integrity during migration. We depend on the legacy team's availability",
    ]

    expected_keys = [
        "project_name",
        "practice_area",
        "primary_deliverable",
        "estimated_duration",
        "team_size",
        "budget_range",
        "top_risk",
        "key_dependency",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        metrics["processing_times"].append(elapsed)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("PROJECT ASSESSMENT SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:35]:35s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )

    print("\n" + "=" * 60)
    print("HANDLER METRICS")
    print("=" * 60)
    print(f"  Phases tracked: {metrics['phases']}")
    print(f"  Data points growth: {metrics['data_points_collected']}")
    print(f"  Processing times: {[f'{t:.1f}s' for t in metrics['processing_times']]}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
