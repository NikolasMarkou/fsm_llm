"""
Incident Response -- Management Pipeline with Handler Hooks
============================================================

Demonstrates an IT incident management process with handler hooks
tracking severity escalation, response timeline, and resolution
tracking across a detailed ITIL-based scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/incident_response/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "response_phases": [],
    "escalation_events": [],
    "processing_log": [],
}


def build_fsm() -> dict:
    return {
        "name": "IncidentResponseBot",
        "description": "IT incident management with ITIL-based response tracking",
        "initial_state": "incident_report",
        "persona": (
            "You are the Incident Manager at DataFlow Systems, a cloud hosting "
            "provider managing infrastructure for 850 enterprise clients across "
            "6 data centers. The company processes 2.3 billion API requests daily "
            "and maintains a 99.99% uptime SLA. Be decisive, clear, and systematic "
            "when managing incidents."
        ),
        "states": {
            "incident_report": {
                "id": "incident_report",
                "description": "Capture initial incident report",
                "purpose": "Document what happened and initial impact",
                "extraction_instructions": "Extract 'incident_title' (brief description) and 'affected_service' (which service/system is impacted).",
                "response_instructions": "Acknowledge the incident report. Incident management follows ITIL v4 with 4 severity levels: SEV-1 (critical, 100+ clients, full outage, 15-min response, CTO notification), SEV-2 (major, 10+ clients, degraded, 30-min response, VP notification), SEV-3 (moderate, 1-10 clients, 2-hour response, manager notification), SEV-4 (low, minor with workaround, 8-hour response). The response team has 4 on-call rotations: NOC, Application Support, DBA, and Security, each with primary and secondary engineers. War room activates for SEV-1/2 with 15-min status updates. Post-incident reviews: 48 hours for SEV-1/2, 1 week for SEV-3. Tools: PagerDuty (alerting), Jira (tracking), Slack #incident-response. MTTR targets: SEV-1 (1 hour), SEV-2 (4 hours), SEV-3 (24 hours), SEV-4 (72 hours). Ask for a brief title and which service is affected.",
                "transitions": [
                    {
                        "target_state": "severity_assessment",
                        "description": "Report captured",
                        "conditions": [
                            {
                                "description": "Incident described",
                                "requires_context_keys": ["incident_title"],
                                "logic": {"has_context": "incident_title"},
                            }
                        ],
                    }
                ],
            },
            "severity_assessment": {
                "id": "severity_assessment",
                "description": "Assess severity and impact",
                "purpose": "Classify incident severity level",
                "extraction_instructions": "Extract 'severity_level' (SEV-1, SEV-2, SEV-3, or SEV-4) and 'clients_affected' (number of clients impacted).",
                "response_instructions": "Assess the severity. Ask how many clients are affected. Apply severity criteria: SEV-1 (100+ clients, full outage), SEV-2 (10+, degraded), SEV-3 (1-10, partial), SEV-4 (minor, workaround exists).",
                "transitions": [
                    {
                        "target_state": "response_team",
                        "description": "Severity classified",
                        "conditions": [
                            {
                                "description": "Severity known",
                                "requires_context_keys": ["severity_level"],
                                "logic": {"has_context": "severity_level"},
                            }
                        ],
                    }
                ],
            },
            "response_team": {
                "id": "response_team",
                "description": "Assign response team",
                "purpose": "Identify the right team and escalation path",
                "extraction_instructions": "Extract 'primary_team' (NOC, application, database, or security) and 'root_cause_hypothesis' (initial theory on cause).",
                "response_instructions": "Based on the affected service, recommend which team should respond (NOC, Application Support, DBA, or Security). Ask for an initial root cause hypothesis.",
                "transitions": [
                    {
                        "target_state": "resolution_tracking",
                        "description": "Team assigned",
                        "conditions": [
                            {
                                "description": "Team and hypothesis set",
                                "requires_context_keys": ["primary_team"],
                                "logic": {"has_context": "primary_team"},
                            }
                        ],
                    }
                ],
            },
            "resolution_tracking": {
                "id": "resolution_tracking",
                "description": "Track resolution progress",
                "purpose": "Document actions taken and current status",
                "extraction_instructions": "Extract 'actions_taken' (what has been done so far) and 'current_status' (ongoing, mitigated, or resolved).",
                "response_instructions": "Ask what actions have been taken so far and the current status. Reference MTTR targets for their severity level. Ask about next steps.",
                "transitions": [
                    {
                        "target_state": "incident_summary",
                        "description": "Resolution tracked",
                        "conditions": [
                            {
                                "description": "Status reported",
                                "requires_context_keys": ["current_status"],
                                "logic": {"has_context": "current_status"},
                            }
                        ],
                    }
                ],
            },
            "incident_summary": {
                "id": "incident_summary",
                "description": "Generate incident summary",
                "purpose": "Present complete incident record",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the incident: title, service, severity, clients affected, response team, root cause hypothesis, actions taken, and current status. Include post-incident review timeline and lessons learned recommendations.",
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
    print("Incident Response -- Management Pipeline with Handlers")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="phase_logger",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["response_phases"].append(
            {"phase": ctx.get("_current_state", "?"), "time": time.strftime("%H:%M:%S")}
        ),
    )

    fsm.create_handler(
        name="escalation_monitor",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["escalation_events"].append(
            ctx.get("severity_level", "unclassified")
        ),
    )

    fsm.create_handler(
        name="processing_logger",
        timing=HandlerTiming.PRE_PROCESSING,
        action=lambda ctx: metrics["processing_log"].append(
            {
                "state": ctx.get("_current_state", "?"),
                "keys": len([k for k in ctx if not k.startswith("_")]),
            }
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "API Gateway returning 503 errors. The authentication service is down",
        "This is SEV-1. We're seeing impact across approximately 200 client applications",
        "Route this to the Application Support team. We suspect a bad deployment from 30 minutes ago",
        "We've rolled back the deployment and restarted services. Status is mitigated, monitoring closely",
    ]

    expected_keys = [
        "incident_title",
        "affected_service",
        "severity_level",
        "clients_affected",
        "primary_team",
        "root_cause_hypothesis",
        "actions_taken",
        "current_status",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("INCIDENT REPORT")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )

    print("\n" + "=" * 60)
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  Response phases: {[p['phase'] for p in metrics['response_phases']]}")
    print(f"  Escalation events: {metrics['escalation_events']}")
    print(f"  Processing log entries: {len(metrics['processing_log'])}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
