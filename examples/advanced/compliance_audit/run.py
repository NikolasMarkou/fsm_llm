"""
Compliance Audit -- Checklist Flow with Handler Tracking
========================================================

Demonstrates a compliance audit checklist process with handler hooks
tracking audit progress, findings severity, and remediation status
across a detailed regulatory compliance scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/compliance_audit/run.py
"""

import os
import time
from typing import Any

from fsm_llm.dialog.api import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "audit_phases": [],
    "findings_count": [],
    "timestamps": [],
}


def build_fsm() -> dict:
    return {
        "name": "ComplianceAuditBot",
        "description": "Guided compliance audit with finding tracking",
        "initial_state": "audit_scope",
        "persona": (
            "You are a senior compliance auditor at Sterling Compliance Partners, a "
            "regulatory consulting firm specializing in financial services compliance. "
            "The firm conducts over 200 audits annually across banking, insurance, and "
            "investment sectors. Be methodical, thorough, and objective when conducting "
            "audit procedures."
        ),
        "states": {
            "audit_scope": {
                "id": "audit_scope",
                "description": "Define audit scope and objectives",
                "purpose": "Establish what will be audited",
                "extraction_instructions": "Extract 'organization_name' (entity being audited) and 'audit_type' (SOX, BSA/AML, GLBA, PCI DSS, or other).",
                "response_instructions": "Introduce the audit engagement. Your expertise covers SOX, BSA/AML, GLBA data privacy, PCI DSS, and state-specific requirements. Findings are classified as: Critical (remediate within 30 days, board notification), Major (60 days, management action plan), Minor (90 days, tracking log), and Observation (best practice, no timeline). The firm uses a risk-based 6-phase process: Planning, Fieldwork, Analysis, Reporting, Remediation Tracking, and Follow-up. Average audit duration is 4-8 weeks. The firm has 98% client retention and zero post-audit enforcement actions. Working papers retained 7 years. Common findings: inadequate access controls (45% of audits), incomplete transaction monitoring (38%), insufficient documentation (52%), outdated policies (35%). Ask which organization is being audited and what type of compliance audit this is.",
                "transitions": [
                    {
                        "target_state": "access_controls",
                        "description": "Scope defined",
                        "conditions": [
                            {
                                "description": "Org and type known",
                                "requires_context_keys": ["organization_name"],
                                "logic": {"has_context": "organization_name"},
                            }
                        ],
                    }
                ],
            },
            "access_controls": {
                "id": "access_controls",
                "description": "Review access controls",
                "purpose": "Assess user access management and authentication",
                "extraction_instructions": "Extract 'access_control_status' (compliant, partial, or non-compliant) and 'mfa_implemented' (yes or no).",
                "response_instructions": "Review access control measures. Ask about user provisioning/deprovisioning processes and multi-factor authentication implementation.",
                "transitions": [
                    {
                        "target_state": "data_protection",
                        "description": "Access reviewed",
                        "conditions": [
                            {
                                "description": "Access status known",
                                "requires_context_keys": ["access_control_status"],
                                "logic": {"has_context": "access_control_status"},
                            }
                        ],
                    }
                ],
            },
            "data_protection": {
                "id": "data_protection",
                "description": "Review data protection controls",
                "purpose": "Assess data encryption, retention, and privacy",
                "extraction_instructions": "Extract 'encryption_at_rest' (yes or no) and 'data_retention_policy' (whether a formal policy exists).",
                "response_instructions": "Assess data protection measures. Ask about encryption for data at rest and in transit. Ask about data retention and disposal policies.",
                "transitions": [
                    {
                        "target_state": "documentation",
                        "description": "Data protection reviewed",
                        "conditions": [
                            {
                                "description": "Encryption status known",
                                "requires_context_keys": ["encryption_at_rest"],
                                "logic": {"has_context": "encryption_at_rest"},
                            }
                        ],
                    }
                ],
            },
            "documentation": {
                "id": "documentation",
                "description": "Review policy documentation",
                "purpose": "Verify completeness of compliance documentation",
                "extraction_instructions": "Extract 'policies_current' (yes or no, whether policies are up to date) and 'last_review_date' (when policies were last reviewed).",
                "response_instructions": "Review policy documentation. Ask if compliance policies are current and when they were last reviewed. Ask about the policy review cadence.",
                "transitions": [
                    {
                        "target_state": "audit_report",
                        "description": "Documentation reviewed",
                        "conditions": [
                            {
                                "description": "Policy status known",
                                "requires_context_keys": ["policies_current"],
                                "logic": {"has_context": "policies_current"},
                            }
                        ],
                    }
                ],
            },
            "audit_report": {
                "id": "audit_report",
                "description": "Generate audit summary report",
                "purpose": "Present findings and recommendations",
                "extraction_instructions": "None",
                "response_instructions": "Summarize audit findings: organization, audit type, access control status, MFA, encryption, data retention, policy currency, and last review. Classify overall compliance posture and list recommended remediation actions with priorities.",
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
    print("Compliance Audit -- Checklist with Handler Tracking")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="audit_phase_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["audit_phases"].append(
            ctx.get("_current_state", "?")
        ),
    )

    fsm.create_handler(
        name="findings_counter",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["findings_count"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    fsm.create_handler(
        name="timestamp_logger",
        timing=HandlerTiming.PRE_PROCESSING,
        action=lambda ctx: metrics["timestamps"].append(time.strftime("%H:%M:%S")),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "We're auditing First National Savings Bank for BSA/AML compliance",
        "Access controls are partially compliant. MFA is implemented for external access but not internal systems",
        "Data at rest is encrypted using AES-256. We have a formal data retention policy with 7-year retention",
        "Policies were last reviewed 14 months ago. Some sections are outdated, so I'd say not fully current",
    ]

    expected_keys = [
        "organization_name",
        "audit_type",
        "access_control_status",
        "mfa_implemented",
        "encryption_at_rest",
        "data_retention_policy",
        "policies_current",
        "last_review_date",
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
    print("AUDIT REPORT")
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
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  Audit phases: {metrics['audit_phases']}")
    print(f"  Findings growth: {metrics['findings_count']}")
    print(f"  Processing timestamps: {metrics['timestamps']}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
