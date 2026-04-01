"""
Medical Triage -- Multi-Level Triage with Handler Analytics
===========================================================

Demonstrates a medical triage system with handler hooks tracking
symptom severity progression, assessment stages, and urgency
classification across a detailed clinical scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/medical_triage/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "triage_stages": [],
    "timing": [],
    "severity_updates": [],
}


def build_fsm() -> dict:
    return {
        "name": "MedicalTriageBot",
        "description": "Multi-level medical triage with severity tracking",
        "initial_state": "initial_assessment",
        "persona": (
            "You are a triage nurse at Metro General Hospital Emergency Department, "
            "a Level II trauma center serving a metropolitan area of 1.2 million "
            "people. Be calm, systematic, and reassuring when triaging patients."
        ),
        "states": {
            "initial_assessment": {
                "id": "initial_assessment",
                "description": "Collect patient identity and chief complaint",
                "purpose": "Identify patient and primary concern",
                "extraction_instructions": "Extract 'patient_name' (full name) and 'chief_complaint' (main reason for visit).",
                "response_instructions": "Greet the patient at Metro General ED. The ED has 45 beds, handles approximately 180 patients daily, and is staffed with board-certified emergency physicians, nurse practitioners, and physician assistants. Triage uses a 5-level Emergency Severity Index (ESI): Level 1 (resuscitation, immediate life threat), Level 2 (emergent, high risk), Level 3 (urgent, multiple resources), Level 4 (less urgent, one resource), Level 5 (non-urgent). Average wait times: Level 1-2 (immediate), Level 3 (45 min), Level 4 (90 min), Level 5 (120 min). On-call specialists available for cardiology, neurology, orthopedics, general surgery, and pediatrics. Lab results within 30 min for stat orders; CT/MRI 1-hour turnaround during peak. The hospital uses Epic for EHR. Ask for their name and what brings them in today.",
                "transitions": [
                    {
                        "target_state": "symptom_detail",
                        "description": "Initial assessment done",
                        "conditions": [
                            {
                                "description": "Complaint stated",
                                "requires_context_keys": ["chief_complaint"],
                                "logic": {"has_context": "chief_complaint"},
                            }
                        ],
                    }
                ],
            },
            "symptom_detail": {
                "id": "symptom_detail",
                "description": "Collect detailed symptom information",
                "purpose": "Assess symptom severity and onset",
                "extraction_instructions": "Extract 'symptom_onset' (when symptoms started) and 'pain_level' (1-10 scale).",
                "response_instructions": "Ask when symptoms started and rate pain on a 1-10 scale. Ask about any associated symptoms.",
                "transitions": [
                    {
                        "target_state": "medical_history",
                        "description": "Symptoms detailed",
                        "conditions": [
                            {
                                "description": "Onset and pain known",
                                "requires_context_keys": ["symptom_onset"],
                                "logic": {"has_context": "symptom_onset"},
                            }
                        ],
                    }
                ],
            },
            "medical_history": {
                "id": "medical_history",
                "description": "Collect relevant medical history",
                "purpose": "Identify risk factors and allergies",
                "extraction_instructions": "Extract 'known_conditions' (pre-existing conditions) and 'current_medications' (medications currently taking).",
                "response_instructions": "Ask about pre-existing medical conditions and current medications. Ask about drug allergies.",
                "transitions": [
                    {
                        "target_state": "vitals",
                        "description": "History collected",
                        "conditions": [
                            {
                                "description": "Conditions known",
                                "requires_context_keys": ["known_conditions"],
                                "logic": {"has_context": "known_conditions"},
                            }
                        ],
                    }
                ],
            },
            "vitals": {
                "id": "vitals",
                "description": "Record vital signs",
                "purpose": "Capture and assess vital signs",
                "extraction_instructions": "Extract 'blood_pressure' (BP reading) and 'temperature' (body temperature).",
                "response_instructions": "Record vital signs. Ask the patient to report their blood pressure if they know it, and whether they've checked their temperature. Provide context about normal ranges.",
                "transitions": [
                    {
                        "target_state": "triage_result",
                        "description": "Vitals recorded",
                        "conditions": [
                            {
                                "description": "Vitals captured",
                                "requires_context_keys": ["blood_pressure"],
                                "logic": {"has_context": "blood_pressure"},
                            }
                        ],
                    }
                ],
            },
            "triage_result": {
                "id": "triage_result",
                "description": "Present triage assessment",
                "purpose": "Classify urgency and provide next steps",
                "extraction_instructions": "None",
                "response_instructions": "Summarize: patient name, chief complaint, symptoms, onset, pain level, medical history, medications, and vitals. Assign an ESI level (1-5) based on the information. Provide estimated wait time and next steps.",
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
    print("Medical Triage -- Multi-Level with Handler Analytics")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="triage_stage_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["triage_stages"].append(
            ctx.get("_current_state", "?")
        ),
    )

    fsm.create_handler(
        name="timing_tracker",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["timing"].append(time.strftime("%H:%M:%S")),
    )

    fsm.create_handler(
        name="severity_monitor",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["severity_updates"].append(
            {
                "pain": ctx.get("pain_level", "N/A"),
                "complaint": ctx.get("chief_complaint", "N/A"),
            }
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm Karen Mitchell. I'm having severe chest tightness and shortness of breath",
        "It started about 2 hours ago. The pain is about a 7 out of 10",
        "I have high blood pressure and take lisinopril and metoprolol daily. No allergies",
        "My blood pressure at home was 158 over 95. My temperature feels normal, maybe 98.6",
    ]

    expected_keys = [
        "patient_name",
        "chief_complaint",
        "symptom_onset",
        "pain_level",
        "known_conditions",
        "current_medications",
        "blood_pressure",
        "temperature",
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
    print("TRIAGE SUMMARY")
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
    print(f"  Triage stages: {metrics['triage_stages']}")
    print(f"  Processing timestamps: {metrics['timing']}")
    print(f"  Severity updates: {len(metrics['severity_updates'])}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
