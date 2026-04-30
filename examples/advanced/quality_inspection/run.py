"""
Quality Inspection -- QA Flow with Handler Metrics
===================================================

Demonstrates a manufacturing quality inspection process with handler
hooks tracking inspection stages, defect counts, and pass/fail rates
across a detailed manufacturing QA scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/quality_inspection/run.py
"""

import os
import time
from typing import Any

from fsm_llm.dialog.api import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "inspection_stages": [],
    "findings_log": [],
    "stage_times": [],
}


def build_fsm() -> dict:
    return {
        "name": "QualityInspectionBot",
        "description": "Manufacturing quality inspection with defect tracking",
        "initial_state": "product_identification",
        "persona": (
            "You are a lead quality inspector at Pinnacle Automotive Parts, a Tier 1 "
            "automotive supplier producing precision-machined brake components, suspension "
            "parts, and steering assemblies for Toyota, Ford, BMW, and Hyundai. "
            "Be precise, systematic, and standards-driven in your assessments."
        ),
        "states": {
            "product_identification": {
                "id": "product_identification",
                "description": "Identify the product being inspected",
                "purpose": "Record part number and production batch",
                "extraction_instructions": "Extract 'part_number' (the part identifier) and 'batch_id' (production batch/lot number).",
                "response_instructions": "Begin the quality inspection process. The facility operates 3 shifts, produces 45,000 parts daily, and maintains IATF 16949 certification. Inspection follows a 4-gate process: Visual Inspection (surface defects, dimensional conformity, labeling), Dimensional Measurement (CMM verification within +/-0.005mm for critical, +/-0.05mm for non-critical), Material Testing (Rockwell C hardness, surface roughness Ra, spectrometry composition), and Functional Testing (assembly fit, 150% load test, 100,000 cycle life sampling). Defect classification: Critical (safety risk, containment + 8D), Major (form/fit/function failure, corrective action), Minor (cosmetic, shippable), Observation (improvement opportunity). Current first-pass yield: 97.3% (target 98.5%). Customer PPM: 12 (target <10). Ask for the part number and production batch ID.",
                "transitions": [
                    {
                        "target_state": "visual_inspection",
                        "description": "Product identified",
                        "conditions": [
                            {
                                "description": "Part identified",
                                "requires_context_keys": ["part_number"],
                                "logic": {"has_context": "part_number"},
                            }
                        ],
                    }
                ],
            },
            "visual_inspection": {
                "id": "visual_inspection",
                "description": "Perform visual and dimensional inspection",
                "purpose": "Check surface quality and basic dimensions",
                "extraction_instructions": "Extract 'visual_result' (pass or fail) and 'defects_found' (number of defects observed).",
                "response_instructions": "Conduct visual inspection. Ask about surface condition, dimensional conformity, and labeling. Record the result as pass/fail and count of any defects found.",
                "transitions": [
                    {
                        "target_state": "material_testing",
                        "description": "Visual done",
                        "conditions": [
                            {
                                "description": "Visual result recorded",
                                "requires_context_keys": ["visual_result"],
                                "logic": {"has_context": "visual_result"},
                            }
                        ],
                    }
                ],
            },
            "material_testing": {
                "id": "material_testing",
                "description": "Perform material property tests",
                "purpose": "Verify hardness and material composition",
                "extraction_instructions": "Extract 'hardness_result' (pass or fail against spec) and 'material_verified' (yes or no).",
                "response_instructions": "Ask about hardness test results (Rockwell C scale) and material composition verification (spectrometry). Record whether each test meets specification.",
                "transitions": [
                    {
                        "target_state": "functional_test",
                        "description": "Material tests done",
                        "conditions": [
                            {
                                "description": "Material tested",
                                "requires_context_keys": ["hardness_result"],
                                "logic": {"has_context": "hardness_result"},
                            }
                        ],
                    }
                ],
            },
            "functional_test": {
                "id": "functional_test",
                "description": "Perform functional and load testing",
                "purpose": "Verify part meets performance requirements",
                "extraction_instructions": "Extract 'load_test_result' (pass or fail) and 'overall_disposition' (accept, rework, or scrap).",
                "response_instructions": "Ask about load testing results (150% rated capacity) and assembly fit check. Based on all inspection results, determine overall disposition: accept, rework, or scrap.",
                "transitions": [
                    {
                        "target_state": "inspection_report",
                        "description": "Functional test done",
                        "conditions": [
                            {
                                "description": "Disposition set",
                                "requires_context_keys": ["overall_disposition"],
                                "logic": {"has_context": "overall_disposition"},
                            }
                        ],
                    }
                ],
            },
            "inspection_report": {
                "id": "inspection_report",
                "description": "Generate inspection report",
                "purpose": "Summarize all inspection results",
                "extraction_instructions": "None",
                "response_instructions": "Generate the complete inspection report: part number, batch, visual result, defects found, hardness, material verification, load test, and overall disposition. Include any corrective action recommendations.",
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
    print("Quality Inspection -- QA Flow with Handler Metrics")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="stage_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["inspection_stages"].append(
            ctx.get("_current_state", "?")
        ),
    )

    fsm.create_handler(
        name="findings_logger",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["findings_log"].append(
            {
                k: v
                for k, v in ctx.items()
                if ("result" in k or "disposition" in k) and not k.startswith("_")
            }
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Inspecting part number BRK-4421-A, batch ID 2026-W14-003",
        "Visual inspection passed. No surface defects, dimensions within spec. Zero defects found",
        "Hardness test passed at 58 HRC, within the 56-60 HRC specification. Material composition verified by spectrometry",
        "Load test passed at 150% rated capacity. Assembly fit check is good. Overall disposition: accept",
    ]

    expected_keys = [
        "part_number",
        "batch_id",
        "visual_result",
        "defects_found",
        "hardness_result",
        "material_verified",
        "load_test_result",
        "overall_disposition",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        metrics["stage_times"].append(elapsed)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("INSPECTION REPORT")
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
    print(f"  Inspection stages: {metrics['inspection_stages']}")
    print(f"  Stage times: {[f'{t:.1f}s' for t in metrics['stage_times']]}")
    print(f"  Findings logged: {len(metrics['findings_log'])} entries")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
