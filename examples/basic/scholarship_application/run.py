from fsm_llm.dialog.api import API

"""
Scholarship Application -- Large Context Multi-Turn Extraction
==============================================================

Tests FSM extraction for a scholarship application process with
detailed program context, eligibility criteria, and academic info.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/scholarship_application/run.py
"""

import os


def build_fsm() -> dict:
    return {
        "name": "ScholarshipBot",
        "description": "Processes scholarship applications with detailed program context",
        "initial_state": "student_info",
        "persona": (
            "You are an admissions coordinator for the Brightpath STEM Scholarship "
            "Program at Pacific Northwest University. "
            "Be encouraging and supportive when collecting application information."
        ),
        "states": {
            "student_info": {
                "id": "student_info",
                "description": "Collect student personal information",
                "purpose": "Gather name and academic program",
                "extraction_instructions": "Extract 'student_name' (full name) and 'field_of_study' (academic major).",
                "response_instructions": "Welcome the applicant to the Brightpath STEM Scholarship Program. The program awards 50 scholarships annually worth $15,000-$25,000 each, covering tuition, books, and a living stipend. Eligible fields: Computer Science, Engineering, Mathematics, Physics, Biology, and Chemistry. Applicants must be enrolled or accepted into a 4-year undergraduate program with minimum 3.2 GPA. Financial need is considered but not required. First-generation and underrepresented students receive priority. The scholarship is renewable for up to 4 years (maintain 3.0 GPA, 20 hours STEM community service per semester). Recipients get industry mentors, summer research fellowship, and guaranteed internship placement with Microsoft, Boeing, and Intel. Deadline: March 15, 2026. Ask for their full name and intended field of study.",
                "transitions": [
                    {
                        "target_state": "academic_record",
                        "description": "Student info collected",
                        "conditions": [
                            {
                                "description": "Name provided",
                                "requires_context_keys": ["student_name"],
                                "logic": {"has_context": "student_name"},
                            }
                        ],
                    }
                ],
            },
            "academic_record": {
                "id": "academic_record",
                "description": "Collect academic details",
                "purpose": "Assess academic eligibility",
                "extraction_instructions": "Extract 'gpa' (grade point average as a number) and 'year_in_school' (freshman, sophomore, junior, or senior).",
                "response_instructions": "Ask about their current GPA and year in school. Mention the minimum GPA requirement of 3.2.",
                "transitions": [
                    {
                        "target_state": "background",
                        "description": "Academic info collected",
                        "conditions": [
                            {
                                "description": "GPA provided",
                                "requires_context_keys": ["gpa"],
                                "logic": {"has_context": "gpa"},
                            }
                        ],
                    }
                ],
            },
            "background": {
                "id": "background",
                "description": "Collect background and motivation",
                "purpose": "Understand personal circumstances and goals",
                "extraction_instructions": "Extract 'first_generation' (yes or no, whether first in family to attend college) and 'career_goal' (intended career path).",
                "response_instructions": "Ask if they are a first-generation college student and about their career goals in STEM. Mention priority for first-generation students.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Background collected",
                        "conditions": [
                            {
                                "description": "Background provided",
                                "requires_context_keys": ["career_goal"],
                                "logic": {"has_context": "career_goal"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize application",
                "purpose": "Present application summary and timeline",
                "extraction_instructions": "None",
                "response_instructions": "Summarize: student name, field of study, GPA, year, first-generation status, and career goal. Mention scholarship value ($15,000-$25,000), renewability, and the March 15 deadline.",
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
    print("Scholarship Application -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm Priya Sharma and I'm studying Computer Science",
        "My GPA is 3.7 and I'm a sophomore",
        "Yes, I'm the first in my family to go to college. I want to work in AI research",
    ]

    expected_keys = [
        "student_name",
        "field_of_study",
        "gpa",
        "year_in_school",
        "first_generation",
        "career_goal",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        response = fsm.converse(msg, conv_id)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state}")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("APPLICATION SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {value!s:30s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )
    print(f"Final state: {fsm.get_current_state(conv_id)}")
    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
