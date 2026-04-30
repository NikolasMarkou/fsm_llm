from fsm_llm.dialog.api import API

"""
Job Application Intake -- Large Context Multi-Turn Extraction
=============================================================

Tests FSM extraction across a detailed multi-turn job application
process with rich contextual information about positions, qualifications,
and candidate background.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/job_application/run.py
"""

import os


def build_fsm() -> dict:
    return {
        "name": "JobApplicationBot",
        "description": "Collects detailed job application information across multiple turns",
        "initial_state": "personal_info",
        "persona": (
            "You are a professional HR intake specialist at TechForward Solutions, "
            "processing applications for a Senior Software Engineer position in the "
            "Cloud Infrastructure team. Be thorough but friendly when collecting "
            "candidate information."
        ),
        "states": {
            "personal_info": {
                "id": "personal_info",
                "description": "Collect candidate personal information",
                "purpose": "Gather name and contact details",
                "extraction_instructions": "Extract 'candidate_name' (full name) and 'email_address' from the message.",
                "response_instructions": "Welcome the candidate to TechForward Solutions. Mention the Senior Software Engineer position in the Cloud Infrastructure team, which requires 5+ years of experience with distributed systems, cloud platforms (AWS, GCP, or Azure), and container orchestration. The salary range is $140,000-$180,000 depending on experience. Benefits include health insurance, 401k matching up to 6%, unlimited PTO, and a $5,000 annual learning budget. The team has 8 engineers growing to 12 by Q3. The role involves microservices architecture, CI/CD pipelines, and Terraform. Remote work with quarterly in-person gatherings in Austin, Texas. Ask for their full name and email address.",
                "transitions": [
                    {
                        "target_state": "experience",
                        "description": "Personal info collected",
                        "conditions": [
                            {
                                "description": "Name and email provided",
                                "requires_context_keys": [
                                    "candidate_name",
                                    "email_address",
                                ],
                                "logic": {
                                    "and": [
                                        {"has_context": "candidate_name"},
                                        {"has_context": "email_address"},
                                    ]
                                },
                            }
                        ],
                    }
                ],
            },
            "experience": {
                "id": "experience",
                "description": "Collect work experience details",
                "purpose": "Understand candidate's professional background",
                "extraction_instructions": "Extract 'years_experience' (number) and 'primary_skill' (main technology area).",
                "response_instructions": "Ask about their years of experience and primary technology skills, especially regarding distributed systems and cloud platforms.",
                "transitions": [
                    {
                        "target_state": "availability",
                        "description": "Experience details collected",
                        "conditions": [
                            {
                                "description": "Experience info provided",
                                "requires_context_keys": ["years_experience"],
                                "logic": {"has_context": "years_experience"},
                            }
                        ],
                    }
                ],
            },
            "availability": {
                "id": "availability",
                "description": "Check availability and salary expectations",
                "purpose": "Confirm start date and compensation alignment",
                "extraction_instructions": "Extract 'start_date' (when they can start) and 'salary_expectation' (desired salary).",
                "response_instructions": "Ask when they can start and their salary expectations. Mention the range is $140,000-$180,000.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Availability confirmed",
                        "conditions": [
                            {
                                "description": "Start date provided",
                                "requires_context_keys": ["start_date"],
                                "logic": {"has_context": "start_date"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize the application",
                "purpose": "Present collected information and confirm",
                "extraction_instructions": "None",
                "response_instructions": "Summarize all collected information: candidate name, email, experience, skills, start date, and salary expectation. Thank them for applying to TechForward Solutions.",
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
    print("Job Application Intake -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, I'm Sarah Chen and my email is sarah.chen@techmail.com",
        "I have 7 years of experience, primarily in cloud infrastructure and Kubernetes",
        "I can start in 3 weeks and I'm looking for around $165,000",
    ]

    expected_keys = [
        "candidate_name",
        "email_address",
        "years_experience",
        "primary_skill",
        "start_date",
        "salary_expectation",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        response = fsm.converse(msg, conv_id)
        print(f"Bot: {response}")

        data = fsm.get_data(conv_id)
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state}")

        if fsm.has_conversation_ended(conv_id):
            break

    # Final summary
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
