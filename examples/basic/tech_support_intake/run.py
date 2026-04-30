from fsm_llm.dialog.api import API
"""
Tech Support Intake -- Large Context Multi-Turn Extraction
==========================================================

Tests FSM extraction for a technical support ticket creation process
with detailed product context, issue classification, and diagnostics.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/tech_support_intake/run.py
"""

import os



def build_fsm() -> dict:
    return {
        "name": "TechSupportIntakeBot",
        "description": "Creates support tickets with detailed product and system context",
        "initial_state": "customer_info",
        "persona": (
            "You are a Level 1 support agent at CloudSync Technologies, a B2B SaaS "
            "company providing enterprise file synchronization and collaboration tools. "
            "Be efficient and empathetic when creating support tickets."
        ),
        "states": {
            "customer_info": {
                "id": "customer_info",
                "description": "Identify the customer and their plan",
                "purpose": "Gather customer name and product",
                "extraction_instructions": "Extract 'customer_name' (full name) and 'product_plan' (drive, collaborate, or enterprise).",
                "response_instructions": "Welcome the customer to CloudSync Technologies support. The product suite includes CloudSync Drive (file storage, $12/user/mo), CloudSync Collaborate (real-time editing, $20/user/mo), and CloudSync Enterprise ($35/user/mo, includes SSO, audit logs, and compliance features). Common issues: sync conflicts (35%), permission errors (25%), SSO problems (20%), and performance issues (15%). Platform supports Windows 10+, macOS 12+, iOS 15+, and Android 12+. Known issues this week: intermittent sync delays on Windows client v4.2.1 (hotfix Thursday), and SSO timeout with Okta SAML 2.0 configs. SLA response times: Critical (1hr), High (4hr), Medium (8hr), Low (24hr). Ask for their name and which product they use (Drive, Collaborate, or Enterprise).",
                "transitions": [
                    {
                        "target_state": "issue_details",
                        "description": "Customer identified",
                        "conditions": [
                            {
                                "description": "Name provided",
                                "requires_context_keys": ["customer_name"],
                                "logic": {"has_context": "customer_name"},
                            }
                        ],
                    }
                ],
            },
            "issue_details": {
                "id": "issue_details",
                "description": "Collect issue description",
                "purpose": "Understand the technical problem",
                "extraction_instructions": "Extract 'issue_category' (sync, permissions, sso, or performance) and 'operating_system' (the OS they're using).",
                "response_instructions": "Ask them to describe their issue. Ask what operating system they're on. Mention common categories: sync conflicts, permission errors, SSO problems, performance.",
                "transitions": [
                    {
                        "target_state": "severity",
                        "description": "Issue documented",
                        "conditions": [
                            {
                                "description": "Issue described",
                                "requires_context_keys": ["issue_category"],
                                "logic": {"has_context": "issue_category"},
                            }
                        ],
                    }
                ],
            },
            "severity": {
                "id": "severity",
                "description": "Assess severity and impact",
                "purpose": "Determine ticket priority",
                "extraction_instructions": "Extract 'severity_level' (critical, high, medium, or low) and 'users_affected' (number of people impacted).",
                "response_instructions": "Ask about the business impact and how many users are affected. Explain SLA times: Critical (1hr), High (4hr), Medium (8hr), Low (24hr).",
                "transitions": [
                    {
                        "target_state": "ticket_summary",
                        "description": "Severity assessed",
                        "conditions": [
                            {
                                "description": "Severity determined",
                                "requires_context_keys": ["severity_level"],
                                "logic": {"has_context": "severity_level"},
                            }
                        ],
                    }
                ],
            },
            "ticket_summary": {
                "id": "ticket_summary",
                "description": "Summarize and create ticket",
                "purpose": "Present ticket details and expected response time",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the support ticket: customer name, product, issue category, OS, severity, and users affected. Provide the expected SLA response time based on severity. Mention any known related issues.",
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
    print("Tech Support Intake -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, I'm James Kim. We use CloudSync Enterprise",
        "We're having sync issues on Windows. Files aren't syncing for the past 2 hours",
        "It's high priority, affecting about 25 users in our engineering department",
    ]

    expected_keys = [
        "customer_name",
        "product_plan",
        "issue_category",
        "operating_system",
        "severity_level",
        "users_affected",
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
    print("TICKET SUMMARY")
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
