from fsm_llm.dialog.api import API

"""
Multi-Turn Extraction Example -- Precise Field Extraction
=========================================================

Tests the core 2-pass architecture's ability to extract specific
named fields across multiple conversation turns, tracking what
gets extracted and what gets missed.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/multi_turn_extraction/run.py
"""

import os


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    print("=" * 60)
    print("Multi-Turn Extraction Test")
    print("=" * 60)
    print(f"Model: {model}")
    print("Expected fields: full_name, age, email, phone, confirmation\n")

    try:
        fsm = API.from_file(
            path=fsm_path, model=model, api_key=api_key, temperature=0.5
        )

        conv_id, response = fsm.start_conversation()
        print(f"Bot: {response}")

        messages = [
            "Hi, I'm John Smith and I'm 28 years old",
            "My email is john.smith@example.com and my phone is 555-0123",
            "Yes, that's all correct",
        ]

        expected_extractions = {
            0: {"full_name": "John Smith", "age": 28},
            1: {"email": "john.smith@example.com", "phone": "555-0123"},
            2: {"confirmation": True},
        }

        for i, msg in enumerate(messages):
            print(f"\nYou: {msg}")

            response = fsm.converse(msg, conv_id)
            print(f"Bot: {response}")

            # Check extractions
            data = fsm.get_data(conv_id)
            state = fsm.get_current_state(conv_id)
            print(f"  State: {state}")

            if i in expected_extractions:
                print("  Extraction check:")
                for key, expected in expected_extractions[i].items():
                    actual = data.get(key)
                    match = (
                        str(expected).lower() in str(actual).lower()
                        if actual
                        else False
                    )
                    status = "OK" if match else "MISS"
                    print(f"    {key}: expected={expected}, got={actual} [{status}]")

            if fsm.has_conversation_ended(conv_id):
                break

        # Final summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        data = fsm.get_data(conv_id)
        all_expected = ["full_name", "age", "email", "phone", "confirmation"]
        extracted = 0
        for key in all_expected:
            value = data.get(key)
            status = "EXTRACTED" if value is not None else "MISSING"
            if value is not None:
                extracted += 1
            print(f"  {key:15s}: {value!s:30s} [{status}]")

        print(
            f"\nExtraction rate: {extracted}/{len(all_expected)} ({100 * extracted / len(all_expected):.0f}%)"
        )
        print(f"Final state: {fsm.get_current_state(conv_id)}")

        fsm.end_conversation(conv_id)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
