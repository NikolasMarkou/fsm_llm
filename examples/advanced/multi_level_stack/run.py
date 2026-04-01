"""
Multi-Level FSM Stacking Example -- 3+ Levels Deep
===================================================

Demonstrates nested FSM stacking beyond 2 levels: a main
customer service FSM pushes a product specialist, which
pushes a warranty sub-specialist, creating a 3-deep stack.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/multi_level_stack/run.py
"""

import os

from fsm_llm import API


def build_warranty_fsm() -> dict:
    """Level 3: Warranty specialist FSM."""
    return {
        "name": "WarrantySpecialist",
        "description": "Handles warranty inquiries",
        "initial_state": "warranty_check",
        "persona": "A knowledgeable warranty specialist",
        "states": {
            "warranty_check": {
                "id": "warranty_check",
                "description": "Check warranty status",
                "purpose": "Determine if product is under warranty",
                "extraction_instructions": "Extract product name and purchase date if mentioned",
                "response_instructions": "Ask about product and purchase date for warranty check",
                "transitions": [
                    {
                        "target_state": "warranty_result",
                        "description": "When warranty info is gathered",
                        "conditions": [
                            {
                                "description": "Product info provided",
                                "requires_context_keys": ["warranty_product"],
                                "logic": {"has_context": "warranty_product"},
                            }
                        ],
                    }
                ],
            },
            "warranty_result": {
                "id": "warranty_result",
                "description": "Provide warranty verdict",
                "purpose": "Give the warranty status and coverage details",
                "extraction_instructions": "None needed",
                "response_instructions": "Inform the customer their product has a standard 2-year warranty with full coverage",
                "transitions": [],
            },
        },
    }


def build_product_fsm() -> dict:
    """Level 2: Product specialist FSM."""
    return {
        "name": "ProductSpecialist",
        "description": "Handles product inquiries and can escalate to warranty",
        "initial_state": "product_inquiry",
        "persona": "A helpful product specialist",
        "states": {
            "product_inquiry": {
                "id": "product_inquiry",
                "description": "Handle product questions",
                "purpose": "Answer product questions or escalate to warranty",
                "extraction_instructions": "Extract what the customer is asking about. Set inquiry_type to 'warranty' if they ask about warranty/guarantee/coverage",
                "response_instructions": "Help with product questions. If warranty-related, let them know you'll connect them with the warranty team",
                "transitions": [
                    {
                        "target_state": "warranty_handoff",
                        "description": "Customer asks about warranty",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Warranty inquiry detected",
                                "requires_context_keys": ["inquiry_type"],
                                "logic": {"==": [{"var": "inquiry_type"}, "warranty"]},
                            }
                        ],
                    },
                    {
                        "target_state": "product_answer",
                        "description": "General product question answered",
                        "priority": 50,
                        "conditions": [
                            {
                                "description": "Product question context available",
                                "requires_context_keys": ["product_question"],
                                "logic": {"has_context": "product_question"},
                            }
                        ],
                    },
                ],
            },
            "warranty_handoff": {
                "id": "warranty_handoff",
                "description": "Transfer to warranty specialist",
                "purpose": "Hand off to warranty team",
                "extraction_instructions": "None",
                "response_instructions": "Let the customer know you're connecting them with the warranty specialist",
                "transitions": [],
            },
            "product_answer": {
                "id": "product_answer",
                "description": "Provide product information",
                "purpose": "Answer the product question",
                "extraction_instructions": "None",
                "response_instructions": "Provide helpful product information",
                "transitions": [],
            },
        },
    }


def build_main_fsm() -> dict:
    """Level 1: Main customer service FSM."""
    return {
        "name": "CustomerService",
        "description": "Main customer service with escalation paths",
        "initial_state": "greeting",
        "persona": "A friendly customer service representative",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Welcome the customer",
                "purpose": "Greet and identify their need",
                "extraction_instructions": "Extract the customer's issue type. Set topic to 'product' if asking about products, features, warranty, etc.",
                "response_instructions": "Welcome the customer and ask how you can help",
                "transitions": [
                    {
                        "target_state": "product_route",
                        "description": "Product-related inquiry",
                        "conditions": [
                            {
                                "description": "Topic is product-related",
                                "requires_context_keys": ["topic"],
                                "logic": {"==": [{"var": "topic"}, "product"]},
                            }
                        ],
                    },
                    {
                        "target_state": "general_help",
                        "description": "General inquiry",
                        "conditions": [
                            {
                                "description": "Any topic identified",
                                "requires_context_keys": ["topic"],
                                "logic": {"has_context": "topic"},
                            }
                        ],
                    },
                ],
            },
            "product_route": {
                "id": "product_route",
                "description": "Route to product specialist",
                "purpose": "Hand off to product team",
                "extraction_instructions": "None",
                "response_instructions": "Let the customer know you're connecting them with a product specialist",
                "transitions": [],
            },
            "general_help": {
                "id": "general_help",
                "description": "Handle general inquiries",
                "purpose": "Answer general questions",
                "extraction_instructions": "None",
                "response_instructions": "Provide helpful general support",
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
    print("Multi-Level FSM Stacking (3 Levels Deep)")
    print("=" * 60)
    print(f"Model: {model}")
    print("Stack: CustomerService -> ProductSpecialist -> WarrantySpecialist\n")

    main_fsm_def = build_main_fsm()
    product_fsm_def = build_product_fsm()
    warranty_fsm_def = build_warranty_fsm()

    try:
        fsm = API.from_definition(
            definition=main_fsm_def, model=model, api_key=api_key, temperature=0.7
        )

        # Start conversation
        conv_id, response = fsm.start_conversation()
        print(f"[Level 1 - CustomerService] {response}")
        print(f"  Stack depth: {fsm.get_stack_depth(conv_id)}")

        # Customer asks about a product
        messages = [
            "I bought a laptop from you and I have questions about it",
            "I want to know about the warranty coverage",
            "It's a ProBook X1 I bought last month",
            "Great, thanks for the info!",
        ]

        for msg in messages:
            print(f"\nCustomer: {msg}")

            state = fsm.get_current_state(conv_id)
            depth = fsm.get_stack_depth(conv_id)

            # Check if we should push to product specialist
            if state == "product_route" and depth == 1:
                print("  [Pushing ProductSpecialist FSM]")
                sub_conv = fsm.push_fsm(conv_id, product_fsm_def)
                _, push_response = fsm.start_conversation(sub_conv)
                print(f"  [Level 2 - ProductSpecialist] {push_response}")

            # Check if we should push to warranty specialist
            if state == "warranty_handoff" and depth == 2:
                print("  [Pushing WarrantySpecialist FSM]")
                sub_conv = fsm.push_fsm(conv_id, warranty_fsm_def)
                _, push_response = fsm.start_conversation(sub_conv)
                print(f"  [Level 3 - WarrantySpecialist] {push_response}")

            try:
                response = fsm.converse(msg, conv_id)
                depth = fsm.get_stack_depth(conv_id)
                state = fsm.get_current_state(conv_id)
                level_name = [
                    "?",
                    "CustomerService",
                    "ProductSpecialist",
                    "WarrantySpecialist",
                ][min(depth, 3)]
                print(f"  [Level {depth} - {level_name}] {response}")
                print(f"  Stack depth: {depth}, State: {state}")
            except Exception as e:
                print(f"  Error: {e}")

            # Pop if conversation at current level ended
            if fsm.has_conversation_ended(conv_id) and depth > 1:
                print(f"  [Popping from level {depth}]")
                fsm.pop_fsm(conv_id)

        # Final summary
        print(f"\nFinal stack depth: {fsm.get_stack_depth(conv_id)}")
        print(f"Final state: {fsm.get_current_state(conv_id)}")

        data = fsm.get_data(conv_id)
        if data:
            print(f"Context keys: {list(data.keys())}")

        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        data = fsm.get_data(conv_id)
        # Keys across all 3 FSM levels:
        # Level 1 (CustomerService): topic
        # Level 2 (ProductSpecialist): inquiry_type, product_question
        # Level 3 (WarrantySpecialist): warranty_product
        expected_keys = [
            "topic",
            "inquiry_type",
            "product_question",
            "warranty_product",
        ]
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
        print(f"Final state: {fsm.get_current_state(conv_id)}")

        fsm.end_conversation(conv_id)
        print("\nConversation ended.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
