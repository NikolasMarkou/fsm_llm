#!/usr/bin/env python3
"""
FSM Stacking Example: Product Recommendation System

This example demonstrates advanced FSM stacking capabilities with a real-world
product recommendation scenario. It shows how to:

1. Start with a main customer service FSM
2. Stack a specialized product recommendation FSM when needed
3. Handle context inheritance and return
4. Use different merge strategies for context handover
5. Preserve conversation history across FSM transitions

The scenario involves a customer service bot that can delegate to a specialized
product recommendation system when the customer needs product suggestions.
"""

from typing import Dict, Any
from llm_fsm import API, ContextMergeStrategy


def create_main_customer_service_fsm() -> Dict[str, Any]:
    """
    Define the main customer service FSM that handles general inquiries
    and can delegate to specialized sub-systems.
    """
    return {
        "name": "customer_service_main",
        "description": "Main customer service FSM handling general inquiries",
        "version": "3.0",
        "persona": "You are a helpful customer service representative. You're friendly, professional, and always try to assist customers with their needs.",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting and customer identification",
                "purpose": "Welcome the customer and identify how to help them",
                "instructions": "Greet the customer warmly and ask how you can help them today. Try to understand their needs.",
                "required_context_keys": ["customer_name", "inquiry_type"],
                "transitions": [
                    {
                        "target_state": "general_help",
                        "description": "Customer has a general question or issue",
                        "priority": 10
                    },
                    {
                        "target_state": "product_inquiry",
                        "description": "Customer is asking about products or wants recommendations",
                        "priority": 5
                    },
                    {
                        "target_state": "farewell",
                        "description": "Customer wants to end the conversation",
                        "priority": 20
                    }
                ]
            },
            "general_help": {
                "id": "general_help",
                "description": "Handle general customer service inquiries",
                "purpose": "Provide general assistance and information",
                "instructions": "Help the customer with their general inquiry. Be informative and helpful.",
                "transitions": [
                    {
                        "target_state": "product_inquiry",
                        "description": "Customer now wants product recommendations",
                        "priority": 5
                    },
                    {
                        "target_state": "resolution",
                        "description": "Customer's issue has been resolved",
                        "priority": 10
                    },
                    {
                        "target_state": "farewell",
                        "description": "Customer wants to end the conversation",
                        "priority": 15
                    }
                ]
            },
            "product_inquiry": {
                "id": "product_inquiry",
                "description": "Customer is interested in product recommendations",
                "purpose": "Identify that customer needs product recommendations and prepare to delegate",
                "instructions": "Acknowledge the customer's interest in products. Explain that you'll connect them with our product recommendation specialist.",
                "required_context_keys": ["product_category", "budget_range"],
                "transitions": [
                    {
                        "target_state": "delegate_to_product_specialist",
                        "description": "Ready to delegate to product recommendation system",
                        "priority": 5
                    },
                    {
                        "target_state": "general_help",
                        "description": "Customer has other questions first",
                        "priority": 10
                    }
                ]
            },
            "delegate_to_product_specialist": {
                "id": "delegate_to_product_specialist",
                "description": "Delegate to specialized product recommendation FSM",
                "purpose": "Hand over to product specialist with context",
                "instructions": "Inform the customer that you're connecting them with a product specialist who will help them find the perfect product.",
                "transitions": [
                    {
                        "target_state": "post_recommendation_followup",
                        "description": "Return from product recommendation system",
                        "priority": 5
                    }
                ]
            },
            "post_recommendation_followup": {
                "id": "post_recommendation_followup",
                "description": "Follow up after product recommendation session",
                "purpose": "Check if customer needs additional help after product recommendations",
                "instructions": "Welcome the customer back and ask if they need any additional assistance with their product selection or have other questions.",
                "transitions": [
                    {
                        "target_state": "resolution",
                        "description": "Customer is satisfied and ready to proceed",
                        "priority": 5
                    },
                    {
                        "target_state": "general_help",
                        "description": "Customer has additional questions",
                        "priority": 10
                    },
                    {
                        "target_state": "farewell",
                        "description": "Customer wants to end the conversation",
                        "priority": 15
                    }
                ]
            },
            "resolution": {
                "id": "resolution",
                "description": "Customer's needs have been addressed",
                "purpose": "Confirm resolution and offer additional help",
                "instructions": "Summarize what was accomplished and ask if there's anything else you can help with.",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Customer is ready to end the conversation",
                        "priority": 5
                    },
                    {
                        "target_state": "general_help",
                        "description": "Customer has additional questions",
                        "priority": 10
                    }
                ]
            },
            "farewell": {
                "id": "farewell",
                "description": "End the conversation",
                "purpose": "Thank the customer and close the conversation",
                "instructions": "Thank the customer for their time, summarize key outcomes, and wish them well.",
                "transitions": []  # Terminal state
            }
        }
    }


def create_product_recommendation_fsm() -> Dict[str, Any]:
    """
    Define a specialized product recommendation FSM that focuses on
    understanding customer needs and providing tailored recommendations.
    """
    return {
        "name": "product_recommendation_specialist",
        "description": "Specialized FSM for product recommendations",
        "version": "3.0",
        "persona": "You are an expert product recommendation specialist. You're knowledgeable about various products, understand customer needs deeply, and provide personalized recommendations based on budget, preferences, and use cases.",
        "initial_state": "specialist_introduction",
        "states": {
            "specialist_introduction": {
                "id": "specialist_introduction",
                "description": "Product specialist introduces themselves",
                "purpose": "Welcome customer and establish expertise",
                "instructions": "Introduce yourself as a product specialist. Acknowledge any context from the previous conversation and ask about specific needs.",
                "transitions": [
                    {
                        "target_state": "needs_assessment",
                        "description": "Begin detailed needs assessment",
                        "priority": 5
                    }
                ]
            },
            "needs_assessment": {
                "id": "needs_assessment",
                "description": "Detailed assessment of customer needs and preferences",
                "purpose": "Gather comprehensive information about customer requirements",
                "instructions": "Ask detailed questions about the customer's needs, use cases, preferences, and constraints. Be thorough but not overwhelming.",
                "required_context_keys": [
                    "primary_use_case",
                    "budget_range",
                    "important_features",
                    "deal_breakers"
                ],
                "transitions": [
                    {
                        "target_state": "recommendation_generation",
                        "description": "Enough information gathered to make recommendations",
                        "priority": 5
                    },
                    {
                        "target_state": "needs_assessment",
                        "description": "Need more information",
                        "priority": 10
                    }
                ]
            },
            "recommendation_generation": {
                "id": "recommendation_generation",
                "description": "Generate and present product recommendations",
                "purpose": "Provide tailored product recommendations based on assessed needs",
                "instructions": "Based on the gathered information, provide 2-3 specific product recommendations. Explain why each product fits their needs and highlight key features.",
                "required_context_keys": ["recommended_products", "recommendation_reasoning"],
                "transitions": [
                    {
                        "target_state": "recommendation_refinement",
                        "description": "Customer wants to refine or modify recommendations",
                        "priority": 5
                    },
                    {
                        "target_state": "recommendation_finalization",
                        "description": "Customer is satisfied with recommendations",
                        "priority": 8
                    },
                    {
                        "target_state": "needs_assessment",
                        "description": "Need to reassess needs based on feedback",
                        "priority": 15
                    }
                ]
            },
            "recommendation_refinement": {
                "id": "recommendation_refinement",
                "description": "Refine recommendations based on customer feedback",
                "purpose": "Adjust recommendations based on customer preferences",
                "instructions": "Listen to customer feedback and adjust recommendations accordingly. Ask clarifying questions if needed.",
                "required_context_keys": ["refinement_feedback", "updated_recommendations"],
                "transitions": [
                    {
                        "target_state": "recommendation_generation",
                        "description": "Present updated recommendations",
                        "priority": 5
                    },
                    {
                        "target_state": "recommendation_finalization",
                        "description": "Customer is satisfied with refined recommendations",
                        "priority": 8
                    }
                ]
            },
            "recommendation_finalization": {
                "id": "recommendation_finalization",
                "description": "Finalize the recommendation process",
                "purpose": "Summarize recommendations and next steps",
                "instructions": "Summarize the final recommendations, provide next steps for purchase or further research, and prepare to hand back to main customer service.",
                "required_context_keys": ["final_recommendations", "next_steps"],
                "transitions": [
                    {
                        "target_state": "specialist_handoff",
                        "description": "Ready to hand back to main customer service",
                        "priority": 5
                    }
                ]
            },
            "specialist_handoff": {
                "id": "specialist_handoff",
                "description": "Hand conversation back to main customer service",
                "purpose": "Smoothly transition back to main customer service",
                "instructions": "Thank the customer for their time, summarize what was accomplished, and let them know you're handing them back to the main customer service team.",
                "transitions": []  # Terminal state - will pop back to main FSM
            }
        }
    }


def run_stacking_example():
    """
    Run the complete FSM stacking example with realistic interactions.
    """
    print("🤖 FSM Stacking Example: Product Recommendation System")
    print("=" * 60)

    # Create the main customer service API
    main_fsm = create_main_customer_service_fsm()
    api = API.from_definition(
        main_fsm,
        model="gpt-4o-mini",  # Using a cost-effective model for the example
        temperature=0.7,
        max_tokens=500
    )

    print("\n📞 Starting customer service conversation...")

    # Start the main conversation
    try:
        conv_id, initial_response = api.start_conversation({
            "session_id": "demo_session_001",
            "timestamp": "2024-01-15T10:30:00Z"
        })

        print(f"🤖 Customer Service: {initial_response}")
        print(f"📊 Stack depth: {api.get_stack_depth(conv_id)}")

        # Simulate customer responses
        customer_messages = [
            "Hi! I'm looking for a laptop for my college studies.",
            "I need something that's good for programming and not too expensive. My budget is around $800-1200.",
        ]

        current_message_index = 0

        while not api.has_conversation_ended(conv_id) and current_message_index < len(customer_messages):
            user_message = customer_messages[current_message_index]
            print(f"\n👤 Customer: {user_message}")

            # Get current state before processing
            current_state = api.get_current_state(conv_id)
            print(f"📍 Current state: {current_state}")

            # Process the message
            response = api.converse(user_message, conv_id)
            print(f"🤖 Response: {response}")

            # Check if we should delegate to product specialist
            new_state = api.get_current_state(conv_id)
            if new_state == "delegate_to_product_specialist":
                print("\n🔄 Delegating to Product Recommendation Specialist...")

                # Get current context to pass to specialist
                current_context = api.get_data(conv_id)
                print(f"📋 Context to pass: {list(current_context.keys())}")

                # Create product recommendation FSM
                product_fsm = create_product_recommendation_fsm()

                # Push the product recommendation FSM onto the stack
                specialist_response = api.push_fsm(
                    conv_id,
                    product_fsm,
                    context_to_pass={
                        "customer_name": current_context.get("customer_name", "Customer"),
                        "initial_budget": current_context.get("budget_range", "$800-1200"),
                        "product_category": current_context.get("product_category", "laptop")
                    },
                    return_context={
                        "specialist_session_completed": True,
                        "referral_source": "main_customer_service"
                    },
                    shared_context_keys=["customer_name", "session_id", "final_recommendations"],
                    preserve_history=True,
                    inherit_context=True
                )

                print(f"🤖 Product Specialist: {specialist_response}")
                print(f"📊 Stack depth after push: {api.get_stack_depth(conv_id)}")

                # Simulate interaction with product specialist
                specialist_messages = [
                    "I mainly need it for programming in Python and Java, some web development, and general college work like writing papers.",
                    "I'd prefer something portable since I'll be carrying it around campus. Battery life is important too.",
                    "Those sound great! I think the ThinkPad would be perfect for my needs."
                ]

                for specialist_msg in specialist_messages:
                    print(f"\n👤 Customer: {specialist_msg}")
                    specialist_resp = api.converse(specialist_msg, conv_id)
                    print(f"🤖 Product Specialist: {specialist_resp}")

                    # Check if we've reached the end of specialist conversation
                    if api.get_current_state(conv_id) == "specialist_handoff":
                        print("\n🔄 Product specialist ready to hand back to main service...")

                        # Get the context from specialist before popping
                        specialist_context = api.get_data(conv_id)
                        print(f"📋 Specialist context: {list(specialist_context.keys())}")

                        # Pop back to main FSM with comprehensive context return
                        return_response = api.pop_fsm(
                            conv_id,
                            context_to_return={
                                "product_recommendations_completed": True,
                                "recommended_product": "Lenovo ThinkPad E14",
                                "specialist_notes": "Customer prefers portable laptop for programming",
                                "estimated_price": "$1100",
                                "customer_satisfaction": "high"
                            },
                            merge_strategy=ContextMergeStrategy.UPDATE
                        )

                        print(f"🤖 Back to Customer Service: {return_response}")
                        print(f"📊 Stack depth after pop: {api.get_stack_depth(conv_id)}")
                        break

            current_message_index += 1

        # Continue with main conversation
        followup_messages = [
            "That was really helpful! Do you have information about warranty options?",
            "Perfect, I think I'm all set. Thank you for your help!"
        ]

        for followup_msg in followup_messages:
            if api.has_conversation_ended(conv_id):
                break

            print(f"\n👤 Customer: {followup_msg}")
            response = api.converse(followup_msg, conv_id)
            print(f"🤖 Customer Service: {response}")

        # Display final results
        print("\n" + "=" * 60)
        print("📊 CONVERSATION SUMMARY")
        print("=" * 60)

        # Show final context
        final_context = api.get_data(conv_id)
        print(f"📋 Final Context Keys: {list(final_context.keys())}")

        # Show context flow
        context_flow = api.get_context_flow(conv_id)
        print(f"🔄 Context Flow: {context_flow}")

        # Show all stack data (should be just main FSM now)
        all_data = api.get_all_stack_data(conv_id)
        print(f"📚 Stack Data Levels: {len(all_data)}")

        print(f"\n✅ Conversation completed successfully!")

    except Exception as e:
        print(f"❌ Error during conversation: {str(e)}")
        raise

    finally:
        # Clean up
        api.close()


if __name__ == "__main__":
    """
    Run the FSM stacking example.

    This example demonstrates:
    1. Complex FSM definitions as Python dictionaries
    2. Context inheritance and handover between FSMs
    3. Realistic conversation flows with state transitions
    4. Professional customer service and product recommendation scenarios
    5. Advanced features like shared context keys and merge strategies
    """

    # Set up environment (you would normally have this in your environment)
    import os

    # Uncomment and set your API key:
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. This example requires an OpenAI API key.")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    run_stacking_example()