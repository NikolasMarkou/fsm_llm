"""
Intent Classification and Routing Example
==========================================

Demonstrates using fsm_llm_classification to classify customer support
messages and route them to appropriate handler functions.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os

from fsm_llm import (
    ClassificationPromptConfig,
    ClassificationSchema,
    Classifier,
    IntentDefinition,
    IntentRouter,
)

# ------------------------------------------------------------------
# 1. Define the classification schema
# ------------------------------------------------------------------

schema = ClassificationSchema(
    intents=[
        IntentDefinition(
            name="order_status",
            description="User is asking about the status, delivery, or tracking of an order",
        ),
        IntentDefinition(
            name="product_info",
            description="User is asking about product details, specs, or availability",
        ),
        IntentDefinition(
            name="payment_issue",
            description="User is reporting a payment failure, charge dispute, or billing error",
        ),
        IntentDefinition(
            name="return_request",
            description="User wants to initiate or track a product return",
        ),
        IntentDefinition(
            name="general_support",
            description="Any query that does not fit the above categories",
        ),
    ],
    fallback_intent="general_support",
    confidence_threshold=0.6,
)


# ------------------------------------------------------------------
# 2. Define handler functions (one per intent)
# ------------------------------------------------------------------


def handle_order_status(message: str, entities: dict) -> str:
    order_id = entities.get("order_id", "unknown")
    return f"[Order Status] Looking up order {order_id}. Your package is on its way!"


def handle_product_info(message: str, entities: dict) -> str:
    product = entities.get("product_name", "the product")
    return f"[Product Info] Here are the details for {product}."


def handle_payment_issue(message: str, entities: dict) -> str:
    return "[Payment Issue] I'm sorry about the trouble. Let me connect you with our billing team."


def handle_return_request(message: str, entities: dict) -> str:
    order_id = entities.get("order_id", "your order")
    return f"[Return Request] Starting a return for {order_id}. You'll receive a shipping label shortly."


def handle_general_support(message: str, entities: dict) -> str:
    return "[General Support] Thanks for reaching out! How can I help you today?"


def handle_clarification(message: str, entities: dict) -> str:
    return "[Clarification] I'm not quite sure what you need. Could you rephrase your question?"


# ------------------------------------------------------------------
# 3. Build classifier and router
# ------------------------------------------------------------------


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    is_ollama = "ollama" in model.lower()

    if not is_ollama and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Export it or add it to your .env file.")
        print(
            "       Or use a local Ollama model: export LLM_MODEL='ollama_chat/qwen3.5:9b'"
        )
        return

    classifier = Classifier(
        schema,
        model=model,
        config=ClassificationPromptConfig(temperature=0.0),
    )

    router = IntentRouter(
        schema,
        clarification_handler=handle_clarification,
    )
    router.register_many(
        {
            "order_status": handle_order_status,
            "product_info": handle_product_info,
            "payment_issue": handle_payment_issue,
            "return_request": handle_return_request,
            "general_support": handle_general_support,
        }
    )

    # ------------------------------------------------------------------
    # 4. Interactive loop
    # ------------------------------------------------------------------

    print("Customer Support Classifier (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        try:
            result = classifier.classify(user_input)

            print(f"  Intent:     {result.intent}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.entities:
                print(f"  Entities:   {result.entities}")

            response = router.route(user_input, result)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
