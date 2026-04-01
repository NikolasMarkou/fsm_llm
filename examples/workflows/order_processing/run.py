"""
Order Processing Workflow Example — Workflows + Core FSM
=========================================================

Demonstrates the workflows DSL to orchestrate a multi-step order processing
pipeline. A ConversationStep runs an FSM to collect order details, then
ConditionStep and APICallStep handle validation and payment processing.

Combines:
    - fsm_llm_workflows: Workflow DSL (create_workflow, conversation_step,
      condition_step, api_step, auto_step)
    - fsm_llm: Core FSM via ConversationStep for order collection

Key Concepts:
    - WorkflowEngine with async execution
    - ConversationStep: embed an FSM conversation inside a workflow
    - ConditionStep: branch based on context values
    - APICallStep: mock external API integration
    - auto_messages: drive an FSM conversation automatically

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import asyncio
import os
from typing import Any

from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    api_step,
    condition_step,
    conversation_step,
    create_workflow,
)

# ------------------------------------------------------------------
# Terminal step — completes the workflow (no outgoing transition)
# ------------------------------------------------------------------


class TerminalStep(WorkflowStep):
    """A step that ends the workflow. Returns no next_state so the engine
    marks the workflow as COMPLETED."""

    action: Any = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.action):
            self.action(context)
        return WorkflowStepResult.success_result(
            data={}, next_state=None, message="Workflow complete"
        )


# ------------------------------------------------------------------
# Mock API functions (simulating external services)
# ------------------------------------------------------------------


async def validate_inventory(
    product_name: str = "unknown", quantity: int = 1, **kwargs
) -> dict:
    """Mock inventory check — always succeeds for demo.

    APICallStep unpacks input_mapping as keyword arguments, so parameters
    must match the keys defined in input_mapping.
    """
    try:
        qty = max(1, int(quantity))
    except (ValueError, TypeError):
        qty = 1
    print(f"  [Inventory API] Checking stock for {qty}x {product_name}...")
    return {
        "in_stock": True,
        "unit_price": 29.99,
        "total_price": 29.99 * qty,
    }


async def process_payment(
    total_price: float = 0, customer_email: str = "unknown", **kwargs
) -> dict:
    """Mock payment processing — always succeeds for demo."""
    try:
        price = float(total_price)
    except (ValueError, TypeError):
        price = 0.0
    print(f"  [Payment API] Processing ${price:.2f} for {customer_email}...")
    return {
        "payment_status": "approved",
        "transaction_id": "TXN-2024-00042",
    }


async def send_confirmation(
    customer_email: str = "unknown", transaction_id: str = "unknown", **kwargs
) -> dict:
    """Mock email confirmation."""
    print(
        f"  [Email API] Sending confirmation to {customer_email} for order {transaction_id}"
    )
    return {"confirmation_sent": True}


# ------------------------------------------------------------------
# Build the workflow
# ------------------------------------------------------------------


def build_order_workflow(model: str) -> "WorkflowEngine":
    """Build the order processing workflow."""
    import json

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm_order_form.json")

    with open(fsm_path) as f:
        fsm_definition = json.load(f)

    # Define the workflow using the DSL
    workflow = create_workflow(
        "order_processing",
        "Order Processing Pipeline",
        "Collect order via FSM conversation, validate, charge, confirm.",
    )

    # Step 1: Collect order details via FSM conversation
    # auto_messages simulate a customer placing an order
    workflow.with_initial_step(
        conversation_step(
            step_id="collect_order",
            name="Collect Order Details",
            fsm_definition=fsm_definition,
            model=model,
            auto_messages=[
                "Hi, I'm Alice Johnson, my email is alice@example.com",
                "I'd like to order 2 wireless headphones, ship to 123 Main St, Springfield, IL 62701",
                "Yes, that looks correct!",
            ],
            context_mapping={
                "customer_name": "customer_name",
                "customer_email": "customer_email",
                "product_name": "product_name",
                "quantity": "quantity",
                "shipping_address": "shipping_address",
            },
            success_state="check_inventory",
            error_state="order_failed",
            description="Run FSM conversation to collect order info",
        )
    )

    # Step 2: Check inventory via mock API
    workflow.with_step(
        api_step(
            step_id="check_inventory",
            name="Inventory Check",
            api_function=validate_inventory,
            success_state="validate_order",
            failure_state="order_failed",
            input_mapping={"product_name": "product_name", "quantity": "quantity"},
            output_mapping={"in_stock": "in_stock", "total_price": "total_price"},
            description="Check product availability",
        )
    )

    # Step 3: Validate the order (condition check)
    workflow.with_step(
        condition_step(
            step_id="validate_order",
            name="Validate Order",
            condition=lambda ctx: (
                ctx.get("in_stock", False) and ctx.get("customer_email")
            ),
            true_state="process_payment",
            false_state="order_failed",
            description="Verify inventory and customer info",
        )
    )

    # Step 4: Process payment via mock API
    workflow.with_step(
        api_step(
            step_id="process_payment",
            name="Process Payment",
            api_function=process_payment,
            success_state="send_confirmation",
            failure_state="order_failed",
            input_mapping={
                "total_price": "total_price",
                "customer_email": "customer_email",
            },
            output_mapping={
                "payment_status": "payment_status",
                "transaction_id": "transaction_id",
            },
            description="Charge the customer",
        )
    )

    # Step 5: Send confirmation email
    workflow.with_step(
        api_step(
            step_id="send_confirmation",
            name="Send Confirmation",
            api_function=send_confirmation,
            success_state="order_complete",
            failure_state="order_complete",  # Still complete even if email fails
            input_mapping={
                "customer_email": "customer_email",
                "transaction_id": "transaction_id",
            },
            output_mapping={"confirmation_sent": "confirmation_sent"},
            description="Email order confirmation",
        )
    )

    # Step 6: Order complete (terminal — ends the workflow)
    workflow.with_step(
        TerminalStep(
            step_id="order_complete",
            name="Order Complete",
            action=lambda ctx: print_order_summary(ctx),
            description="Final step — display summary",
        )
    )

    # Step 7: Order failed (terminal — ends the workflow)
    workflow.with_step(
        TerminalStep(
            step_id="order_failed",
            name="Order Failed",
            action=lambda ctx: print(
                f"  [ORDER FAILED] Reason: {ctx.get('error', 'Unknown')}"
            ),
            description="Handle order failure",
        )
    )

    # Register and return
    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


def print_order_summary(ctx: dict) -> dict:
    """Print the final order summary."""
    print("\n" + "=" * 50)
    print("ORDER COMPLETE")
    print("=" * 50)
    print(f"  Customer:     {ctx.get('customer_name', 'N/A')}")
    print(f"  Email:        {ctx.get('customer_email', 'N/A')}")
    print(f"  Product:      {ctx.get('product_name', 'N/A')}")
    print(f"  Quantity:     {ctx.get('quantity', 'N/A')}")
    print(f"  Ship to:      {ctx.get('shipping_address', 'N/A')}")
    print(f"  Total:        ${ctx.get('total_price', 0):.2f}")
    print(f"  Transaction:  {ctx.get('transaction_id', 'N/A')}")
    print(f"  Payment:      {ctx.get('payment_status', 'N/A')}")
    print(f"  Confirmed:    {ctx.get('confirmation_sent', False)}")
    return {}


# ------------------------------------------------------------------
# Run the workflow
# ------------------------------------------------------------------


async def run():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not os.getenv("OPENAI_API_KEY") and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    print("Order Processing Workflow")
    print("=" * 50)
    print("This workflow will:")
    print("  1. Collect order details via FSM conversation (auto-driven)")
    print("  2. Check inventory")
    print("  3. Validate the order")
    print("  4. Process payment")
    print("  5. Send confirmation email")
    print()

    try:
        engine = build_order_workflow(model)

        # Start the workflow with empty initial context
        instance_id = await engine.start_workflow(
            "order_processing", initial_context={}
        )
        print(f"Workflow started: {instance_id}\n")

        # The workflow runs automatically — check status
        instance = engine.get_workflow_instance(instance_id)
        if instance:
            print(f"\nFinal status: {instance.status.value}")
            print(f"Final context keys: {list(instance.context.keys())}")

            # ── Verification ──
            ctx = instance.context
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            checks = {
                "workflow_completed": instance.status.value == "completed",
                "customer_name": ctx.get("customer_name"),
                "customer_email": ctx.get("customer_email"),
                "product_name": ctx.get("product_name"),
                "in_stock": ctx.get("in_stock"),
                "total_price": ctx.get("total_price"),
                "payment_status": ctx.get("payment_status"),
                "transaction_id": ctx.get("transaction_id"),
                "confirmation_sent": ctx.get("confirmation_sent"),
            }
            extracted = 0
            for key, value in checks.items():
                passed = value is not None and value not in (False, 0, "", "failed")
                status = "EXTRACTED" if passed else "MISSING"
                if passed:
                    extracted += 1
                print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")
            print(
                f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
            )
    except Exception as e:
        print(f"Workflow error: {e}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
