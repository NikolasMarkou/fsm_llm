"""
Handler Hooks Example -- All 8 Handler Timing Points
=====================================================

Demonstrates registering handlers for all 8 timing points:
START_CONVERSATION, PRE_PROCESSING, POST_PROCESSING,
PRE_TRANSITION, POST_TRANSITION, CONTEXT_UPDATE,
END_CONVERSATION, ERROR.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/handler_hooks/run.py
"""

import json
import os
import time

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming


def build_fsm() -> dict:
    """Simple 3-state FSM for testing hooks."""
    return {
        "name": "HookTestBot",
        "description": "Test all handler timing hooks",
        "initial_state": "ask_name",
        "persona": "A friendly bot collecting your name and favorite color",
        "states": {
            "ask_name": {
                "id": "ask_name",
                "description": "Ask for name",
                "purpose": "Collect the user's name",
                "extraction_instructions": "Extract the user's name as 'user_name'",
                "response_instructions": "Ask the user for their name",
                "transitions": [
                    {
                        "target_state": "ask_color",
                        "description": "Name provided",
                        "conditions": [
                            {
                                "description": "Name available",
                                "requires_context_keys": ["user_name"],
                                "logic": {"has_context": "user_name"},
                            }
                        ],
                    }
                ],
            },
            "ask_color": {
                "id": "ask_color",
                "description": "Ask for favorite color",
                "purpose": "Collect favorite color",
                "extraction_instructions": "Extract the user's favorite color as 'favorite_color'",
                "response_instructions": "Ask for their favorite color",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Color provided",
                        "conditions": [
                            {
                                "description": "Color available",
                                "requires_context_keys": ["favorite_color"],
                                "logic": {"has_context": "favorite_color"},
                            }
                        ],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Say goodbye",
                "purpose": "Thank them and end",
                "extraction_instructions": "None",
                "response_instructions": "Thank them using their name and favorite color, then say goodbye",
                "transitions": [],
            },
        },
    }


# Track all hook invocations
hook_log: list[dict] = []


def log_hook(timing: str, context: dict, extra: str = ""):
    """Record a hook invocation."""
    entry = {
        "timing": timing,
        "time": time.strftime("%H:%M:%S"),
        "state": context.get("_current_state", "?"),
        "extra": extra,
    }
    hook_log.append(entry)
    print(f"  [{timing}] state={entry['state']} {extra}")


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Handler Hooks -- All 8 Timing Points")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm_def = build_fsm()
    fsm = API.from_definition(
        definition=fsm_def, model=model, api_key=api_key, temperature=0.7
    )

    # Register handlers for ALL 8 timing points
    fsm.create_handler(
        name="on_start",
        timing=HandlerTiming.START_CONVERSATION,
        action=lambda ctx: log_hook("START_CONVERSATION", ctx),
    )

    fsm.create_handler(
        name="on_pre_process",
        timing=HandlerTiming.PRE_PROCESSING,
        action=lambda ctx: log_hook(
            "PRE_PROCESSING", ctx,
            f"input='{ctx.get('_user_message', '')[:40]}'"
        ),
    )

    fsm.create_handler(
        name="on_post_process",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: log_hook("POST_PROCESSING", ctx),
    )

    fsm.create_handler(
        name="on_pre_transition",
        timing=HandlerTiming.PRE_TRANSITION,
        action=lambda ctx: log_hook(
            "PRE_TRANSITION", ctx,
            f"target={ctx.get('_target_state', '?')}"
        ),
    )

    fsm.create_handler(
        name="on_post_transition",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: log_hook(
            "POST_TRANSITION", ctx,
            f"from={ctx.get('_previous_state', '?')}"
        ),
    )

    fsm.create_handler(
        name="on_context_update",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: log_hook(
            "CONTEXT_UPDATE", ctx,
            f"keys={[k for k in ctx if not k.startswith('_')]}"
        ),
    )

    fsm.create_handler(
        name="on_end",
        timing=HandlerTiming.END_CONVERSATION,
        action=lambda ctx: log_hook("END_CONVERSATION", ctx),
    )

    fsm.create_handler(
        name="on_error",
        timing=HandlerTiming.ERROR,
        action=lambda ctx: log_hook(
            "ERROR", ctx,
            f"error={ctx.get('_error', 'unknown')}"
        ),
    )

    # Run the conversation
    conv_id, response = fsm.start_conversation()
    print(f"\nBot: {response}\n")

    messages = ["My name is Alice", "Blue is my favorite color", "Bye!"]

    for msg in messages:
        print(f"You: {msg}")
        try:
            response = fsm.converse(msg, conv_id)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

        if fsm.has_conversation_ended(conv_id):
            break

    fsm.end_conversation(conv_id)

    # Summary
    print("=" * 60)
    print("HOOK INVOCATION SUMMARY")
    print("=" * 60)
    from collections import Counter

    counts = Counter(h["timing"] for h in hook_log)
    all_timings = [
        "START_CONVERSATION", "PRE_PROCESSING", "POST_PROCESSING",
        "PRE_TRANSITION", "POST_TRANSITION", "CONTEXT_UPDATE",
        "END_CONVERSATION", "ERROR",
    ]
    for timing in all_timings:
        count = counts.get(timing, 0)
        status = "FIRED" if count > 0 else "not triggered"
        print(f"  {timing:25s} : {count:2d}x  ({status})")
    print(f"\n  Total hook invocations: {len(hook_log)}")


if __name__ == "__main__":
    main()
