"""
Reply builders: the three ``CompletionResponse`` shapes a scripted double returns.

A model turn is a final text, a request for tool calls or a malformed tool-call
turn; these three functions build each one so a test does not spell out a
``CompletionResponse`` and its ``ModelToolCall`` tuple by hand. The ``kind``
strings come from ``fsm_llm.codes.CompletionKind``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.codes import CompletionKind
from fsm_llm.core.definitions import CompletionResponse, ModelToolCall


def final_reply(
    text: str | None, finish_reason: str | None = None
) -> CompletionResponse:
    """A reply with no tool call: what a model answers when it is done.

    Interface contract (queue item of ``ScriptedLLM``, or any ``complete`` fake):
        - ``text``: the reply, or ``None`` for a reply with no text at all.
        - ``finish_reason``: the provider's reason the reply ended, ``None``
          when it sent none; ``"length"`` makes ``truncated`` true (a reply cut
          at ``max_tokens``).
        - Returns a frozen ``CompletionResponse`` of kind
          ``CompletionKind.FINAL``. Never raises.
    """
    return CompletionResponse(
        kind=CompletionKind.FINAL.value, text=text, finish_reason=finish_reason
    )


def tool_call_reply(
    *calls: tuple[str, dict[str, Any]],
    text: str | None = None,
    finish_reason: str | None = None,
) -> CompletionResponse:
    """A reply that asks for one or more tool calls.

    Interface contract (queue item of ``ScriptedLLM``, or any ``complete`` fake):
        - ``calls``: at least one ``(tool_name, arguments)`` pair, ``arguments``
          a dict (it is copied). The calls get the ids ``call_1``, ``call_2``,
          ... in order.
        - ``text``: any text the model wrote beside the calls, else ``None``.
        - ``finish_reason``: as for ``final_reply``.
        - Returns a frozen ``CompletionResponse`` of kind
          ``CompletionKind.CALLS``.
        - Raises ``ValueError`` when no call is given (a ``calls`` reply with
          no call does not exist), and the response model's own
          ``ValidationError`` (a ``ValueError``) for a blank tool name.
    """
    if not calls:
        raise ValueError(
            "tool_call_reply needs at least one (tool_name, arguments) pair; "
            "use final_reply for a reply with no tool call"
        )
    return CompletionResponse(
        kind=CompletionKind.CALLS.value,
        text=text,
        finish_reason=finish_reason,
        calls=tuple(
            ModelToolCall(id=f"call_{i}", name=name, arguments=dict(arguments))
            for i, (name, arguments) in enumerate(calls, start=1)
        ),
    )


def malformed_reply(text: str | None = None) -> CompletionResponse:
    """A tool-call turn that cannot be run (bad arguments, no name, no id).

    Interface contract (queue item of ``ScriptedLLM``, or any ``complete`` fake):
        - ``text``: whatever text came with the broken turn, else ``None``.
        - Returns a frozen ``CompletionResponse`` of kind
          ``CompletionKind.MALFORMED`` that carries no calls, as the real
          interface returns it. Never raises.
    """
    return CompletionResponse(kind=CompletionKind.MALFORMED.value, text=text)
