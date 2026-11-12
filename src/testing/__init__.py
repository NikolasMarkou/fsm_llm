"""
FSM-LLM Testing
===============

Public test doubles, a scripted user and assertions for code built on
``fsm_llm``: fake LLM interfaces that need no network, a scripted user that
drives a conversation, and assertion helpers that name a state, a context
value, a visited path or a run result in one line.

Exported today: the fake ``LLMInterface`` implementations ``MockLLM2Interface``
(fixed answers), ``PromptGroundedLLM`` (a fact comes back only when the prompt
shows its evidence) and ``ScriptedLLM`` (a queue of replies, fields from a
dict, the Pass-2 reply per state, metered, and an unscripted call recorded and
raised as ``UnscriptedCallError``); the reply builders ``final_reply``,
``tool_call_reply`` and ``malformed_reply``; ``state_in_prompt``, the one
reader of the state tag of a response prompt; and ``hash_embed``, a pure
embedder. ``ScriptedUser``, assertions and ``PathRecorder`` follow in the next
steps of the plan.

Import it by name, ``from fsm_llm.testing import ...``; ``import fsm_llm`` never
loads it and no library module imports it. It imports the core and
``fsm_llm.codes`` only, never ``pytest``, ``litellm`` or another subpackage,
and importing it has no side effect (no clock, no randomness, no socket).

Install:
    From a clone of https://github.com/NikolasMarkou/fsm_llm (no extra deps):
    pip install -e .
"""

from __future__ import annotations

# Version info: imported via __version__.py to stay in sync
from .__version__ import __version__  # noqa: F401
from .builders import final_reply, malformed_reply, tool_call_reply
from .doubles import (
    MockLLM2Interface,
    PromptGroundedLLM,
    ScriptedLLM,
    UnscriptedCallError,
    hash_embed,
    state_in_prompt,
)

__all__ = [
    "MockLLM2Interface",
    "PromptGroundedLLM",
    "ScriptedLLM",
    "UnscriptedCallError",
    "final_reply",
    "hash_embed",
    "malformed_reply",
    "state_in_prompt",
    "tool_call_reply",
]
