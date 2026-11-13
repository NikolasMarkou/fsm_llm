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
embedder; ``ScriptedUser``, which drives a conversation from fixed messages and
rules, ``advance_until_terminal`` for message-free FSMs, and the
``ConversationRun`` record of either (the visited path included); the plain
assertions ``assert_state``, ``assert_context``, ``assert_path`` and
``assert_result`` (explicit ``AssertionError``, never ``assert``); and
``PathRecorder``, an opt-in handler that records the states code under test
visits.

Import it by name, ``from fsm_llm.testing import ...``; ``import fsm_llm`` never
loads it and no library module imports it. It imports the core and
``fsm_llm.core.codes`` only, never ``pytest``, ``litellm`` or another subpackage,
and importing it has no side effect (no clock, no randomness, no socket).

Install:
    From a clone of https://github.com/NikolasMarkou/fsm_llm (no extra deps):
    pip install -e .
"""

from __future__ import annotations

# Version info: imported via __version__.py to stay in sync
from .__version__ import __version__  # noqa: F401
from .assertions import (
    PathRecorder,
    assert_context,
    assert_path,
    assert_result,
    assert_state,
)
from .builders import final_reply, malformed_reply, tool_call_reply
from .doubles import (
    MockLLM2Interface,
    PromptGroundedLLM,
    ScriptedLLM,
    UnscriptedCallError,
    hash_embed,
    state_in_prompt,
)
from .user import ConversationRun, ScriptedUser, advance_until_terminal

__all__ = [
    "ConversationRun",
    "MockLLM2Interface",
    "PathRecorder",
    "PromptGroundedLLM",
    "ScriptedLLM",
    "ScriptedUser",
    "UnscriptedCallError",
    "advance_until_terminal",
    "assert_context",
    "assert_path",
    "assert_result",
    "assert_state",
    "final_reply",
    "hash_embed",
    "malformed_reply",
    "state_in_prompt",
    "tool_call_reply",
]
