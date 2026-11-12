"""
FSM-LLM Testing
===============

Public test doubles, a scripted user and assertions for code built on
``fsm_llm``: fake LLM interfaces that need no network, a scripted user that
drives a conversation, and assertion helpers that name a state, a context
value, a visited path or a run result in one line.

Exported today: the two fake ``LLMInterface`` implementations promoted from
the repository's own test configuration, ``MockLLM2Interface`` (fixed answers)
and ``PromptGroundedLLM`` (a fact comes back only when the prompt shows its
evidence). ``ScriptedLLM``, reply builders, ``ScriptedUser``, assertions and
``PathRecorder`` follow in the next steps of the plan.

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
from .doubles import MockLLM2Interface, PromptGroundedLLM

__all__ = [
    "MockLLM2Interface",
    "PromptGroundedLLM",
]
