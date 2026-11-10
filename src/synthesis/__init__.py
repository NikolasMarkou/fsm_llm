"""
FSM-LLM Synthesis
=================

Runtime synthesis of conversation procedures: a strong model compiles a goal
into a finite state machine, gates check it, a library stores it, and a cheap
model runs it as a pushed child of a host conversation.

So far: the exception hierarchy, the shared constants, the policy (limits, the
G5 gate and abort injection). The compiler, library and runner arrive in later
steps.

Import it by name, ``from fsm_llm.synthesis import SynthesisError``;
``import fsm_llm`` never loads it.

Install:
    From a clone of https://github.com/NikolasMarkou/fsm_llm (no extra deps):
    pip install -e .
"""

from __future__ import annotations

# Version info — imported via __version__.py to stay in sync
from .__version__ import __version__  # noqa: F401
from .exceptions import LibraryError, SynthesisError, SynthesisRefusedError
from .policy import (
    PolicyFinding,
    SynthesisPolicy,
    check_policy,
    inject_abort,
    needs_approval,
    strip_abort,
)

__all__ = [
    "LibraryError",
    "PolicyFinding",
    "SynthesisError",
    "SynthesisPolicy",
    "SynthesisRefusedError",
    "check_policy",
    "inject_abort",
    "needs_approval",
    "strip_abort",
]
