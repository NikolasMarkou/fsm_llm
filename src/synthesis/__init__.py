"""
FSM-LLM Synthesis
=================

Runtime synthesis of conversation procedures: a strong model compiles a goal
into a finite state machine, gates check it, a library stores it, and a cheap
model runs it as a pushed child of a host conversation.

So far: the exception hierarchy, the shared constants, the policy (limits, the
G5 gate and abort injection), the library, the gate composition, the compile
and repair loop, ``obtain`` with the approval gate, the ``ProcedureRunner``
that pushes, runs and pops a procedure under a ``synthesize`` state, and the
agent tools ``obtain_procedure`` and ``run_procedure``.

Import it by name, ``from fsm_llm.synthesis import SynthesisError``;
``import fsm_llm`` never loads it.

Install:
    From a clone of https://github.com/NikolasMarkou/fsm_llm (no extra deps):
    pip install -e .
"""

from __future__ import annotations

# Version info: imported via __version__.py to stay in sync
from .__version__ import __version__  # noqa: F401
from .exceptions import LibraryError, SynthesisError, SynthesisRefusedError
from .gates import GateFinding, GateReport, run_gates, verify_stored
from .library import FSMLibrary, LibraryMatch, LibraryRecord, digest_of
from .policy import (
    PolicyFinding,
    SynthesisPolicy,
    check_policy,
    inject_abort,
    needs_approval,
    strip_abort,
)
from .runner import ProcedureRunner
from .synthesizer import (
    CompileResult,
    FSMSynthesizer,
    Procedure,
    ProcedureApprovalRequest,
    reviewer_from_hitl,
)
from .tools import create_procedure_tools

__all__ = [
    "CompileResult",
    "FSMLibrary",
    "FSMSynthesizer",
    "GateFinding",
    "GateReport",
    "LibraryError",
    "LibraryMatch",
    "LibraryRecord",
    "PolicyFinding",
    "Procedure",
    "ProcedureApprovalRequest",
    "ProcedureRunner",
    "SynthesisError",
    "SynthesisPolicy",
    "SynthesisRefusedError",
    "check_policy",
    "create_procedure_tools",
    "digest_of",
    "inject_abort",
    "needs_approval",
    "reviewer_from_hitl",
    "run_gates",
    "strip_abort",
    "verify_stored",
]
