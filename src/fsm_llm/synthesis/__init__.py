"""
FSM-LLM Synthesis
=================

Runtime synthesis of conversation procedures: a strong model compiles a goal
into a finite state machine, gates check it, a library stores it, and a cheap
model runs it as a pushed child of a host conversation.

So far: the exception hierarchy, the shared constants, the policy (limits, the
G5 gate and abort injection), the library, the gate composition (G1 to G5, the
G6a dead-edge gate and the G6b ``Simulator``), the compile and repair loop,
``RunTrace`` and ``TraceRecorder``, ``diagnose`` over run traces,
``apply_patch`` (an RFC 6902 subset), ``propose_patch`` and
``repair_procedure``, ``obtain`` with the approval gate, the
``ProcedureRunner`` that pushes, runs and pops a procedure under a
``synthesize`` state, and the agent tools ``obtain_procedure`` and
``run_procedure``.

Import it by name, ``from fsm_llm.synthesis import SynthesisError``;
``import fsm_llm`` never loads it.

Install:
    From a clone of https://github.com/NikolasMarkou/fsm_llm (no extra deps):
    pip install -e .
"""

from __future__ import annotations

# Version info: imported via __version__.py to stay in sync
from .__version__ import __version__  # noqa: F401
from .diagnose import Diagnosis, diagnose
from .dryrun import analyze_dryrun
from .exceptions import (
    LibraryError,
    PatchError,
    SynthesisError,
    SynthesisRefusedError,
)
from .gates import GateFinding, GateReport, run_gates, verify_stored
from .library import FSMLibrary, LibraryMatch, LibraryRecord, digest_of
from .patch import apply_patch
from .policy import (
    PolicyFinding,
    SynthesisPolicy,
    check_policy,
    inject_abort,
    needs_approval,
    strip_abort,
)
from .repair import PromotionResult, propose_patch, repair_procedure
from .runner import ProcedureRunner
from .simulation import SimulationReport, Simulator
from .synthesizer import (
    CompileResult,
    FSMSynthesizer,
    Procedure,
    ProcedureApprovalRequest,
    reviewer_from_hitl,
)
from .tools import create_procedure_tools
from .traces import RunTrace, TraceRecorder

__all__ = [
    "CompileResult",
    "Diagnosis",
    "FSMLibrary",
    "FSMSynthesizer",
    "GateFinding",
    "GateReport",
    "LibraryError",
    "LibraryMatch",
    "LibraryRecord",
    "PatchError",
    "PolicyFinding",
    "Procedure",
    "ProcedureApprovalRequest",
    "ProcedureRunner",
    "PromotionResult",
    "RunTrace",
    "SimulationReport",
    "Simulator",
    "SynthesisError",
    "SynthesisPolicy",
    "SynthesisRefusedError",
    "TraceRecorder",
    "analyze_dryrun",
    "apply_patch",
    "check_policy",
    "create_procedure_tools",
    "diagnose",
    "digest_of",
    "inject_abort",
    "needs_approval",
    "propose_patch",
    "repair_procedure",
    "reviewer_from_hitl",
    "run_gates",
    "strip_abort",
    "verify_stored",
]
