from __future__ import annotations

"""
FSM JSON → λ-AST compiler (M2).

# DECISION D-003 — Callbacks, not Leaf nodes, for FSM runtime.
# The compiler emits a λ-term whose effectful steps are plain App-chains
# over host-callable Vars (executor.py:234 handles Python callables as
# first-class values under App). Leaf is reserved for native-λ programs
# in M4+. See plans/plan_2026-04-24_69c9ca79/decisions.md#D-003.

# DECISION D-004 — No Fix emission for push_fsm stacking.
# FSM stacking is an API/manager concern (the stack picks which compiled
# term to run per turn); the compiler emits no Fix for it. See D-004.

The compile result is a closed term with the following shape::

    λ state_id.
      λ message.
        λ conv_id.
          λ instance.
            case state_id of
              "s1" → <body for s1>
              "s2" → <body for s2>
              ...

Each state body is a ``Let``-chain that sequences the pipeline stages by
calling callbacks bound in the executor env at run time:

- ``_cb_extract``       — Pass 1 data extraction (mutates instance.context)
- ``_cb_field_extract``  — Field extractions (per state)
- ``_cb_class_extract``  — Classification extractions (per state)
- ``_cb_eval_transit``   — Transition evaluation; returns a discriminant
                          string ("deterministic:<target>", "ambiguous",
                          "blocked") which the following Case branches on.
- ``_cb_resolve_ambig``  — Classifier-backed ambiguity resolution.
- ``_cb_transit``        — Execute the state transition (mutates instance.current_state).
- ``_cb_respond``        — Pass 2 response generation, returns the user-facing string.

Reserved env var names (must be bound at ``executor.run`` time):

- ``state_id``, ``message``, ``conv_id``, ``instance`` — the 4 per-turn inputs.
- All ``_cb_*`` names above — the pipeline callbacks.

No callback does I/O directly; each is a bound method on ``MessagePipeline``
reusing the existing implementation. The compiled term owns control flow;
the callbacks own side effects. This preserves the Tier-3 test surface
(``manager._pipeline._execute_*`` methods) without semantic change.

M2 scope: compile_fsm returns a Term. The pipeline rewrite that calls
``executor.run(compile_fsm(defn), env)`` lives in step S8.
"""

from dataclasses import dataclass, field

from fsm_llm.definitions import FSMDefinition

from .ast import Term
from .errors import ASTConstructionError

# Reserved env-var names. Kept as module constants so the pipeline
# (which builds the env in S8) and the compiler agree on spelling.

VAR_STATE_ID: str = "state_id"
VAR_MESSAGE: str = "message"
VAR_CONV_ID: str = "conv_id"
VAR_INSTANCE: str = "instance"

CB_EXTRACT: str = "_cb_extract"
CB_FIELD_EXTRACT: str = "_cb_field_extract"
CB_CLASS_EXTRACT: str = "_cb_class_extract"
CB_EVAL_TRANSIT: str = "_cb_eval_transit"
CB_RESOLVE_AMBIG: str = "_cb_resolve_ambig"
CB_TRANSIT: str = "_cb_transit"
CB_RESPOND: str = "_cb_respond"

RESERVED_VARS: frozenset[str] = frozenset(
    {
        VAR_STATE_ID,
        VAR_MESSAGE,
        VAR_CONV_ID,
        VAR_INSTANCE,
        CB_EXTRACT,
        CB_FIELD_EXTRACT,
        CB_CLASS_EXTRACT,
        CB_EVAL_TRANSIT,
        CB_RESOLVE_AMBIG,
        CB_TRANSIT,
        CB_RESPOND,
    }
)


@dataclass
class _CompileCtx:
    """Per-compile-run context. Carries diagnostics and internal sequencing."""

    fsm_name: str
    # Internal fresh-name counter for synthesized Let-bindings.
    _gensym_count: int = field(default=0)

    def gensym(self, prefix: str) -> str:
        """Allocate a fresh binding name unique to this compile run."""
        self._gensym_count += 1
        return f"__{prefix}_{self._gensym_count}"


def compile_fsm(defn: FSMDefinition) -> Term:
    """Compile an ``FSMDefinition`` to a closed λ-term.

    The result has shape::

        λ state_id. λ message. λ conv_id. λ instance. case state_id of { ... }

    The returned term is ready for ``Executor.run`` with an env binding
    each of the 4 inputs plus the 7 ``_cb_*`` callbacks (see module
    docstring).

    Raises ``ASTConstructionError`` for malformed definitions that the
    FSMDefinition validator somehow let through (e.g., empty states dict).
    """
    if not defn.states:
        raise ASTConstructionError(
            f"compile_fsm: FSM {defn.name!r} has no states to compile"
        )
    # Step S2 (base case) and beyond extend this to produce a real term.
    # The scaffold lands here so S2's implementation has a clear insertion
    # point and imports elsewhere can reference the symbol.
    raise NotImplementedError(
        "compile_fsm: base case lands in S2; scaffold only in S1."
    )


__all__ = [
    "compile_fsm",
    "VAR_STATE_ID",
    "VAR_MESSAGE",
    "VAR_CONV_ID",
    "VAR_INSTANCE",
    "CB_EXTRACT",
    "CB_FIELD_EXTRACT",
    "CB_CLASS_EXTRACT",
    "CB_EVAL_TRANSIT",
    "CB_RESOLVE_AMBIG",
    "CB_TRANSIT",
    "CB_RESPOND",
    "RESERVED_VARS",
]
