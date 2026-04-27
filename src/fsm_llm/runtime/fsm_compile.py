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

Callback contract for ``_cb_eval_transit`` (S5):

- signature: ``(instance: FSMInstance) -> str``
- side effect: when the transition evaluator returns DETERMINISTIC, the
  callback mutates ``instance.current_state`` in place *before*
  returning (mirroring pipeline.py:1063). On AMBIGUOUS the callback
  defers the mutation to ``_cb_resolve_ambig`` (S6).
- return value: one of ``"advanced" | "blocked" | "ambiguous"``. The
  emitted ``Case`` dispatches on this string; the ``"ambiguous"`` branch
  is specialized in S6.

Callback contract for ``_cb_resolve_ambig`` (S6):

- signature: **curried** — ``(instance) -> (message) -> None``. The host
  callable must return a callable: ``def cb(instance): return lambda
  message: ...``. The AST has no multi-arg application and no tuple
  constructor, so curried ``App(App(CB, instance), message)`` is the
  only faithful encoding. See D-S6-01.
- side effect: runs classifier-backed disambiguation; on a valid
  non-fallback target, mutates ``instance.current_state`` in place
  (mirroring pipeline.py:1252-1338 followed by the
  ``_execute_state_transition`` call at pipeline.py:1103). On fallback
  (classifier low-confidence or exception), leaves ``current_state``
  untouched.
- return value: discarded (consumed by the surrounding ``Let`` seq).
- error semantics: exceptions propagate through the ``Let`` and abort
  the current-turn evaluation. Silent recovery is reserved for the
  classifier's fallback path, not wrapping try/except at this layer.

``CB_TRANSIT`` is reserved but unused at S5/S6 — see D-S5-01 in
``_compile_state`` for why eval+apply are bundled atomically.

M2 scope: compile_fsm returns a Term. The pipeline rewrite that calls
``executor.run(compile_fsm(defn), env)`` lives in step S8.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

from fsm_llm.definitions import FSMDefinition, State

from .ast import Term
from .dsl import abs_, app, case_, let_, var
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
    ctx = _CompileCtx(fsm_name=defn.name)
    branches: dict[str, Term] = {
        state_id: _compile_state(state, ctx) for state_id, state in defn.states.items()
    }
    body = case_(var(VAR_STATE_ID), branches)
    # Outer-to-inner: λ state_id. λ message. λ conv_id. λ instance. <body>
    return abs_(
        VAR_STATE_ID,
        abs_(
            VAR_MESSAGE,
            abs_(VAR_CONV_ID, abs_(VAR_INSTANCE, body)),
        ),
    )


def _compile_state(state: State, ctx: _CompileCtx) -> Term:
    """Compile a single FSM state to its per-turn body term.

    Shape (S5):

    - **Terminal state** (``state.transitions`` empty): an optional
      ``Let``-chain for extraction stages, terminating in
      ``App(CB_RESPOND, instance)``.
    - **Non-terminal state** (``state.transitions`` non-empty): the same
      extraction ``Let``-chain, terminating in a transition-dispatch
      ``Let``+``Case`` pair::

          let __disc_k = (_cb_eval_transit instance) in
            case __disc_k of
              "advanced"  → _cb_respond instance
              "blocked"   → _cb_respond instance
              "ambiguous" → _cb_respond instance   # S6 will specialize
              default     → _cb_respond instance

    The ``_cb_eval_transit`` callback bundles transition evaluation with
    the apply-on-deterministic side effect (it mutates
    ``instance.current_state`` in place when DETERMINISTIC). S6 will
    replace the ``"ambiguous"`` branch with a classifier-resolution
    sub-term; S5's three branches are intentionally identical so
    dispatch shape is present for S6 to refine.

    Callbacks mutate ``instance`` / ``instance.context`` in place and may
    return anything (return value other than the eval-transit
    discriminant is dropped by the surrounding Let-chain). All side
    effects — LLM calls, context cleaning, handler hook fires — happen
    inside the callback.

    # DECISION D-S5-01 — eval+apply atomicity
    # The AST has no string-literal node, so a compile-time-known target
    # state id cannot be passed to a separate ``_cb_transit`` callback
    # as an AST arg. Instead ``_cb_eval_transit`` mirrors
    # pipeline.py:1063's existing atomic eval+apply behavior: evaluate,
    # mutate current_state on DETERMINISTIC, return a discriminant
    # string for Case dispatch. ``CB_TRANSIT`` remains reserved but
    # unused at S5. See plans/plan_2026-04-24_7d0db3e4/decisions.md#D-S5-01.
    """
    # Base: terminal response call.
    body: Term = app(var(CB_RESPOND), var(VAR_INSTANCE))

    # Non-terminal states wrap the response in a transition-dispatch
    # Let+Case. Extraction Let-chain (below) nests this whole structure
    # so extractions run before eval_transit.
    if state.transitions:
        disc_name = ctx.gensym("disc")
        # S6: specialize the "ambiguous" branch. The other branches share
        # the same ``body`` reference (respond only) as in S5.
        # # DECISION D-S6-01 — curried two-arg callback
        # # The AST has no multi-arg App; the resolve-ambig callback takes
        # # (instance, message). Encode as curried application
        # # App(App(CB_RESOLVE_AMBIG, instance), message), which the
        # # executor (_apply at executor.py:230) reduces to
        # # callback(instance)(message). The host-bound callable must be
        # # curried at the S8 binding site. See
        # # plans/plan_2026-04-24_28a819cd/decisions.md#D-S6-01.
        ambig_body = let_(
            ctx.gensym("ambig"),
            app(
                app(var(CB_RESOLVE_AMBIG), var(VAR_INSTANCE)),
                var(VAR_MESSAGE),
            ),
            body,
        )
        body = let_(
            disc_name,
            app(var(CB_EVAL_TRANSIT), var(VAR_INSTANCE)),
            case_(
                var(disc_name),
                branches={
                    "advanced": body,
                    "blocked": body,
                    "ambiguous": ambig_body,
                },
                default=body,
            ),
        )

    # Build backwards: wrap the response in Let-chains for each upstream
    # stage that this state declares. Innermost wrap first so the
    # resulting nesting matches runtime evaluation order: bulk → field →
    # classification → response.
    if state.classification_extractions:
        body = let_(
            ctx.gensym("seq"),
            app(var(CB_CLASS_EXTRACT), var(VAR_INSTANCE)),
            body,
        )

    if state.field_extractions:
        body = let_(
            ctx.gensym("seq"),
            app(var(CB_FIELD_EXTRACT), var(VAR_INSTANCE)),
            body,
        )

    extract_inst = state.extraction_instructions
    # Emit CB_EXTRACT whenever legacy `_build_field_configs_from_state`
    # would synthesize extraction work: explicit extraction_instructions,
    # required_context_keys, or any transition condition's
    # requires_context_keys. Matches the legacy semantics exercised by
    # test_functional.py::test_converse_calls_extract_and_generate.
    # Cohort gate (pipeline._check_compiled_cohort) mirrors this predicate
    # at tier 0 to keep sentinel semantics (D-S8b-02).
    has_extract_inst = bool(extract_inst and extract_inst.strip())
    has_required_keys = bool(state.required_context_keys)
    has_transition_required_keys = any(
        bool(getattr(cond, "requires_context_keys", None))
        for t in (state.transitions or [])
        for cond in (t.conditions or [])
    )
    if has_extract_inst or has_required_keys or has_transition_required_keys:
        body = let_(
            ctx.gensym("seq"),
            app(var(CB_EXTRACT), var(VAR_INSTANCE)),
            body,
        )

    return body


# --------------------------------------------------------------
# R2 — kernel-level compile cache (per plan v3 step 8, D-PLAN-07).
# --------------------------------------------------------------

# DECISION D-002 — kernel-level FSM compile cache.
# `compile_fsm_cached(fsm, fsm_id=None)` is the canonical entry for
# callers that want a memoised compile. Cache identity is the pair
# `(fsm_id, fsm.model_dump_json())`: fsm_id provides telemetry/log
# coherence, model_dump_json() is the actual content fingerprint.
# When fsm_id is omitted we synthesise it from a sha256 prefix of the
# JSON, matching API.process_fsm_definition (api.py:299-337).
#
# Cost: lru_cache eviction is on key tuples (LRU on (fsm_id, json))
# rather than the OrderedDict.move_to_end(fsm_id) pattern that
# FSMManager._compiled_terms used. The behavioural assertion delta is
# documented in plan v3 D-PLAN-06: TestFSMManagerCompileCache is
# rewritten to assert kernel-cache behaviour (cache_info hits/misses)
# rather than internal-attribute ordering.
#
# See plans/plan_2026-04-27_a426f667/decisions.md D-PLAN-07.

_COMPILE_FSM_CACHE_MAXSIZE: int = 64


@lru_cache(maxsize=_COMPILE_FSM_CACHE_MAXSIZE)
def _compile_fsm_by_id(fsm_id: str, fsm_json: str) -> Term:
    """LRU-cached compile keyed on (fsm_id, content-fingerprint).

    Internal helper. Callers should go through :func:`compile_fsm_cached`
    rather than invoking this directly — that helper handles the
    fsm_id default and JSON serialisation.

    Equality on lru_cache args: ``fsm_id`` is a string (interned-eligible);
    ``fsm_json`` is the canonical JSON of the FSMDefinition (Pydantic
    ``model_dump_json``). Two definitions with byte-equal JSON produce
    the same cached Term.
    """
    defn = FSMDefinition.model_validate_json(fsm_json)
    return compile_fsm(defn)


def compile_fsm_cached(fsm: FSMDefinition, fsm_id: str | None = None) -> Term:
    """Memoised compile-FSM front-door.

    Compiles ``fsm`` to a Term, caching the result under
    ``(fsm_id, fsm.model_dump_json())`` via an LRU cache of size 64.

    When ``fsm_id`` is None, a stable id is synthesised from the
    sha256 prefix of the JSON: ``f"defn_{sha256(json)[:8]}"`` —
    matching the pattern in ``api.py:process_fsm_definition``.

    Two callers passing the same logical FSM definition (same JSON
    content) but different fsm_id strings will produce independent
    cache entries. This is intentional: fsm_id identifies the *source*
    (e.g., file path vs in-memory dict), and bench/log coherence
    requires separate cache slots per source.

    Direct callers from outside ``FSMManager``:

    - ``Program.from_fsm`` (R1) routes via ``API`` → ``FSMManager`` →
      this function transitively.
    - Stdlib script callers (``scripts/eval.py`` etc.) call this
      directly when they need a Term but no per-conversation state.

    See plans/plan_2026-04-27_a426f667/decisions.md D-PLAN-07.
    """
    fsm_json = fsm.model_dump_json()
    if fsm_id is None:
        digest = hashlib.sha256(fsm_json.encode()).hexdigest()[:8]
        fsm_id = f"defn_{digest}"
    return _compile_fsm_by_id(fsm_id, fsm_json)


__all__ = [
    "compile_fsm",
    "compile_fsm_cached",
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
