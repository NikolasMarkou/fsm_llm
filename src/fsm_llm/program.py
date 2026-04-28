"""
fsm_llm.program — `Program` facade (R1 + R8).

`Program` is the unified entry point for running λ-terms in any of the
three Category surfaces:

- **Category A (FSM dialog)** — `Program.from_fsm(fsm_def | dict | str)`
  delegates to `API`. `.invoke(message=..., conversation_id=...)` (or the
  legacy `.converse(msg, conv_id)` alias) runs one β-reduction step on
  the compiled term, persisting per-conversation state.
- **Category B/C (term / factory)** — `Program.from_term(term)` and
  `Program.from_factory(factory, ...)` wrap a pre-authored λ-term.
  `.invoke(inputs={...})` (or the legacy `.run(**env)` alias) is one
  stateless evaluation.

R8 promotes `.invoke(...)` to the single user-visible verb spanning both
modes. `.run` and `.converse` are preserved as thin deprecation aliases
delegating to `.invoke`. Per Invariant I5 of plan
plan_2026-04-27_32652286: term-mode `.run` and FSM-mode `.converse`
remain (back-compat), AND FSM-mode `.run` and term-mode `.converse` no
longer raise NotImplementedError — they route through `.invoke` so the
correct mode-specific path executes.

References:
- plans/plan_2026-04-27_a426f667/plan.md (R1 success criteria SC1-SC11)
- plans/plan_2026-04-27_32652286/plan.md (R8 — Program.invoke + Result + ProgramModeError)
- plans/plan_2026-04-27_a426f667/findings/program-facade-r1.md
- plans/plan_2026-04-27_a426f667/decisions.md (D-PLAN-02, D-PLAN-03)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .runtime import Executor, LiteLLMOracle, Oracle, Plan, Term

if TYPE_CHECKING:
    from .api import API
    from .definitions import FSMDefinition
    from .handlers import FSMHandler
    from .session import SessionStore


__all__ = ["ExplainOutput", "Program", "ProgramModeError", "Result"]


# ---------------------------------------------------------------------------
# ExplainOutput — value object returned by Program.explain()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainOutput:
    """Static, planner-derived description of a Program's term.

    Returned by ``Program.explain()``. Captures the AST shape (a
    string rendering of the top-level term skeleton), one ``Plan`` per
    discovered ``Fix`` subtree, and the schema declared by every
    ``Leaf`` (keyed by leaf id — synthesised if not user-provided).
    """

    plans: list[Plan] = field(default_factory=list)
    leaf_schemas: dict[str, type | None] = field(default_factory=dict)
    ast_shape: str = ""


# ---------------------------------------------------------------------------
# Result — value object returned by Program.invoke() in term/factory mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """Uniform value object returned by ``Program.invoke`` in **every** mode.

    Per merge-spec §4 CAND-A (M1 of plan plan_2026-04-28_6597e394):
    ``Program.invoke()`` returns ``Result`` for FSM mode, term mode, and
    factory mode alike — eliminating the pre-M1 ``Result | str`` union
    leak in the public surface.

    Fields:

    - ``value`` — the user-visible payload. In FSM mode, the response
      ``str`` (what ``API.converse`` returned pre-M1). In term/factory
      mode, whatever the term reduces to (string for unstructured
      Leaves, a Pydantic model for structured Leaves, or whatever a
      Combinator chain produces).
    - ``conversation_id`` — set in FSM mode (the id of the conversation
      this turn ran against, whether explicit or auto-started).
      ``None`` in term/factory mode.
    - ``plan`` — populated by term/factory mode when the executor
      attached a planner :class:`fsm_llm.lam.Plan` (per-Fix-subtree
      closed-form prediction). ``None`` in FSM mode and for term-mode
      programs without a Fix subtree.
    - ``leaf_calls`` — number of Leaf evaluations the executor issued
      this run (term/factory mode). ``0`` in FSM mode.
    - ``oracle_calls`` — number of Oracle invocations the executor
      issued this run (term/factory mode). ``0`` in FSM mode. After
      M3 (response-Leaf lift), FSM mode will populate this too.
    - ``explain`` — populated only when ``Program.invoke(explain=True)``
      was passed, otherwise ``None``. The :class:`ExplainOutput`
      describes the AST shape, leaf schemas, and (when (n,K) supplied)
      planner output.

    The legacy ``.run(**env)`` and ``.converse(msg, conv_id)`` aliases
    keep their pre-M1 return types (``Any`` / ``str`` respectively) by
    unwrapping ``result.value`` — so users on the old surface are
    unaffected.
    """

    value: Any = None
    conversation_id: str | None = None
    plan: Plan | None = None
    leaf_calls: int = 0
    oracle_calls: int = 0
    explain: ExplainOutput | None = None


# ---------------------------------------------------------------------------
# ProgramModeError — mode-mismatch on the unified .invoke surface
# ---------------------------------------------------------------------------


# Lazy import: FSMError lives in dialog.definitions (which transitively
# pulls in pydantic models). Resolved at class-definition time so the
# inheritance chain is fixed but no extra imports surface at the top of
# this module beyond what was already required.
from .dialog.definitions import FSMError


class ProgramModeError(FSMError):
    """Raised when ``Program.invoke`` is called with arguments that
    don't match the Program's mode.

    Examples:
        >>> Program.from_term(t).invoke(message="hi")
        ProgramModeError: term-mode invoke requires inputs= not message=

        >>> Program.from_fsm(d).invoke(inputs={"x": 1})
        ProgramModeError: FSM-mode invoke requires message= not inputs=
    """

    pass


# ---------------------------------------------------------------------------
# Program — the facade
# ---------------------------------------------------------------------------


class Program:
    """Unified facade over (term, oracle, optional session, optional handlers).

    Three constructors:

    - :meth:`from_fsm` — build from an FSM JSON definition. Internally
      constructs an :class:`fsm_llm.api.API` and delegates ``.invoke`` /
      ``.register_handler`` to it. See ``# DECISION D-001`` below for
      why API-delegation is the right shape in R1.
    - :meth:`from_term` — wrap a pre-authored λ-term directly.
      ``.invoke(inputs={...})`` is supported.
    - :meth:`from_factory` — invoke a stdlib factory and wrap its
      returned term. ``factory_args`` and ``factory_kwargs`` are
      explicit (per Q1 in `findings/program-facade-r1.md`); facade
      kwargs are kw-only.

    The bare ``Program(term=…, oracle=…)`` constructor is the term-mode
    shape and is the simplest path for users who already hold a
    ``Term`` and an ``Oracle``.

    R8 promoted :meth:`invoke` to the single user-visible verb. The
    legacy :meth:`run` and :meth:`converse` are preserved as thin
    deprecation aliases routing to :meth:`invoke` — see ``# DECISION
    D-008`` below.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        term: Term | None = None,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: list[FSMHandler] | None = None,
        # Internal-only: set by from_fsm to enable .converse delegation.
        _api: API | None = None,
    ):
        self._term = term
        self._oracle = oracle
        self._session = session
        self._handlers = list(handlers) if handlers else []
        self._api = _api

        # Sanity: a Program is either term-mode (term is set) or
        # FSM-mode (api is set). It is never both, never neither.
        if (term is None) == (_api is None):
            raise ValueError(
                "Program must be constructed with either a `term` "
                "(term/factory mode) or an internal `_api` (FSM mode), "
                "but not both. Use Program.from_fsm / from_term / "
                "from_factory rather than calling __init__ directly."
            )

        # R5 step 3 — if constructed with handlers in term-mode, compose
        # them into the term up-front. FSM-mode handlers flow through API
        # (via from_fsm registering each handler before returning) and the
        # composition happens lazily in FSMManager. compose() is idempotent
        # for an empty handler list, so this branch is a no-op when no
        # handlers were supplied.
        if self._term is not None and self._handlers:
            from .handlers import compose

            self._term = compose(self._term, self._handlers)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fsm(
        cls,
        fsm_definition: FSMDefinition | dict | str,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: list[FSMHandler] | None = None,
        **api_kwargs: Any,
    ) -> Program:
        """Build a Program backed by an FSM definition.

        # DECISION D-001 — Program.from_fsm constructs an internal API
        # and delegates .converse / .register_handler to it. This keeps
        # R1 strictly additive: no edits to api.py, fsm.py, pipeline.py,
        # or handlers.py. The FSM compile pipeline already provides the
        # full (extract → evaluate → respond) machinery; reusing it is
        # the right shape until R5 collapses both paths into one.
        #
        # Cost: when ``oracle=`` is supplied, it must be a LiteLLMOracle
        # (we unwrap to its underlying LiteLLMInterface so API can use
        # it). Non-LiteLLM oracles raise TypeError. See
        # plans/plan_2026-04-27_a426f667/decisions.md D-PLAN-02.

        ``api_kwargs`` flow through to :class:`fsm_llm.api.API` (model,
        temperature, max_tokens, transition_config, …). They must not
        collide with `llm_interface`, `session_store`, or `handlers`,
        which are derived from this constructor's kw-only args.
        """
        # Lazy local import — avoids pulling api.py at module load
        # (api.py imports the FSM compile pipeline, which transitively
        # touches litellm).
        from .api import API

        # Translate facade kw-only args into the API constructor shape.
        if oracle is not None:
            if not isinstance(oracle, LiteLLMOracle):
                raise TypeError(
                    "Program.from_fsm currently supports only "
                    "LiteLLMOracle instances (which wrap an "
                    "LLMInterface that API can use directly). Got: "
                    f"{type(oracle).__name__}. To use a custom oracle, "
                    "build a Program from a kernel term via "
                    "Program.from_term — see D-PLAN-02 for the R1 "
                    "rationale; R5 will collapse the two paths."
                )
            # Unwrap the underlying LLMInterface for API.
            api_kwargs["llm_interface"] = oracle._llm

        if session is not None:
            api_kwargs["session_store"] = session
        if handlers:
            api_kwargs["handlers"] = list(handlers)

        api = API(fsm_definition, **api_kwargs)
        return cls(_api=api, oracle=oracle, session=session, handlers=handlers)

    @classmethod
    def from_term(
        cls,
        term: Term,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: list[FSMHandler] | None = None,
    ) -> Program:
        """Build a Program from a pre-authored λ-term."""
        return cls(
            term=term,
            oracle=oracle,
            session=session,
            handlers=handlers,
        )

    @classmethod
    def from_factory(
        cls,
        factory: Callable[..., Term],
        factory_args: tuple = (),
        factory_kwargs: dict[str, Any] | None = None,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: list[FSMHandler] | None = None,
    ) -> Program:
        """Build a Program by invoking a stdlib factory."""
        kwargs = factory_kwargs or {}
        term = factory(*factory_args, **kwargs)
        return cls(
            term=term,
            oracle=oracle,
            session=session,
            handlers=handlers,
        )

    # ------------------------------------------------------------------
    # Runtime surface — R8 unified verb
    # ------------------------------------------------------------------

    def invoke(
        self,
        message: str | None = None,
        *,
        inputs: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        explain: bool = False,
    ) -> Result:
        """Single user-visible execution verb (R8 + M1).

        Mode dispatch is automatic. **Returns :class:`Result` in every
        mode** post-M1 (plan plan_2026-04-28_6597e394) — eliminating
        the pre-M1 ``Result | str`` union leak.

        - **FSM mode** (built via :meth:`from_fsm`) — pass ``message=``
          (and optionally ``conversation_id=``). Returns
          ``Result(value=<reply_str>, conversation_id=<id>, ...)``.
          ``inputs=`` is rejected as a mode mismatch.
          ``conversation_id=None`` auto-starts a conversation and
          caches the id on this Program for subsequent calls.
          ``plan`` / ``leaf_calls`` / ``oracle_calls`` are
          ``None`` / ``0`` / ``0`` until M3 lifts the response Leaf.
        - **Term/factory mode** (built via :meth:`from_term` or
          :meth:`from_factory`) — pass ``inputs=`` (a dict unpacked as
          ``**env`` to :class:`fsm_llm.lam.Executor`). Returns
          ``Result(value=<reduction>, leaf_calls=..., oracle_calls=...,
          explain=...)``. ``message=`` is rejected as a mode mismatch.

        Edge cases per plan_2026-04-27_32652286:

        - E1: FSM-mode + no conversation_id → auto-starts.
        - E2: term-mode + ``inputs=None`` → empty env.
        - E3: ``explain=True`` → :class:`Result` with non-None explain.
        - E4: mode mismatch → :class:`ProgramModeError`.

        Raises
        ------
        ProgramModeError
            If ``message=`` is passed in term-mode, or ``inputs=`` is
            passed in FSM-mode.
        """
        if self._api is not None:
            # FSM mode
            if inputs is not None:
                raise ProgramModeError(
                    "FSM-mode invoke takes message= and conversation_id=, "
                    "not inputs=. Build the Program with from_term / "
                    "from_factory if you need an inputs-based call."
                )
            if message is None:
                raise ProgramModeError(
                    "FSM-mode invoke requires message= (the user message "
                    "to send to the FSM)."
                )

            if conversation_id is None:
                # Lazily start (or reuse) a conversation. Stash the id on
                # the Program so subsequent calls without an explicit id
                # continue the same conversation rather than spinning up a
                # fresh one each time (which would be surprising).
                cached = getattr(self, "_default_conv_id", None)
                if cached is None:
                    conversation_id, _greeting = self._api.start_conversation()
                    self._default_conv_id = conversation_id
                else:
                    conversation_id = cached

            # M1 (plan plan_2026-04-28_6597e394 §M1): wrap the FSM-mode
            # reply string in Result so .invoke() returns Result in every
            # mode. plan / leaf_calls / oracle_calls are None / 0 / 0
            # until M3 lifts response generation into a Leaf and we can
            # account oracle calls per-turn against the planner.
            reply = self._api.converse(message, conversation_id)
            return Result(
                value=reply,
                conversation_id=conversation_id,
                plan=None,
                leaf_calls=0,
                oracle_calls=0,
                explain=None,
            )

        # Term/factory mode
        if message is not None:
            raise ProgramModeError(
                "term-mode invoke takes inputs= (a dict unpacked as the "
                "Executor env), not message=. Build the Program with "
                "from_fsm if you need a conversational entry point."
            )
        env = dict(inputs) if inputs else {}
        assert self._term is not None  # invariant: term-mode has term
        executor = self._executor()
        value = executor.run(self._term, env)
        # M1: surface executor accounting on Result. Executor exposes
        # `oracle_calls` directly (the per-run Oracle invocation count
        # used for Theorem-2 checks); leaf-call cardinality is recorded
        # on the CostAccumulator (`total_calls`). plan stays None for
        # term-mode invoke — the planner is exposed via .explain(n=, K=)
        # on demand, not eagerly.
        oracle_calls = getattr(executor, "oracle_calls", 0)
        cost_accum = getattr(executor, "cost_accumulator", None)
        leaf_calls = cost_accum.total_calls if cost_accum is not None else 0
        if explain:
            # Forward the same env as inputs= to .explain so any (n, K)
            # extracted from inputs is honored. R8 contract: when
            # explain=True with inputs= supplied, .explain may use those
            # inputs to populate plans. For now, simple shape-only.
            explain_out = self.explain(inputs=env)
            return Result(
                value=value,
                conversation_id=None,
                plan=None,
                leaf_calls=leaf_calls,
                oracle_calls=oracle_calls,
                explain=explain_out,
            )
        return Result(
            value=value,
            conversation_id=None,
            plan=None,
            leaf_calls=leaf_calls,
            oracle_calls=oracle_calls,
            explain=None,
        )

    # ------------------------------------------------------------------
    # Legacy aliases — preserved for back-compat per Invariant I5
    # ------------------------------------------------------------------

    def run(self, **env: Any) -> Any:
        """Term/factory-mode one-shot evaluation (legacy alias for invoke).

        # DECISION D-008 — `.run(**env)` is preserved as a thin wrapper
        # around `.invoke(inputs=env)` for back-compat per plan
        # plan_2026-04-27_32652286 Invariant I5. Scheduled for
        # deprecation in 0.5.0; removal in 0.6.0 (out of scope here).
        # Users should prefer `program.invoke(inputs={...})`.
        #
        # Behavior change vs R1: FSM-mode `.run(**env)` no longer raises
        # NotImplementedError; instead it routes through `.invoke` which
        # raises ProgramModeError("FSM-mode invoke requires message=...")
        # if `env` doesn't include a `message` key. To preserve the
        # historical "FSM-mode is not stateless" diagnostic, FSM-mode
        # `.run` still raises ProgramModeError with a clear redirect.

        Returns the term's reduction value (NOT a :class:`Result`,
        unlike :meth:`invoke`) so existing call sites continue to work.
        """
        if self._term is None:
            # FSM mode — preserve the historical "not for stateless eval"
            # diagnostic but as ProgramModeError (the new exception type).
            raise ProgramModeError(
                "Program.run is not supported for FSM-backed Programs. "
                "Use .invoke(message=..., conversation_id=...) (or the "
                "legacy .converse alias) instead, or build the Program "
                "with .from_term / .from_factory for stateless one-shot "
                "evaluation."
            )
        # Term mode — unwrap inputs and return value (not Result) for
        # back-compat with the R1 .run signature.
        result = self.invoke(inputs=env)
        assert isinstance(result, Result)
        return result.value

    def converse(self, message: str, conversation_id: str | None = None) -> str:
        """FSM-mode conversational entry (legacy alias for invoke).

        # DECISION D-008 — `.converse(msg, conv_id)` is preserved as a
        # thin wrapper around `.invoke(message=msg, conversation_id=...)`
        # for back-compat per plan plan_2026-04-27_32652286 Invariant
        # I5. Scheduled for deprecation in 0.5.0; removal in 0.6.0
        # (out of scope here). Users should prefer
        # `program.invoke(message="...", conversation_id="...")`.
        #
        # Behavior change vs R1: term-mode `.converse(...)` no longer
        # raises NotImplementedError. Instead it raises ProgramModeError
        # with a clear redirect — term-mode is fundamentally stateless,
        # so a "converse" call has no coherent meaning in that mode.
        """
        if self._api is None:
            raise ProgramModeError(
                "Program.converse is supported only for Programs built "
                "via Program.from_fsm. Term-mode programs (.from_term, "
                ".from_factory) should call .invoke(inputs={...}) (or "
                "the legacy .run(**env) alias) instead — they are "
                "stateless one-shot evaluations."
            )
        # M1 (plan plan_2026-04-28_6597e394): .invoke now returns Result
        # in every mode. Unwrap result.value to preserve the legacy
        # .converse(...) -> str return type. (D-008 back-compat.)
        result = self.invoke(message=message, conversation_id=conversation_id)
        assert isinstance(result, Result)
        assert isinstance(result.value, str)
        return result.value

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        n: int | None = None,
        K: int | None = None,
        plan_kwargs: dict[str, Any] | None = None,
    ) -> ExplainOutput:
        """Static description of the wrapped term.

        Walks the AST and returns:
        - ``ast_shape``: a multi-line indented rendering of the term's
          node-kind skeleton (no template strings, no env values).
        - ``plans``: list of :class:`fsm_llm.lam.Plan` instances —
          **one per discovered ``Fix`` subtree** when ``n`` and ``K``
          are both supplied. When either is ``None`` (the default),
          returns an empty list — :func:`fsm_llm.lam.plan` requires
          runtime quantities ``(n, K)`` that aren't available at
          static-inspection time, so we honour the original R1
          contract. To get a populated ``plans`` list, call
          ``program.explain(n=<input_size>, K=<context_window>)``.
        - ``leaf_schemas``: maps a synthesised leaf-id (template prefix
          + position index) to the leaf's ``schema_ref`` (or ``None``
          for unstructured leaves). The id is stable as long as the
          term is unchanged.

        Parameters
        ----------
        inputs : dict, optional
            R8 addition. When supplied AND no explicit ``n`` is given,
            ``n`` is inferred from ``inputs`` if it contains a top-level
            ``"n"`` (or a ``len(...)``-able first value). Cheap shim;
            users who need precise control should pass ``n`` and ``K``
            directly. ``inputs`` itself is NOT executed by ``.explain``
            — this is a static-walk method.
        n : int, optional
            Input rank for planner. When supplied together with ``K``,
            populates ``plans`` with one :class:`fsm_llm.lam.Plan` per
            ``Fix`` subtree (closed-form planner output). When
            omitted, ``plans`` stays empty.
        K : int, optional
            Oracle context-window budget (tokens). See ``n``.
        plan_kwargs : dict, optional
            Extra :class:`fsm_llm.lam.PlanInputs` kwargs forwarded to
            :func:`fsm_llm.lam.plan` (e.g. ``tau``, ``alpha``,
            ``leaf_accuracy``). Useful for non-default planning
            scenarios.

        For FSM-mode programs, walks the term cached on the underlying
        ``API``'s ``FSMManager`` (compiled at API construction time).
        FSM-mode programs typically have no ``Fix`` subtrees today —
        ``plans`` is empty even when ``(n, K)`` are supplied.
        """
        # R8: if inputs supplied and n omitted, attempt cheap inference.
        if inputs and n is None:
            inferred = inputs.get("n")
            if isinstance(inferred, int):
                n = inferred

        # Resolve which term to walk.
        term: Term | None = self._term
        if term is None and self._api is not None:
            mgr = self._api.fsm_manager
            try:
                term = mgr.get_composed_term(self._api.fsm_id)
            except Exception:  # pragma: no cover — defensive fallback
                try:
                    term = mgr.get_compiled_term(self._api.fsm_id)
                except Exception:
                    term = None
        if term is None:
            return ExplainOutput()

        shape_lines: list[str] = []
        leaf_schemas: dict[str, type | None] = {}
        leaf_counter = [0]
        fix_subtrees: list[Any] = []

        def _walk(node: Any, indent: int) -> None:
            pad = "  " * indent
            kind = type(node).__name__
            if kind == "Var":
                shape_lines.append(f"{pad}Var({node.name!r})")
            elif kind == "Abs":
                shape_lines.append(f"{pad}Abs(param={node.param!r})")
                _walk(node.body, indent + 1)
            elif kind == "App":
                shape_lines.append(f"{pad}App")
                _walk(node.fn, indent + 1)
                _walk(node.arg, indent + 1)
            elif kind == "Let":
                shape_lines.append(f"{pad}Let(name={node.name!r})")
                _walk(node.value, indent + 1)
                _walk(node.body, indent + 1)
            elif kind == "Case":
                shape_lines.append(
                    f"{pad}Case(branches={list(node.branches.keys())!r})"
                )
                _walk(node.scrutinee, indent + 1)
                for key, branch in node.branches.items():
                    shape_lines.append(f"{pad}  ⊢ {key!r}:")
                    _walk(branch, indent + 2)
                if node.default is not None:
                    shape_lines.append(f"{pad}  ⊢ default:")
                    _walk(node.default, indent + 2)
            elif kind == "Combinator":
                shape_lines.append(f"{pad}Combinator(op={node.op})")
                for arg in node.args:
                    _walk(arg, indent + 1)
            elif kind == "Fix":
                shape_lines.append(f"{pad}Fix")
                fix_subtrees.append(node)
                _walk(node.body, indent + 1)
            elif kind == "Leaf":
                idx = leaf_counter[0]
                leaf_counter[0] += 1
                tpl_preview = node.template[:30].replace("\n", " ")
                leaf_id = f"leaf_{idx:03d}_{tpl_preview!r}"
                leaf_schemas[leaf_id] = node.schema_ref
                shape_lines.append(
                    f"{pad}Leaf(template={tpl_preview!r}..., "
                    f"input_vars={list(node.input_vars)!r}, "
                    f"schema_ref={node.schema_ref!r})"
                )
            else:
                shape_lines.append(f"{pad}{kind}(?)")

        _walk(term, 0)

        plans: list[Plan] = []
        if n is not None and K is not None and fix_subtrees:
            from .runtime import PlanInputs
            from .runtime import plan as _plan

            extra = dict(plan_kwargs or {})
            for _fix_node in fix_subtrees:
                inputs_obj = PlanInputs(n=n, K=K, **extra)
                plans.append(_plan(inputs_obj))

        return ExplainOutput(
            plans=plans,
            leaf_schemas=leaf_schemas,
            ast_shape="\n".join(shape_lines),
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def register_handler(self, handler: FSMHandler) -> None:
        """Register a handler.

        FSM-mode (:meth:`from_fsm`): delegates to
        :meth:`fsm_llm.dialog.api.API.register_handler`. Cache invalidation
        is handled by the underlying ``FSMManager``.

        Term-mode (:meth:`from_term` / :meth:`from_factory`): R5 splices
        the handler into ``self._term`` via
        :func:`fsm_llm.handlers.compose` and updates the term in place.
        Subsequent :meth:`invoke` calls evaluate the composed term.

        The handler is also tracked on the Program's own
        ``self._handlers`` list so callers can introspect what's been
        registered without reaching into ``self._api`` internals.
        """
        # DECISION D-STEP-03 — Program.register_handler in term-mode
        # composes handlers into self._term via handlers.compose. FSM-mode
        # delegates to API.register_handler.
        if self._api is None:
            from .handlers import compose

            assert self._term is not None
            self._handlers.append(handler)
            self._term = compose(self._term, self._handlers)
            return

        self._api.register_handler(handler)
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _executor(self) -> Executor:
        """Construct an Executor with a lazily-defaulted oracle.

        Term/factory mode only. FSM mode runs through the API's own
        executor pipeline.
        """
        oracle = self._oracle if self._oracle is not None else _default_oracle()
        return Executor(oracle=oracle)


# ---------------------------------------------------------------------------
# Default-oracle factory — lazy so importing Program never touches network.
# ---------------------------------------------------------------------------


def _default_oracle() -> LiteLLMOracle:
    """Lazy default oracle.

    Mirrors the defaults `API` uses (DEFAULT_LLM_MODEL, temperature=0.5,
    max_tokens=1000) per Q4 in findings/program-facade-r1.md.

    Constructed on first use, not at import time, so importing
    `fsm_llm.program` is side-effect free even when no LLM credentials
    are present.
    """
    from .constants import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE
    from .llm import LiteLLMInterface

    iface = LiteLLMInterface(
        model=DEFAULT_LLM_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=1000,
    )
    return LiteLLMOracle(iface)
