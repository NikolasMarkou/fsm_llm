"""
fsm_llm.program — the `Program` facade.

`Program` is the unified entry point for running λ-terms across the three
Category surfaces. Mode is fixed at construction time via one of three
classmethods; ``.invoke(...)`` is the single user-visible execution verb
and returns ``Result`` uniformly.

* **Category A (FSM dialog)** — ``Program.from_fsm(fsm_def | dict | str)``
  compiles the FSM and runs ``.invoke(message=..., conversation_id=...)``
  one β-reduction step at a time, persisting per-conversation state.
* **Category B (term)** — ``Program.from_term(term)`` wraps a
  pre-authored λ-term. ``.invoke(inputs={...})`` is a single stateless
  evaluation.
* **Category C (factory)** — ``Program.from_factory(factory, ...)`` calls
  the factory at construction time and wraps the resulting term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .runtime import Executor, LiteLLMOracle, Oracle, Plan, Term

if TYPE_CHECKING:
    from .dialog.api import API
    from .dialog.definitions import FSMDefinition
    from .dialog.session import SessionStore
    from .handlers import FSMHandler
    from .profiles import HarnessProfile


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
    """Uniform value object returned by ``Program.invoke`` in every mode.

    Fields:

    * ``value`` — the user-visible payload. In FSM mode, the response
      string. In term/factory mode, whatever the term reduces to (a
      string for unstructured Leaves, a Pydantic model for structured
      Leaves, or whatever a Combinator chain produces).
    * ``conversation_id`` — set in FSM mode (the id of the conversation
      this turn ran against, whether explicit or auto-started). ``None``
      in term/factory mode.
    * ``plan`` — populated by term/factory mode when the executor
      attached a planner :class:`fsm_llm.runtime.Plan` (per-Fix-subtree
      closed-form prediction). ``None`` in FSM mode and for term-mode
      programs without a Fix subtree.
    * ``leaf_calls`` — number of Leaf evaluations the executor issued
      this run (term/factory mode). ``0`` in FSM mode.
    * ``oracle_calls`` — number of Oracle invocations the executor
      issued this run (term/factory mode). ``0`` in FSM mode.
    * ``explain`` — populated only when ``Program.invoke(explain=True)``
      was passed, otherwise ``None``. The :class:`ExplainOutput`
      describes the AST shape, leaf schemas, and (when (n,K) supplied)
      planner output.
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


# FSMError lives in the neutral ``fsm_llm._models`` layer (since 0.7.0); the
# class hierarchy below extends it for mode-mismatch errors on .invoke.
from ._models import FSMError


class ProgramModeError(FSMError):
    """Raised when ``Program.invoke`` is called with arguments that
    don't match the Program's mode.

    Mode is fixed at construction:

    * **FSM mode** (``Program.from_fsm``) accepts ``message=`` and
      ``conversation_id=``. Passing ``inputs=`` raises
      ``ProgramModeError``.
    * **Term/factory mode** (``Program.from_term`` / ``from_factory``)
      accepts ``inputs=``. Passing ``message=`` raises
      ``ProgramModeError``.

    Examples:
        >>> Program.from_term(t).invoke(message="hi")
        ProgramModeError: term-mode invoke takes inputs=, not message=

        >>> Program.from_fsm(d).invoke(inputs={"x": 1})
        ProgramModeError: FSM-mode invoke takes message=, not inputs=
    """

    pass


# ---------------------------------------------------------------------------
# Program — the facade
# ---------------------------------------------------------------------------


class Program:
    """Unified facade over (term, oracle, optional session, optional handlers).

    Three constructors:

    * :meth:`from_fsm` — build from an FSM JSON definition (Category A).
    * :meth:`from_term` — wrap a pre-authored λ-term (Category B).
    * :meth:`from_factory` — invoke a stdlib factory and wrap the
      returned term (Category C).

    Mode is fixed at construction. ``.invoke(...)`` is the single user-
    visible execution verb and returns ``Result`` uniformly in every mode.
    Calling ``.invoke`` with the wrong argument shape for the program's
    mode raises :class:`ProgramModeError`.

    Direct construction via ``Program(term=..., oracle=...)`` is supported
    for callers that already hold a kernel ``Term`` and an ``Oracle``.
    FSM-mode construction is only available through :meth:`from_fsm` —
    the internal ``API`` instance is private state.
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
    ):
        """Construct a term-mode Program directly.

        For FSM-mode construction use :meth:`from_fsm`. For term-mode
        callers that already hold a kernel ``Term``, this is the simple
        path; ``handlers`` are spliced into the term up-front via
        ``handlers.compose``.
        """
        if term is None:
            raise ValueError(
                "Program(...) requires a `term` argument. Use "
                "Program.from_fsm(...) for FSM-mode construction or "
                "Program.from_term / from_factory for term-mode."
            )
        self._term = term
        self._oracle = oracle
        self._session = session
        self._handlers = list(handlers) if handlers else []
        self._api: API | None = None
        self._profile: HarnessProfile | None = None

        # If constructed with handlers in term-mode, compose them into the
        # term up-front. compose() is idempotent for an empty handler list,
        # so this branch is a no-op when no handlers were supplied.
        if self._handlers:
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
        profile: HarnessProfile | str | None = None,
        # ----- API constructor passthroughs (explicit since 0.8.0) -----
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_history_size: int = 5,
        max_message_length: int = 1000,
        handler_error_mode: str = "continue",
        transition_config: Any | None = None,
        # Additional LiteLLM kwargs (top_p, presence_penalty, …).
        **llm_kwargs: Any,
    ) -> Program:
        """Build a Program backed by an FSM definition.

        Parameters
        ----------
        fsm_definition
            The FSM JSON, dict, or path to a JSON file.
        oracle
            Optional ``LiteLLMOracle`` to thread through the dialog
            pipeline. When supplied, must be a ``LiteLLMOracle``; non-
            LiteLLM oracles raise ``TypeError``. To use a custom oracle,
            construct a kernel term and call :meth:`from_term` instead.
        session
            Optional :class:`SessionStore` for per-conversation
            persistence (defaults to in-memory).
        handlers
            Optional list of :class:`FSMHandler` instances spliced into
            the compiled FSM term at construction.
        profile
            Optional :class:`HarnessProfile` name or instance applied
            to the compiled term once at construction. See
            ``profiles.apply_to_term`` for the Theorem-2 contract.
        model, api_key, temperature, max_tokens
            LLM configuration forwarded to the default
            :class:`LiteLLMInterface`.
        max_history_size, max_message_length
            Conversation-history bounds.
        handler_error_mode
            Either ``"continue"`` (skip failed handlers) or ``"raise"``.
        transition_config
            Optional :class:`TransitionEvaluatorConfig` for fine-grained
            transition resolution.
        **llm_kwargs
            Additional keyword arguments forwarded to
            :class:`LiteLLMInterface` (e.g. ``top_p``,
            ``presence_penalty``).
        """
        # Lazy local import — avoids pulling api.py at module load.
        from .dialog.api import API

        api_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history_size": max_history_size,
            "max_message_length": max_message_length,
            "handler_error_mode": handler_error_mode,
            "transition_config": transition_config,
            **llm_kwargs,
        }
        # Drop ``None`` defaults so ``API`` applies its own defaults.
        api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}

        if oracle is not None:
            if not isinstance(oracle, LiteLLMOracle):
                raise TypeError(
                    "Program.from_fsm currently supports only "
                    "LiteLLMOracle instances (which wrap an "
                    "LLMInterface that API can use directly). Got: "
                    f"{type(oracle).__name__}. To use a custom oracle, "
                    "build a Program from a kernel term via "
                    "Program.from_term."
                )
            # Unwrap the underlying LLMInterface for API; pass the
            # Oracle itself so identity propagates from
            # Program → API → FSMManager → MessagePipeline.
            api_kwargs["llm_interface"] = oracle._llm
            api_kwargs["oracle"] = oracle

        if session is not None:
            api_kwargs["session_store"] = session
        if handlers:
            api_kwargs["handlers"] = list(handlers)

        api = API(fsm_definition, **api_kwargs)

        # Profile application (apply-once). For FSM mode, the compiled
        # term lives on the API's FSMManager. We resolve the per-Manager
        # composed term, apply Leaf overrides, and stash the result in
        # ``FSMManager._composed_term_cache`` keyed on
        # ``(fsm_id, _handlers_version)``.
        resolved_profile = _resolve_profile(profile)
        if resolved_profile is not None:
            from .profiles import apply_to_term

            mgr = api.fsm_manager
            try:
                composed = mgr.get_composed_term(api.fsm_id)
                rewritten = apply_to_term(composed, resolved_profile)
                if rewritten is not composed and hasattr(mgr, "_composed_term_cache"):
                    key = (api.fsm_id, mgr._handlers_version)
                    mgr._composed_term_cache[key] = rewritten
            except Exception:  # pragma: no cover — defensive
                pass

        return cls._from_api(
            api=api,
            oracle=oracle,
            session=session,
            handlers=handlers,
            profile=resolved_profile,
        )

    @classmethod
    def from_term(
        cls,
        term: Term,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: list[FSMHandler] | None = None,
        profile: HarnessProfile | str | None = None,
    ) -> Program:
        """Build a Program from a pre-authored λ-term."""
        resolved_profile = _resolve_profile(profile)
        if resolved_profile is not None:
            from .profiles import apply_to_term

            term = apply_to_term(term, resolved_profile)
        instance = cls(
            term=term,
            oracle=oracle,
            session=session,
            handlers=handlers,
        )
        instance._profile = resolved_profile
        return instance

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
        profile: HarnessProfile | str | None = None,
    ) -> Program:
        """Build a Program by invoking a stdlib factory."""
        kwargs = factory_kwargs or {}
        term = factory(*factory_args, **kwargs)
        resolved_profile = _resolve_profile(profile)
        if resolved_profile is not None:
            from .profiles import apply_to_term

            term = apply_to_term(term, resolved_profile)
        instance = cls(
            term=term,
            oracle=oracle,
            session=session,
            handlers=handlers,
        )
        instance._profile = resolved_profile
        return instance

    # ------------------------------------------------------------------
    # Private FSM-mode construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_api(
        cls,
        *,
        api: API,
        oracle: Oracle | None,
        session: SessionStore | None,
        handlers: list[FSMHandler] | None,
        profile: HarnessProfile | None,
    ) -> Program:
        """Internal FSM-mode constructor — bypasses the public
        ``__init__`` to bind the private ``_api`` slot.
        """
        instance = cls.__new__(cls)
        instance._term = None
        instance._oracle = oracle
        instance._session = session
        instance._handlers = list(handlers) if handlers else []
        instance._api = api
        instance._profile = profile
        return instance

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
        # R8 contract: when explain=True with inputs= supplied, .explain
        # may use those inputs to populate plans. For now, simple shape-only.
        explain_out = self.explain(inputs=env) if explain else None
        return Result(
            value=value,
            conversation_id=None,
            plan=None,
            leaf_calls=leaf_calls,
            oracle_calls=oracle_calls,
            explain=explain_out,
        )

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


def _resolve_profile(
    profile: HarnessProfile | str | None,
) -> HarnessProfile | None:
    """Coerce ``profile`` (HarnessProfile | str | None) to HarnessProfile or None.

    String specs are resolved via :attr:`fsm_llm.profile_registry`.
    A string spec that resolves to ``None`` (no registered profile)
    raises :class:`KeyError` — silently ignoring would surprise the
    caller who passed a name expecting a registered profile.
    """
    if profile is None:
        return None
    from .profiles import HarnessProfile, profile_registry

    if isinstance(profile, HarnessProfile):
        return profile
    if isinstance(profile, str):
        resolved = profile_registry.get(profile, kind="harness")
        if resolved is None:
            raise KeyError(
                f"No HarnessProfile registered under {profile!r}. "
                "Use profile_registry.register(name, profile) before "
                "constructing the Program, or pass a HarnessProfile "
                "instance directly."
            )
        return resolved  # type: ignore[return-value]
    raise TypeError(
        f"profile must be a HarnessProfile, str, or None; got {type(profile).__name__}"
    )


def _default_oracle() -> LiteLLMOracle:
    """Lazy default oracle.

    Mirrors the defaults `API` uses (DEFAULT_LLM_MODEL, temperature=0.5,
    max_tokens=1000) per Q4 in findings/program-facade-r1.md.

    Constructed on first use, not at import time, so importing
    `fsm_llm.program` is side-effect free even when no LLM credentials
    are present.
    """
    from .constants import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE
    from .runtime._litellm import LiteLLMInterface

    iface = LiteLLMInterface(
        model=DEFAULT_LLM_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=1000,
    )
    return LiteLLMOracle(iface)
