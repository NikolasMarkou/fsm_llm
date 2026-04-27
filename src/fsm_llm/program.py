"""
fsm_llm.program — `Program` facade (R1).

`Program` is the unified entry point for running λ-terms in any of the
three Category surfaces:

- **Category A (FSM dialog)** — `Program.from_fsm(fsm_def | dict | str)`
  delegates to `API`. `.converse(msg, conv_id)` runs one β-reduction step
  on the compiled term, persisting per-conversation state.
- **Category B/C (term / factory)** — `Program.from_term(term)` and
  `Program.from_factory(factory, ...)` wrap a pre-authored λ-term.
  `.run(**env)` is one stateless evaluation. `.converse` is not supported
  in R1 — see `# DECISION D-001` below.

R1 is purely additive. No edits to `api.py`, `fsm.py`, `pipeline.py`, or
`handlers.py`. R2 will route `from_fsm` directly through the kernel
compile-cache; R5 will collapse the two execution paths into one.

References:
- plans/plan_2026-04-27_a426f667/plan.md (R1 success criteria SC1-SC11)
- plans/plan_2026-04-27_a426f667/findings/program-facade-r1.md
- plans/plan_2026-04-27_a426f667/decisions.md (D-PLAN-02, D-PLAN-03)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .lam import Executor, LiteLLMOracle, Oracle, Plan, Term

if TYPE_CHECKING:
    from .api import API
    from .definitions import FSMDefinition
    from .handlers import FSMHandler
    from .session import SessionStore


__all__ = ["ExplainOutput", "Program"]


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
# Program — the facade
# ---------------------------------------------------------------------------


class Program:
    """Unified facade over (term, oracle, optional session, optional handlers).

    Three constructors:

    - :meth:`from_fsm` — build from an FSM JSON definition. Internally
      constructs an :class:`fsm_llm.api.API` and delegates ``.converse`` /
      ``.register_handler`` to it. See ``# DECISION D-001`` below for
      why API-delegation is the right shape in R1.
    - :meth:`from_term` — wrap a pre-authored λ-term directly.
      ``.run(**env)`` is supported; ``.converse`` raises
      :class:`NotImplementedError` (R5 territory).
    - :meth:`from_factory` — invoke a stdlib factory and wrap its
      returned term. ``factory_args`` and ``factory_kwargs`` are
      explicit (per Q1 in `findings/program-facade-r1.md`); facade
      kwargs are kw-only.

    The bare ``Program(term=…, oracle=…)`` constructor is the term-mode
    shape and is the simplest path for users who already hold a
    ``Term`` and an ``Oracle``.
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

    # ------------------------------------------------------------------
    # Constructors — to be filled in by subsequent steps (2-5)
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
        """Build a Program from a pre-authored λ-term.

        Implementation arrives in step 2 (`run` path) and step 5
        (`register_handler` path).
        """
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
        """Build a Program by invoking a stdlib factory.

        Implementation arrives in step 2.
        """
        kwargs = factory_kwargs or {}
        term = factory(*factory_args, **kwargs)
        return cls(
            term=term,
            oracle=oracle,
            session=session,
            handlers=handlers,
        )

    # ------------------------------------------------------------------
    # Runtime surface — stubs filled in by steps 2-5
    # ------------------------------------------------------------------

    def run(self, **env: Any) -> Any:
        """One-shot stateless evaluation of the wrapped term.

        Term/factory mode only. FSM-mode programs (built via
        :meth:`from_fsm`) raise :class:`NotImplementedError` because the
        FSM compile pipeline binds env to per-conversation context, not
        to user-supplied kwargs — use :meth:`converse` instead.

        ``env`` is passed through as the env dict to
        :meth:`fsm_llm.lam.Executor.run` (free Vars are resolved
        against this dict). Returns whatever the term reduces to —
        typically a string for unstructured Leaves, a Pydantic model
        for structured Leaves, or whatever a Combinator chain produces.

        The Executor is constructed fresh on each call (oracle-call
        counter resets per run, per
        :class:`fsm_llm.lam.executor.Executor` semantics). When
        ``self._oracle is None``, a lazy default
        :class:`fsm_llm.lam.LiteLLMOracle` is built — see
        :func:`_default_oracle` for the defaults.
        """
        if self._term is None:
            # FSM-mode (constructed via from_fsm)
            raise NotImplementedError(
                "Program.run is not supported for FSM-backed Programs. "
                "Use .converse(message, conversation_id) instead, or "
                "build the Program with .from_term / .from_factory for "
                "stateless one-shot evaluation."
            )
        return self._executor().run(self._term, env)

    def converse(self, message: str, conversation_id: str | None = None) -> str:
        """Stateful conversational entry. FSM-mode only.

        # DECISION D-001 — In R1, only Programs constructed via
        # ``from_fsm`` support ``.converse``. ``from_term`` /
        # ``from_factory`` programs raise NotImplementedError because
        # term-mode lacks the per-conversation session protocol that
        # the FSM compile pipeline supplies. Generic env-from-session
        # round-tripping is R5 territory (handlers as AST transformers).
        # See plans/plan_2026-04-27_a426f667/decisions.md D-PLAN-02.

        When ``conversation_id`` is None, a new conversation is started
        automatically (the initial greeting is discarded — only the
        response to ``message`` is returned). The auto-started
        conversation_id is then stored on the Program so subsequent
        calls without an explicit id continue the same conversation.
        Pass an explicit id to multiplex multiple conversations on the
        same Program.
        """
        if self._api is None:
            raise NotImplementedError(
                "Program.converse is supported only for Programs built "
                "via Program.from_fsm. Term-mode programs (.from_term, "
                ".from_factory) should call .run(**env) instead — they "
                "are stateless one-shot evaluations. See D-PLAN-02 in "
                "plans/plan_2026-04-27_a426f667/decisions.md."
            )

        if conversation_id is None:
            # Lazily start (or reuse) a conversation. We stash the id on
            # the Program so subsequent calls without an explicit id
            # continue the same conversation rather than spinning up a
            # fresh one each time (which would be surprising).
            cached = getattr(self, "_default_conv_id", None)
            if cached is None:
                conversation_id, _greeting = self._api.start_conversation()
                self._default_conv_id = conversation_id
            else:
                conversation_id = cached

        return self._api.converse(message, conversation_id)

    def explain(self) -> ExplainOutput:
        """Static description of the wrapped term.

        Walks the AST and returns:
        - ``ast_shape``: a multi-line indented rendering of the term's
          node-kind skeleton (no template strings, no env values).
        - ``plans``: list of :class:`fsm_llm.lam.Plan` instances. R1
          returns an empty list — :func:`fsm_llm.lam.plan` requires
          runtime quantities (n, K) that aren't available at static-
          inspection time. A future R5/R6 step may add a
          ``Program.explain(n=…, K=…)`` overload that runs ``plan()``
          on each ``Fix`` subtree.
        - ``leaf_schemas``: maps a synthesised leaf-id (template prefix
          + position index) to the leaf's ``schema_ref`` (or ``None``
          for unstructured leaves). The id is stable as long as the
          term is unchanged.

        For FSM-mode programs, walks the term cached on the underlying
        ``API``'s ``FSMManager`` (compiled at API construction time).
        """
        # Resolve which term to walk.
        term: Term | None = self._term
        if term is None and self._api is not None:
            # Reach into the FSMManager's compiled-term cache.
            try:
                term = self._api.fsm_manager.get_compiled_term(self._api.fsm_id)
            except Exception:  # pragma: no cover — defensive
                term = None
        if term is None:
            # Should be unreachable under the __init__ XOR invariant.
            return ExplainOutput()

        shape_lines: list[str] = []
        leaf_schemas: dict[str, type | None] = {}
        leaf_counter = [0]  # Mutable cell for nested closures.

        def _walk(node: Any, indent: int) -> None:
            pad = "  " * indent
            kind = type(node).__name__
            # Per-kind summary.
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
                _walk(node.body, indent + 1)
            elif kind == "Leaf":
                idx = leaf_counter[0]
                leaf_counter[0] += 1
                # Synthesise a stable id: position index + template prefix.
                tpl_preview = node.template[:30].replace("\n", " ")
                leaf_id = f"leaf_{idx:03d}_{tpl_preview!r}"
                leaf_schemas[leaf_id] = node.schema_ref
                shape_lines.append(
                    f"{pad}Leaf(template={tpl_preview!r}..., "
                    f"input_vars={list(node.input_vars)!r}, "
                    f"schema_ref={node.schema_ref!r})"
                )
            else:
                # Defensive: unknown node kind (would mean an out-of-band
                # AST extension). Render kind + repr-prefix.
                shape_lines.append(f"{pad}{kind}(?)")

        _walk(term, 0)

        return ExplainOutput(
            plans=[],
            leaf_schemas=leaf_schemas,
            ast_shape="\n".join(shape_lines),
        )

    def register_handler(self, handler: FSMHandler) -> None:
        """Register a handler. FSM-mode only in R1.

        Implementation arrives in step 5.
        """
        raise NotImplementedError(
            "Program.register_handler is not yet implemented (R1 step 5).",
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


def _default_oracle() -> LiteLLMOracle:
    """Lazy default oracle.

    Mirrors the defaults `API` uses (DEFAULT_LLM_MODEL, temperature=0.5,
    max_tokens=1000) per Q4 in findings/program-facade-r1.md.

    Constructed on first use, not at import time, so importing
    `fsm_llm.program` is side-effect free even when no LLM credentials
    are present.
    """
    # Lazy local imports — avoid pulling LiteLLMInterface (and litellm)
    # at module import time.
    from .constants import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE
    from .llm import LiteLLMInterface

    iface = LiteLLMInterface(
        model=DEFAULT_LLM_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=1000,
    )
    return LiteLLMOracle(iface)
