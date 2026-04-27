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

        Implementation arrives in step 3 (plan.md R1 step 3).
        """
        raise NotImplementedError(
            "Program.from_fsm is not yet implemented (R1 step 3)."
        )

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

        Implementation arrives in step 2.
        """
        raise NotImplementedError("Program.run is not yet implemented (R1 step 2).")

    def converse(self, message: str, conversation_id: str | None = None) -> str:
        """Stateful conversational entry. FSM-mode only.

        # DECISION D-001 — In R1, only Programs constructed via
        # ``from_fsm`` support ``.converse``. ``from_term`` /
        # ``from_factory`` programs raise NotImplementedError because
        # term-mode lacks the per-conversation session protocol that
        # the FSM compile pipeline supplies. Generic env-from-session
        # round-tripping is R5 territory (handlers as AST transformers).
        # See plans/plan_2026-04-27_a426f667/decisions.md D-PLAN-02.

        Implementation arrives in step 3.
        """
        raise NotImplementedError(
            "Program.converse is not yet implemented (R1 step 3)."
        )

    def explain(self) -> ExplainOutput:
        """Static description of the wrapped term.

        Implementation arrives in step 4.
        """
        raise NotImplementedError(
            "Program.explain is not yet implemented (R1 step 4).",
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
