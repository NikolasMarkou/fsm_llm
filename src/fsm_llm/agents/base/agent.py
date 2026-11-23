"""``BaseAgent``, the shell of the abstract base of every fsm_llm agent pattern, and ``accepts_tools``.

The common conversation loop, budget enforcement, answer extraction, trace building and context
filtering shared by the FSM agent patterns. Split at plan d454fd62 step 87: the class is a shell
(construction, the context seed, ``_run_outcome``, the abstract hooks) composed of four mixins:
``HitlGateMixin`` (``hitl_gate.py``), ``OutcomeMixin`` (``outcome.py``), ``RunsMixin`` (``runs.py``)
and ``RegistrationMixin`` (``registration.py``). ``_run_outcome`` and ``accepts_tools`` stay here
because they name ``BaseAgent`` itself.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from fsm_llm import API
from fsm_llm.agents.constants import ContextKeys, ErrorMessages, StopReason
from fsm_llm.agents.definitions import (
    AgentConfig,
    AgentResult,
    AgentTrace,
    default_agent_config,
)
from fsm_llm.aio import aoffload
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT
from fsm_llm.core.definitions import AgentError
from fsm_llm.logging import logger

from .helpers import (
    _output_response_format,
    pattern_run_output_keys,
    reject_misplaced_kwargs,
    strip_caller_context,
)
from .hitl_gate import HitlGateMixin
from .outcome import OutcomeMixin
from .registration import RegistrationMixin
from .runs import RunsMixin

_H = TypeVar("_H")


class BaseAgent(HitlGateMixin, OutcomeMixin, RunsMixin, RegistrationMixin, ABC):
    """Abstract base class for FSM-LLM agents.

    Provides the common conversation loop, budget enforcement, answer
    extraction, and trace building. Subclasses implement only the
    pattern-specific logic: FSM building, handler registration, and
    context setup.

    Usage for end-users is unchanged — all existing agent constructors
    and ``run()`` signatures are preserved.  Additionally, agents now
    support ``__call__``::

        result = agent("What is 2+2?")
    """

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: each pattern lists the
    # context keys its run writes for itself (answer, draft, verdict, success
    # and progress keys). Caller context and graph/swarm hand-offs never seed
    # them: core skip-if-set never re-extracts a set key, so a forged
    # `draft_output` or `generated_output` shipped as a successful answer with
    # no work done. Do NOT fold these into the global RUN_OUTPUT_KEYS: a
    # pattern's run output can be a legitimate input of another pattern.
    # Do NOT list a key a caller legitimately supplies (task, domain hints).
    _run_output_keys: ClassVar[frozenset[str]] = frozenset()

    # States in which core's seconds budget does not stop the run (core
    # `seconds_exempt_states`, D-032 of plan 944e2692): a pattern's wind-down
    # that must finish once reached. Empty: the clock is checked every step.
    # It may name states of every shape a pattern builds; each run passes
    # only those its built FSM has (`_seconds_exempt_for`), since core
    # refuses an id that is not a state of the running definition (D-039).
    _seconds_exempt_states: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        self.config = config or default_agent_config(api_kwargs.get("llm_interface"))
        self._api_kwargs = api_kwargs
        self._refuse_prompt_cache_with_injected_interface()

    def _seconds_exempt_for(self, api: API) -> frozenset[str]:
        """The ``_seconds_exempt_states`` that the run's built FSM has.

        Contract: the intersection of the class attribute with the states of
        ``api.fsm_definition`` (the root definition the run starts in); never
        raises.
        """
        return self._seconds_exempt_states & frozenset(api.fsm_definition.states)

    @staticmethod
    def _require_tools(tools: Any) -> None:
        """Raise ``AgentError`` for an empty registry (tool patterns' ``__init__``)."""
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

    def _require_handlers(self, handlers: _H | None) -> _H:
        """Return *handlers*; raise ``AgentError`` when a ReAct-family
        ``_register_handlers`` got none.

        ``run()`` always passes its call-local handlers (D-014 of plan
        2026-09-12T065608-089d0ec7); ``None`` here is a programming error.
        """
        if handlers is None:
            raise AgentError(
                f"{type(self).__name__}._register_handlers called without a "
                "handlers instance — this is a programming error, not a "
                "runtime condition; run() must always pass one."
            )
        return handlers

    def _react_context(
        self,
        task: str,
        initial_context: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``_init_context`` for the ReAct family: no observations yet, the
        iteration budget seeded, plus the pattern's own *extra* keys."""
        return self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                ContextKeys.ITERATION_BUDGET: self.config.max_iterations,
                **(extra or {}),
            },
        )

    def _refuse_prompt_cache_with_injected_interface(self) -> None:
        """Raise ``AgentError`` when ``enable_prompt_cache`` meets ``llm_interface``.

        Contract: called at construction and again each time a run builds its
        ``API`` (``config`` is a plain attribute a caller may replace), so the
        refusal names the config flag the caller set, never core's "takes no
        LLM settings ['caching']". Returns ``None`` otherwise.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-038: refuse, never drop.
        # Do NOT add `caching` only when no interface is injected (the flag
        # would then be ignored silently), and do NOT let core's refusal name
        # a `caching` kwarg the caller never passed. See D-038.
        if (
            self.config.enable_prompt_cache
            and self._api_kwargs.get("llm_interface") is not None
        ):
            raise AgentError(ErrorMessages.PROMPT_CACHE_WITH_INTERFACE)

    @abstractmethod
    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent on a task. Implemented by each agent pattern."""
        ...

    # DECISION plan-2026-11-20T055240-ace7a8c8/D-065
    # The default ``arun`` hops to ``run`` with ``aoffload(self.run, ...)``. Do NOT write
    # ``acall(self, "run", ...)``: in async mode ``acall`` finds ``self.arun`` first and calls itself
    # forever. Do NOT make it abstract (a sync-only subclass must still instantiate), and do NOT let
    # ``AgentStep`` treat it as native (``_fsm_llm_offloads_run`` marks it: the step keeps the engine
    # ``executor=``). A subclass with its OWN ``async def arun`` is native and its ``run`` is bypassed.
    async def arun(self, task: str, *args: Any, **kwargs: Any) -> AgentResult:
        """The awaitable name of ``run``: ``run`` on a worker thread.

        Interface contract: the arguments are forwarded verbatim to ``run`` and its result or error
        comes back unchanged. Async mode runs ``run`` through ``aoffload`` (``asyncio.to_thread``); a
        cancelled await returns at once while ``run`` keeps going to its end on the worker and its late
        result is dropped (tools and handlers of that run are not stopped). Sync mode
        (``run_sync(agent.arun(...))``) runs it inline. A pattern with its own ``async def arun``
        replaces this.
        """
        return await aoffload(self.run, task, *args, **kwargs)

    arun._fsm_llm_offloads_run = True  # type: ignore[attr-defined]

    def __call__(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Callable shorthand: ``agent("task")`` → ``agent.run("task")``."""
        return self.run(task, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"

    def _init_context(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the standard initial context for an agent run.

        Sets ``TASK``, ``AGENT_TRACE``, ``ITERATION_COUNT`` and
        ``MAX_ITERATIONS_REACHED`` (False; so a run's ``final_context`` carries it).
        Seeds ``OBSERVATION_COUNT`` to 0. Caller keys in ``RUN_OUTPUT_KEYS``
        (including the driver-only approval grant) and in the pattern's own
        ``_run_output_keys`` are dropped with a WARNING by
        ``strip_caller_context``; other internal-prefix keys pass through.
        Warns if *initial_context* already contains reserved keys.
        """
        context = strip_caller_context(
            initial_context,
            source="initial_context",
            run_keys=pattern_run_output_keys(self),
        )
        reserved = {
            ContextKeys.TASK,
            ContextKeys.AGENT_TRACE,
            ContextKeys.ITERATION_COUNT,
            ContextKeys.MAX_ITERATIONS_REACHED,
        }
        conflicts = reserved & context.keys()
        if conflicts:
            logger.warning(
                f"initial_context contains reserved keys that will be "
                f"overwritten: {conflicts}"
            )
        context[ContextKeys.TASK] = task
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: seed the forced-stop
        # flag False. Bulk extraction fills only UNSET keys, so the model cannot
        # write True and pass every evidence guard (P2-W2). Only the limiter
        # and stall handlers write True. Do NOT drop the seed or set it None.
        context[ContextKeys.MAX_ITERATIONS_REACHED] = False
        # Evidence counter starts at zero for every run; the caller's value was
        # dropped above (D-002 of plan 06a5ec0a, LOOP-12).
        context[ContextKeys.OBSERVATION_COUNT] = 0

        # Schema-enforced output: when output_schema is set, store the
        # JSON schema as response_format so the pipeline can pass it to
        # the LLM for constrained decoding on the conclude state.
        response_format = _output_response_format(self.config.output_schema)
        if response_format is not None:
            context[CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT] = response_format

        if extra:
            context.update(extra)
        return context

    @staticmethod
    def _run_outcome(
        final_context: dict[str, Any],
        trace: AgentTrace,
        extra_answer_keys: list[str] | None,
        execution_evidence_keys: list[str] | None = None,
        judged: bool = False,
        answer_cut_off: bool = False,
    ) -> tuple[bool, StopReason]:
        """Compute ``(success, stop_reason)`` for a finished FSM run.

        Interface contract (callers: ``_standard_run``; ``ADaPTAgent.run``
        for an undecomposed run): the one success rule for every pattern
        whose result comes from an FSM run.

        Returns:
            ``(False, <forced reason>)`` when the run was forced to stop
            (``_forced_stop_reason``, ``judged`` passed through);
            ``(False, "truncated")`` with ``answer_cut_off`` (the caller's
            ``_answer_was_cut_off``; no forced reason outranks it, so a judged
            pattern's bare limiter flag does not hide a cut);
            ``(False, "no_result")`` when a handler recorded
            ``StopReason.NO_RESULT`` under ``forced_stop_reason`` (the run
            ended without a result its own rule accepts: native_fc); else, in
            planner mode
            (``execution_evidence_keys``), ``(True, "evidence")`` with real
            execution evidence; otherwise ``(True, "answered")`` when
            ``_completion_is_real`` (an answer key or an executed tool call);
            and ``(False, "no_result")`` in every other case. Never raises.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-011: a forced stop, forced
        # pass or stall reports success=False with its reason, and the caller
        # still ships the last answer. Do NOT raise here (c1d5bfbc D-022: a
        # forced budget stop ships its output), and do NOT let the answer key
        # or a tool call override the forced flag (a forced EvalOpt/MakerChecker
        # pass always has an answer key, which is how it read as success=True).
        # Do NOT compute success per pattern: this is the one rule.
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: `judged` patterns
        # (EvalOpt, MakerChecker) report forced only on the judge handler's
        # recorded override. Do NOT read their bare limiter flag here: a
        # genuine pass on the limiter's round read as forced_pass.
        forced = BaseAgent._forced_stop_reason(final_context, judged=judged)
        if forced is not None:
            return False, forced
        if answer_cut_off:
            return False, StopReason.TRUNCATED
        # DECISION plan-2026-10-01T093600-944e2692/D-028: a pattern whose
        # success rule is stricter than "an answer key or a tool call" (native
        # function calling: the loop reached a tool-call-free final turn with
        # a non-empty answer, bf7ffe24/D-005) states its verdict through a
        # handler-recorded `no_result`, read HERE. Do NOT compute success in
        # the pattern, add a per-pattern keyword or put NO_RESULT in
        # FORCED_STOP_REASONS (it is not a forced stop: no warning, no limiter
        # reads it). The key is handler-only and a run output, so a model or
        # caller cannot write it. See decisions.md D-028.
        if final_context.get(ContextKeys.FORCED_STOP_REASON) == StopReason.NO_RESULT:
            return False, StopReason.NO_RESULT
        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: when
        # ``execution_evidence_keys`` is supplied (planner patterns:
        # orchestrator/rewoo/plan_execute), the answer-key/tool-call test is NOT
        # sufficient. A planner can reach a synthesis state that writes an answer
        # (and record a ``delegate`` control action that _build_trace turns into
        # a fake ToolCall) while having executed ZERO real work, so success
        # requires genuine EXECUTION EVIDENCE (a successful worker / non-empty
        # tool evidence / executed steps). Do NOT run ``_completion_is_real``
        # for a planner: filler must report success=False.
        if execution_evidence_keys:
            if BaseAgent._has_execution_evidence(
                final_context, execution_evidence_keys
            ):
                return True, StopReason.EVIDENCE
            return False, StopReason.NO_RESULT
        if BaseAgent._completion_is_real(final_context, trace, extra_answer_keys):
            return True, StopReason.ANSWERED
        return False, StopReason.NO_RESULT

    @abstractmethod
    def _register_handlers(self, api: API) -> None:
        """Register pattern-specific handlers. Implemented by each agent.

        Most overrides keep exactly this ``(self, api)`` shape. A handful
        of patterns that need call-local handler state (``react.py``,
        ``parallel_react.py`` — see D-014) widen their OWN override with an
        extra optional ``handlers`` parameter; adding an optional parameter
        to an override is LSP-compatible (every caller of the narrower base
        signature still works unchanged), so this abstract signature is
        deliberately left unwidened — see ``_standard_run``'s docstring for
        the full ``handlers=`` threading contract.
        """
        ...


def accepts_tools(agent_cls: type) -> bool:
    """Tell whether *agent_cls*'s constructor takes a ``tools`` argument.

    Interface contract (callers: ``create_agent`` and its tests):

    Args:
        agent_cls: An agent class (any class; non-agents are fine).

    Returns:
        ``True`` when some ``__init__`` on the MRO names a ``tools`` parameter
        and every ``__init__`` before it forwards ``**kwargs`` (VerifiedReact and
        AutoMemory forward to ReactAgent). ``BaseAgent.__init__`` ends the walk:
        its ``**api_kwargs`` is not a tools sink (it rejects ``tools``). Never
        raises; an uninspectable signature counts as ``False``.
    """
    for klass in agent_cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if klass is BaseAgent or klass is object:
            return False
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):
            return False
        if "tools" in params and params["tools"].kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return False
    return False
