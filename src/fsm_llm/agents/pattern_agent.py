"""``PatternAgent``: an agent that runs a pattern file (plan a555dcfe step 9, D-015).

A pattern is an executable document of kind ``pattern`` (see ``pattern_files``): a static FSM
template, named variants, typed holes, handler bindings and a run block. ``PatternAgent`` is the
generic run body on ``BaseAgent._standard_run``: per run it instantiates the template
(``pattern_inputs.pattern_fsm``), builds a fresh ``HandlerContext`` and registers what the file
binds from the handler library (``fsm_llm.agents.handlers.library``, resolved by name through the
``handlers`` namespace). ``ReactAgent`` is the first class made thin on it; a user file that
binds only library handlers runs through it as it stands.

Interface contract:

- ``PatternAgent(envelope, tools=None, config=None, hitl=None, options=None, callables=None,
  **api_kwargs)``: ``envelope`` is a LOADED ``PatternEnvelope`` (a template or a generator body, D-019).
  ``tools`` is required and non-empty when the file lists ``tools`` in
  ``requires.callables``. ``options`` are the caller's options: a boolean option of the same name as
  a variant selects it; an option named ``approval`` or like an input the agent derives is refused
  (``AgentError``). ``callables`` are named host objects a factory may read (``HandlerContext``).
  The run block honours ``agent_type``, ``answer_keys``, ``success_keys``, ``evidence_keys``,
  ``judged``, ``context_seed`` and the hooks ``at: after_run`` (D-022: library entries called, in
  declared order, when the loop ends, also when it raised; an error of a hook propagates; a hook may replace
  the run's result in ``handlers.scope[RUN_RESULT]``, which ``run`` then returns);
  ``step_budget``, ``run_output_keys``, ``seconds_exempt`` and the hooks ``before_run`` and
  ``on_iteration`` have no seam yet and are REFUSED at construction (``AgentError``), never ignored.
- HITL, fail closed (census D1): a file that binds ``agents.approval_gate`` runs
  ``_refuse_unapprovable_flagged_tools`` at construction and at the start of every run and selects the
  ``approval`` variant exactly under ``_hitl_active``; a file whose ``requires.tools`` is ``some`` (or that binds ``agents.tool_executor``) and
  that binds no gate runs ``_refuse_flagged_tools`` at both places (D-021); a ``HumanInTheLoop`` given to a file with no
  gate is refused at construction (a file whose ``requires.tools`` is ``none`` refuses ``tools=`` and ``hitl=``
  with the base ``TypeError`` of ``reject_misplaced_kwargs``). The grant check stays in ``AgentHandlers.approval_refusal``.
- ``run(task, initial_context=None) -> AgentResult`` and ``run_stream(...)``: one call-local
  ``HandlerContext`` per call (never stored on ``self``, D-014 of plan 089d0ec7). A subclass that
  needs another way to build the FSM overrides ``_fsm_for_run(task, initial_context)``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import Defaults, LogMessages
from fsm_llm.agents.handlers.context import RUN_RESULT, HandlerContext
from fsm_llm.agents.pattern_inputs import pattern_fsm, refuse_reserved_options
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.core.definitions.executable_pattern import APPROVAL_GATE_HANDLER
from fsm_llm.engine.executables.namespaces import HANDLERS
from fsm_llm.logging import logger
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolRegistry

from .base import (
    BaseAgent,
    caller_prompt_keys,
    pattern_run_output_keys,
    reject_misplaced_kwargs,
)
from .definitions import AgentConfig, AgentResult

__all__ = ["PatternAgent"]

_TOOL_EXECUTOR = "agents.tool_executor"
_THINK_LIMITER = "agents.think_turn_limiter"
_NO_SEAM = ("step_budget", "run_output_keys", "seconds_exempt")
_AFTER_RUN = "after_run"


class PatternAgent(BaseAgent):
    """Run a pattern file on ``_standard_run``; see the module docstring."""

    def __init__(
        self,
        envelope: PatternEnvelope,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        hitl: HumanInTheLoop | None = None,
        options: Mapping[str, Any] | None = None,
        callables: Mapping[str, Any] | None = None,
        **api_kwargs: Any,
    ) -> None:
        body = envelope.body
        if unsupported := [f for f in _NO_SEAM if getattr(body.run, f)]:
            raise AgentError(
                f"pattern {envelope.id!r} sets run fields {unsupported} that have no "
                "seam in this version; they are refused, not ignored"
            )
        if early := [hook.at for hook in body.run.hooks if hook.at != _AFTER_RUN]:
            raise AgentError(
                f"pattern {envelope.id!r} binds hooks at {early} that have no seam in "
                f"this version (only {_AFTER_RUN!r}); they are refused, not ignored"
            )
        bound = {binding.handler for binding in body.handlers}
        self._gate = APPROVAL_GATE_HANDLER in bound
        # a file that says it needs tools, or binds the executor, runs them (D-021); only the ReAct
        # loop (the executor with the think-turn limiter) takes the react context seed
        self._executes_tools = (
            envelope.requires.tools == "some" or _TOOL_EXECUTOR in bound
        )
        self._react_loop = _TOOL_EXECUTOR in bound and _THINK_LIMITER in bound
        if "tools" in envelope.requires.callables:
            self._require_tools(tools)
        refuse_reserved_options(envelope, options or {})
        if envelope.requires.tools == "none":
            # a pattern with no tools: `tools=` and `hitl=` are named here, so the base denylist
            # (which looks at **api_kwargs) would never see them (D-003 of plan 06a5ec0a)
            reject_misplaced_kwargs(
                type(self).__name__,
                {k: v for k, v in (("tools", tools), ("hitl", hitl)) if v is not None},
            )
        if hitl is not None and not self._gate:
            raise AgentError(
                f"pattern {envelope.id!r} binds no {APPROVAL_GATE_HANDLER}: a HumanInTheLoop "
                "given to it could never be consulted"
            )
        super().__init__(config, **api_kwargs)
        self.envelope = envelope
        self.tools = tools
        self.hitl = hitl
        self._options = dict(options or {})
        self._callables = dict(callables or {})
        self._refuse_run_start()
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014
        # No `self._handlers` here (and none is ever assigned anywhere in this class): a
        # per-instance handler object shared across concurrent run()/run_stream() calls is the
        # D-004/D-014 race. Each call builds its own call-LOCAL HandlerContext (see `_run_inputs`).
        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=len(tools) if tools is not None else 0,
                model=self.config.model,
            )
        )

    @property
    def _agent_type(self) -> str:
        return self.envelope.body.run.agent_type or self.envelope.id

    def _refuse_run_start(self) -> None:
        """The flagged-tool refusal of the file's HITL shape (construction and every run start)."""
        if self._gate:
            self._refuse_unapprovable_flagged_tools()
        elif self._executes_tools:
            self._refuse_flagged_tools()

    def _fsm_for_run(
        self, task: str, initial_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The FSM of one run: the file instantiated under the agent's state."""
        return pattern_fsm(
            self.envelope,
            tools=self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            approval=self._gate and self._hitl_active,
            options=self._options,
            output_schema=self.config.output_schema,
            # the ReAct loop lists every caller key; the other patterns leave out their own outputs
            context_keys=caller_prompt_keys(
                initial_context,
                () if self._react_loop else pattern_run_output_keys(self),
            ),
            variables=self._text_variables(),
        )

    def _text_variables(self) -> dict[str, str] | None:
        """The agent's own text variables of the file (the maker and checker prompts of
        ``MakerCheckerAgent``, D-027); none by default."""
        return None

    def _handler_context(self) -> HandlerContext:
        """A new per-run context: the agent's tools, its own ``hitl`` and its one predicate."""
        return HandlerContext(
            tools=self.tools,
            config=self.config,
            hitl=self.hitl,
            hitl_active=self._hitl_active,
            approval_predicate=self._approval_predicate,
            options=self._options,
            callables=self._callables,
        )

    def _run_inputs(
        self, task: str, initial_context: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any], HandlerContext]:
        """The FSM, initial context and call-local handler context of one run, shared by ``run``
        and ``run_stream`` so the two cannot drift.

        The refusal comes first (a registry can gain a flagged tool after construction, D-052),
        and the ``await_approval`` variant is chosen under the SAME predicate that registers the
        gate (``BaseAgent._hitl_active``, D-001 of plan c1d5bfbc).
        """
        self._refuse_run_start()
        handlers = self._handler_context()
        fsm_def = self._fsm_for_run(task, initial_context)
        seed = self._context_seed() or None
        if self._react_loop:
            context = self._react_context(task, initial_context, extra=seed)
        else:
            context = self._init_context(task, initial_context, extra=seed)
        return fsm_def, context, handlers

    def _context_seed(self) -> dict[str, Any]:
        """The seed of one run: the file's, a deep copy (a pattern with a config-bound key adds it)."""
        return copy.deepcopy(dict(self.envelope.body.run.context_seed))

    def _budget_kwargs(self) -> dict[str, Any]:
        """The ``max_iterations`` of a run when the pattern has its own (default: none, the
        config's); a pattern whose budget depends on its input overrides it (``PromptChainAgent``:
        three per step)."""
        return {}

    def _run_kwargs(self) -> dict[str, Any]:
        run = self.envelope.body.run
        return {
            "extra_answer_keys": list(run.answer_keys) or None,
            "execution_evidence_keys": list(run.evidence_keys) or None,
            "judged": run.judged,
            "success_keys": list(run.success_keys) or None,
        }

    def run(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        """Run the pattern on *task* (one call-local ``HandlerContext``; see the module docstring)."""
        fsm_def, context, handlers = self._run_inputs(task, initial_context)
        try:
            handlers.scope[RUN_RESULT] = self._standard_run(
                task,
                fsm_def,
                context,
                self._agent_type,
                handlers=handlers,
                **self._run_kwargs(),
                **self._budget_kwargs(),
            )
        finally:
            self._after_run(handlers)
        result: AgentResult = handlers.scope[RUN_RESULT]  # a hook may have replaced it
        return result

    def run_stream(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> Iterator[str]:
        """Run the pattern, streaming the reply of each speaking state (lazy, like ``run``)."""
        fsm_def, context, handlers = self._run_inputs(task, initial_context)
        try:
            yield from self._standard_run_stream(
                task,
                fsm_def,
                context,
                self._agent_type,
                handlers=handlers,
                **self._budget_kwargs(),
            )
        finally:
            self._after_run(handlers)

    # DECISION plan-2026-11-24T043150-a555dcfe/D-022
    # The `after_run` hooks run in a `finally`, in declared order: a worker's budget or timeout error
    # left in the run's holder must re-raise even when the loop itself raised (the `finally` the
    # orchestrator had). Do NOT run them only on success, do NOT swallow a hook error, do NOT accept a
    # handler factory as a hook (or the reverse), and do NOT honour `before_run` or `on_iteration`
    # until a seam and a client exist (refused at construction). A model-origin document never binds
    # a hook (V7). A hook may replace the result of a finished run through `handlers.scope[RUN_RESULT]`
    # (D-025 of this plan: REWOO's cut-off answer); on a raising loop or a stream the slot is empty.
    def _after_run(self, handlers: HandlerContext) -> None:
        """Call the file's ``after_run`` hooks, in declared order, each built from this run's context.

        A hook is a library entry (``HOOKS``) whose factory returns a zero-argument callable; an
        error of one propagates (the hooks after it do not run), as the ``finally`` it replaced did.
        """
        for hook in self.envelope.body.run.hooks:
            call = HANDLERS.resolve(hook.hook)(handlers, hook)
            if not callable(call):
                raise AgentError(f"hook {hook.hook!r} is a handler, not a hook")
            call()

    # DECISION plan-2026-09-12T065608-089d0ec7/D-014
    # `handlers` is REQUIRED in practice: `run()`/`run_stream()` always pass their own call-local
    # `HandlerContext` through `_standard_run`'s `handlers=` parameter. The `| None = None` default
    # exists only so this override's signature stays compatible with `BaseAgent._register_handlers`'s
    # narrower abstract signature. Do NOT fall back to a context read from `self` when `handlers`
    # is None: that attribute does not exist so this method cannot regress to the D-004 shared
    # mutable slot. The factories are called here, once per run, in the order the file binds them
    # (the handler order is part of the run: `test_pattern_handler_order`).
    def _register_handlers(
        self, api: API, handlers: HandlerContext | None = None
    ) -> None:
        """Register the handlers the file binds, built from the library for this run."""
        context = self._require_handlers(handlers)
        for binding in self.envelope.body.handlers:
            if binding.handler not in HANDLERS:
                raise AgentError(f"handler {binding.handler!r} is not in the library")
            built = HANDLERS.resolve(binding.handler)(context, binding)
            if not isinstance(built, list):
                raise AgentError(
                    f"handler {binding.handler!r} is a hook, not a handler"
                )
            for handler in built:
                api.register_handler(handler)
