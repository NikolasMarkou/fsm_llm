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
  ``judged``, ``context_seed`` and the hooks (D-022, D-032): ``before_run`` rewrites the task once,
  ``after_run`` hooks run in declared order when each attempt's loop ends, also when it raised, and may
  replace the result in ``handlers.scope[RUN_RESULT]`` or return a ``Retry`` (the loop runs again on the
  task plus its note), ``on_iteration`` hooks run at every step AFTER the HITL driver and write only
  through a writer that refuses an internal-prefixed key; ``step_budget``, ``run_output_keys`` and
  ``seconds_exempt`` have no seam yet and are REFUSED at construction (``AgentError``), never ignored.
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
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import Defaults, LogMessages
from fsm_llm.agents.handlers.context import (
    RUN_ATTEMPT,
    RUN_RESULT,
    RUN_TASK,
    HandlerContext,
    Retry,
)
from fsm_llm.agents.pattern_inputs import pattern_fsm, refuse_reserved_options
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.core.definitions.executable_pattern import (
    APPROVAL_GATE_HANDLER,
    PatternBody,
)
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
from .definitions import AgentConfig, AgentResult, config_with, default_agent_config

__all__ = ["PatternAgent"]

_TOOL_EXECUTOR = "agents.tool_executor"
_THINK_LIMITER = "agents.think_turn_limiter"
_NO_SEAM = ("step_budget", "run_output_keys", "seconds_exempt")
_BEFORE_RUN, _AFTER_RUN, _ON_ITERATION = "before_run", "after_run", "on_iteration"


def _config_from_options(
    config: AgentConfig | None,
    body: PatternBody,
    given: Mapping[str, Any],
    llm_interface: Any,
) -> AgentConfig | None:
    """*config* with the options of the file that are ``AgentConfig`` field names applied (D-033).

    Only a ``PatternAgent`` itself applies them (a preset or a user file): a thin class such as
    ``DebateAgent`` owns its options (its ``max_iterations`` is a step budget, not the config's).
    Precedence: an option the caller gave, then a field the host's *config* set itself, then the
    file's default. A preset never names a model (``model`` is no option), so the interface or
    ``LLM_MODEL`` still decides it. No such option: *config* as it is (possibly ``None``).
    """
    declared = body.options_schema.properties
    fields = [name for name in declared if name in AgentConfig.model_fields]
    if not fields:
        return config
    base = config or default_agent_config(llm_interface)
    own = base.model_fields_set if config is not None else set()
    updates = {
        n: given[n] if n in given else body.options[n]
        for n in fields
        if n in given or (n in body.options and n not in own)
    }
    return config_with(base, **updates) if updates else config


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
        wanted = envelope.requires.tools
        if isinstance(wanted, list) and (
            absent := [n for n in wanted if tools is None or n not in tools]
        ):
            raise AgentError(
                f"pattern {envelope.id!r} requires the tools {absent}, which the registry lacks"
            )
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
        # a class that stands behind a file owns its options (D-033)
        if type(self) is PatternAgent:
            config = _config_from_options(
                config, body, options or {}, api_kwargs.get("llm_interface")
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
        self,
        task: str,
        initial_context: dict[str, Any] | None,
        handlers: HandlerContext | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], HandlerContext]:
        """The FSM, initial context and call-local handler context of one run, shared by ``run``
        and ``run_stream`` so the two cannot drift.

        The refusal comes first (a registry can gain a flagged tool after construction, D-052),
        and the ``await_approval`` variant is chosen under the SAME predicate that registers the
        gate (``BaseAgent._hitl_active``, D-001 of plan c1d5bfbc).
        """
        self._refuse_run_start()
        handlers = handlers or self._handler_context()
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

    def _run_loop(
        self,
        task: str,
        initial_context: dict[str, Any] | None,
        handlers: HandlerContext,
    ) -> AgentResult:
        """One attempt: the FSM of the file run on *task* with this attempt's call-local context
        (the seam a ``before_run`` or retry chain wraps, D-032)."""
        fsm_def, context, _ = self._run_inputs(task, initial_context, handlers)
        return self._standard_run(
            task,
            fsm_def,
            context,
            self._agent_type,
            handlers=handlers,
            **self._run_kwargs(),
            **self._budget_kwargs(),
        )

    # DECISION plan-2026-11-24T043150-a555dcfe/D-032
    # `run` is the ONE seam of the run hooks: `before_run` once, then attempts, each with its own
    # context (`RUN_TASK`, `RUN_ATTEMPT`, `RUN_RESULT` in its scope) and its `after_run` hooks in a
    # `finally`; a `Retry` runs another attempt on the base task plus its note. Do NOT add a second
    # wrapper around `run` (a class override that loops or persists is the pattern the file
    # replaced), do NOT cap the attempts here (the retry hook owns its bound), do NOT hand an
    # `on_iteration` hook the `api` (its writer refuses internal-prefixed keys: no grant path) and do
    # NOT run an `on_iteration` hook before `super()._on_loop_iteration`, the HITL driver.
    def run(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        """Run the pattern on *task* (one call-local ``HandlerContext`` per attempt; see the module docstring)."""
        base = self._before_run(task)
        note: str | None = None
        attempt = 0
        while True:
            handlers = self._handler_context()
            handlers.scope.update({RUN_TASK: task, RUN_ATTEMPT: attempt})
            try:
                handlers.scope[RUN_RESULT] = self._run_loop(
                    base if note is None else f"{base}\n\n{note}",
                    initial_context,
                    handlers,
                )
            finally:
                retry = self._after_run(handlers)
            if retry is None:
                result: AgentResult = handlers.scope[
                    RUN_RESULT
                ]  # a hook may have replaced it
                return result
            note, attempt = retry.note, attempt + 1

    def run_stream(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> Iterator[str]:
        """Run the pattern, streaming the reply of each speaking state (lazy, like ``run``).

        ``before_run`` applies; the ``after_run`` hooks run with an empty result slot (D-025, D-032).
        """
        task = self._before_run(task)
        fsm_def, context, handlers = self._run_inputs(task, initial_context)
        handlers.scope[RUN_TASK] = task
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

    def _hook_calls(
        self, at: str, handlers: HandlerContext | None = None
    ) -> Iterator[Callable[..., Any]]:
        """The hooks of the file at position *at*, each built when reached from the library entry
        (a context is made only when some hook is bound there); a handler factory is no hook."""
        bound = [hook for hook in self.envelope.body.run.hooks if hook.at == at]
        context = (handlers or self._handler_context()) if bound else None
        for hook in bound:
            call = HANDLERS.resolve(hook.hook)(context, hook)
            if not callable(call):
                raise AgentError(f"hook {hook.hook!r} is a handler, not a hook")
            yield call

    def _before_run(self, task: str) -> str:
        """The task the attempts run on: the ``before_run`` hooks applied in declared order."""
        for call in self._hook_calls(_BEFORE_RUN):
            task = call(task)
        return task

    # DECISION plan-2026-11-24T043150-a555dcfe/D-022
    # The `after_run` hooks run in a `finally`, in declared order: a worker's budget or timeout error
    # left in the run's holder must re-raise even when the loop itself raised (the `finally` the
    # orchestrator had). Do NOT run them only on success, do NOT swallow a hook error, do NOT accept a
    # handler factory as a hook (or the reverse). A model-origin document never binds a hook (V7). A
    # hook may replace the result of a finished run through `handlers.scope[RUN_RESULT]` (D-025 of
    # this plan: REWOO's cut-off answer); on a raising loop or a stream the slot is empty. D-032 adds
    # the other two positions and the `Retry` verdict on top of this one seam.
    def _after_run(self, handlers: HandlerContext) -> Retry | None:
        """Call the file's ``after_run`` hooks, in declared order, each built from this attempt's
        context; the first ``Retry`` ends the chain (the attempt is discarded), an error of a hook
        propagates (the hooks after it do not run), as the ``finally`` it replaced did."""
        for call in self._hook_calls(_AFTER_RUN, handlers):
            verdict = call()
            if isinstance(verdict, Retry):
                return verdict
            if verdict is not None:
                raise AgentError(f"an after_run hook returned {verdict!r}, not a Retry")
        return None

    def _on_loop_iteration(
        self, api: API, conversation_id: str, iteration: int
    ) -> None:
        """The HITL driver first and always, then the ``on_iteration`` hooks with the step number, a
        reader of the conversation data and a writer that refuses an internal-prefixed key."""
        super()._on_loop_iteration(api, conversation_id, iteration)
        for call in self._hook_calls(_ON_ITERATION):
            call(
                iteration,
                partial(api.get_data, conversation_id),
                partial(self._hook_write, api, conversation_id),
            )

    @staticmethod
    def _hook_write(api: API, conversation_id: str, update: Mapping[str, Any]) -> None:
        """The one writer of an ``on_iteration`` hook: no internal-prefixed key (the grant is one)."""
        if internal := sorted(k for k in update if has_internal_prefix(k)):
            raise AgentError(f"a hook may not write the internal keys {internal}")
        api.update_context(conversation_id, dict(update))

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
