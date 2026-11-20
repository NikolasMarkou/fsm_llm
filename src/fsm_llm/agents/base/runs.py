"""The run drivers of ``BaseAgent`` (``RunsMixin``).

The conversation loop on core's bounded run, its budgets, the API factory and the standard run with its
stream form. Every read of the clock for a run budget is in this module (``_check_budgets``,
``_run_budgets``, ``_standard_run_stream``, ``_standard_run``), so a test binds one ``time``. The method
bodies are the old ones (plan d454fd62 step 87).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import partial
from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import (
    FORCED_STOP_REASONS,
    RESULT_DROPPED_CONTEXT_KEYS,
    ContextKeys,
    Defaults,
    LogMessages,
    StopReason,
)
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.core.definitions import (
    CONTEXT_KEY_CUT_OFF,
    AgentError,
    FSMError,
    RunBudgetExceededError,
)
from fsm_llm.engine.api.helpers import llm_settings_for
from fsm_llm.logging import logger

from .helpers import _prompt_overflow_error, _with_instructions
from .surface import _AgentSurface


class RunsMixin(_AgentSurface):
    """The run drivers of ``BaseAgent``: loop, budgets, API factory, standard run and stream."""

    def _run_conversation_loop(
        self,
        api: API,
        context: dict[str, Any],
        start_time: float,
        agent_type: str,
        max_iterations: int | None = None,
    ) -> tuple[list[str], dict[str, Any], int]:
        """Run the agent's FSM to its terminal state on core's bounded run.

        Returns:
            Tuple of (responses, final_context, step_count). ``responses``
            holds the reply of every state that spoke, in order (a silent
            state contributes nothing).

        Raises:
            AgentTimeoutError, BudgetExhaustedError: a run budget was spent
                (``_run_budgets``, ``_budget_error``).
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-030
        # The run loop, the step cap and the wall-clock budget are core's
        # (`API.run_until_terminal`). Do NOT bring back a `while` loop, a step
        # counter or a synthetic `converse` turn here or in
        # `_standard_run_stream`, and do NOT filter `[state]` markers in this
        # package (supersedes 06a5ec0a/D-030): a step has no user message and
        # a silent state returns no text. If a pattern needs a budget or loop
        # behaviour core cannot express, extend core's loop with a core test.
        # See decisions.md D-030.
        conv_id, greeting = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id,
            package="fsm_llm.agents",
            agent_type=agent_type,
        )

        try:
            max_steps, max_seconds = self._run_budgets(start_time, max_iterations)
            try:
                steps = api.run_until_terminal(
                    conv_id,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    before_step=partial(self._on_loop_iteration, api, conv_id),
                    seconds_exempt_states=self._seconds_exempt_for(api),
                )
            except RunBudgetExceededError as exc:
                raise self._budget_error(exc, max_iterations) from exc

            replies = [greeting or None, *(step.response for step in steps)]
            final_context = api.get_data(conv_id)
            # D-050: core ends the run normally when the conversation is
            # closed from outside (a hook, another thread, a monitor). That
            # run reached no final state, so it is a forced stop, not an
            # answer: the conversation is no longer active and its last step
            # did not end it.
            if conv_id not in api.list_active_conversations() and not (
                steps and steps[-1].ended
            ):
                final_context[ContextKeys.FORCED_STOP_REASON] = StopReason.ENDED.value
            else:
                # DECISION plan-2026-11-06T093029-0068b055/D-025 (answer source:
                # D-027): the LAST turn's cut-off record rides out in
                # `final_context` and is judged ONCE, by `_answer_was_cut_off`
                # inside `_run_outcome`, where the answer source is known. Do
                # NOT turn it into a stop reason here (an answer read from a
                # context key is whole when only an unused reply was cut), do
                # NOT repeat this per pattern, and do NOT read an earlier turn's
                # cut (the record is cleared every turn). See decisions.md D-027.
                cut = self._last_turn_cut_off(api, conv_id)
                if cut:
                    final_context[CONTEXT_KEY_CUT_OFF] = cut
            log.info(LogMessages.AGENT_COMPLETE.format(iterations=len(steps)))
            return [r for r in replies if r is not None], final_context, len(steps)

        finally:
            self._end_run_conversation(api, conv_id)

    @staticmethod
    def _last_turn_cut_off(api: API, conversation_id: str) -> dict[str, Any]:
        """Core's cut-off record of the run's last turn, or ``{}``.

        Interface contract (caller: ``_run_conversation_loop``): reads
        ``CONTEXT_KEY_CUT_OFF`` (``{"reply": True}`` and/or
        ``{"fields": [names]}``) from the top frame's UNFILTERED context
        (``fsm_manager.copy_raw_context_data``: ``api.get_data`` drops internal
        keys) and returns a copy. ``{}`` when the conversation is no longer
        active or the key is absent, empty or not a mapping.
        """
        if conversation_id not in api.list_active_conversations():
            return {}
        raw = api.fsm_manager.copy_raw_context_data(
            api.get_sub_conversation_id(conversation_id)
        )
        record = raw.get(CONTEXT_KEY_CUT_OFF) if raw else None
        return dict(record) if isinstance(record, dict) else {}

    @staticmethod
    def _end_run_conversation(api: API, conversation_id: str) -> None:
        """End the run's conversation unless something else already ended it.

        Interface contract (callers: the ``finally`` of
        ``_run_conversation_loop`` and ``_standard_run_stream``): calls
        ``api.end_conversation(conversation_id)``. A failure is re-raised unless
        ``conversation_id`` is no longer an active conversation (a hook, another
        thread or a monitor ended it during the run). A refused end
        (``ConversationBusyError``) leaves the conversation active, so it is
        re-raised.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-050: a conversation that
        # was ended from outside the run (the run itself stops cleanly, D-044)
        # must not fail the run in its own cleanup with "Conversation not
        # found". Do NOT catch every error here: only an end that failed
        # because the conversation is already gone is tolerated. See D-050.
        try:
            api.end_conversation(conversation_id)
        except FSMError:
            if conversation_id in api.list_active_conversations():
                raise
            logger.debug(f"Conversation {conversation_id} was already ended")

    def _on_loop_iteration(
        self, api: API, conversation_id: str, iteration: int
    ) -> None:
        """Hook called before each step of the run (core's ``before_step``).

        ``iteration`` is the step number, from 1. Runs the HITL approval driver
        (a no-op for a pattern with no ``hitl``); override to add other
        between-step logic and call ``super()``.
        """
        self._handle_hitl_approval(api, conversation_id)

    def _check_budgets(self, start_time: float) -> None:
        """Raise ``AgentTimeoutError`` when the wall clock of a run is spent.

        For callers that are not inside a core bounded run (SelfConsistency
        samples). FSM runs pass their budgets to core instead
        (``_run_budgets``).
        """
        if time.monotonic() - start_time > self.config.timeout_seconds:
            raise AgentTimeoutError(self.config.timeout_seconds)

    def _run_budgets(
        self, start_time: float, max_iterations: int | None = None
    ) -> tuple[int, float]:
        """The ``(max_steps, max_seconds)`` of a core bounded run.

        Interface contract (callers: ``_run_conversation_loop``,
        ``_standard_run_stream``): ``max_steps`` is the pattern's
        ``_step_ceiling`` for ``max_iterations`` (default
        ``config.max_iterations``); ``max_seconds`` is what is left of
        ``config.timeout_seconds`` since ``start_time`` (always above 0).
        Raises ``AgentTimeoutError`` when nothing is left: core refuses a
        non-positive time budget.
        """
        remaining = self.config.timeout_seconds - (time.monotonic() - start_time)
        if remaining <= 0:
            raise AgentTimeoutError(self.config.timeout_seconds)
        ceiling, _ = self._step_ceiling(max_iterations or self.config.max_iterations)
        return ceiling, remaining

    def _step_ceiling(self, max_iterations: int) -> tuple[int, str]:
        """The core step ceiling of a run with *max_iterations*, and its formula.

        Interface contract (callers: ``_run_budgets``, ``_budget_error``; a
        pattern whose loop turn is not ``FSM_BUDGET_MULTIPLIER`` steps
        overrides it, as ``NativeFunctionCallingReactAgent`` does): returns
        ``(ceiling, formula)``, ``ceiling >= 1`` and ``formula`` the text the
        budget error cites after ``"<ceiling> loop turns = "``. Never raises.
        """
        multiplier = Defaults.FSM_BUDGET_MULTIPLIER
        return (
            max_iterations * multiplier,
            f"max_iterations {max_iterations} x FSM_BUDGET_MULTIPLIER {multiplier}",
        )

    def _budget_error(
        self, exc: RunBudgetExceededError, max_iterations: int | None = None
    ) -> AgentError:
        """The public agent error for a spent core run budget.

        Interface contract (same callers as ``_run_budgets``): the seconds
        budget maps to ``AgentTimeoutError(config.timeout_seconds)``, the
        steps budget to ``BudgetExhaustedError`` citing the step ceiling.
        Returns the error; the caller raises it ``from exc``.
        """
        if exc.budget == "seconds":
            return AgentTimeoutError(self.config.timeout_seconds)
        ceiling, formula = self._step_ceiling(
            max_iterations or self.config.max_iterations
        )
        # LOOP-17: cite the loop-turn ceiling that was hit, not max_iterations.
        return BudgetExhaustedError(
            "iterations", ceiling, detail=f"{ceiling} loop turns = {formula}"
        )

    def _create_api(self, fsm_def: dict[str, Any]) -> API:
        """Create an API instance from an FSM definition.

        ``config.model``, ``temperature`` and ``max_tokens`` configure the
        interface core builds; with an injected ``llm_interface`` they are not
        passed (the interface owns them, ``llm_settings_for``), and
        ``enable_prompt_cache`` is refused (``AgentError``).
        """
        self._refuse_prompt_cache_with_injected_interface()
        kwargs = dict(self._api_kwargs)
        if self.config.transition_config is not None:
            kwargs["transition_config"] = self.config.transition_config
        # Additive opt-in config passthroughs (defaults reproduce prior behavior).
        if (
            self.config.max_history_size is not None
            and "max_history_size" not in kwargs
        ):
            kwargs["max_history_size"] = self.config.max_history_size
        if self.config.enable_prompt_cache and "caching" not in kwargs:
            # litellm response-cache flag; no-op where the provider/cache is unset.
            kwargs["caching"] = True
        try:
            return API.from_definition(
                _with_instructions(fsm_def, self.config.instructions),
                **llm_settings_for(
                    kwargs,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                **kwargs,
            )
        except ValueError as exc:
            error = _prompt_overflow_error(
                exc, self.config.instructions, getattr(self, "tools", None)
            )
            if error is None:
                raise
            raise error from exc

    def _prepare_run_api(
        self, fsm_def: dict[str, Any], agent_type: str, handlers: Any
    ) -> API:
        """The API of one run: built from *fsm_def*, with the pattern's handlers
        and the lifecycle handlers registered.

        Shared by ``_standard_run`` and ``_standard_run_stream``; *handlers* is
        the call-local handler-state object of ``_standard_run`` (see there).
        """
        api = self._create_api(fsm_def)
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014
        # Do NOT collapse this to an unconditional `self._register_handlers(api,
        # handlers)` — most subclasses' `_register_handlers(self, api)` override
        # takes only one argument, and only react.py/parallel_react.py's
        # override accepts (and requires) `handlers`. The abstract base
        # signature is deliberately left at `(self, api)` (LSP: subclasses
        # may only WIDEN with optional params, not the base), so this
        # 2-argument call is only type-correct for the subclasses that
        # actually declare it — hence the `type: ignore[call-arg]` below,
        # guarded by the same `handlers is not None` runtime check those
        # subclasses require. See decisions.md D-014.
        if handlers is not None:
            self._register_handlers(api, handlers)  # type: ignore[call-arg]
        else:
            self._register_handlers(api)
        self._register_lifecycle_handlers(api, agent_type)
        return api

    def _standard_run_stream(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        handlers: Any = None,
    ) -> Iterator[str]:
        """Streaming variant of ``_standard_run``.

        Runs the same FSM on ``API.run_until_terminal_stream`` and streams
        each speaking state's reply (silent states yield nothing). Yields raw
        text only; use ``run()`` for the ``AgentResult``/trace. Errors are
        wrapped like ``_standard_run``'s. ``handlers``: see there.
        """
        start_time = time.monotonic()
        api = self._prepare_run_api(fsm_def, agent_type, handlers)

        try:
            conv_id, greeting = api.start_conversation(context)
            try:
                if greeting:
                    yield greeting
                max_steps, max_seconds = self._run_budgets(start_time, max_iterations)
                yield from api.run_until_terminal_stream(
                    conv_id,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    before_step=partial(self._on_loop_iteration, api, conv_id),
                    seconds_exempt_states=self._seconds_exempt_for(api),
                )
            finally:
                self._end_run_conversation(api, conv_id)
        except RunBudgetExceededError as exc:
            raise self._budget_error(exc, max_iterations) from exc
        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e

    def _standard_run(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        extra_answer_keys: list[str] | None = None,
        execution_evidence_keys: list[str] | None = None,
        handlers: Any = None,
        judged: bool = False,
        success_keys: list[str] | None = None,
    ) -> AgentResult:
        """Standard run() implementation shared by most agents.

        Handles API creation, handler registration, conversation loop,
        answer extraction, trace building, and error wrapping.

        ``extra_answer_keys``: context keys the answer is read from, before
        the replies. ``success_keys``: keys whose non-empty value proves a
        result (default ``extra_answer_keys``); a pattern whose answer is not
        one of its keys (Debate, PromptChain, native_fc) passes only this.

        ``judged``: see ``_forced_stop_reason`` (EvalOpt, MakerChecker).

        ``handlers``: optional, opaque, call-local handler-state object
        (e.g. ``AgentHandlers``) created fresh by the caller's own ``run()``
        and threaded straight through to ``_register_handlers`` as an
        explicit parameter. Interface contract: when a subclass's ``run()``
        passes ``handlers``, that subclass's ``_register_handlers`` override
        MUST accept it as a second positional parameter (see
        ``react.py``/``parallel_react.py``); subclasses that never pass
        ``handlers`` (the default, ``None``) are unaffected and keep their
        original ``_register_handlers(self, api)`` signature. This exists so
        per-call handler state never has to round-trip through a ``self``
        attribute between construction and use — see decisions.md D-014 for
        why that round-trip was itself a data race (D-004's insufficient
        first fix).
        """
        start_time = time.monotonic()
        api = self._prepare_run_api(fsm_def, agent_type, handlers)

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, agent_type, max_iterations
            )

            answer = self._extract_answer(final_context, responses, extra_answer_keys)
            trace = self._build_trace(final_context, iteration)

            structured = self._try_parse_structured_output(answer, final_context)

            # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: a run that produced
            # neither a designated answer key (a pattern-specific
            # extra_answer_key) NOR any tool call is degenerate — the `answer`
            # came from the prose-fallback in _extract_answer (a planner state's
            # Pass-2 text leaking as the result). Report success=False rather
            # than passing leaked filler off as a completed task. Mirrors
            # _extract_answer's primary/secondary sources, so any pattern that
            # concludes properly (sets an answer key) or runs a tool is unaffected.
            success, stop_reason = self._run_outcome(
                final_context,
                trace,
                extra_answer_keys if success_keys is None else success_keys,
                execution_evidence_keys,
                judged=judged,
                answer_cut_off=self._answer_was_cut_off(
                    final_context, extra_answer_keys
                ),
            )
            if stop_reason in FORCED_STOP_REASONS:
                logger.warning(
                    f"Agent '{agent_type}' was forced to stop ({stop_reason.value}); "
                    f"shipping its last answer with success=False."
                )
            elif not success:
                missing = (
                    f"execution evidence ({execution_evidence_keys})"
                    if execution_evidence_keys
                    else "an answer key and any tool call"
                )
                logger.warning(
                    f"Agent '{agent_type}' completed with no {missing}: its "
                    f"answer is filler; marking success=False."
                )

            return AgentResult(
                answer=answer,
                trace=trace,
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the redo
                # stash leaves the result here, not in _filter_context (shared
                # with adapt) and not via an internal prefix (prompts need it).
                final_context={
                    k: v
                    for k, v in self._filter_context(final_context).items()
                    if k not in RESULT_DROPPED_CONTEXT_KEYS
                },
                structured_output=structured,
                stop_reason=stop_reason,
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e
