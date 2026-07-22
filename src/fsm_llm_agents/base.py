from __future__ import annotations

"""
BaseAgent — Abstract base class for all fsm_llm agents.

Extracts the common conversation loop, budget enforcement, answer extraction,
trace building, and context filtering from the 12 agent implementations.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, cast

from pydantic import BaseModel

from fsm_llm import API
from fsm_llm.constants import has_internal_prefix
from fsm_llm.context import ContextCompactor
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError


def _output_response_format(schema: Any) -> dict[str, Any] | None:
    """Build the ``response_format`` envelope for a Pydantic *schema*.

    Interface contract (two call sites: ``_init_context`` here, and
    ``native_fc``'s post-loop repair turn):

    Args:
        schema: ``AgentConfig.output_schema`` — a Pydantic ``BaseModel``
            subclass, ``None``, or any object (duck-typed on
            ``model_json_schema``).

    Returns:
        The ``{"type": "json_schema", ...}`` dict litellm/OpenAI accept as
        ``response_format``, or ``None`` when *schema* is ``None`` or does not
        expose ``model_json_schema``.  Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-002
    # This helper is the ONE new abstraction that plan's Complexity Budget
    # allows inside the five existing packages (1/1), and it is earned by
    # EXACTLY two call sites: `_init_context` below, and `native_fc.run`'s
    # post-loop repair turn. It was extracted rather than copied because the
    # alternative -- native_fc building its own `{"type": "json_schema", ...}`
    # envelope -- is a second builder of the same provider contract, kept in
    # lockstep by hand, which is the drift `hardening.py`'s D-059 block already
    # records this repo paying for once.
    # Do NOT add a third caller by reflex: a call site that can set
    # `AgentConfig.output_schema` and go through `BaseAgent` gets this for free
    # and should. Do NOT make it raise on a bad schema either -- both callers
    # treat `None` as "no constrained decoding available", and an exception here
    # would turn a missing capability into a failed run.
    # See decisions.md D-002.
    if schema is None or not hasattr(schema, "model_json_schema"):
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }


class BaseAgent(ABC):
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

    def __init__(
        self,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        self.config = config or AgentConfig()
        self._api_kwargs = api_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent on a task. Implemented by each agent pattern."""
        ...

    def __call__(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Callable shorthand: ``agent("task")`` → ``agent.run("task")``."""
        return self.run(task, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"

    # ------------------------------------------------------------------
    # Common conversation loop
    # ------------------------------------------------------------------

    def _run_conversation_loop(
        self,
        api: API,
        context: dict[str, Any],
        start_time: float,
        agent_type: str,
        max_iterations: int | None = None,
    ) -> tuple[list[str], dict[str, Any], int]:
        """Run the standard FSM conversation loop.

        Returns:
            Tuple of (responses, final_context, iteration_count).
        """
        max_iters = max_iterations or self.config.max_iterations

        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id,
            package="fsm_llm_agents",
            agent_type=agent_type,
        )

        try:
            responses = [initial_response]
            iteration = 0

            while not api.has_conversation_ended(conv_id):
                iteration += 1

                self._check_budgets(start_time, iteration, max_iters)

                # Hook for mid-loop processing (e.g. HITL approval)
                self._on_loop_iteration(api, conv_id, iteration)

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            final_context = api.get_data(conv_id)
            log.info(LogMessages.AGENT_COMPLETE.format(iterations=iteration))
            return responses, final_context, iteration

        finally:
            api.end_conversation(conv_id)

    def _on_loop_iteration(  # noqa: B027
        self,
        api: API,
        conv_id: str,
        iteration: int,
    ) -> None:
        """Hook called each loop iteration before converse().

        Override for HITL approval gates or other mid-loop logic.
        Default is a no-op.
        """

    # ------------------------------------------------------------------
    # Context initialisation
    # ------------------------------------------------------------------

    def _init_context(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the standard initial context for an agent run.

        Sets ``TASK``, ``AGENT_TRACE``, and ``ITERATION_COUNT``.
        Warns if *initial_context* already contains reserved keys.
        """
        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        reserved = {
            ContextKeys.TASK,
            ContextKeys.AGENT_TRACE,
            ContextKeys.ITERATION_COUNT,
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

        # Schema-enforced output: when output_schema is set, store the
        # JSON schema as response_format so the pipeline can pass it to
        # the LLM for constrained decoding on the conclude state.
        response_format = _output_response_format(self.config.output_schema)
        if response_format is not None:
            context["_output_response_format"] = response_format

        if extra:
            context.update(extra)
        return context

    # ------------------------------------------------------------------
    # Handler registration helpers
    # ------------------------------------------------------------------

    def _register_iteration_limiter(
        self,
        api: API,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard iteration-limiter handler."""
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .with_priority(HandlerPriorities.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(handler_fn)
        )

    def _register_tool_executor(
        self,
        api: API,
        state: str,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard tool-executor handler on a state entry."""
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(state)
            .do(handler_fn)
        )

    def _register_hitl_gate(
        self,
        api: API,
        checker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the HITL approval-gate handler."""
        api.register_handler(
            api.create_handler(HandlerNames.HITL_GATE)
            .with_priority(HandlerPriorities.HITL_GATE)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.TOOL_NAME)
            .do(checker_fn)
        )

    # ------------------------------------------------------------------
    # HITL loop-iteration helper
    # ------------------------------------------------------------------

    def _handle_hitl_approval(self, api: API, conv_id: str) -> None:
        """Check and process HITL approval for the current context.

        Shared logic for agents that use synchronous HITL approval gates
        (ReactAgent, ReflexionAgent).  Subclasses must set ``self.hitl``
        to a :class:`HumanInTheLoop` instance (or ``None``).
        """
        from .hitl import HumanInTheLoop
        from .tools import normalize_tool_input

        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None:
            return

        current_context = api.get_data(conv_id)
        if not (
            current_context.get(ContextKeys.APPROVAL_REQUIRED)
            and not current_context.get(ContextKeys.APPROVAL_GRANTED)
        ):
            return

        tool_name = current_context.get(ContextKeys.TOOL_NAME, "")
        tool_input = normalize_tool_input(current_context.get(ContextKeys.TOOL_INPUT))
        reasoning = current_context.get(ContextKeys.REASONING, "")

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=tool_input,
            reasoning=reasoning,
        )

        approved = hitl.request_approval(tool_call, current_context)
        api.update_context(
            conv_id,
            {
                ContextKeys.APPROVAL_GRANTED: approved,
                ContextKeys.APPROVAL_REQUIRED: False,
            },
        )
        if not approved:
            api.update_context(
                conv_id,
                {
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                },
            )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budgets(
        self,
        start_time: float,
        iteration: int,
        max_iterations: int | None = None,
    ) -> None:
        """Raise if time or iteration budget exceeded."""
        if time.monotonic() - start_time > self.config.timeout_seconds:
            raise AgentTimeoutError(self.config.timeout_seconds)

        max_iters = max_iterations or self.config.max_iterations
        if iteration > max_iters * Defaults.FSM_BUDGET_MULTIPLIER:
            raise BudgetExhaustedError("iterations", max_iters)

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract answer with a fallback chain.

        1. Try ``ContextKeys.FINAL_ANSWER``
        2. Try each key in *extra_keys* (e.g. ``JUDGE_VERDICT``)
        3. Try responses in reverse order
        4. Return default message
        """
        # Primary: final_answer
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and answer.strip():
            return str(answer)

        # Secondary: extra context keys (pattern-specific)
        for key in extra_keys or []:
            val = final_context.get(key)
            if val and isinstance(val, str) and val.strip():
                return str(val).strip()

        # Tertiary: last non-empty response
        for response in reversed(responses):
            if response and response.strip():
                return response.strip()

        return "Agent could not determine an answer."

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    @staticmethod
    def _has_execution_evidence(
        final_context: dict[str, Any],
        evidence_keys: list[str],
    ) -> bool:
        """True if any ``evidence_keys`` context value proves real execution.

        Generic over the three planner evidence shapes:
        - a non-empty dict (e.g. ReWOO ``evidence`` mapping) → real;
        - a list whose dict entries all carry a ``success`` flag (e.g.
          Orchestrator ``worker_results``) → real only if ≥1 entry succeeded
          (placeholder/failed-worker entries are NOT evidence);
        - any other non-empty list (e.g. plan_execute ``step_results``, which
          have no per-entry success flag) → real.
        """
        for key in evidence_keys:
            value = final_context.get(key)
            if not value:
                continue
            if isinstance(value, dict):
                return True
            if isinstance(value, list):
                dict_entries = [e for e in value if isinstance(e, dict)]
                if dict_entries and all("success" in e for e in dict_entries):
                    if any(e.get("success") for e in dict_entries):
                        return True
                    # all entries failed/placeholder → not evidence; keep looking
                else:
                    return True
        return False

    @staticmethod
    def _completion_is_real(
        final_context: dict[str, Any],
        trace: AgentTrace,
        extra_answer_keys: list[str] | None,
        execution_evidence_keys: list[str] | None = None,
    ) -> bool:
        """True if the run produced a genuine result.

        A real completion has either a designated answer key
        (``FINAL_ANSWER`` or a pattern-specific ``extra_answer_key``) or at
        least one executed tool call. When BOTH are absent the ``answer``
        can only have come from the prose-fallback in ``_extract_answer``
        (a planner state's Pass-2 text leaking as the result) — that is a
        degenerate completion, not a success.

        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: when ``execution_evidence_keys``
        # is supplied (planner patterns: orchestrator/rewoo/plan_execute), the
        # answer-key/tool-call test above is NOT sufficient. A planner can reach
        # a synthesis state that sets ``final_answer`` (and record a ``delegate``
        # control action that _build_trace turns into a fake ToolCall) while
        # having executed ZERO real work — weak 4b decomposition routes straight
        # to synthesis via the fallback transitions. That filler must report
        # success=False. So in planner mode we require genuine EXECUTION EVIDENCE
        # (a successful worker / non-empty tool evidence / executed steps)
        # instead. This extends D-001 (plan_26c9510a) from "no answer key AND no
        # tool" to also cover "answer key but no real execution". Opt-in per
        # call-site → every non-planner pattern (execution_evidence_keys=None)
        # keeps the original has_answer_key-OR-tools_executed behavior unchanged.
        """
        if execution_evidence_keys:
            return BaseAgent._has_execution_evidence(
                final_context, execution_evidence_keys
            )
        has_answer_key = bool(
            str(final_context.get(ContextKeys.FINAL_ANSWER) or "").strip()
        ) or any(
            str(final_context.get(k) or "").strip() for k in (extra_answer_keys or [])
        )
        tools_executed = len(trace.tool_calls) > 0
        return has_answer_key or tools_executed

    def _build_trace(
        self,
        final_context: dict[str, Any],
        iteration: int,
    ) -> AgentTrace:
        """Build an AgentTrace from the AGENT_TRACE context entries.

        Agents that don't use tools get an empty trace (just iteration count).
        """
        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )

        for step in trace_data:
            if isinstance(step, dict) and "action" in step:
                tool_name = step.get("action", "").split("(")[0]
                if tool_name and tool_name != ContextKeys.NO_TOOL:
                    tool_input = step.get("tool_input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {"input": tool_input}
                    trace.tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            parameters=tool_input,
                            reasoning=step.get("thought", ""),
                        )
                    )

        return trace

    # ------------------------------------------------------------------
    # Context filtering
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
    # This filter's output feeds `AgentResult.final_context`, which is returned
    # straight to the agent's caller. Do NOT re-inline `k.startswith("_")` here:
    # that check is case-SENSITIVE and only sees the literal `_` prefix, so
    # `system_password`, `internal_token` and `__dunder` all leaked through it
    # (F-13, measured). `has_internal_prefix` is the single canonical predicate
    # over INTERNAL_KEY_PREFIXES and case-folds. See decisions.md D-003.
    @staticmethod
    def _filter_context(context: dict[str, Any]) -> dict[str, Any]:
        """Remove internal-prefixed keys from context.

        Top-level only: nested dict values are not recursed into (that is a
        separate contract, see `fsm_llm.context.clean_context_keys`).
        """
        return {k: v for k, v in context.items() if not has_internal_prefix(k)}

    # ------------------------------------------------------------------
    # API factory helper
    # ------------------------------------------------------------------

    def _create_api(self, fsm_def: dict[str, Any]) -> API:
        """Create an API instance from an FSM definition."""
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
        return API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Lifecycle handler registration (shared by run + run_stream)
    # ------------------------------------------------------------------

    def _register_lifecycle_handlers(self, api: API, agent_type: str) -> None:
        """Register the END_CONVERSATION, ERROR, and context-compactor handlers.

        Extracted from ``_standard_run`` so the streaming path registers the
        exact same lifecycle handlers (no behavior drift between run paths).
        """
        api.register_handler(
            api.create_handler(HandlerNames.END_CONVERSATION)
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.END_CONVERSATION)
            .do(
                lambda ctx: {
                    "_agent_completed": True,
                    "_agent_type": agent_type,
                }
            )
        )

        def _error_handler(ctx: dict[str, Any]) -> dict[str, Any]:
            logger.warning(
                f"Agent error in {agent_type}: state={ctx.get('_current_state', '?')}"
            )
            return {}

        api.register_handler(
            api.create_handler(HandlerNames.ERROR)
            .with_priority(HandlerPriorities.ERROR)
            .at(HandlerTiming.ERROR)
            .do(_error_handler)
        )

        compactor = ContextCompactor(
            transient_keys={
                ContextKeys.TOOL_RESULT,
                ContextKeys.TOOL_STATUS,
                ContextKeys.TOOL_ERROR,
            },
        )
        api.register_handler(
            api.create_handler("AgentContextCompactor")
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(compactor.compact)
        )

        # Opt-in observation summarization (config.auto_summarize_after).
        # No-op for agents that don't accumulate observations.
        if self.config.auto_summarize_after:
            from .summarization import make_observation_summarizer

            summarizer = make_observation_summarizer(self.config.auto_summarize_after)
            api.register_handler(
                api.create_handler("AgentObservationSummarizer")
                .with_priority(HandlerPriorities.END_CONVERSATION)
                .at(HandlerTiming.PRE_PROCESSING)
                .do(summarizer)
            )

    # ------------------------------------------------------------------
    # Streaming run() implementation
    # ------------------------------------------------------------------

    def _standard_run_stream(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
    ) -> Iterator[str]:
        """Streaming variant of ``_standard_run``.

        Drives the same FSM loop but streams each turn's Pass-2 output token by
        token via ``API.converse_stream``. Intermediate states with empty
        ``response_instructions`` (think/act) yield nothing; the final answer
        state (conclude) streams its output. Yields raw text only — callers
        needing the structured ``AgentResult``/trace should use ``run()``.
        """
        start_time = time.monotonic()
        api = self._create_api(fsm_def)
        self._register_handlers(api)
        self._register_lifecycle_handlers(api, agent_type)

        max_iters = max_iterations or self.config.max_iterations
        conv_id, initial_response = api.start_conversation(context)
        if initial_response:
            yield initial_response

        iteration = 0
        try:
            while not api.has_conversation_ended(conv_id):
                iteration += 1
                self._check_budgets(start_time, iteration, max_iters)
                self._on_loop_iteration(api, conv_id, iteration)
                yield from api.converse_stream(Defaults.CONTINUE_MESSAGE, conv_id)
        finally:
            api.end_conversation(conv_id)

    # ------------------------------------------------------------------
    # Standard run() implementation
    # ------------------------------------------------------------------

    def _standard_run(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        extra_answer_keys: list[str] | None = None,
        execution_evidence_keys: list[str] | None = None,
    ) -> AgentResult:
        """Standard run() implementation shared by most agents.

        Handles API creation, handler registration, conversation loop,
        answer extraction, trace building, and error wrapping.
        """
        start_time = time.monotonic()
        api = self._create_api(fsm_def)
        self._register_handlers(api)
        self._register_lifecycle_handlers(api, agent_type)

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, agent_type, max_iterations
            )

            answer = self._extract_answer(final_context, responses, extra_answer_keys)
            trace = self._build_trace(final_context, iteration)

            structured = self._try_parse_structured_output(answer, final_context)

            # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: a run that produced
            # neither a designated answer key (FINAL_ANSWER or a pattern-specific
            # extra_answer_key) NOR any tool call is degenerate — the `answer`
            # came from the prose-fallback in _extract_answer (a planner state's
            # Pass-2 text leaking as the result). Report success=False rather
            # than passing leaked filler off as a completed task. Mirrors
            # _extract_answer's primary/secondary sources, so any pattern that
            # concludes properly (sets an answer key) or runs a tool is unaffected.
            success = self._completion_is_real(
                final_context, trace, extra_answer_keys, execution_evidence_keys
            )
            if not success:
                if execution_evidence_keys:
                    logger.warning(
                        f"Agent '{agent_type}' completed with no execution "
                        f"evidence ({execution_evidence_keys}) — planner produced "
                        f"synthesis filler without running planned work; marking "
                        f"success=False."
                    )
                else:
                    logger.warning(
                        f"Agent '{agent_type}' completed with no answer key and no "
                        f"tool calls — answer is prose-fallback only; marking "
                        f"success=False."
                    )

            return AgentResult(
                answer=answer,
                success=success,
                trace=trace,
                final_context=self._filter_context(final_context),
                structured_output=structured,
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def _try_parse_structured_output(
        self, answer: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Validate *answer* against ``config.output_schema`` if set.

        Returns a Pydantic model instance on success, ``None`` on failure
        or when no schema is configured.
        """
        # `output_schema` is declared `type | None` but `AgentConfig.validate_output_schema`
        # (definitions.py) guarantees any non-None value is a Pydantic BaseModel subclass,
        # so `.model_fields` is always present at runtime. Cast narrows for mypy (683,693).
        # Annotation-only — no runtime change.
        schema = cast("type[BaseModel] | None", self.config.output_schema)
        if schema is None:
            return None

        # 1. Try constructing from context keys (most reliable — uses Pass 1 data)
        # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: emit a diagnostic instead of
        # silently swallowing the validation error — a structured_output of None
        # otherwise gives no clue which fields were missing/invalid.
        if context:
            try:
                fields = {
                    k: context[k]
                    for k in schema.model_fields
                    if k in context and context[k] is not None
                }
                if fields:
                    return schema(**fields)
            except Exception as e:
                logger.debug(
                    f"Structured output: context-key construction of "
                    f"{schema.__name__} failed ({e}); "
                    f"present keys={sorted(fields)}, "
                    f"schema keys={sorted(schema.model_fields)}. "
                    f"Falling back to JSON parse."
                )

        # 2. Try parsing JSON from the answer string
        try:
            import json as _json

            from fsm_llm.utilities import extract_json_from_text

            data = extract_json_from_text(answer)
            if data is None:
                # Try direct JSON parse
                data = _json.loads(answer)

            if isinstance(data, dict):
                return schema(**data)

            logger.warning(
                f"Structured output: expected dict, got {type(data).__name__}"
            )
        except Exception as e:
            logger.warning(f"Structured output validation failed: {e}")

        # 3. Scan tool observations for JSON matching the schema
        if context:
            import json as _json

            from fsm_llm.utilities import extract_json_from_text

            observations = context.get("observations", [])
            if isinstance(observations, list):
                for obs in reversed(observations):
                    if not isinstance(obs, str):
                        continue
                    # Entries are formatted as "[Step n] Tool: ... | Result: ...";
                    # scan only the trailing result segment so embedded JSON is found.
                    segment = obs.rsplit("Result:", 1)[-1] if "Result:" in obs else obs
                    data = extract_json_from_text(segment)
                    if isinstance(data, dict):
                        try:
                            return schema(**data)
                        except Exception:
                            continue

        return None

    @abstractmethod
    def _register_handlers(self, api: API) -> None:
        """Register pattern-specific handlers. Implemented by each agent."""
        ...
