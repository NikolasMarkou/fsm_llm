"""
Workflow step implementations for the FSM-LLM Workflow System.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Executor
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from fsm_llm.constants import has_internal_prefix
from fsm_llm.logging import logger

from .constants import KEY_USER_INPUT, PARALLEL_DEEPCOPY_WARNING_THRESHOLD
from .exceptions import WorkflowStepError
from .models import WaitEventConfig, WorkflowStepResult

# --------------------------------------------------------------

#: Executor used for synchronous user callables. The engine sets it for the
#: duration of a step when it was constructed with ``executor=...``; ``None``
#: means the event loop's default executor.
_STEP_EXECUTOR: contextvars.ContextVar[Executor | None] = contextvars.ContextVar(
    "fsm_llm_workflows_step_executor", default=None
)


#: ``fsm_llm.definitions.ResponseGenerationRequest.user_message`` max_length.
_MAX_CORE_USER_MESSAGE = 10000


def _is_async_callable(fn: Any) -> bool:
    """True for ``async def`` functions (incl. partials) and objects whose
    ``__call__`` is ``async def``."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)  # noqa: B004
    return call is not None and inspect.iscoroutinefunction(call)


async def _call_user_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a user callable without blocking the event loop.

    Async callables are awaited directly. Everything else runs in the step
    executor, and if it RETURNS an awaitable (a lambda wrapping an async
    function, a sync factory returning a coroutine) that awaitable is awaited
    too, so a coroutine object is never mistaken for a plain value.

    # DECISION plan-2026-09-27T120000-5d1e7a3b/D-004
    # Do NOT go back to `inspect.iscoroutinefunction(fn)` alone: a lambda that
    # returns a coroutine was run in the executor and its (truthy, never
    # awaited) coroutine object was used as the result, so a ConditionStep
    # always took `true_state` and an APICallStep silently mapped nothing.

    Note: a synchronous callable that is still running when a timeout fires
    cannot be stopped (Python threads are not cancellable); it keeps running
    in its worker thread. Keep sync callables short or make them async.
    """
    if _is_async_callable(fn):
        result = fn(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _STEP_EXECUTOR.get(), functools.partial(fn, *args, **kwargs)
        )
    if inspect.isawaitable(result):
        result = await result
    return result


class WorkflowStep(BaseModel, ABC):
    """Base class for workflow steps."""

    step_id: str
    name: str
    description: str = ""
    timeout: float | None = None
    """Maximum seconds this step may run. ``None`` (the default) disables the
    timeout; set a finite value for steps that call out to slow services."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    async def _with_timeout(self, coro):
        """Wrap a coroutine with ``asyncio.wait_for`` if timeout is configured."""
        if self.timeout is not None:
            try:
                return await asyncio.wait_for(coro, timeout=self.timeout)
            except asyncio.TimeoutError as e:
                raise WorkflowStepError(
                    step_id=self.step_id,
                    message=f"Step timed out after {self.timeout}s",
                ) from e
        return await coro


class AutoTransitionStep(WorkflowStep):
    """A step that automatically transitions to the next state.

    ``action`` may return a dict (merged into the context) or ``None``. When
    the action fails, the step routes to ``error_state`` if one is set and
    raises ``WorkflowStepError`` otherwise (the instance then FAILS).
    """

    next_state: str
    action: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    error_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the step and transition automatically."""
        try:
            data = {}
            if self.action:
                data = await self._with_timeout(
                    _call_user_callable(self.action, context)
                )
                if data is None:
                    data = {}
                elif not isinstance(data, dict):
                    raise TypeError(
                        f"action must return a dict or None, got {type(data).__name__}"
                    )

            return WorkflowStepResult.success_result(
                data=data,
                next_state=self.next_state,
                message=f"Auto-transitioned to {self.next_state}",
            )
        except Exception as e:
            logger.error(f"Error in auto transition step {self.step_id}: {e!s}")
            if self.error_state:
                return WorkflowStepResult.failure_result(
                    error=_error_text(e),
                    next_state=self.error_state,
                    message=f"Auto-transition failed, transitioning to {self.error_state}",
                )
            raise WorkflowStepError(
                step_id=self.step_id, message="Auto-transition failed", cause=e
            ) from e


class APICallStep(WorkflowStep):
    """A step that calls an external API.

    ``input_mapping`` is ``{api_param: context_key}``. ``output_mapping`` is
    ``{context_key: result_key}``; ``result_key`` may be a dotted path into
    nested dicts (``"user.name"``) or ``""`` for the whole result (which also
    captures non-dict results such as lists or scalars).
    """

    api_function: Callable[..., Any]
    success_state: str
    failure_state: str
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the API call and process the result."""
        try:
            # Prepare API parameters from context
            params = self._map_input_parameters(context)

            # Call the API
            api_result = await self._call_api(params)

            # Process the result
            output_data = self._map_output_data(api_result)

            return WorkflowStepResult.success_result(
                data=output_data,
                next_state=self.success_state,
                message=f"API call successful, transitioning to {self.success_state}",
            )
        except Exception as e:
            logger.error(f"Error in API call step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_state=self.failure_state,
                message=f"API call failed, transitioning to {self.failure_state}",
            )

    def _map_input_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        """Map context keys to API parameters."""
        params = {}
        for api_param, context_key in self.input_mapping.items():
            if context_key in context:
                params[api_param] = context[context_key]
            else:
                logger.debug(
                    f"APICallStep [{self.step_id}]: input_mapping key "
                    f"'{context_key}' not in context; '{api_param}' omitted"
                )
        return params

    async def _call_api(self, params: dict[str, Any]) -> Any:
        """Call the API function."""
        return await self._with_timeout(
            _call_user_callable(self.api_function, **params)
        )

    def _map_output_data(self, api_result: Any) -> dict[str, Any]:
        """Map API response to context keys."""
        output_data = {}
        for context_key, result_key in self.output_mapping.items():
            found, value = _lookup_path(api_result, result_key)
            if found:
                output_data[context_key] = value
            else:
                logger.debug(
                    f"APICallStep [{self.step_id}]: output_mapping key "
                    f"'{result_key}' not in API result; '{context_key}' not set"
                )
        return output_data


class ConditionStep(WorkflowStep):
    """A step that evaluates a condition and transitions accordingly.

    When the condition raises, the step routes to ``error_state`` if one is
    set and raises ``WorkflowStepError`` otherwise.
    """

    condition: Callable[[dict[str, Any]], bool]
    true_state: str
    false_state: str
    error_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the condition and determine the next state."""
        try:
            result = await self._with_timeout(
                _call_user_callable(self.condition, context)
            )

            # Determine the next state
            next_state = self.true_state if result else self.false_state

            return WorkflowStepResult.success_result(
                next_state=next_state,
                message=f"Condition evaluated to {result}, transitioning to {next_state}",
                data={"condition_result": result},
            )
        except Exception as e:
            logger.error(f"Error in condition step {self.step_id}: {e!s}")
            if self.error_state:
                return WorkflowStepResult.failure_result(
                    error=_error_text(e),
                    next_state=self.error_state,
                    message=f"Condition evaluation failed, transitioning to {self.error_state}",
                )
            raise WorkflowStepError(
                step_id=self.step_id, message="Condition evaluation failed", cause=e
            ) from e


class LLMProcessingStep(WorkflowStep):
    """A step that processes data using an LLM.

    ``llm_interface`` may be a core ``fsm_llm.LLMInterface`` (its
    ``generate_response`` is called with ``system_prompt`` and the rendered
    prompt as the user message) or any object with a ``generate(prompt)``
    method, sync or async, returning a string.

    ``context_mapping`` is the INPUT mapping ``{prompt_var: context_key}``
    used to fill ``prompt_template`` (``str.format`` syntax: write literal
    braces as ``{{`` and ``}}``). ``output_mapping`` is
    ``{context_key: regex}``: group 1 (or the whole match) is stored; an
    empty regex stores the whole response; a regex that does not match
    leaves the key unset. On failure the step routes to ``error_state``, or
    FAILS the instance when none is set.
    """

    llm_interface: Any
    prompt_template: str
    context_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    next_state: str
    error_state: str | None = None
    system_prompt: str = (
        "You are a precise assistant. Follow the instructions in the user "
        "message and answer with exactly what they ask for."
    )

    @field_validator("output_mapping")
    @classmethod
    def _validate_output_patterns(cls, v: dict[str, str]) -> dict[str, str]:
        for key, pattern in v.items():
            if pattern:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise ValueError(
                        f"output_mapping[{key!r}] is not a valid regex: {e}"
                    ) from e
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Process data with the LLM."""
        try:
            # Prepare the prompt
            prompt = self._prepare_prompt(context)

            # Call the LLM
            llm_response = await self._call_llm(prompt)

            # Process the result
            output_data = self._process_llm_response(llm_response)

            return WorkflowStepResult.success_result(
                data=output_data,
                next_state=self.next_state,
                message=f"LLM processing successful, transitioning to {self.next_state}",
            )
        except Exception as e:
            logger.error(f"Error in LLM processing step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_state=self.error_state,
                message=f"LLM processing failed: {e!s}",
            )

    def _prepare_prompt(self, context: dict[str, Any]) -> str:
        """Prepare the prompt from the template and context."""
        prompt_vars = {}
        for prompt_var, context_key in self.context_mapping.items():
            if context_key in context:
                prompt_vars[prompt_var] = context[context_key]
        try:
            return self.prompt_template.format(**prompt_vars)
        except KeyError as e:
            raise WorkflowStepError(
                step_id=self.step_id,
                message=f"Prompt template variable {e} not found in context mapping",
                cause=e,
            ) from e
        except (ValueError, IndexError) as e:
            raise WorkflowStepError(
                step_id=self.step_id,
                message=(
                    f"Prompt template is not a valid format string ({e}); "
                    "write literal braces as '{{' and '}}'"
                ),
                cause=e,
            ) from e

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface and return the response text."""
        llm = self.llm_interface
        generate = getattr(llm, "generate", None)
        if callable(generate):
            raw = await self._with_timeout(_call_user_callable(generate, prompt))
        elif callable(getattr(llm, "generate_response", None)):
            from fsm_llm.definitions import ResponseGenerationRequest

            system_prompt = self.system_prompt or "Follow the user's instructions."
            if len(prompt) > _MAX_CORE_USER_MESSAGE:
                # ResponseGenerationRequest caps user_message; the system
                # prompt allows much more, so carry a long prompt there.
                request = ResponseGenerationRequest(
                    system_prompt=f"{system_prompt}\n\n{prompt}",
                    user_message="Follow the instructions above.",
                )
            else:
                request = ResponseGenerationRequest(
                    system_prompt=system_prompt, user_message=prompt
                )
            raw = await self._with_timeout(
                _call_user_callable(llm.generate_response, request)
            )
        else:
            raise TypeError(
                "llm_interface must provide generate(prompt) or "
                "generate_response(ResponseGenerationRequest)"
            )
        if isinstance(raw, str):
            return raw
        message = getattr(raw, "message", None)
        if isinstance(message, str):
            return message
        raise TypeError(
            f"LLM returned {type(raw).__name__}, expected a string or an object "
            "with a string .message"
        )

    def _process_llm_response(self, response: str) -> dict[str, Any]:
        """Process the LLM response and extract data using regex patterns."""
        output_data = {}
        for context_key, pattern in self.output_mapping.items():
            if pattern:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    output_data[context_key] = (
                        match.group(1) if match.lastindex else match.group(0)
                    )
                else:
                    logger.warning(
                        f"LLMProcessingStep [{self.step_id}]: pattern for "
                        f"'{context_key}' did not match the response; key not set"
                    )
            else:
                output_data[context_key] = response
        return output_data


class WaitForEventStep(WorkflowStep):
    """A step that waits for an external event (see ``WaitEventConfig``)."""

    config: WaitEventConfig

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Set up the workflow to wait for an event."""
        waiting_info = {
            "waiting_for_event": True,
            "event_type": self.config.event_type,
            "timeout_seconds": self.config.timeout_seconds,
            "timeout_state": self.config.timeout_state,
            "success_state": self.config.success_state,
            "event_mapping": self.config.event_mapping,
            "correlation_key": self.config.correlation_key,
            "waiting_since": datetime.now(timezone.utc).isoformat(),
        }

        return WorkflowStepResult.success_result(
            data={"_waiting_info": waiting_info},
            message=f"Waiting for event of type {self.config.event_type}",
        )


class TimerStep(WorkflowStep):
    """A step that waits for a specified time before transitioning."""

    delay_seconds: float
    next_state: str

    @field_validator("delay_seconds")
    @classmethod
    def _validate_delay_seconds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("delay_seconds must be >= 0")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Set up a timer to transition after a delay."""
        timer_info = {
            "waiting_for_timer": True,
            "delay_seconds": self.delay_seconds,
            "next_state": self.next_state,
            "timer_start": datetime.now(timezone.utc).isoformat(),
            "timer_end": (
                datetime.now(timezone.utc) + timedelta(seconds=self.delay_seconds)
            ).isoformat(),
        }

        return WorkflowStepResult.success_result(
            data={"_timer_info": timer_info},
            message=f"Timer set for {self.delay_seconds} seconds, will transition to {self.next_state}",
        )


class ConversationStep(WorkflowStep):
    """A step that runs an FSM conversation using the fsm_llm API.

    This enables deep integration between workflows and FSM-LLM core:
    workflows can invoke full FSM conversations (including reasoning)
    as steps in a larger workflow.

    The conversation is started, driven by ``auto_messages`` (at most
    ``max_turns`` turns; plus the ``advance_workflow`` user input when
    ``use_user_input`` is set), and its collected data is returned. It is NOT
    guaranteed to reach a terminal state: ``conversation_<step_id>_ended``
    reports whether it did, and ``require_completion=True`` turns a
    conversation that did not end into a failure.

    Exactly one of ``fsm_file`` / ``fsm_definition`` (a dict or an
    ``FSMDefinition``) must be given. ``initial_context`` maps
    ``{conversation_key: workflow_key}`` (input); ``context_mapping`` maps
    ``{workflow_key: conversation_key}`` (output). The effective time limit
    is the smaller of ``timeout`` and ``conversation_timeout``. Failures
    route to ``error_state``, or FAIL the instance when none is set.
    """

    fsm_file: str | None = None
    fsm_definition: Any | None = None
    model: str | None = None
    initial_context: dict[str, str] = Field(default_factory=dict)
    context_mapping: dict[str, str] = Field(default_factory=dict)
    success_state: str = ""
    error_state: str | None = None
    max_turns: int = 20
    conversation_timeout: float | None = None
    require_completion: bool = False
    use_user_input: bool = False

    @field_validator("max_turns")
    @classmethod
    def _validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be >= 1")
        return v

    @field_validator("fsm_definition")
    @classmethod
    def _validate_fsm_definition(cls, v: Any) -> Any:
        if v is None or isinstance(v, dict) or callable(getattr(v, "model_dump", None)):
            return v
        raise ValueError(
            f"fsm_definition must be a dict or an FSMDefinition, got {type(v).__name__}"
        )

    @model_validator(mode="after")
    def _validate_fsm_source(self):
        has_def = self.fsm_definition is not None
        has_file = bool(self.fsm_file)
        if has_def == has_file:
            raise ValueError(
                "ConversationStep requires either fsm_file or fsm_definition "
                "(exactly one)"
            )
        return self

    auto_messages: list[str] = Field(default_factory=list)

    def _effective_timeout(self) -> float | None:
        limits = [t for t in (self.timeout, self.conversation_timeout) if t is not None]
        return min(limits) if limits else None

    def _failure_state(self) -> str | None:
        return self.error_state

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute an FSM conversation and return collected data."""
        limit = self._effective_timeout()
        try:
            # Run the blocking conversation body in an executor so the event
            # loop is not frozen and the time limit can actually fire.
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                _STEP_EXECUTOR.get(), self._run_conversation, dict(context)
            )
            if limit is None:
                return await fut
            # asyncio.wait (not wait_for) so a TimeoutError RAISED by the
            # conversation itself is not mistaken for this step's time limit.
            done, _ = await asyncio.wait({fut}, timeout=limit)
            if fut not in done:
                fut.cancel()
                logger.error(
                    f"ConversationStep [{self.step_id}] timed out after {limit}s"
                )
                return WorkflowStepResult.failure_result(
                    error=f"Conversation timed out after {limit}s",
                    next_state=self._failure_state(),
                    message="Conversation timed out",
                )
            return fut.result()
        except Exception as e:
            logger.error(f"Error in conversation step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_state=self._failure_state(),
                message=f"Conversation failed: {e!s}",
            )

    def _run_conversation(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the FSM conversation loop (synchronous; offloaded to an executor
        by ``execute`` so the event loop stays responsive)."""
        from fsm_llm import API

        # Build initial context from workflow context using mapping
        conv_context: dict[str, Any] = {}
        for conv_key, workflow_key in self.initial_context.items():
            if workflow_key in context:
                conv_context[conv_key] = context[workflow_key]

        # Create API instance
        if self.fsm_definition is not None:
            fsm = API.from_definition(self.fsm_definition, model=self.model)
        else:
            # The model validator guarantees fsm_file is set here.
            fsm = API.from_file(str(self.fsm_file), model=self.model)

        messages = list(self.auto_messages)
        user_input = context.get(KEY_USER_INPUT)
        if self.use_user_input and isinstance(user_input, str) and user_input:
            messages.append(user_input)

        # Start conversation
        conv_id, response = fsm.start_conversation(initial_context=conv_context)
        logger.info(
            f"ConversationStep [{self.step_id}] started conversation: "
            f"{str(response or '')[:100]}"
        )
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-039
        # `last_response` / `final_answer` hold the last SPOKEN reply. Do NOT
        # store the last turn's return value as is: a turn that ends on a
        # silent state (no `response_instructions`) returns "", and that
        # would overwrite the reply a workflow maps out of the conversation.
        last_spoken = response or None

        try:
            # Drive the conversation with the messages
            turn = 0
            for message in messages:
                if fsm.has_conversation_ended(conv_id) or turn >= self.max_turns:
                    break
                response = fsm.converse(user_message=message, conversation_id=conv_id)
                if response:
                    last_spoken = response
                logger.debug(
                    f"ConversationStep [{self.step_id}] turn {turn}: "
                    f"{str(response or '')[:100]}"
                )
                turn += 1

            ended = bool(fsm.has_conversation_ended(conv_id))
            # Collect results
            collected_data = fsm.get_data(conv_id)
            # Inject the last spoken reply so context_mapping can reference
            # it; a conversation that never spoke adds neither key.
            if last_spoken is not None:
                collected_data.setdefault("last_response", last_spoken)
                collected_data.setdefault("final_answer", last_spoken)
        finally:
            fsm.end_conversation(conv_id)

        # Map collected data back to workflow context
        output_data: dict[str, Any] = {}
        for workflow_key, conv_key in self.context_mapping.items():
            if conv_key in collected_data:
                output_data[workflow_key] = collected_data[conv_key]
            else:
                logger.warning(
                    f"ConversationStep [{self.step_id}] context_mapping key "
                    f"'{conv_key}' not found in collected data. "
                    f"Available keys: {list(collected_data.keys())}"
                )
        # Also include raw collected data under a namespaced key.
        # NOTE (W-ISSUE-001): key must NOT start with "_" — the workflow engine
        # filters out underscore-prefixed keys unless whitelisted.
        output_data[f"conversation_{self.step_id}_data"] = collected_data
        output_data[f"conversation_{self.step_id}_ended"] = ended

        if self.require_completion and not ended:
            return WorkflowStepResult(
                success=False,
                data=output_data,
                error=f"Conversation did not reach a terminal state in {turn} turns",
                next_state=self._failure_state(),
                message="Conversation did not complete",
            )

        return WorkflowStepResult.success_result(
            data=output_data,
            next_state=self.success_state,
            message=f"Conversation completed in {turn} turns",
        )


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel.

    Children get their own copy of the context (``deepcopy``, falling back
    to a shallow copy for values that cannot be deep-copied). Their
    ``next_state`` is ignored; their data is merged as ``step_<index>_<key>``
    (internal-prefix keys are dropped first) or by ``aggregation_function``.
    Event and timer steps cannot be children (they only pause a workflow).

    If any child fails, the step fails: it routes to ``error_state``, or
    FAILS the instance when none is set, and still reports the data of the
    children that succeeded.
    """

    steps: list[WorkflowStep]
    next_state: str
    error_state: str | None = None
    aggregation_function: (
        Callable[[list[WorkflowStepResult]], dict[str, Any]] | None
    ) = None

    @field_validator("steps")
    @classmethod
    def _validate_children(cls, v: list[WorkflowStep]) -> list[WorkflowStep]:
        for child in v:
            inner = child
            while isinstance(inner, RetryStep):
                inner = inner.step
            if isinstance(inner, (WaitForEventStep, TimerStep)):
                raise ValueError(
                    f"ParallelStep child '{child.step_id}' is a "
                    f"{type(inner).__name__}; event and timer steps cannot run "
                    "inside a ParallelStep"
                )
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute multiple steps in parallel and aggregate results."""
        try:
            # Execute all steps in parallel
            results = await self._execute_parallel_steps(context)

            # Check for errors
            errors = self._collect_errors(results)
            if errors:
                error_msg = "; ".join(errors)
                # DECISION plan-2026-09-27T120000-5d1e7a3b/D-003
                # Do NOT fall back to next_state when error_state is unset: a
                # failed child must not continue down the success route. With
                # no error_state the failure has no next_state and the engine
                # FAILS the instance.
                return WorkflowStepResult(
                    success=False,
                    data=self._default_aggregate(results),
                    error=error_msg,
                    next_state=self.error_state,
                    message=f"Parallel step had {len(errors)} error(s)",
                )

            # Aggregate results
            aggregated_data = self._aggregate_results(results)

            return WorkflowStepResult.success_result(
                data=aggregated_data,
                next_state=self.next_state,
                message="Parallel step completed successfully",
            )
        except Exception as e:
            logger.error(f"Error in parallel step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_state=self.error_state,
                message=f"Parallel step failed: {e!s}",
            )

    def _copy_context(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            return copy.deepcopy(context)
        except Exception as e:
            logger.warning(
                f"ParallelStep [{self.step_id}] could not deep-copy the context "
                f"({e!s}); using a shallow copy"
            )
            return dict(context)

    async def _execute_parallel_steps(
        self, context: dict[str, Any]
    ) -> list[WorkflowStepResult]:
        """Execute all steps in parallel with isolated context copies."""
        if len(self.steps) > PARALLEL_DEEPCOPY_WARNING_THRESHOLD:
            logger.warning(
                f"ParallelStep [{self.step_id}] deep-copying context for "
                f"{len(self.steps)} parallel steps — consider reducing parallelism "
                "if memory usage is a concern"
            )
        contexts = [self._copy_context(context) for _ in self.steps]
        tasks = [
            step.execute(ctx) for step, ctx in zip(self.steps, contexts, strict=True)
        ]
        results = await self._with_timeout(
            asyncio.gather(*tasks, return_exceptions=True)
        )

        # Convert exceptions (including a cancelled child) to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=_error_text(result),
                        message=f"Parallel step {i} failed: {_error_text(result)}",
                    )
                )
            elif not isinstance(result, WorkflowStepResult):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=f"returned {type(result).__name__}, not a WorkflowStepResult",
                        message=f"Parallel step {i} returned an invalid result",
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _collect_errors(self, results: list[WorkflowStepResult]) -> list[str]:
        """Collect one error message per failed result.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-003
        # A failure counts whether or not it carries error text: do NOT go
        # back to filtering on `r.error` (an exception with an empty message,
        # such as `TimeoutError()`, then aggregated as a success).
        """
        return [
            r.error or r.message or f"step {i} failed"
            for i, r in enumerate(results)
            if not r.success
        ]

    def _default_aggregate(self, results: list[WorkflowStepResult]) -> dict[str, Any]:
        aggregated_data = {}
        for i, result in enumerate(results):
            if result.success and result.data:
                prefix = f"step_{i}_"
                for key, value in result.data.items():
                    # Drop internal keys BEFORE prefixing: "step_0__x" would no
                    # longer look internal and would slip past the engine's filter.
                    if has_internal_prefix(key):
                        continue
                    aggregated_data[f"{prefix}{key}"] = value
        return aggregated_data

    def _aggregate_results(self, results: list[WorkflowStepResult]) -> dict[str, Any]:
        """Aggregate results from parallel steps."""
        if self.aggregation_function:
            return self.aggregation_function(results)
        return self._default_aggregate(results)


# ------------------------------------------------------------------
# AgentStep — run an fsm_llm.agents BaseAgent as a workflow step
# ------------------------------------------------------------------


class AgentStep(WorkflowStep):
    """Execute an FSM-LLM agent as a workflow step.

    Wraps any ``BaseAgent.run()`` call so agents can participate in
    workflows alongside other step types. ``agent.run`` must return an
    ``AgentResult``-like object (``answer``, ``success``, optional
    ``final_context`` / ``structured_output``) or a plain string answer.

    Example::

        from fsm_llm.agents import ReactAgent
        agent = ReactAgent(tools=registry)
        step = AgentStep(
            step_id="research",
            name="Research Step",
            agent=agent,
            task_template="Research {topic}",
            success_state="analyze",
            context_mapping={"findings": "answer"},
        )

    Output keys: ``agent_answer``/``agent_success`` (latest agent step) and
    ``agent_<step_id>_answer``/``agent_<step_id>_success`` (per step). An
    agent that reports ``success=False`` fails the step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any = Field(exclude=True)
    """A ``BaseAgent`` instance (or any object with a ``run(task)`` method)."""
    task_template: str = "{task}"
    """Format string for the agent task.  Placeholders are filled from context."""
    success_state: str = ""
    context_mapping: dict[str, str] = Field(default_factory=dict)
    """OUTPUT mapping ``{workflow_key: agent_result_key}``. ``agent_result_key``
    is looked up in ``final_context``; ``"answer"``, ``"success"`` and
    ``"structured_output"`` also map the matching result attribute."""
    input_mapping: dict[str, str] = Field(default_factory=dict)
    """INPUT mapping ``{agent_context_key: workflow_key}`` passed to
    ``agent.run(task, initial_context=...)`` (only when non-empty)."""
    error_state: str | None = None

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, v: Any) -> Any:
        if not callable(getattr(v, "run", None)):
            raise ValueError("agent must provide a run(task) method")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the agent and map its result back into workflow context."""
        try:
            # Format the task from context
            try:
                task = self.task_template.format(**context)
            except KeyError as e:
                raise WorkflowStepError(
                    step_id=self.step_id,
                    message=f"Missing context key for task template: {e}",
                ) from e

            kwargs: dict[str, Any] = {}
            if self.input_mapping:
                kwargs["initial_context"] = {
                    agent_key: context[wf_key]
                    for agent_key, wf_key in self.input_mapping.items()
                    if wf_key in context
                }

            result = await self._with_timeout(
                _call_user_callable(self.agent.run, task, **kwargs)
            )

            answer: Any
            success: bool
            final_context: Any
            structured: Any
            if isinstance(result, str):
                answer, success, final_context, structured = result, True, {}, None
            else:
                answer = getattr(result, "answer", None)
                success = bool(getattr(result, "success", False))
                final_context = getattr(result, "final_context", None) or {}
                structured = getattr(result, "structured_output", None)

            attrs = {
                "answer": answer,
                "success": success,
                "structured_output": structured,
            }
            data: dict[str, Any] = {
                "agent_answer": answer,
                "agent_success": success,
                f"agent_{self.step_id}_answer": answer,
                f"agent_{self.step_id}_success": success,
            }
            for wf_key, agent_key in self.context_mapping.items():
                if isinstance(final_context, dict) and agent_key in final_context:
                    data[wf_key] = final_context[agent_key]
                elif agent_key in attrs:
                    data[wf_key] = attrs[agent_key]

            if not success:
                return WorkflowStepResult(
                    success=False,
                    data=data,
                    error=f"Agent reported failure: {str(answer or '')[:200]}",
                    next_state=self.error_state,
                    message="Agent step failed: agent reported success=False",
                )

            return WorkflowStepResult.success_result(
                data=data,
                next_state=self.success_state,
                message=f"Agent completed: {str(answer or '')[:100]}",
            )

        except Exception as e:
            logger.error(f"Agent step '{self.step_id}' failed: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_state=self.error_state,
                message=f"Agent step failed: {e!s}",
            )


# ------------------------------------------------------------------
# RetryStep — wrap any step with retry logic
# ------------------------------------------------------------------


class RetryStep(WorkflowStep):
    """Wrap another step with automatic retry on failure.

    Makes ``max_retries + 1`` attempts; the delay before retry ``n`` (1-based)
    is ``backoff_factor * n`` seconds (linear). ``timeout`` applies to each
    attempt. The last failure is returned (or re-raised) unchanged.

    Example::

        inner = api_step("call_api", "Call API", my_api, "done", "error")
        step = RetryStep(
            step_id="retry_call",
            name="Retry API Call",
            step=inner,
            max_retries=3,
            backoff_factor=2.0,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: Any = Field(exclude=True)
    """The inner ``WorkflowStep`` to retry."""
    max_retries: int = 3
    backoff_factor: float = 1.0
    """Delay multiplier between retries (seconds). Delay = backoff_factor * attempt."""

    @field_validator("step")
    @classmethod
    def _validate_step(cls, v: Any) -> Any:
        # Duck typing on purpose: non-WorkflowStep objects with an async
        # execute(context) are supported (see D-010 below).
        if not callable(getattr(v, "execute", None)):
            raise ValueError(
                "RetryStep.step must be a WorkflowStep (or an object with an "
                f"execute(context) method), got {type(v).__name__}"
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be >= 0")
        return v

    @field_validator("backoff_factor")
    @classmethod
    def _validate_backoff_factor(cls, v: float) -> float:
        if v < 0:
            raise ValueError("backoff_factor must be >= 0")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the inner step with retries."""
        last_result: WorkflowStepResult | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result: WorkflowStepResult = await self._with_timeout(
                    self.step.execute(context)
                )
            # DECISION plan-2026-09-12T065608-089d0ec7/D-010
            # Catch bare Exception (not just WorkflowStepError): a custom/
            # non-conforming inner WorkflowStep can raise any exception type,
            # and this must be retried the same way, matching the bare
            # `except Exception` idiom already used by 6+ other step types
            # (AutoTransitionStep, APICallStep, ConditionStep,
            # LLMProcessingStep, ConversationStep, ParallelStep, AgentStep).
            # See decisions.md D-010.
            except Exception:
                # Steps that signal failure by RAISING (ConditionStep,
                # AutoTransitionStep, LLMProcessingStep, _with_timeout) must
                # also be retried, not propagated on the first attempt.
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (attempt + 1)
                    logger.debug(
                        f"Retry step '{self.step_id}': attempt {attempt + 1} "
                        f"raised, retrying in {delay:.1f}s "
                        f"({self.max_retries - attempt - 1} left)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            if result.success:
                return result
            last_result = result
            if attempt < self.max_retries:
                delay = self.backoff_factor * (attempt + 1)
                logger.debug(
                    f"Retry step '{self.step_id}': attempt {attempt + 1} failed, "
                    f"retrying in {delay:.1f}s ({self.max_retries - attempt - 1} left)"
                )
                await asyncio.sleep(delay)
        return last_result  # type: ignore[return-value]


# ------------------------------------------------------------------
# SwitchStep — n-way branching on a context key
# ------------------------------------------------------------------


class SwitchStep(WorkflowStep):
    """Route to different states based on a context key value.

    Unlike ``ConditionStep`` which only supports binary branching,
    ``SwitchStep`` supports arbitrary n-way routing. The context value is
    compared as ``str(value)`` (so ``True`` matches the case ``"True"``);
    a missing key compares as ``""``.

    Example::

        step = SwitchStep(
            step_id="route",
            name="Route by intent",
            key="user_intent",
            cases={"buy": "checkout", "browse": "catalog", "help": "support"},
            default_state="fallback",
        )

    Output keys: ``switch_<step_id>_matched`` (the compared value) and
    ``switch_<step_id>_target`` (the chosen state).
    """

    key: str
    """Context key whose value determines the target state."""
    cases: dict[str, str]
    """Mapping of key values to target state IDs."""
    default_state: str | None = None
    """State to transition to when the key value doesn't match any case.
    ``None`` means no default (the instance FAILS). Use ``""`` for terminal."""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the context key and route to the matching state."""
        value = context.get(self.key)
        value_str = str(value) if value is not None else ""

        target = self.cases.get(value_str, self.default_state)
        if target is None:
            return WorkflowStepResult.failure_result(
                error=f"No matching case for {self.key}={value_str!r} and no default_state",
                next_state="",
                message=f"Switch step '{self.step_id}' has no route for value {value_str!r}",
            )

        return WorkflowStepResult.success_result(
            data={
                f"switch_{self.step_id}_matched": value_str,
                f"switch_{self.step_id}_target": target,
            },
            next_state=target,
            message=f"Routed to '{target}' (key={self.key}, value={value_str!r})",
        )


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _error_text(exc: BaseException) -> str:
    """Non-empty error text for an exception (``str()`` can be empty)."""
    text = str(exc)
    return text if text else type(exc).__name__


def _lookup_path(source: Any, path: str) -> tuple[bool, Any]:
    """Resolve ``path`` in ``source``: ``""`` is the whole value, an exact
    dict key wins, otherwise a dotted path walks nested dicts."""
    if path == "":
        return True, source
    if not isinstance(source, dict):
        return False, None
    if path in source:
        return True, source[path]
    current: Any = source
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current
