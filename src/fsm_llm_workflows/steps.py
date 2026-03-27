from __future__ import annotations

"""
Workflow step implementations for the FSM-LLM Workflow System.
"""

import asyncio
import copy
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from fsm_llm.logging import logger

from .exceptions import WorkflowStepError
from .models import WaitEventConfig, WorkflowStepResult

# --------------------------------------------------------------


class WorkflowStep(BaseModel, ABC):
    """Base class for workflow steps."""

    step_id: str
    name: str
    description: str = ""
    timeout: float | None = None
    """Maximum seconds this step may run. ``None`` disables timeout (default).
    Use ``constants.DEFAULT_STEP_TIMEOUT`` (120 s) for safety."""

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
    """A step that automatically transitions to the next state."""

    next_state: str
    action: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the step and transition automatically."""
        try:
            data = {}
            if self.action:
                if inspect.iscoroutinefunction(self.action):
                    data = await self._with_timeout(self.action(context))
                else:
                    # Wrap sync action to respect timeout
                    loop = asyncio.get_running_loop()
                    coro = loop.run_in_executor(None, self.action, context)
                    data = await self._with_timeout(coro)

            return WorkflowStepResult.success_result(
                data=data,
                next_state=self.next_state,
                message=f"Auto-transitioned to {self.next_state}",
            )
        except Exception as e:
            logger.error(f"Error in auto transition step {self.step_id}: {e!s}")
            raise WorkflowStepError(
                step_id=self.step_id, message="Auto-transition failed", cause=e
            ) from e


class APICallStep(WorkflowStep):
    """A step that calls an external API."""

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
                error=str(e),
                next_state=self.failure_state,
                message=f"API call failed, transitioning to {self.failure_state}",
            )

    def _map_input_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        """Map context keys to API parameters."""
        params = {}
        for api_param, context_key in self.input_mapping.items():
            if context_key in context:
                params[api_param] = context[context_key]
        return params

    async def _call_api(self, params: dict[str, Any]) -> Any:
        """Call the API function."""
        if inspect.iscoroutinefunction(self.api_function):
            return await self._with_timeout(self.api_function(**params))
        else:
            loop = asyncio.get_running_loop()
            coro = loop.run_in_executor(None, lambda: self.api_function(**params))
            return await self._with_timeout(coro)

    def _map_output_data(self, api_result: Any) -> dict[str, Any]:
        """Map API response to context keys."""
        output_data = {}
        if isinstance(api_result, dict):
            for context_key, result_key in self.output_mapping.items():
                if result_key in api_result:
                    output_data[context_key] = api_result[result_key]
        return output_data


class ConditionStep(WorkflowStep):
    """A step that evaluates a condition and transitions accordingly."""

    condition: Callable[[dict[str, Any]], bool]
    true_state: str
    false_state: str

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the condition and determine the next state."""
        try:
            # Evaluate the condition
            if inspect.iscoroutinefunction(self.condition):
                result = await self._with_timeout(self.condition(context))
            else:
                result = self.condition(context)

            # Determine the next state
            next_state = self.true_state if result else self.false_state

            return WorkflowStepResult.success_result(
                next_state=next_state,
                message=f"Condition evaluated to {result}, transitioning to {next_state}",
                data={"condition_result": result},
            )
        except Exception as e:
            logger.error(f"Error in condition step {self.step_id}: {e!s}")
            raise WorkflowStepError(
                step_id=self.step_id, message="Condition evaluation failed", cause=e
            ) from e


class LLMProcessingStep(WorkflowStep):
    """A step that processes data using an LLM."""

    llm_interface: Any
    prompt_template: str
    context_mapping: dict[str, str]
    output_mapping: dict[str, str]
    next_state: str
    error_state: str | None = None

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
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"LLM processing failed, transitioning to {next_state}",
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

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface."""
        result: str = await self._with_timeout(self.llm_interface.generate(prompt))
        return result

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
                    output_data[context_key] = response
            else:
                output_data[context_key] = response
        return output_data


class WaitForEventStep(WorkflowStep):
    """A step that waits for an external event."""

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
            "waiting_since": datetime.now(timezone.utc).isoformat(),
        }

        return WorkflowStepResult.success_result(
            data={"_waiting_info": waiting_info},
            message=f"Waiting for event of type {self.config.event_type}",
        )


class TimerStep(WorkflowStep):
    """A step that waits for a specified time before transitioning."""

    delay_seconds: int
    next_state: str

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

    The conversation runs to completion (until terminal state), and the
    collected context data is returned as the step result.
    """

    fsm_file: str | None = None
    fsm_definition: dict[str, Any] | None = None
    model: str | None = None
    initial_context: dict[str, str] = Field(default_factory=dict)
    context_mapping: dict[str, str] = Field(default_factory=dict)
    success_state: str = ""
    error_state: str | None = None
    max_turns: int = 20
    conversation_timeout: float | None = None
    auto_messages: list[str] = Field(default_factory=list)

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute an FSM conversation and return collected data."""
        try:
            coro = self._run_conversation(context)
            if self.conversation_timeout is not None:
                return await asyncio.wait_for(coro, timeout=self.conversation_timeout)
            return await coro
        except asyncio.TimeoutError:
            logger.error(
                f"ConversationStep [{self.step_id}] timed out after {self.conversation_timeout}s"
            )
            next_state = self.error_state if self.error_state else self.success_state
            return WorkflowStepResult.failure_result(
                error=f"Conversation timed out after {self.conversation_timeout}s",
                next_state=next_state,
                message=f"Conversation timed out, transitioning to {next_state}",
            )
        except WorkflowStepError:
            raise
        except Exception as e:
            logger.error(f"Error in conversation step {self.step_id}: {e!s}")
            next_state = self.error_state if self.error_state else self.success_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"Conversation failed: {e!s}",
            )

    async def _run_conversation(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the FSM conversation loop."""
        from fsm_llm import API

        # Build initial context from workflow context using mapping
        conv_context: dict[str, Any] = {}
        for conv_key, workflow_key in self.initial_context.items():
            if workflow_key in context:
                conv_context[conv_key] = context[workflow_key]

        # Create API instance
        if self.fsm_definition:
            fsm = API.from_definition(self.fsm_definition, model=self.model)
        elif self.fsm_file:
            fsm = API.from_file(self.fsm_file, model=self.model)
        else:
            raise WorkflowStepError(
                step_id=self.step_id,
                message="ConversationStep requires either fsm_file or fsm_definition",
            )

        # Start conversation
        conv_id, response = fsm.start_conversation(initial_context=conv_context)
        logger.info(
            f"ConversationStep [{self.step_id}] started conversation: {response[:100]}"
        )

        try:
            # Drive the conversation with auto_messages
            turn = 0
            for message in self.auto_messages:
                if fsm.has_conversation_ended(conv_id) or turn >= self.max_turns:
                    break
                response = fsm.converse(user_message=message, conversation_id=conv_id)
                logger.debug(
                    f"ConversationStep [{self.step_id}] turn {turn}: {response[:100]}"
                )
                turn += 1

            # Collect results
            collected_data = fsm.get_data(conv_id)
        finally:
            fsm.end_conversation(conv_id)

        # Map collected data back to workflow context
        output_data: dict[str, Any] = {}
        for workflow_key, conv_key in self.context_mapping.items():
            if conv_key in collected_data:
                output_data[workflow_key] = collected_data[conv_key]
        # Also include raw collected data under a namespaced key
        output_data[f"_conversation_{self.step_id}_data"] = collected_data

        return WorkflowStepResult.success_result(
            data=output_data,
            next_state=self.success_state,
            message=f"Conversation completed in {turn} turns",
        )


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel."""

    steps: list[WorkflowStep]
    next_state: str
    error_state: str | None = None
    aggregation_function: (
        Callable[[list[WorkflowStepResult]], dict[str, Any]] | None
    ) = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute multiple steps in parallel and aggregate results."""
        try:
            # Execute all steps in parallel
            results = await self._execute_parallel_steps(context)

            # Check for errors
            errors = self._collect_errors(results)
            if errors:
                error_msg = "; ".join(errors)
                next_state = self.error_state if self.error_state else self.next_state
                if not self.error_state:
                    logger.warning(
                        f"Parallel step '{self.step_id}' had {len(errors)} error(s) "
                        f"but no error_state configured; transitioning to "
                        f"next_state='{self.next_state}' despite failure"
                    )
                return WorkflowStepResult.failure_result(
                    error=error_msg,
                    next_state=next_state,
                    message=f"Parallel step had {len(errors)} error(s), transitioning to {next_state}",
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
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"Parallel step failed, transitioning to {next_state}",
            )

    async def _execute_parallel_steps(
        self, context: dict[str, Any]
    ) -> list[WorkflowStepResult]:
        """Execute all steps in parallel with isolated context copies."""
        if len(self.steps) > 10:
            logger.warning(
                f"ParallelStep [{self.step_id}] deep-copying context for "
                f"{len(self.steps)} parallel steps — consider reducing parallelism "
                "if memory usage is a concern"
            )
        tasks = [step.execute(copy.deepcopy(context)) for step in self.steps]
        results = await self._with_timeout(
            asyncio.gather(*tasks, return_exceptions=True)
        )

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=str(result),
                        message=f"Parallel step {i} failed: {result!s}",
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _collect_errors(self, results: list[WorkflowStepResult]) -> list[str]:
        """Collect error messages from failed results."""
        return [r.error for r in results if not r.success and r.error]

    def _aggregate_results(self, results: list[WorkflowStepResult]) -> dict[str, Any]:
        """Aggregate results from parallel steps."""
        if self.aggregation_function:
            return self.aggregation_function(results)

        # Default aggregation
        aggregated_data = {}
        for i, result in enumerate(results):
            if result.data:
                prefix = f"step_{i}_"
                for key, value in result.data.items():
                    aggregated_data[f"{prefix}{key}"] = value

        return aggregated_data


# ------------------------------------------------------------------
# AgentStep — run an fsm_llm_agents BaseAgent as a workflow step
# ------------------------------------------------------------------


class AgentStep(WorkflowStep):
    """Execute an FSM-LLM agent as a workflow step.

    Wraps any ``BaseAgent.run()`` call so agents can participate in
    workflows alongside other step types.

    Example::

        from fsm_llm_agents import ReactAgent
        agent = ReactAgent(tools=registry)
        step = AgentStep(
            step_id="research",
            name="Research Step",
            agent=agent,
            task_template="Research {topic}",
            success_state="analyze",
            context_mapping={"findings": "answer"},
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any = Field(exclude=True)
    """A ``BaseAgent`` instance (or any object with a ``run(task)`` method)."""
    task_template: str = "{task}"
    """Format string for the agent task.  Placeholders are filled from context."""
    success_state: str = ""
    context_mapping: dict[str, str] = Field(default_factory=dict)
    """Maps workflow context keys to agent result keys.
    ``{workflow_key: agent_result_key}``."""
    error_state: str | None = None

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
                )

            # Run agent (sync — use executor to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.agent.run, task)

            # Map agent result to workflow context
            data: dict[str, Any] = {
                "agent_answer": result.answer,
                "agent_success": result.success,
            }
            if result.final_context:
                for wf_key, agent_key in self.context_mapping.items():
                    if agent_key in result.final_context:
                        data[wf_key] = result.final_context[agent_key]
                    elif agent_key == "answer":
                        data[wf_key] = result.answer

            next_state = self.success_state
            return WorkflowStepResult.success_result(
                data=data,
                next_state=next_state,
                message=f"Agent completed: {result.answer[:100]}",
            )

        except Exception as e:
            logger.error(f"Agent step '{self.step_id}' failed: {e!s}")
            next_state = self.error_state if self.error_state else self.success_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"Agent step failed: {e!s}",
            )


# ------------------------------------------------------------------
# RetryStep — wrap any step with retry logic
# ------------------------------------------------------------------


class RetryStep(WorkflowStep):
    """Wrap another step with automatic retry on failure.

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

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the inner step with retries."""
        last_result = None
        for attempt in range(self.max_retries + 1):
            result = await self.step.execute(context)
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
    ``SwitchStep`` supports arbitrary n-way routing.

    Example::

        step = SwitchStep(
            step_id="route",
            name="Route by intent",
            key="user_intent",
            cases={"buy": "checkout", "browse": "catalog", "help": "support"},
            default_state="fallback",
        )
    """

    key: str
    """Context key whose value determines the target state."""
    cases: dict[str, str]
    """Mapping of key values to target state IDs."""
    default_state: str = ""
    """State to transition to when the key value doesn't match any case."""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the context key and route to the matching state."""
        value = context.get(self.key)
        value_str = str(value) if value is not None else ""

        target = self.cases.get(value_str, self.default_state)
        if not target:
            return WorkflowStepResult.failure_result(
                error=f"No matching case for {self.key}={value_str!r} and no default_state",
                next_state="",
                message=f"Switch step '{self.step_id}' has no route for value {value_str!r}",
            )

        return WorkflowStepResult.success_result(
            data={"_switch_matched": value_str, "_switch_target": target},
            next_state=target,
            message=f"Routed to '{target}' (key={self.key}, value={value_str!r})",
        )
