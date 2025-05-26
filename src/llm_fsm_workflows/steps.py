"""
Workflow step implementations for the LLM-FSM Workflow System.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field, ConfigDict

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from llm_fsm.logging import logger
from .exceptions import WorkflowStepError
from .models import WorkflowStepResult, WaitEventConfig

# --------------------------------------------------------------


class WorkflowStep(BaseModel, ABC):
    """Base class for workflow steps."""
    step_id: str
    name: str
    description: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Execute the step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    def _safe_execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute a function, handling both sync and async."""
        try:
            if inspect.iscoroutinefunction(func):
                return asyncio.create_task(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)
        except Exception as e:
            raise WorkflowStepError(
                step_id=self.step_id,
                message=f"Function execution failed: {str(e)}",
                cause=e
            )


class AutoTransitionStep(WorkflowStep):
    """A step that automatically transitions to the next state."""
    next_state: str
    action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Execute the step and transition automatically."""
        try:
            data = {}
            if self.action:
                if inspect.iscoroutinefunction(self.action):
                    data = await self.action(context)
                else:
                    data = self.action(context)

            return WorkflowStepResult.success_result(
                data=data,
                next_state=self.next_state,
                message=f"Auto-transitioned to {self.next_state}"
            )
        except Exception as e:
            logger.error(f"Error in auto transition step {self.step_id}: {str(e)}")
            raise WorkflowStepError(
                step_id=self.step_id,
                message="Auto-transition failed",
                cause=e
            )


class APICallStep(WorkflowStep):
    """A step that calls an external API."""
    api_function: Callable
    success_state: str
    failure_state: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
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
                message=f"API call successful, transitioning to {self.success_state}"
            )
        except Exception as e:
            logger.error(f"Error in API call step {self.step_id}: {str(e)}")
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=self.failure_state,
                message=f"API call failed, transitioning to {self.failure_state}"
            )

    def _map_input_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Map context keys to API parameters."""
        params = {}
        for api_param, context_key in self.input_mapping.items():
            if context_key in context:
                params[api_param] = context[context_key]
        return params

    async def _call_api(self, params: Dict[str, Any]) -> Any:
        """Call the API function."""
        if inspect.iscoroutinefunction(self.api_function):
            return await self.api_function(**params)
        else:
            return self.api_function(**params)

    def _map_output_data(self, api_result: Any) -> Dict[str, Any]:
        """Map API response to context keys."""
        output_data = {}
        if isinstance(api_result, dict):
            for context_key, result_key in self.output_mapping.items():
                if result_key in api_result:
                    output_data[context_key] = api_result[result_key]
        return output_data


class ConditionStep(WorkflowStep):
    """A step that evaluates a condition and transitions accordingly."""
    condition: Callable[[Dict[str, Any]], bool]
    true_state: str
    false_state: str

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the condition and determine the next state."""
        try:
            # Evaluate the condition
            if inspect.iscoroutinefunction(self.condition):
                result = await self.condition(context)
            else:
                result = self.condition(context)

            # Determine the next state
            next_state = self.true_state if result else self.false_state

            return WorkflowStepResult.success_result(
                next_state=next_state,
                message=f"Condition evaluated to {result}, transitioning to {next_state}",
                data={"condition_result": result}
            )
        except Exception as e:
            logger.error(f"Error in condition step {self.step_id}: {str(e)}")
            raise WorkflowStepError(
                step_id=self.step_id,
                message="Condition evaluation failed",
                cause=e
            )


class LLMProcessingStep(WorkflowStep):
    """A step that processes data using an LLM."""
    llm_interface: Any
    prompt_template: str
    context_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    next_state: str
    error_state: Optional[str] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
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
                message=f"LLM processing successful, transitioning to {self.next_state}"
            )
        except Exception as e:
            logger.error(f"Error in LLM processing step {self.step_id}: {str(e)}")
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"LLM processing failed, transitioning to {next_state}"
            )

    def _prepare_prompt(self, context: Dict[str, Any]) -> str:
        """Prepare the prompt from the template and context."""
        prompt_vars = {}
        for prompt_var, context_key in self.context_mapping.items():
            if context_key in context:
                prompt_vars[prompt_var] = context[context_key]
        return self.prompt_template.format(**prompt_vars)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface."""
        return await self.llm_interface.generate(prompt)

    def _process_llm_response(self, response: str) -> Dict[str, Any]:
        """Process the LLM response and extract data."""
        output_data = {}
        # Simplified parsing - implement proper parsing based on needs
        for context_key, pattern in self.output_mapping.items():
            # Placeholder implementation
            output_data[context_key] = response
        return output_data


class WaitForEventStep(WorkflowStep):
    """A step that waits for an external event."""
    config: WaitEventConfig

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Set up the workflow to wait for an event."""
        waiting_info = {
            "waiting_for_event": True,
            "event_type": self.config.event_type,
            "timeout_seconds": self.config.timeout_seconds,
            "timeout_state": self.config.timeout_state,
            "success_state": self.config.success_state,
            "event_mapping": self.config.event_mapping,
            "waiting_since": datetime.now().isoformat()
        }

        return WorkflowStepResult.success_result(
            data={"_waiting_info": waiting_info},
            message=f"Waiting for event of type {self.config.event_type}"
        )


class TimerStep(WorkflowStep):
    """A step that waits for a specified time before transitioning."""
    delay_seconds: int
    next_state: str

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Set up a timer to transition after a delay."""
        timer_info = {
            "waiting_for_timer": True,
            "delay_seconds": self.delay_seconds,
            "next_state": self.next_state,
            "timer_start": datetime.now().isoformat(),
            "timer_end": (datetime.now() + timedelta(seconds=self.delay_seconds)).isoformat()
        }

        return WorkflowStepResult.success_result(
            data={"_timer_info": timer_info},
            message=f"Timer set for {self.delay_seconds} seconds, will transition to {self.next_state}"
        )


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel."""
    steps: List[WorkflowStep]
    next_state: str
    error_state: Optional[str] = None
    aggregation_function: Optional[Callable[[List[WorkflowStepResult]], Dict[str, Any]]] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """Execute multiple steps in parallel and aggregate results."""
        try:
            # Execute all steps in parallel
            results = await self._execute_parallel_steps(context)

            # Check for errors
            errors = self._collect_errors(results)
            if errors and self.error_state:
                return WorkflowStepResult.failure_result(
                    error="; ".join(errors),
                    next_state=self.error_state,
                    message=f"Parallel step errors, transitioning to {self.error_state}"
                )

            # Aggregate results
            aggregated_data = self._aggregate_results(results)

            success = all(r.success for r in results if isinstance(r, WorkflowStepResult))
            return WorkflowStepResult.success_result(
                data=aggregated_data,
                next_state=self.next_state,
                message=f"Parallel step completed ({'successfully' if success else 'with errors'})"
            )
        except Exception as e:
            logger.error(f"Error in parallel step {self.step_id}: {str(e)}")
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult.failure_result(
                error=str(e),
                next_state=next_state,
                message=f"Parallel step failed, transitioning to {next_state}"
            )

    async def _execute_parallel_steps(self, context: Dict[str, Any]) -> List[WorkflowStepResult]:
        """Execute all steps in parallel."""
        tasks = [step.execute(context) for step in self.steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=str(result),
                        message=f"Parallel step {i} failed: {str(result)}"
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _collect_errors(self, results: List[WorkflowStepResult]) -> List[str]:
        """Collect error messages from failed results."""
        return [r.error for r in results if not r.success and r.error]

    def _aggregate_results(self, results: List[WorkflowStepResult]) -> Dict[str, Any]:
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