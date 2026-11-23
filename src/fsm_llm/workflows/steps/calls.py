"""The steps that call out: ``APIStep`` (a user function) and ``LLMStep`` (a model).

``_lookup_path`` resolves an ``output_mapping`` result key against an API result; only ``APIStep`` reads it.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator

from fsm_llm.core.definitions import USER_MESSAGE_MAX_CHARS, ResponseGenerationRequest
from fsm_llm.logging import logger

from ..exceptions import WorkflowStepError
from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _call_user_callable, _error_text, _is_async_callable


class APIStep(WorkflowStep):
    """A step that calls an external API.

    ``input_mapping`` is ``{api_param: context_key}``. ``output_mapping`` is
    ``{context_key: result_key}``; ``result_key`` may be a dotted path into
    nested dicts (``"user.name"``) or ``""`` for the whole result (which also
    captures non-dict results such as lists or scalars).
    """

    api_function: Callable[..., Any]
    success_step_id: str
    failure_step_id: str
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
                next_step_id=self.success_step_id,
                message=f"API call successful, transitioning to {self.success_step_id}",
            )
        except Exception as e:
            logger.error(f"Error in API call step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.failure_step_id,
                message=f"API call failed, transitioning to {self.failure_step_id}",
            )

    def _map_input_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        """Map context keys to API parameters."""
        params = {}
        for api_param, context_key in self.input_mapping.items():
            if context_key in context:
                params[api_param] = context[context_key]
            else:
                logger.debug(
                    f"APIStep [{self.step_id}]: input_mapping key "
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
                    f"APIStep [{self.step_id}]: output_mapping key "
                    f"'{result_key}' not in API result; '{context_key}' not set"
                )
        return output_data


class LLMStep(WorkflowStep):
    """A step that processes data using an LLM.

    ``llm_interface`` may be a core ``fsm_llm.LLMInterface`` (its
    ``agenerate_response`` is awaited on the loop, with ``system_prompt`` and the rendered
    prompt as the user message; an interface with no async twin, a duck-typed object, runs its
    ``generate_response`` through the step executor) or any object with a ``generate(prompt)``
    method, sync or async, returning a string.

    ``context_mapping`` is the INPUT mapping ``{prompt_var: context_key}``
    used to fill ``prompt_template`` (``str.format`` syntax: write literal
    braces as ``{{`` and ``}}``). ``output_mapping`` is
    ``{context_key: regex}``: group 1 (or the whole match) is stored; an
    empty regex stores the whole response; a regex that does not match
    leaves the key unset. On failure the step routes to ``error_step_id``, or
    FAILS the instance when none is set.
    """

    llm_interface: Any
    prompt_template: str
    context_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    next_step_id: str
    error_step_id: str | None = None
    system_prompt: str = (
        "You are a precise assistant. Follow the instructions in the user "
        "message and answer with exactly what they ask for."
    )

    @field_validator("llm_interface")
    @classmethod
    def _validate_llm_interface(cls, v: Any) -> Any:
        if not (
            callable(getattr(v, "generate", None))
            or callable(getattr(v, "generate_response", None))
        ):
            raise ValueError(
                "llm_interface must provide generate(prompt) or "
                "generate_response(ResponseGenerationRequest)"
            )
        return v

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
                next_step_id=self.next_step_id,
                message=f"LLM processing successful, transitioning to {self.next_step_id}",
            )
        except Exception as e:
            logger.error(f"Error in LLM processing step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
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
                f"Prompt template variable {e} not found in context mapping",
                step_id=self.step_id,
                cause=e,
            ) from e
        except (ValueError, IndexError) as e:
            raise WorkflowStepError(
                f"Prompt template is not a valid format string ({e}); "
                "write literal braces as '{{' and '}}'",
                step_id=self.step_id,
                cause=e,
            ) from e

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM interface and return the response text."""
        llm = self.llm_interface
        generate = getattr(llm, "generate", None)
        if callable(generate):
            raw = await self._with_timeout(_call_user_callable(generate, prompt))
        else:
            system_prompt = self.system_prompt or "Follow the user's instructions."
            if len(prompt) > USER_MESSAGE_MAX_CHARS:
                # ResponseGenerationRequest caps user_message; the system
                # prompt allows much more, so carry a long prompt there.
                request = ResponseGenerationRequest.checked(
                    system_prompt=f"{system_prompt}\n\n{prompt}",
                    user_message="Follow the instructions above.",
                )
            else:
                request = ResponseGenerationRequest.checked(
                    system_prompt=system_prompt, user_message=prompt
                )
            agenerate: Any = getattr(llm, "agenerate_response", None)
            if _is_async_callable(agenerate):
                # a real LLMInterface: awaited on the loop, cancelled at the time limit
                call = agenerate(request)
            else:
                call = _call_user_callable(llm.generate_response, request)
            raw = await self._with_timeout(call)
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
                        f"LLMStep [{self.step_id}]: pattern for "
                        f"'{context_key}' did not match the response; key not set"
                    )
            else:
                output_data[context_key] = response
        return output_data


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
