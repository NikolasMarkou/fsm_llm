"""The step that runs an executable by id: ``ExecutableStep`` (roadmap E5.2, plan ace7a8c8 step 26, X5.2).

``ExecutableStep`` reads ``registry.get(target)``, maps workflow keys to the executable's inputs, awaits
``arun`` under a ``RunContext`` and maps the ``RunResult`` back into the workflow context. It owns no
conversation lifecycle: the executable does (``FsmBodyExecutable`` ends its conversation in ``finally``), so this
module imports no ``API``. It reads the executable side only through ``fsm_llm.engine.executables`` and
``fsm_llm.engine.context``, never ``agents`` or ``reasoning`` (a built-in arrives through the registry).

Part of the package (plan ace7a8c8 step 26; the type tag in a serialized workflow is the class name).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import Field, field_validator, model_validator

from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH, has_internal_prefix
from fsm_llm.core.codes import Status
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import RunContext
from fsm_llm.logging import logger

from ..constants import CALLABLES_NAMESPACE
from ..exceptions import WorkflowStepError
from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _RUN_CONTEXT, _RUN_DEPTH, _error_text


class ExecutableStep(WorkflowStep):
    """Run an executable (a document or a built-in) by id as a workflow step.

    ``registry`` is an ``fsm_llm.engine.executables.ExecutableRegistry`` (anything with ``get(id)``) and
    ``target`` the id of the executable in it. ``inputs`` maps ``{executable_input: workflow_key}`` (a workflow
    key the context lacks is omitted, and the executable judges its own contract); ``returns`` maps
    ``{workflow_key: executable_output}`` and copies only outputs that are not ``None``. A ``returns`` entry
    that names an internal-prefixed workflow key is refused (876e7164/D-003: the engine keeps a whitelist of
    internal keys, a step must not be able to name one). ``callables`` is the ``NameRegistry`` (namespace
    ``callables``) that supplies the host objects of a code-backed built-in through ``RunContext.callables``;
    an executable whose ``requires.callables`` is not empty is refused when the step has none. ``max_steps`` and
    ``max_seconds`` are the budgets of the ONE run (``RunContext``); ``timeout_seconds`` is the step's own time
    limit and cancels the run in flight.

    The model of the run comes from the executable (an FSM body takes ``LLM_MODEL``, as ``API`` does; no model
    is a failure of the step, never a silent default). A ``failed`` or ``aborted`` result (any status that is not
    ``ok``) is a failure that routes to ``error_step_id`` and NEVER to the success route (5d1e7a3b/D-003); with
    no ``error_step_id`` the instance FAILS.

    Output keys: ``executable_status``, ``executable_identity`` (``{"template_identity", "instance_identity"}``)
    and ``executable_answer`` (latest executable step) and ``executable_<step_id>_status``/``_identity``/``_answer``
    (per step), plus the ``returns`` keys on success.
    """

    target: str = Field(min_length=1)
    """The id of the executable in ``registry``."""
    registry: Any = Field(exclude=True)
    """An ``ExecutableRegistry`` (or anything with ``get(id)``); named through ``callables`` in a document."""
    callables: Any | None = Field(default=None, exclude=True)
    """A ``NameRegistry`` of namespace ``callables``: the host objects of a code-backed executable."""
    inputs: dict[str, str] = Field(default_factory=dict)
    """INPUT mapping ``{executable_input: workflow_key}``."""
    returns: dict[str, str] = Field(default_factory=dict)
    """OUTPUT mapping ``{workflow_key: executable_output}``; non-None outputs only."""
    success_step_id: str = ""
    error_step_id: str | None = None
    max_steps: int | None = Field(default=None, ge=1)
    max_seconds: float | None = Field(default=None, gt=0)

    @field_validator("registry")
    @classmethod
    def _validate_registry(cls, v: Any) -> Any:
        if not callable(getattr(v, "get", None)):
            raise ValueError(
                "registry must provide get(id), as an ExecutableRegistry does"
            )
        return v

    @field_validator("callables")
    @classmethod
    def _validate_callables(cls, v: Any) -> Any:
        if v is not None and not (
            isinstance(v, NameRegistry) and v.namespace == CALLABLES_NAMESPACE
        ):
            raise ValueError(
                f"callables must be a NameRegistry of namespace {CALLABLES_NAMESPACE!r}"
            )
        return v

    @field_validator("returns")
    @classmethod
    def _validate_returns(cls, v: dict[str, str]) -> dict[str, str]:
        _refuse_internal_keys(v)
        return v

    @model_validator(mode="after")
    def _validate_budgets(self) -> ExecutableStep:
        """Refuse a budget the executable names in ``refused_budgets``, when the registry has it.

        A target the registry does not serve yet (or a registry that raises) is judged at run time,
        which still refuses the same budgets (an assignment skips this validator).
        """
        try:
            refused = tuple(
                getattr(self.registry.get(self.target), "refused_budgets", ())
            )
        except Exception:
            return self
        for budget in refused:
            if getattr(self, budget) is not None:
                raise ValueError(
                    f"executable {self.target!r} takes no {budget} (it refuses that budget "
                    "of a run context): leave it unset"
                )
        return self

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the executable and map its result back into the workflow context."""
        try:
            # DECISION plan-2026-11-25T195353-7d9c0955/D-012
            # Composed workflow runs nest at most DEFAULT_MAX_STACK_DEPTH deep (`run_context_scope` counts
            # them). Do NOT drop this check (a self-targeting workflow or a ParallelStep cycle recurses
            # until the interpreter fails), repeat the literal, or name a document value in the refusal.
            if _RUN_DEPTH.get() >= DEFAULT_MAX_STACK_DEPTH:
                raise WorkflowStepError(
                    f"composed workflow runs nest at most {DEFAULT_MAX_STACK_DEPTH} deep"
                )
            # the field validator does not run on an assignment: check again at run time
            _refuse_internal_keys(self.returns)
            executable = self.registry.get(self.target)
            needed = list(executable.requires.callables)
            parent = _RUN_CONTEXT.get()
            callables = self.callables
            if callables is None and parent is not None:
                callables = parent.callables
            if needed and callables is None:
                raise WorkflowStepError(
                    f"executable {self.target!r} needs the callables {needed} and this step "
                    "names no callables registry",
                    step_id=self.step_id,
                )
            inputs = {
                name: context[key]
                for name, key in self.inputs.items()
                if key in context
            }
            steps, seconds = self.max_steps, self.max_seconds
            # a limit above the parent's makes derive raise: no clamp (O-4)
            run_context = dataclasses.replace(
                RunContext(max_steps=steps, max_seconds=seconds)
                if parent is None
                else parent.derive(
                    parent=self.step_id, max_steps=steps, max_seconds=seconds
                ),
                callables=callables,
            )
            result = await self._with_timeout(
                executable.arun(inputs, context=run_context)
            )
        except Exception as e:
            logger.error(f"Executable step '{self.step_id}' failed: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
                message=f"Executable step failed: {e!s}",
            )

        identity = {
            "template_identity": result.identity.template_identity,
            "instance_identity": result.identity.instance_identity,
        }
        data: dict[str, Any] = {}
        for prefix in ("executable", f"executable_{self.step_id}"):
            data[f"{prefix}_status"] = result.status.value
            data[f"{prefix}_identity"] = identity
            data[f"{prefix}_answer"] = result.answer
        if result.status is not Status.OK:
            reason = f" ({result.reason.value})" if result.reason is not None else ""
            ended = f"{result.status.value}{reason}"
            return WorkflowStepResult(
                success=False,
                data=data,
                error=f"Executable {self.target!r} ended {ended}",
                next_step_id=self.error_step_id,
                message=f"Executable step failed: {ended}",
            )
        for workflow_key, output_key in self.returns.items():
            value = result.outputs.get(output_key)
            if value is not None:
                data[workflow_key] = value
        return WorkflowStepResult.success_result(
            data=data,
            next_step_id=self.success_step_id,
            message=f"Executable {self.target!r} completed",
        )


def _refuse_internal_keys(returns: dict[str, str]) -> None:
    internal = sorted(key for key in returns if has_internal_prefix(key))
    if internal:
        raise ValueError(
            f"returns names the internal key(s) {internal}: a workflow key with an internal "
            "prefix is the engine's, never a step's"
        )
