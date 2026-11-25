"""The body of a ``workflow`` executable: step documents as data, and the names they hold.

A workflow document (roadmap E5.1, plan 7d9c0955) is an envelope of kind ``workflow`` whose
``body`` is ``{initial_step_id, steps: {step_id: {type, ...fields}}}``: the ``serialize(callables=)``
form of a ``WorkflowDefinition`` minus ``workflow_id``, ``name``, ``description`` and ``metadata``,
which the envelope header carries (``workflow_id`` is ``envelope.id``). The steps stay UNTYPED dicts
here: core is below ``fsm_llm.workflows``, so the step classes, the loader and V2 (structure) run in
the workflows adapter, and V1 judges shape and size only.

This leaf module holds what core and ``workflows`` both need and must not copy: the two tables of
step fields that hold a NAME instead of an object (``STEP_CALLABLE_EXCLUDE_FIELDS``,
``NAMED_OBJECT_FIELDS``: ``workflows.definitions`` and ``workflows.loader`` import them), the one
iterative walker over the step documents and the name collector built on it.

Interface contract:

- ``walk_steps(steps) -> Iterator[(step, level, owner)]``: every step document under the mapping
  *steps*, the nested ``step`` (a mapping) and ``steps`` (a list of mappings) of a step included, in
  document order. ``level`` is 1 for a top-level step, ``owner`` the key of the top-level step it sits
  under. Iterative (no recursion), lazy (a caller stops it), skips what is no mapping; it follows
  whatever the dict holds, so a caller that was not given a ``WorkflowBody`` bounds it.
- ``workflow_names(steps) -> WorkflowNames``: the names held by the fields of ``NAMED_OBJECT_FIELDS``
  (``callables``) and the ``target`` of every ``ExecutableStep`` (``executables``), each sorted and
  unique. Reads the BODY, never a claim of the header. Non-text values are ignored.
- ``WorkflowBody``: V1. A ``ValueError`` (pydantic wraps it) names keys and levels, never a value of a
  step. Imports no ``executable`` module. Lazy: nothing at ``import fsm_llm`` loads it, and
  ``fsm_llm.workflows`` imports only the two tables.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .limits import EXECUTABLE_ID_MAX_LENGTH

__all__ = [
    "MAX_STEP_NESTING",
    "MAX_WORKFLOW_STEPS",
    "NAMED_OBJECT_FIELDS",
    "STEP_CALLABLE_EXCLUDE_FIELDS",
    "WORKFLOW_FIELD_VERSIONS",
    "WorkflowBody",
    "WorkflowNames",
    "walk_steps",
    "workflow_names",
]

#: The callable/object fields no step's serialized form can carry (functions are not JSON). The
#: default ``serialize()`` strips them; ``serialize(callables=)`` writes their registered NAME.
STEP_CALLABLE_EXCLUDE_FIELDS = frozenset(
    {"action", "api_function", "condition", "aggregation_function", "llm_interface"}
)

#: The fields a document names instead of holding: the five above plus a step's ``agent`` and an
#: ``ExecutableStep``'s ``registry`` and ``callables`` (``Field(exclude=True)``, so the default dump
#: never carried them). Read by ``serialize(callables=)``, by ``load_workflow`` and by V3 and V7.
NAMED_OBJECT_FIELDS = STEP_CALLABLE_EXCLUDE_FIELDS | {"agent", "registry", "callables"}

#: Most step documents a body holds, counted through every nesting, and the deepest nesting of
#: ``step`` and ``steps`` (a top-level step is level 1).
MAX_WORKFLOW_STEPS = 500
MAX_STEP_NESTING = 16

_STEP_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_MAX_PROBLEMS = 10  # lines one body reports
_SHOWN_CHARS = 60
_StepId = Annotated[str, Field(min_length=1, max_length=EXECUTABLE_ID_MAX_LENGTH)]


class WorkflowNames(NamedTuple):
    """The names a body uses: ``callables`` (named object fields) and ``executables`` (targets)."""

    callables: tuple[str, ...]
    executables: tuple[str, ...]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _shown(text: object) -> str:
    return repr(str(text)[:_SHOWN_CHARS])


def walk_steps(
    steps: Mapping[str, Any],
) -> Iterator[tuple[Mapping[str, Any], int, str]]:
    """Every step document under *steps* with its level and top-level owner (see the module text)."""
    stack: list[tuple[Any, int, str]] = [
        (step, 1, str(key)) for key, step in reversed(list(steps.items()))
    ]
    while stack:
        step, level, owner = stack.pop()
        if not isinstance(step, Mapping):
            continue
        yield step, level, owner
        nested: list[Any] = []
        if isinstance(step.get("step"), Mapping):
            nested.append(step["step"])
        if isinstance(step.get("steps"), (list, tuple)):
            nested.extend(step["steps"])
        stack.extend((child, level + 1, owner) for child in reversed(nested))


def workflow_names(steps: Mapping[str, Any]) -> WorkflowNames:
    """The callable names and executable targets the step documents of *steps* use."""
    callables: set[str] = set()
    executables: set[str] = set()
    for step, _, _ in walk_steps(steps):
        for field in NAMED_OBJECT_FIELDS:
            if isinstance(step.get(field), str):
                callables.add(step[field])
        target = step.get("target")
        if step.get("type") == "ExecutableStep" and isinstance(target, str):
            executables.add(target)
    return WorkflowNames(tuple(sorted(callables)), tuple(sorted(executables)))


def _shape_problems(initial_step_id: str, steps: Mapping[str, Any]) -> list[str]:
    """One line per refusal of V1 for a body (at most ``_MAX_PROBLEMS``)."""
    problems: list[str] = []
    if initial_step_id not in steps:
        problems.append(f"initial_step_id {_shown(initial_step_id)} is not a step")
    for key, step in steps.items():
        step_id = step.get("step_id")
        if "step_id" in step and step_id != key:
            problems.append(f"step {_shown(key)} holds a step_id that is not its key")
    count = 0
    for step, level, owner in walk_steps(steps):
        count += 1
        if count > MAX_WORKFLOW_STEPS:
            problems.append(f"more than {MAX_WORKFLOW_STEPS} steps")
            break
        if level > MAX_STEP_NESTING:
            problems.append(
                f"step {_shown(owner)} nests deeper than {MAX_STEP_NESTING}"
            )
            break
        tag = step.get("type")
        if not (isinstance(tag, str) and _STEP_TYPE.fullmatch(tag)):
            problems.append(f"a step under {_shown(owner)} has no readable type")
        if "step" in step and not isinstance(step["step"], Mapping):
            problems.append(f"step {_shown(owner)}: step is not a step document")
        children = step.get("steps")
        if "steps" in step and not (
            isinstance(children, list) and all(isinstance(c, Mapping) for c in children)
        ):
            problems.append(
                f"step {_shown(owner)}: steps is not a list of step documents"
            )
        if len(problems) >= _MAX_PROBLEMS:
            break
    return problems


# DECISION plan-2026-11-25T195353-7d9c0955/D-005
# The workflow body is DATA in the serialize() shape: steps stay untyped dicts, V1 judges shape and
# size only, and the two tables of named-object fields live HERE and nowhere else (core cannot
# import `workflows`, and `workflows` imports them from this module). Do NOT copy a table into
# another module or the verifier (a copy drifts from the loader), do NOT type the steps with the
# step classes or import `workflows` to check them (V2 runs in the adapter factory), do NOT let a
# step document reach the FSM loader (a body that looks like an FSM never loads as one), and do NOT
# put a value of a step in an error line: a line names keys and levels only. Decision D-005.
class WorkflowBody(_Model):
    """``{initial_step_id, steps}``: the steps of a workflow as ``serialize(callables=)`` writes them."""

    initial_step_id: _StepId = Field(description="The key of the first step")
    steps: dict[_StepId, dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_WORKFLOW_STEPS,
        description="step id -> step document ({type, ...fields}); checked by the adapter",
    )

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if problems := _shape_problems(self.initial_step_id, self.steps):
            raise ValueError("; ".join(problems))
        return self


#: model -> {envelope version it was introduced in -> its field names}: merged into the envelope
#: table of ``executable`` (the workflow kind is part of format 1.0, which reserved it).
WORKFLOW_FIELD_VERSIONS: dict[str, dict[str, str]] = {
    "WorkflowBody": {"1.0": "initial_step_id steps"},
}
