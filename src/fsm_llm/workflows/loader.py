"""The workflow loader: a document back into a ``WorkflowDefinition`` (roadmap E5.1).

``load_workflow(document, *, objects, step_classes=None)`` is the inverse of
``WorkflowDefinition.serialize(callables=objects)``. A document is data and never runs code:

- a step's ``type`` resolves ONLY through the step-class table: the built-in step classes
  (``builtin_step_classes``, derived from ``fsm_llm.workflows.steps``) plus the classes the
  caller passes in ``step_classes``, keyed by class name. Nothing is imported, evaluated or
  looked up on a module by document text; the tag is matched exactly (case-sensitive);
- a callable or object field (``action``, ``api_function``, ``condition``,
  ``aggregation_function``, ``llm_interface``, ``agent``) holds a NAME, resolved only through
  ``objects``, a ``NameRegistry`` of namespace ``callables``. A name the registry lacks is an
  error, never a ``None``;
- ``fsm_definition`` of a conversation step is normalised to a dict (an ``FSMDefinition`` object
  in the document becomes its ``model_dump()``): a document holds data only, so equality with the
  object-form definition does not hold, and ``serialize(load(x)) == x`` does;
- the loader never changes the document (it works on a deep copy), refuses an unknown top-level
  key itself (``WorkflowDefinition.model_config`` is unchanged), builds every step through its
  class (``extra="forbid"`` refuses an unknown field), builds the definition and calls
  ``validate_structure()``.

Interface contract: ``load_workflow(document, *, objects, step_classes=None) ->
WorkflowDefinition``. ``document`` is a mapping (a parsed JSON object); ``objects`` a
``NameRegistry`` of namespace ``callables``; ``step_classes`` an iterable of ``WorkflowStep``
subclasses (the tag of a class is its ``__name__``, the one ``serialize`` writes). Failure mode:
``WorkflowDefinitionError`` for a registry that is not a ``callables`` ``NameRegistry`` (a
caller bug); ONE ``WorkflowValidationError`` carrying every failure of the document otherwise
(unknown top-level key, unknown or missing ``type``, unresolved name, pydantic errors per
step, a custom class that repeats a name, structure errors). Structure validation runs only
when every step built: a structure check over a partial step set would report errors the
failed steps caused. A callable field of a custom class that is not one of the shared names
is not resolved by the loader (its name reaches pydantic and is refused there, loudly).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from fsm_llm.core.definitions.executable_workflow import NAMED_OBJECT_FIELDS
from fsm_llm.core.names import NameRegistry

from . import steps as _steps
from .definitions import WorkflowDefinition, check_callables_registry
from .exceptions import WorkflowValidationError
from .steps import ConversationStep, ParallelStep, RetryStep, WorkflowStep

__all__ = ["builtin_step_classes", "load_workflow"]


def builtin_step_classes() -> dict[str, type[WorkflowStep]]:
    """The built-in step classes by class name (``AutoStep`` ... ``AgentStep``).

    Derived from the classes ``fsm_llm.workflows.steps`` exports, so a new step class adds
    its name here with no edit.
    """
    return {
        cls.__name__: cls
        for cls in vars(_steps).values()
        if isinstance(cls, type)
        and issubclass(cls, WorkflowStep)
        and cls is not WorkflowStep
    }


def _step_table(
    step_classes: Iterable[type[WorkflowStep]] | None, errors: list[str]
) -> dict[str, type[WorkflowStep]]:
    table = builtin_step_classes()
    custom: set[str] = set()
    for cls in step_classes or ():
        name = getattr(cls, "__name__", repr(cls))
        if not (isinstance(cls, type) and issubclass(cls, WorkflowStep)):
            errors.append(f"step_classes entry {name!r} is not a WorkflowStep subclass")
        elif name in table and name not in custom:
            errors.append(
                f"custom step class {name!r} has the name of a built-in step class"
            )
        elif name in custom:
            errors.append(f"custom step class {name!r} is given twice")
        else:
            table[name] = cls
            custom.add(name)
    return table


def _pydantic_text(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '(step)'}: {item['msg']}"
        for item in error.errors()
    )


class _Builder:
    """One load: the step table, the registry and the list every failure goes to."""

    def __init__(
        self,
        table: dict[str, type[WorkflowStep]],
        objects: NameRegistry[Any],
        errors: list[str],
    ) -> None:
        self.table = table
        self.objects = objects
        self.errors = errors

    def build(self, raw: object, where: str) -> WorkflowStep | None:
        """One step from its document form, or ``None`` after recording why it failed."""
        if not isinstance(raw, Mapping):
            self.errors.append(
                f"{where}: a step must be a mapping, got {type(raw).__name__}"
            )
            return None
        fields = dict(raw)
        tag = fields.pop("type", None)
        if tag is None:
            self.errors.append(f"{where}: the step has no 'type'")
            return None
        cls = self.table.get(tag) if isinstance(tag, str) else None
        if cls is None:
            self.errors.append(
                f"{where}: unknown step type {tag!r} (known: {', '.join(sorted(self.table))})"
            )
            return None
        ok = self._resolve_objects(cls, fields, where)
        ok = self._build_nested(cls, fields, where) and ok
        if issubclass(cls, ConversationStep):
            dump = getattr(fields.get("fsm_definition"), "model_dump", None)
            if callable(dump):
                fields["fsm_definition"] = dump()
        if not ok:
            return None
        try:
            return cls(**fields)
        except ValidationError as error:
            self.errors.append(f"{where} ({tag}): {_pydantic_text(error)}")
            return None

    def _resolve_objects(
        self, cls: type[WorkflowStep], fields: dict[str, Any], where: str
    ) -> bool:
        ok = True
        for field in sorted(NAMED_OBJECT_FIELDS & set(cls.model_fields)):
            if fields.get(field) is None:
                continue
            name = fields[field]
            if not isinstance(name, str):
                self.errors.append(
                    f"{where} field '{field}': expected the registered name of an object "
                    f"(a string), got {type(name).__name__}"
                )
                ok = False
            elif name not in self.objects:
                self.errors.append(
                    f"{where} field '{field}': {name!r} is not registered in the "
                    f"{self.objects.namespace!r} registry"
                )
                ok = False
            else:
                fields[field] = self.objects.resolve(name)
        return ok

    def _build_nested(
        self, cls: type[WorkflowStep], fields: dict[str, Any], where: str
    ) -> bool:
        if issubclass(cls, ParallelStep) and "steps" in fields:
            raw_children = fields["steps"]
            if not isinstance(raw_children, list):
                return True  # pydantic reports the shape
            children = [
                self.build(child, f"{where} steps[{index}]")
                for index, child in enumerate(raw_children)
            ]
            fields["steps"] = children
            return all(child is not None for child in children)
        if issubclass(cls, RetryStep) and "step" in fields:
            inner = self.build(fields["step"], f"{where} step")
            fields["step"] = inner
            return inner is not None
        return True


# DECISION plan-2026-11-20T055240-ace7a8c8/D-069
# A step `type` resolves ONLY through the class table (`builtin_step_classes()` plus the caller's
# `step_classes`, keyed by `__name__`), and a callable or object a document names resolves ONLY
# through the caller's `NameRegistry`. A document never chooses what runs by a dynamic import.
# Do NOT resolve a type or a name from a dotted path of the document (the loader's source row of
# `test_workflow_loader` and `test_string_imports` ban the import machinery here), do NOT add a
# global step registry, and do NOT let a custom class shadow a built-in name (one collected error).
def load_workflow(
    document: Mapping[str, Any],
    *,
    objects: NameRegistry[Any],
    step_classes: Iterable[type[WorkflowStep]] | None = None,
) -> WorkflowDefinition:
    """Build a ``WorkflowDefinition`` from its document. See the module docstring."""
    check_callables_registry(objects)
    errors: list[str] = []
    table = _step_table(step_classes, errors)
    if not isinstance(document, Mapping):
        errors.append(f"the document must be a mapping, got {type(document).__name__}")
        raise WorkflowValidationError(
            "Workflow document is invalid", validation_errors=errors
        )
    doc = copy.deepcopy(dict(document))
    known = set(WorkflowDefinition.model_fields)
    for key in sorted(str(k) for k in doc if k not in known):
        errors.append(f"unknown top-level key {key!r}")
    raw_steps = doc.get("steps", {})
    steps: dict[str, WorkflowStep] = {}
    if not isinstance(raw_steps, Mapping):
        errors.append(f"'steps' must be a mapping, got {type(raw_steps).__name__}")
    else:
        builder = _Builder(table, objects, errors)
        for step_id, raw in raw_steps.items():
            step = builder.build(raw, f"step '{step_id}'")
            if step is not None:
                steps[step_id] = step
    if errors:
        raise WorkflowValidationError(
            "Workflow document is invalid", validation_errors=errors
        )
    try:
        definition = WorkflowDefinition(
            **{key: value for key, value in doc.items() if key != "steps"}, steps=steps
        )
    except ValidationError as error:
        errors.append(f"workflow: {_pydantic_text(error)}")
        raise WorkflowValidationError(
            "Workflow document is invalid", validation_errors=errors
        ) from error
    try:
        definition.validate_structure()
    except WorkflowValidationError as error:
        errors.extend(str(e) for e in error.validation_errors)
        raise WorkflowValidationError(
            "Workflow document is invalid", validation_errors=errors
        ) from error
    return definition
