"""The checks of a ``workflow`` executable that ``load_executable`` runs (plan 7d9c0955, D-006).

``core.verify.executable`` runs the load sequence; the workflow rules live here as one pure
function, so the sequence stays one reading order. The steps are untyped dicts in core (the step
classes live in ``fsm_llm.workflows``, a layer above), so these rules read the DOCUMENT through the
one iterative walker of ``core.definitions.executable_workflow``; V2 (structure) runs in the
workflows adapter when a registry loads the document, and V1 (shape and size) was done by the body
model. Per step of the sequence:

- V3 names: the callable names and executable targets the BODY uses against ``requires`` (a used
  name missing from the claim is an error, a claimed name no step uses a warning), a named-object
  field that holds no text, a name that is blank or starts with the reserved ``$run.`` prefix (the
  adapter binds names of its own under it), then the caller's resolver. A claim is never what decides.
- V4: a workflow takes no variables, so a ``{{name}}`` anywhere in the steps is filled by nothing.
- V5: one info line (no flow analysis exists for workflows; structure is the adapter's).
- Host-bound, EVERY origin: a step that sets ``registry`` or ``callables`` (they bind to the run
  context at run time, never to a document).
- V7 (``origin="model"``): a closed allow-list of step types applied to every nested step, every
  body callable and target on the caller's allow-list, no handler, no tool, ``draft`` only.

Interface contract: ``workflow_findings(envelope, resolver=None) -> list[(tag, severity, text)]``
(tags ``names``, ``variables``, ``body``; severities ``error``, ``warning``, ``info``; never
raises) and ``model_problems(envelope, allow) -> list[str]`` (V7, deduplicated). A name or key that
came from a document reaches a line only through ``capped_line``; no line holds a value of a step.
Imports no workflows module, no engine module; lazy like ``core.verify.executable``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    ExecutableEnvelope,
    WorkflowEnvelope,
    scan_placeholders,
)
from fsm_llm.core.definitions.executable_workflow import (
    NAMED_OBJECT_FIELDS,
    RESERVED_NAME_PREFIX,
    walk_steps,
    workflow_names,
)
from fsm_llm.text import capped_line

__all__ = [
    "ALLOWED_STEP_TYPES",
    "HOST_BOUND_FIELDS",
    "model_problems",
    "workflow_findings",
]

#: V7: the step types a model-written workflow may use. Closed and keyed by the exact tag; every
#: step of the walk (a wrapper's child included) must be on it. Held a subset of the loader's
#: ``builtin_step_classes()`` by a row of the tests (core may not import ``workflows``).
ALLOWED_STEP_TYPES = frozenset(
    {
        "ExecutableStep",
        "SwitchStep",
        "AutoStep",
        "ConditionStep",
        "ParallelStep",
        "RetryStep",
    }
)

#: Step fields the host binds to the run context at run time: a document never sets them.
HOST_BOUND_FIELDS = ("registry", "callables")

_NAME_CHARS = 60
_MAX_LINES = 20  # lines one rule reports


def _shown(value: object) -> str:
    return repr(capped_line(value, _NAME_CHARS))


def _listed(allow: Mapping[str, Collection[str]] | None, category: str) -> set[str]:
    names = (allow or {}).get(category, ())
    return {names} if isinstance(names, str) else set(names)


def _steps(envelope: WorkflowEnvelope) -> Mapping[str, Any]:
    return envelope.body.steps


def workflow_findings(
    envelope: WorkflowEnvelope, resolver: Any = None
) -> list[tuple[str, str, str]]:
    """V3, V4, V5 and the host-bound rule for *envelope*: ``(tag, severity, text)`` lines."""
    out: list[tuple[str, str, str]] = []
    steps = _steps(envelope)
    out.append(
        (
            "body",
            "info",
            "workflow structure is checked when a registry loads the document",
        )
    )
    used = workflow_names(steps)
    for category, names in (
        ("callables", used.callables),
        ("executables", used.executables),
    ):
        claim = set(getattr(envelope.requires, category))
        out += [
            (
                "names",
                "error",
                f"the body uses {category} name {_shown(n)} that requires.{category} lacks",
            )
            for n in names
            if n not in claim
        ]
        out += [
            (
                "names",
                "warning",
                f"requires.{category} names {_shown(n)} and no step uses it",
            )
            for n in sorted(claim - set(names))
        ]
        out += [
            (
                "names",
                "error",
                f"the body uses {category} name {_shown(n)}: a name is not blank and does "
                f"not start with {RESERVED_NAME_PREFIX!r}, which the run binds itself",
            )
            for n in names
            if not n.strip() or n.startswith(RESERVED_NAME_PREFIX)
        ]
    for step, _, owner in walk_steps(steps):
        where = f"step {_shown(owner)}"
        out += [
            ("names", "error", f"{where}: {field} holds no name")
            for field in sorted(NAMED_OBJECT_FIELDS)
            if step.get(field) is not None and not isinstance(step[field], str)
        ]
        out += [
            ("body", "error", f"{where} sets {field}: the host binds it at run time")
            for field in HOST_BOUND_FIELDS
            if field in step
        ]
    leaves, keys = scan_placeholders(steps)
    out += [
        (
            "variables",
            "error",
            f"the body uses {{{{{name}}}}}: a workflow takes no variables, so nothing fills it",
        )
        for name in sorted(leaves | keys)[:_MAX_LINES]
    ]
    if resolver is not None:
        for category in ("tools", "handlers", "callables", "executables"):
            claimed = getattr(envelope.requires, category)
            out += [
                ("names", "error", f"{category} name {_shown(n)} does not resolve")
                for n in (claimed if isinstance(claimed, list) else ())
                if not resolver(category, n)
            ]
    else:
        out.append(("names", "info", "names are not resolved: no resolver was given"))
    return out


# DECISION plan-2026-11-25T195353-7d9c0955/D-006
# V3 and V7 read the BODY of the workflow, never a claim: `requires`, `capabilities`, `lifecycle`
# and `contract` lift no refusal and there is NO early return (the ace7a8c8 bug skipped the body
# check when `requires.tools` said "none"). The step-type list is closed and every nested step is
# judged, so a RetryStep or ParallelStep is allowed only around allowed steps. Do NOT add a type to
# the list without a row against `builtin_step_classes()`, do NOT skip the walk for a nested step,
# do NOT let a callable or target off the caller's allow-list through because `requires` names
# it, and do NOT put a value of a step in a line (a type tag, a key or a capped name only).
def model_problems(
    envelope: ExecutableEnvelope, allow: Mapping[str, Collection[str]] | None
) -> list[str]:
    """V7: what a workflow document a model wrote may not carry (empty for another kind)."""
    if not isinstance(envelope, WorkflowEnvelope):
        return []
    problems: list[str] = []
    for step, _, owner in walk_steps(_steps(envelope)):
        tag = step.get("type")
        if not (isinstance(tag, str) and tag in ALLOWED_STEP_TYPES):
            problems.append(
                f"step {_shown(owner)} has type {_shown(tag)}: not a step type a model may write"
            )
    used = workflow_names(_steps(envelope))
    for category, names in (
        ("callables", used.callables),
        ("executables", used.executables),
    ):
        listed = _listed(allow, category)
        problems += [
            f"the body uses {category} name {_shown(n)}: not on the allow-list"
            for n in names
            if n not in listed
        ]
    requires = envelope.requires
    if requires.handlers:
        problems.append(
            f"requires.handlers names {_shown(requires.handlers[0])}: a workflow "
            "binds no handler"
        )
    if requires.tools not in ("none", []):
        problems.append("requires.tools is not none: a workflow runs no tool")
    if envelope.lifecycle is not Lifecycle.DRAFT:
        problems.append(
            f"lifecycle {_shown(envelope.lifecycle.value)} is a claim a model may not "
            "make; only 'draft'"
        )
    return list(dict.fromkeys(problems))[:_MAX_LINES]
