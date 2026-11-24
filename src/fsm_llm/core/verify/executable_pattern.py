"""The checks of a ``pattern`` executable that ``load_executable`` adds (plan a555dcfe D-012).

``core.verify.executable`` runs the load sequence; this module holds the pattern rules as pure
functions that return problem lines, so the sequence stays one reading order and the E4.6
meta-builder can reuse a rule without the whole load. Each rule says which step it belongs to:

- V2 ``build_instances``: the FSM a pattern would run with every default (the DEFAULT instance)
  and with EVERY variant applied once in declaration order; the caller validates both as it
  validates an FSM body. An option with no default, a refused variant patch or a hole of the
  wrong type is one line each (``InstantiationError`` text, which never holds a value).
- V3 ``pattern_names`` and ``hitl_problems``: the handler, hook and generator names a pattern
  uses, for the caller's resolver (the seam step 7 fills), and the rule that holds three things
  equal: the ``hitl`` capability, the ``agents.approval_gate`` binding and an ``await_approval``
  state.
- V4 ``hole_text_problems``: a hole whose option is free text may not stand where instruction
  text is read; only a declared ``{{variable}}`` is the text channel of a template.
- V7 ``model_problems``: what a document written by a model may not carry (handlers, hooks,
  variants, a generator, an options schema, and the run fields that name code or the driver
  channel).

Interface contract: ``build_instances(envelope, variables) -> PatternInstances`` (never raises;
``problems`` are ``[body]`` error lines, ``notes`` info lines); ``pattern_names(envelope) ->
dict[str, list[str]]`` (keys ``handlers`` and ``generators``, each name once, in document order);
``hitl_problems(envelope, bodies) -> list[str]``; ``hole_text_problems(body) -> list[str]``;
``model_problems(envelope) -> list[str]``. A name that came from a document reaches a line only
through ``capped_line``. Imports no engine module; lazy like ``core.verify.executable``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fsm_llm.core.definitions.executable import (
    Capability,
    ExecutableEnvelope,
    InstantiationError,
    PatternEnvelope,
)
from fsm_llm.core.definitions.executable_instance import instantiate_body
from fsm_llm.core.definitions.executable_pattern import (
    APPROVAL_GATE_HANDLER,
    APPROVAL_STATE,
    HOLE_KEY,
    PatternBody,
)
from fsm_llm.text import capped_line

__all__ = [
    "TEXT_KEYS",
    "PatternInstances",
    "build_instances",
    "hitl_problems",
    "hole_text_problems",
    "model_problems",
    "pattern_names",
]

#: The keys under which a string is instruction or persona text that a model reads. A hole of a
#: free-text option may not stand under one of them (V4). Closed on purpose: a field that is
#: added to the definition models and read by a prompt joins this tuple.
TEXT_KEYS = (
    "description",
    "purpose",
    "persona",
    "instructions",
    "extraction_instructions",
    "response_instructions",
)

_NAME_CHARS = 60


def _shown(value: object) -> str:
    return repr(capped_line(value, _NAME_CHARS))


@dataclass(frozen=True)
class PatternInstances:
    """The FSM dicts a pattern is checked on (V2) and the lines that building them produced."""

    default: dict[str, Any] | None = None
    every_variant: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def bodies(self) -> list[dict[str, Any]]:
        """Every instance that could be built, default first."""
        return [b for b in (self.default, self.every_variant) if b is not None]


def _why(error: Exception) -> str:
    # an InstantiationError message never holds a value; any other error shows its type only
    if isinstance(error, InstantiationError):
        return capped_line(str(error), 300)
    return f"({type(error).__name__})"


def build_instances(
    envelope: PatternEnvelope, variables: Mapping[str, str] | None
) -> PatternInstances:
    """V2: the default instance and the every-variant instance of *envelope* (never raises).

    *variables* ``None`` leaves each declared ``{{name}}`` as its own text (the body is checked
    as written); a mapping fills them (the caller has already refused a bad one). A generator
    body has no template: one note and no instance.
    """
    body = envelope.body
    if body.fsm is None:
        return PatternInstances(
            notes=["a generator body builds its FSM at run time: nothing to check here"]
        )
    values = (
        dict(variables)
        if variables is not None
        else {name: "{{" + name + "}}" for name in envelope.variables}
    )
    try:
        default = instantiate_body(envelope, variables=values)
    except (ValueError, TypeError, RecursionError) as error:
        return PatternInstances(
            problems=[f"the default instance cannot be built: {_why(error)}"]
        )
    if not body.variants:
        return PatternInstances(default=default)
    try:
        everything = instantiate_body(
            envelope, variants=tuple(body.variants), variables=values
        )
    except (ValueError, TypeError, RecursionError) as error:
        return PatternInstances(
            default=default,
            problems=[
                f"the instance with every variant applied cannot be built: {_why(error)}"
            ],
        )
    return PatternInstances(default=default, every_variant=everything)


def pattern_names(envelope: PatternEnvelope) -> dict[str, list[str]]:
    """The names *envelope* uses that a resolver must know (V3), each once, in document order.

    ``handlers``: ``requires.handlers``, the bindings and the run hooks (hooks live in the same
    namespace, D-004); ``generators``: the generator, when the body has one.
    """
    body = envelope.body
    handlers = [
        *envelope.requires.handlers,
        *(binding.handler for binding in body.handlers),
        *(hook.hook for hook in body.run.hooks),
    ]
    return {
        "handlers": list(dict.fromkeys(handlers)),
        "generators": [] if body.generator is None else [body.generator],
    }


def _state_ids(fsm: Mapping[str, Any] | None) -> set[str]:
    states = fsm.get("states") if isinstance(fsm, Mapping) else None
    return (
        {k for k in states if isinstance(k, str)} if isinstance(states, dict) else set()
    )


def hitl_problems(
    envelope: PatternEnvelope, bodies: Sequence[Mapping[str, Any] | None]
) -> list[str]:
    """V3: the ``hitl`` capability, the approval gate binding and the gate state agree.

    *bodies* are the template, the default instance and the every-variant instance (``None``
    entries are skipped): the state counts when ANY of them has it. A generator body has no
    template, so its state side is not checked here. Each broken side is one line.
    """
    # DECISION plan-2026-11-24T043150-a555dcfe/D-012: the three are judged against each other
    # and never against a claim alone. Do NOT trust `capabilities` (a claim; LESSONS, the V7 bug
    # of ace7a8c8), do NOT accept a gate with no `await_approval` state (a gate that never
    # fires), do NOT accept the state with no gate bound (a wait nothing ends and no grant
    # check), and do NOT let a variant or an option decide whether the gate is bound (approval
    # is chosen by the run, never by data).
    body = envelope.body
    bound = any(b.handler == APPROVAL_GATE_HANDLER for b in body.handlers)
    claimed = Capability.HITL in envelope.capabilities
    problems = []
    if claimed and not bound:
        problems.append(
            f"capabilities claims hitl but {APPROVAL_GATE_HANDLER!r} is not bound in "
            "body.handlers"
        )
    if bound and not claimed:
        problems.append(
            f"{APPROVAL_GATE_HANDLER!r} is bound but capabilities does not claim hitl"
        )
    if body.fsm is not None:
        has_state = any(APPROVAL_STATE in _state_ids(b) for b in bodies)
        if bound and not has_state:
            problems.append(
                f"{APPROVAL_GATE_HANDLER!r} is bound but no instance has an "
                f"{APPROVAL_STATE!r} state"
            )
        if has_state and not bound:
            problems.append(
                f"an {APPROVAL_STATE!r} state is present but {APPROVAL_GATE_HANDLER!r} "
                "is not bound in body.handlers"
            )
    return problems


def _holes_with_parent_key(
    node: Any, parent_key: str | None
) -> Iterator[tuple[str, str | None]]:
    """``(option name, key the hole sits under)`` of every well-formed hole in *node*.

    Iterative, document order. Inside a list the key is the list's own key. A malformed hole
    (several keys, a name that is no string) was refused at parse and is skipped here.
    """
    stack: list[tuple[Any, str | None]] = [(node, parent_key)]
    while stack:
        current, key = stack.pop()
        if isinstance(current, Mapping):
            if HOLE_KEY in current:
                if isinstance(current[HOLE_KEY], str):
                    yield current[HOLE_KEY], key
                continue
            stack.extend(
                (child, str(k)) for k, child in reversed(list(current.items()))
            )
        elif isinstance(current, list):
            stack.extend((child, key) for child in reversed(current))


def hole_text_problems(body: PatternBody) -> list[str]:
    """V4: one line per hole of a free-text option that sits under a text key.

    The hole of a string option with no ``enum`` would put text the caller chose into
    instruction or persona text; only a declared ``{{variable}}`` may. An ``enum`` string is a
    vocabulary the author listed and may sit anywhere. The ops of the variants are judged with
    the last segment of the op ``path`` as the key of an op ``value``.
    """
    properties = body.options_schema.properties
    sites: list[tuple[str, str | None, str]] = [
        (name, key, "fsm") for name, key in _holes_with_parent_key(body.fsm, None)
    ]
    for variant, ops in body.variants.items():
        for op in ops:
            path = op.get("path")
            leaf = path.rsplit("/", 1)[-1] if isinstance(path, str) else None
            sites += [
                (name, key, f"variant {_shown(variant)}")
                for name, key in _holes_with_parent_key(op.get("value"), leaf)
            ]
    problems = []
    for name, key, where in sites:
        spec = properties.get(name)
        if (
            spec is not None
            and spec.type == "string"
            and spec.enum is None
            and key in TEXT_KEYS
        ):
            problems.append(
                f"the hole of option {_shown(name)} sits under {key!r} in {where}: only a "
                "declared {{variable}} may put text into instruction text"
            )
    return list(dict.fromkeys(problems))


def model_problems(envelope: ExecutableEnvelope) -> list[str]:
    """V7: what a pattern document written by a model may not carry (empty for another kind)."""
    # DECISION plan-2026-11-24T043150-a555dcfe/D-012: the content is judged and no claim is read
    # here: `requires`, `capabilities`, `lifecycle` and `contract` never lift a refusal and there
    # is NO early return (the ace7a8c8 bug skipped the body check when `requires.tools` said
    # "none"). Do NOT refuse the kind as a whole (each field below is its own proof) and do NOT
    # drop a field because another one was refused. A binding, a hook, a variant selection, a
    # generator and an options schema come from a template a person wrote; `agent_type` names a
    # class, `run_output_keys` and `context_seed` reach the driver.
    if not isinstance(envelope, PatternEnvelope):
        return []
    body = envelope.body
    run = body.run
    problems: list[str] = []
    if body.handlers:
        problems.append(
            f"body.handlers binds {_shown(body.handlers[0].handler)}: a handler binding "
            "comes from a template, never from a model"
        )
    if run.hooks:
        problems.append(
            f"body.run.hooks binds {_shown(run.hooks[0].hook)}: a hook comes from a "
            "template, never from a model"
        )
    if body.variants:
        problems.append(
            "body.variants is set: a variant is selected by the code that runs a "
            "template, never by a model's document"
        )
    if body.generator is not None:
        problems.append(
            f"body.generator names {_shown(body.generator)}: a generator is code a template binds"
        )
    if body.options_schema.properties:
        problems.append(
            "body.options_schema declares options: an option is a template feature, "
            "never a model's"
        )
    if run.agent_type is not None:
        problems.append(
            f"body.run.agent_type names {_shown(run.agent_type)}: it selects a class, "
            "never a model's choice"
        )
    if run.run_output_keys:
        problems.append("body.run.run_output_keys is set: it names driver-owned keys")
    if run.context_seed:
        problems.append("body.run.context_seed is set: a seed reaches the driver")
    return problems
