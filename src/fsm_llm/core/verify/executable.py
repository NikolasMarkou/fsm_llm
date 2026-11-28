"""``load_executable``: the load sequence V1 to V5 and V7 of an executable (roadmap 6.5.6, E1.3).

One call reports EVERY failure it can find in a document, each message tagged by its step:
``[schema]`` (V1: the generated models, ``parse_executable``; a failure ends the call), ``[body]``
(V2: ``FSMValidator`` over the FSM body, errors and warnings copied), ``[variables]`` and
``[inputs]`` (V4: variables and the Principle 14 rule: no caller input renders into instruction or
persona text), ``[contract]`` (V5: the data-flow and liveness analyses against the declared
contract, WARNINGS only, so a built-in is never refused by a finding), ``[names]`` (V3: the
completion tool names against ``requires.tools``, and every required name through an injected
resolver) and ``[model]`` (V7, only for ``origin="model"``). V6 (strict mode) is not built.

A ``workflow`` document (plan 7d9c0955 D-006) takes its own rules, read from the BODY and never from
a claim: names, placeholders, the host-bound fields and, for a model, a closed list of step types
(``core.verify.executable_workflow``); V2 for it runs in the workflows adapter.

A ``pattern`` document (plan a555dcfe D-012) takes the same sequence over INSTANCES: V2 checks the
default instance (every option at its default, every variant unselected) and the instance with
every variant applied once, each as an FSM body, and a template with a required option (no default)
is one ``[body]`` error; V4 refuses a ``{{name}}`` that is not a declared variable and a hole of a
free-text option under an instruction-text key; V3 resolves the handler, hook and generator names
it uses and holds the ``hitl`` capability, an ``agents.approval_gate`` binding and an
``await_approval`` state equal; V7 refuses what binds code. The rules live in
``core.verify.executable_pattern``. NO name of a pattern resolves without a resolver: a pattern that
uses any handler, hook or generator name and gets none is refused (the namespaces arrive with the
agents handler library; until then nothing loads with a name it cannot resolve).

Interface contract:

``load_executable(data, *, variables=None, resolver=None, origin="operator", allow=None,
generators=None, bases=None)
-> (envelope | None, FSMValidationResult)``.

- ``data``: anything; a non-document, a document nested too deep or one the models refuse is an
  invalid result, never an exception. The envelope is returned only when the result is valid.
- ``variables``: ``None`` checks the template as written (declared variables are noted, not
  required); a mapping, possibly empty, fills the variables first: a missing, undeclared or
  non-text value is a ``[variables]`` error, and V2 then checks the FILLED body.
- ``resolver``: ``Callable[[str, str], bool]`` called with ``(category, name)`` pairs, the
  category one of ``tools``, ``handlers``, ``callables``, ``executables`` (and ``generators``
  for a pattern; the names of its bindings and run hooks go under ``handlers``); build it from the
  ``NameRegistry`` namespaces (``lambda ns, n: n in namespaces[ns]``). ``None`` skips resolution
  and says so once, except for a pattern that uses a handler, hook or generator name (refused).
  This module reads no registry.
- ``bases`` (``id -> PatternEnvelope | None``, D-033): where a PRESET (a pattern whose body
  ``extends`` another) finds its base; the result is the FLAT envelope (the base with the preset
  applied, ``core.definitions.executable_preset``), checked like any pattern. A preset with no
  ``bases``, an unknown base, a cycle or a chain over ``MAX_EXTENDS_DEPTH`` is one ``[schema]``
  error. V7 judges the preset AS WRITTEN: its ``extends`` is an implied ``requires.executables``
  name, so a model-origin preset may extend only a base on ``allow["executables"]``.
- ``origin``: ``"operator"`` (a file a person wrote: V7 never applies) or ``"model"`` (model-written
  text, the synthesis and meta-builder output of roadmap E4.6: V7 applies). ``allow``: category
  to the names a model may require; ``None`` is an empty list, so every name is refused (fail
  closed). V7: every ``requires`` name must be on the list; every tool the BODY declares in a
  completion state (read from the body as written and from the variable-filled body) must be on
  the ``tools`` list whatever ``requires.tools`` says, and ``requires.tools`` may not be ``"some"``
  nor ``"none"`` beside declared tools (a PRESET may not write ``requires.tools`` ``"none"`` at all:
  it would replace its base's claim, D-035); the body may not set a persona, a ``model`` or
  ``prompt_config`` on a classification entry, a validation-rule ``pattern`` or a ``synthesize``
  state (a copy of what synthesis gate G5 refuses, D-053); ``requires.handlers`` must be empty (a
  handler binding comes from a template, never from a model), and ``lifecycle`` may only be
  ``draft``. V7 is NOT gate G5: text length, key names, unknown keys, priorities and the abort edge
  stay G5's. Lifecycle, capabilities and contract are CLAIMS of the document, never
  authorization. A pattern document a model wrote may carry no ``handlers``, run ``hooks``,
  ``variants``, ``generator``, ``options_schema``, ``run.agent_type``, ``run.run_output_keys`` or
  ``run.context_seed`` (one message each, whatever the claims say); an ``fsm`` body key that binds
  code (``handlers``, ``variants``, ``generator``) is refused by V1 for every origin.
- Model-authored text (names, keys, state ids) is capped and control-stripped before it reaches a
  message (``fsm_llm.text.capped_line``): a message may be shown to a model by a repair loop.

Imports no engine module, calls no model and executes nothing from the document. Lazy: nothing
imports it at module top (``fsm_llm.core.verify.cli`` imports it inside the function), so
``import fsm_llm`` and ``fsm-llm-validate`` on an FSM file never load it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Literal

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    PROBLEM_LINE_CHARS,
    ExecutableDocumentError,
    ExecutableEnvelope,
    PatternEnvelope,
    WorkflowEnvelope,
    parse_executable,
    scan_placeholders,
)
from fsm_llm.core.definitions.executable_instance import (
    GeneratorLookup,
    substitute_variables,
    variable_problems,
)
from fsm_llm.core.definitions.executable_pattern import PatternBodyError
from fsm_llm.core.definitions.executable_preset import BaseLookup, resolve_extends
from fsm_llm.core.state_kinds import raw_kinds
from fsm_llm.text import capped_line

from .executable_pattern import (
    build_instances,
    hitl_problems,
    hole_text_problems,
    pattern_names,
)
from .executable_pattern import model_problems as pattern_model_problems
from .executable_workflow import model_problems as workflow_model_problems
from .executable_workflow import workflow_findings
from .flow import analyze_data_flow, analyze_liveness
from .result import FSMValidationResult
from .validator import FSMValidator

__all__ = ["load_executable"]

#: ``(category, name) -> resolves?`` with the category ``tools``, ``handlers``, ``callables`` or
#: ``executables``.
Resolver = Callable[[str, str], bool]

_TAGS = ("schema", "body", "variables", "inputs", "contract", "names", "model")
_CATEGORIES = ("tools", "handlers", "callables", "executables")
_PATTERN_CATEGORIES = (*_CATEGORIES, "generators")
_NAME_CHARS = 60  # one model-authored name inside a message


def _placeholder(name: str) -> str:
    return "{{" + name + "}}"


def _shown(value: object) -> str:
    """A model-authored name as it may appear in a message: capped, control-stripped, quoted."""
    return repr(capped_line(value, _NAME_CHARS))


def _declared_entries(
    fsm: Mapping[str, Any],
    reader: Literal["raw_declared_tools", "raw_declared_targets"],
) -> list[str | None]:
    """What the RAW states declare through the ``KindSpec`` reader named *reader*
    (``raw_declared_tools`` or ``raw_declared_targets``): one entry per name, ``None`` for a name
    that cannot be read."""
    states = fsm.get("states")
    entries: list[str | None] = []
    if isinstance(states, dict):
        for state in states.values():
            if isinstance(state, dict):
                for spec in raw_kinds(state):
                    entries.extend(getattr(spec, reader)(state))
    return entries


def _tool_names(fsm: Mapping[str, Any]) -> set[str]:
    """The tool names the RAW states declare (so a body that does not load is still read)."""
    return {
        n for n in _declared_entries(fsm, "raw_declared_tools") if isinstance(n, str)
    }


def _items(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


# The settings a model-written body may not carry. A COPY of what synthesis gate G5 refuses
# (``synthesis/policy.py``: ``persona-set``, ``model-setting`` with ``_MODEL_SETTING_KEYS``,
# ``regex-pattern`` and ``nested-synthesize``): ``core.verify`` may not import ``synthesis``. Key
# presence decides, as in G5; a message names the key and never its value (D-053).
_MODEL_SETTING_KEYS = ("model", "prompt_config")


def _body_setting_problems(fsm: Mapping[str, Any]) -> list[str]:
    """V7: the persona, model settings, regex patterns and synthesize states of a body."""
    problems: list[str] = []
    if fsm.get("persona") is not None:
        problems.append("body.fsm sets a persona: a model may not")
    states = fsm.get("states")
    for sid, state in states.items() if isinstance(states, dict) else ():
        if not isinstance(state, dict):
            continue
        where = f"state {_shown(sid)}"
        if "synthesize" in state:
            problems.append(f"{where} is a synthesize state: a model may not write one")
        for entry in (
            e
            for e in _items(state.get("classification_extractions"))
            if isinstance(e, dict)
        ):
            problems += [
                f"{where}: a classification entry sets {key}: a model may not"
                for key in _MODEL_SETTING_KEYS
                if key in entry
            ]
        for extraction in _items(state.get("field_extractions")):
            rules = (
                extraction.get("validation_rules")
                if isinstance(extraction, dict)
                else None
            )
            if isinstance(rules, dict) and "pattern" in rules:
                problems.append(
                    f"{where}: a validation rule has a pattern: a model may not write a regex"
                )
    return problems


def _model_problems(
    envelope: ExecutableEnvelope,
    allow: Mapping[str, Collection[str]] | None,
    bodies: Sequence[Mapping[str, Any]],
) -> list[str]:
    """V7: what a model-origin document may not claim, require, declare or set.

    *bodies* are the FSM body as written and, when variables were filled, the filled one: the
    tools and settings of the body are judged whatever ``requires`` says (D-053).
    """
    requires = envelope.requires
    problems = []
    listed = {}
    for category in ("tools", "callables", "executables"):
        names = (allow or {}).get(category, ())
        listed[category] = {names} if isinstance(names, str) else set(names)
    declared = [
        entry
        for body in bodies
        for entry in _declared_entries(body, "raw_declared_tools")
    ]
    if requires.tools == "some":
        problems.append("requires.tools 'some' cannot be checked; name the tools")
    elif requires.tools == "none" and declared:
        problems.append(
            "requires.tools 'none' disagrees with the body, which declares completion tools"
        )
    reported: set[str | None] = set()
    for category in ("tools", "callables", "executables"):
        names = getattr(requires, category)
        for n in names if isinstance(names, list) else ():
            if n not in listed[category]:
                problems.append(f"{category} name {_shown(n)} is not on the allow-list")
                if category == "tools":
                    reported.add(n)
    for n in dict.fromkeys(declared):
        if n in reported:
            continue
        if n is None:
            problems.append("a completion tool of the body has no readable name")
        elif n not in listed["tools"]:
            problems.append(f"body tool {_shown(n)} is not on the allow-list")
    # DECISION plan-2026-11-28T050702-8525c797/D-003: a call target of the body is judged
    # like a tool: against the allow-list AND against `requires.executables`, whatever the
    # claim says (an empty claim beside a body that calls an executable is a disagreement).
    # Do NOT leave it to the runtime resolver: `ExecutableRegistry.resolve_target` serves any
    # registered id and has no origin, so a model-written `call` state would reach all of
    # them. Gate G5 refuses a compiled `call` state outright (`synthesis/policy.py`).
    claimed = {n for n in requires.executables if isinstance(n, str)}
    for n in dict.fromkeys(
        t for body in bodies for t in _declared_entries(body, "raw_declared_targets")
    ):
        if n is None:
            problems.append("a call target of the body has no readable name")
        elif n not in listed["executables"]:
            if n not in claimed:  # a claimed name was reported above
                problems.append(
                    f"body call target {_shown(n)} is not on the allow-list"
                )
        elif n not in claimed:
            problems.append(
                f"body call target {_shown(n)} is not in requires.executables"
            )
    for body in bodies:
        problems += _body_setting_problems(body)
    base = getattr(envelope.body, "extends", None)
    if base is not None and base not in listed["executables"]:
        problems.append(f"extends names {_shown(base)}: not on the allow-list")
    if (
        base is not None
        and "tools" in requires.model_fields_set
        and requires.tools == "none"
    ):
        # a preset replaces its base's tool claim (D-033): a model may name allow-listed tools, but
        # may not rewrite the claim to "no tools" over a base that runs them (D-035)
        problems.append(
            "requires.tools 'none' on a preset: it would replace its base's tool claim"
        )
    if requires.handlers:
        problems.append(
            f"requires.handlers names {_shown(requires.handlers[0])}: a handler "
            "binding comes from a template, never from a model"
        )
    if envelope.lifecycle is not Lifecycle.DRAFT:
        problems.append(
            f"lifecycle {_shown(envelope.lifecycle.value)} is a claim a model "
            "may not make; only 'draft'"
        )
    return list(dict.fromkeys(problems))


def load_executable(
    data: object,
    *,
    variables: Mapping[str, str] | None = None,
    resolver: Resolver | None = None,
    origin: str = "operator",
    allow: Mapping[str, Collection[str]] | None = None,
    generators: GeneratorLookup | None = None,
    bases: BaseLookup | None = None,
) -> tuple[ExecutableEnvelope | None, FSMValidationResult]:
    """Load *data* as an executable and report every failure at once (see the module text).

    *generators* (``name -> generator | None``, D-019) lets V2 build the default instance of a
    pattern whose body is a generator; without it such a body is not instantiated.
    """
    if origin not in ("operator", "model"):
        raise ValueError(f"origin must be 'operator' or 'model', not {origin!r}")
    doc_id = data.get("id") if isinstance(data, Mapping) else None
    result = FSMValidationResult(capped_line(doc_id, 80) or "Unnamed executable")
    notes: dict[str, list[tuple[str, str]]] = {tag: [] for tag in _TAGS}

    def note(tag: str, severity: str, text: str) -> None:
        notes[tag].append(
            (severity, f"[{tag}] {capped_line(text, PROBLEM_LINE_CHARS)}")
        )

    def finish(
        envelope: ExecutableEnvelope | None,
    ) -> tuple[ExecutableEnvelope | None, FSMValidationResult]:
        for tag in _TAGS:
            for severity, text in notes[tag]:
                getattr(result, f"add_{severity}")(text)
        return (envelope if result.is_valid else None), result

    # V1
    try:
        envelope = parse_executable(data)
    except ExecutableDocumentError as error:
        for line in error.problems:
            note("schema", "error", line)
        return finish(None)
    if isinstance(envelope, WorkflowEnvelope):
        # a workflow has no `body.fsm`: its own rules, read from the steps (D-006)
        for tag, severity, text in workflow_findings(envelope, resolver):
            note(tag, severity, text)
        for line in (
            workflow_model_problems(envelope, allow) if origin == "model" else ()
        ):
            note("model", "error", line)
        return finish(envelope)

    written = envelope
    if isinstance(envelope, PatternEnvelope) and envelope.body.extends is not None:
        try:  # a preset (D-033): everything below sees the flat envelope
            if bases is None:
                raise PatternBodyError(
                    "a preset needs bases=: its base cannot be found"
                )
            envelope = parse_executable(resolve_extends(envelope, bases))
        except (PatternBodyError, ExecutableDocumentError) as error:
            for line in getattr(error, "problems", None) or (str(error),):
                note("schema", "error", line)
            return finish(None)
    pattern = envelope if isinstance(envelope, PatternEnvelope) else None

    # V4, the values: fill the declared variables before the body is checked
    declared = set(envelope.variables)
    template: dict[str, Any] | None = envelope.body.fsm  # None: a generator pattern
    fsm = template
    buildable = True
    if variables is None:
        if declared:
            note(
                "variables",
                "info",
                f"variables {sorted(declared)} are not filled: the body is checked as written",
            )
    else:
        problems = variable_problems(envelope, variables)
        for line in problems:
            note("variables", "error", line)
        buildable = not problems
        if buildable and pattern is None:
            fsm = substitute_variables(template, variables)

    # V2: an FSM body as written or filled; for a pattern its DEFAULT instance and its instance
    # with every variant applied once (D-012)
    instances: list[dict[str, Any]] = [fsm] if pattern is None and fsm else []
    if pattern is not None and buildable:
        built = build_instances(pattern, variables, generators)
        for line in built.problems:
            note("body", "error", line)
        for line in built.notes:
            note("body", "info", line)
        instances = built.bodies()
    definition = None
    default_errors: set[str] = set()
    for index, instance in enumerate(instances):
        validator = FSMValidator(instance)
        try:
            body = validator.validate()
        except (
            Exception
        ) as error:  # a validator never raises on a mapping; a belt, not a path
            note(
                "body",
                "error",
                f"the body could not be validated ({type(error).__name__})",
            )
            continue
        if index == 0:
            default_errors = set(body.errors)
            for severity, items in (
                ("error", body.errors),
                ("warning", body.warnings),
                ("info", body.info),
            ):
                for text in items:
                    note("body", severity, text)
            definition = validator._definition
        else:  # the every-variant instance says only what the default instance did not
            for text in dict.fromkeys(body.errors):
                if text not in default_errors:
                    note("body", "error", f"with every variant applied: {text}")
    read = [template, *instances] if pattern is not None else [fsm]

    # V4, the text: no caller input renders into instruction or persona text
    leaves, keys = scan_placeholders(
        template if pattern is None else [template, pattern.body.variants]
    )
    contract = envelope.contract
    arguments = (
        set(contract.inputs)
        | scan_placeholders(list(contract.input_templates.values()))[0]
    )
    for name in sorted(leaves & arguments):
        note(
            "inputs",
            "error",
            f"the body uses {_placeholder(name)}, a caller input: no caller input "
            "renders into instruction or persona text",
        )
    for name in sorted(leaves - declared - arguments):
        if pattern is None:
            note(
                "variables",
                "warning",
                f"{_placeholder(name)} is not a declared variable",
            )
        else:  # the declared variables are the one text channel of a template (D-012)
            note(
                "variables",
                "error",
                f"{_placeholder(name)} is not a declared variable: only a declared "
                "variable may put text into a template",
            )
    for name in sorted(keys & declared):
        note(
            "variables",
            "warning",
            f"{_placeholder(name)} sits in a key and is left alone (text values only)",
        )
    if pattern is not None:
        for line in hole_text_problems(pattern.body):
            note("inputs", "error", line)

    # V5: warnings only
    if definition is not None:
        findings = (
            *analyze_data_flow(
                definition, inputs=contract.inputs, return_keys=contract.outputs
            ),
            *analyze_liveness(definition, inputs=contract.inputs),
        )
        for f in findings:
            note(
                "contract",
                "warning",
                f"{f.analysis} {f.code} at state {f.state!r}: {f.message}",
            )

    # V3
    tools = envelope.requires.tools
    declared_tools = {name for body_ in read if body_ for name in _tool_names(body_)}
    if isinstance(tools, list) and (absent := sorted(declared_tools - set(tools))):
        shown = ", ".join(_shown(n) for n in absent)
        note("names", "error", f"completion tools {shown} are not in requires.tools")
    categories: tuple[str, ...] = _CATEGORIES
    names: dict[str, Any] = {c: getattr(envelope.requires, c) for c in _CATEGORIES}
    if pattern is not None:
        categories = _PATTERN_CATEGORIES
        names.update(pattern_names(pattern))
        for line in hitl_problems(pattern, read):
            note("names", "error", line)
    if resolver is None:
        note("names", "info", "names are not resolved: no resolver was given")
        unresolved = (
            [n for c in ("handlers", "generators") for n in names[c]]
            if pattern is not None
            else []
        )
        if unresolved:
            # fail closed (D-012): until a namespace is given no name of a template resolves,
            # so a pattern that binds one does not load
            more = f" and {len(unresolved) - 1} more" if len(unresolved) > 1 else ""
            note(
                "names",
                "error",
                f"the pattern uses {_shown(unresolved[0])}{more} and no resolver was "
                "given: a name that cannot be resolved is refused",
            )
    else:
        for category in categories:
            for name in names[category] if isinstance(names[category], list) else ():
                if not resolver(category, name):
                    note(
                        "names",
                        "error",
                        f"{category} name {_shown(name)} does not resolve",
                    )

    # V7
    if origin == "model":
        bodies = [b for b in ([template, fsm] if pattern is None else read) if b]
        if (
            written is not envelope
        ):  # a preset: judged as written, its base is allow-listed
            bodies = []
        for line in dict.fromkeys(
            [
                *_model_problems(written, allow, bodies),
                *pattern_model_problems(written),
            ]
        ):
            note("model", "error", line)
    return finish(envelope)
