"""``load_executable``: the load sequence V1 to V5 and V7 of an executable (roadmap 6.5.6, E1.3).

One call reports EVERY failure it can find in a document, each message tagged by its step:
``[schema]`` (V1: the generated models, ``parse_executable``; a failure ends the call), ``[body]``
(V2: ``FSMValidator`` over the FSM body, errors and warnings copied), ``[variables]`` and
``[inputs]`` (V4: variables and the Principle 14 rule: no caller input renders into instruction or
persona text), ``[contract]`` (V5: the data-flow and liveness analyses against the declared
contract, WARNINGS only, so a built-in is never refused by a finding), ``[names]`` (V3: the
completion tool names against ``requires.tools``, and every required name through an injected
resolver) and ``[model]`` (V7, only for ``origin="model"``). V4 variants wait for the pattern kind
and V6 (strict mode) is not built.

Interface contract:

``load_executable(data, *, variables=None, resolver=None, origin="operator", allow=None)
-> (envelope | None, FSMValidationResult)``.

- ``data``: anything; a non-document, a document nested too deep or one the models refuse is an
  invalid result, never an exception. The envelope is returned only when the result is valid.
- ``variables``: ``None`` checks the template as written (declared variables are noted, not
  required); a mapping, possibly empty, fills the variables first: a missing, undeclared or
  non-text value is a ``[variables]`` error, and V2 then checks the FILLED body.
- ``resolver``: ``Callable[[str, str], bool]`` called with ``(category, name)`` pairs, the
  category one of ``tools``, ``handlers``, ``callables``, ``executables``; build it from the
  ``NameRegistry`` namespaces (``lambda ns, n: n in namespaces[ns]``). ``None`` skips resolution
  and says so once. This module reads no registry.
- ``origin``: ``"operator"`` (a file a person wrote: V7 never applies) or ``"model"`` (model-written
  text, the synthesis and meta-builder output of roadmap E4.6: V7 applies). ``allow``: category
  to the names a model may require; ``None`` is an empty list, so every name is refused (fail
  closed). V7: every ``requires`` name must be on the list, ``requires.tools`` may not be the bare
  claim ``"some"``, ``requires.handlers`` must be empty (a handler binding comes from a template,
  never from a model), and ``lifecycle`` may only be ``draft``. Lifecycle, capabilities and
  contract are CLAIMS of the document, never authorization. A body key that binds code
  (``handlers``, ``variants``, ``generator``) is refused by V1 for every origin.
- Model-authored text (names, keys, state ids) is capped and control-stripped before it reaches a
  message (``fsm_llm.text.capped_line``): a message may be shown to a model by a repair loop.

Imports no engine module, calls no model and executes nothing from the document. Lazy: nothing
imports it at module top (``fsm_llm.core.verify.cli`` imports it inside the function), so
``import fsm_llm`` and ``fsm-llm-validate`` on an FSM file never load it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    ExecutableDocumentError,
    ExecutableEnvelope,
    parse_executable,
    scan_placeholders,
    substitute_variables,
    variable_problems,
)
from fsm_llm.core.state_kinds import raw_kinds
from fsm_llm.text import capped_line

from .flow import analyze_data_flow, analyze_liveness
from .result import FSMValidationResult
from .validator import FSMValidator

__all__ = ["load_executable"]

#: ``(category, name) -> resolves?`` with the category ``tools``, ``handlers``, ``callables`` or
#: ``executables``.
Resolver = Callable[[str, str], bool]

_TAGS = ("schema", "body", "variables", "inputs", "contract", "names", "model")
_CATEGORIES = ("tools", "handlers", "callables", "executables")
_LINE_CHARS = 300  # one message, whatever the document put in it
_NAME_CHARS = 60  # one model-authored name inside a message


def _placeholder(name: str) -> str:
    return "{{" + name + "}}"


def _shown(value: object) -> str:
    """A model-authored name as it may appear in a message: capped, control-stripped, quoted."""
    return repr(capped_line(value, _NAME_CHARS))


def _tool_names(fsm: Mapping[str, Any]) -> set[str]:
    """The tool names the RAW states declare (so a body that does not load is still read)."""
    states = fsm.get("states")
    names: set[str] = set()
    if isinstance(states, dict):
        for state in states.values():
            if isinstance(state, dict):
                for spec in raw_kinds(state):
                    names.update(
                        n for n in spec.raw_declared_tools(state) if isinstance(n, str)
                    )
    return names


def _model_problems(
    envelope: ExecutableEnvelope, allow: Mapping[str, Collection[str]] | None
) -> list[str]:
    """V7: what a model-origin document may not claim or require."""
    requires = envelope.requires
    problems = []
    if requires.tools == "some":
        problems.append("requires.tools 'some' cannot be checked; name the tools")
    for category in ("tools", "callables", "executables"):
        names = getattr(requires, category)
        listed = (allow or {}).get(category, ())
        listed = {listed} if isinstance(listed, str) else set(listed)
        if isinstance(names, list):
            problems += [
                f"{category} name {_shown(n)} is not on the allow-list"
                for n in names
                if n not in listed
            ]
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
    return problems


def load_executable(
    data: object,
    *,
    variables: Mapping[str, str] | None = None,
    resolver: Resolver | None = None,
    origin: str = "operator",
    allow: Mapping[str, Collection[str]] | None = None,
) -> tuple[ExecutableEnvelope | None, FSMValidationResult]:
    """Load *data* as an executable and report every failure at once (see the module text)."""
    if origin not in ("operator", "model"):
        raise ValueError(f"origin must be 'operator' or 'model', not {origin!r}")
    doc_id = data.get("id") if isinstance(data, Mapping) else None
    result = FSMValidationResult(capped_line(doc_id, 80) or "Unnamed executable")
    notes: dict[str, list[tuple[str, str]]] = {tag: [] for tag in _TAGS}

    def note(tag: str, severity: str, text: str) -> None:
        notes[tag].append((severity, f"[{tag}] {capped_line(text, _LINE_CHARS)}"))

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

    # V4, the values: fill the declared variables before the body is checked
    declared = set(envelope.variables)
    fsm = envelope.body.fsm
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
        if not problems:
            fsm = substitute_variables(fsm, variables)

    # V2
    validator = FSMValidator(fsm)
    definition = None
    try:
        body = validator.validate()
    except (
        Exception
    ) as error:  # a validator never raises on a mapping; a belt, not a path
        note(
            "body", "error", f"the body could not be validated ({type(error).__name__})"
        )
    else:
        for severity, items in (
            ("error", body.errors),
            ("warning", body.warnings),
            ("info", body.info),
        ):
            for text in items:
                note("body", severity, text)
        definition = validator._definition

    # V4, the text: no caller input renders into instruction or persona text
    leaves, keys = scan_placeholders(envelope.body.fsm)
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
        note("variables", "warning", f"{_placeholder(name)} is not a declared variable")
    for name in sorted(keys & declared):
        note(
            "variables",
            "warning",
            f"{_placeholder(name)} sits in a key and is left alone (text values only)",
        )

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
    if isinstance(tools, list) and (absent := sorted(_tool_names(fsm) - set(tools))):
        shown = ", ".join(_shown(n) for n in absent)
        note("names", "error", f"completion tools {shown} are not in requires.tools")
    if resolver is None:
        note("names", "info", "names are not resolved: no resolver was given")
    else:
        for category in _CATEGORIES:
            names = getattr(envelope.requires, category)
            for name in names if isinstance(names, list) else ():
                if not resolver(category, name):
                    note(
                        "names",
                        "error",
                        f"{category} name {_shown(name)} does not resolve",
                    )

    # V7
    if origin == "model":
        for line in _model_problems(envelope, allow):
            note("model", "error", line)
    return finish(envelope)
