"""The executable envelope: one JSON document around an FSM body (roadmap E1.1, plan ace7a8c8 C7).

An executable is a header (``schema``, ``schema_version``, ``kind``, ``id``, ``version``,
``name``, ``description``, ``contract``, ``requires``, ``capabilities``, ``lifecycle``,
``variables``, ``metadata``) and a ``body`` that the ``kind`` discriminates: ``fsm`` (``{fsm}``),
``strategy`` (``{fsm, strategy}``) and ``procedure`` (``{fsm, provenance}``). ``pattern`` and
``workflow`` are refused by name until their adapters exist. ``FSMDefinition`` and ``State`` gain
no field: ``body.fsm`` is the FSM dict, checked by the body loader (V2, ``core.verify.executable``)
and loaded exactly as ``API`` loads it by ``instantiate_envelope``.

Two identities (roadmap C11): the TEMPLATE identity is ``canonical_digest`` of the envelope's
canonical dump (``exclude_defaults``, by alias, so a later defaulted field moves nothing), the
INSTANCE identity is ``definition_digest`` of the instantiated FSM body. ``identity_of`` yields
both as a ``core.results.Identity`` and is the only non-FSM recipe. A caller's task never enters
either: ``contract.input_templates`` render caller arguments into an input VALUE, ``variables`` are
operator values and may not be named like an input (Principle 14).

``lifecycle``, ``capabilities``, ``contract`` and ``requires`` are CLAIMS of the document, never
authorization. Document text is data: nothing here executes, imports or resolves it by name.

Interface contract:

- ``parse_executable(data) -> ExecutableEnvelope``: V1. Raises ``ExecutableDocumentError`` (its
  ``problems`` are one line per refusal) for a non-mapping, a document nested deeper than
  ``MAX_DOCUMENT_DEPTH``, a ``kind`` not built yet, or any model error. Never raises anything else.
- ``instantiate_envelope(envelope, *, options=None, variables=None) -> (FSMDefinition, record)``:
  ``{{name}}`` placeholders of DECLARED variables are replaced in the string leaves of
  ``body.fsm`` (never keys; a value is never rescanned); a declared name without a value, a value
  for an undeclared name, a non-text value, non-empty ``options`` (no kind has an option schema
  yet) or a body that does not load raise ``InstantiationError``. ``record`` is
  ``{"options": {}, "variables": {sorted}}``. ``rebuild_instance`` re-instantiates from a record
  and raises ``InstanceIdentityMismatchError`` unless the digest equals the logged one.
- ``newer_envelope_fields(envelope)``: findings for fields newer than the declared
  ``schema_version`` (its own table, never ``schema.newer_fields``, which reads ``.version``).
- ``executable_json_schema()`` / ``schema_text()``: the generated JSON Schema and its canonical
  text; ``SCHEMA_PATH`` is the shipped file, written by ``python -m tests.executables_corpus
  --write``, and held equal to the models by a test (its bytes depend on the pinned pydantic).

Imported by name (``from fsm_llm.core.definitions.executable import parse_executable``); neither
``fsm_llm/__init__.py`` nor ``core/definitions/__init__.py`` imports it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.results import Identity
from fsm_llm.text import capped_line

from .errors import FSMError
from .fsm import FSMDefinition, definition_digest
from .limits import nests_deeper_than
from .schema import SchemaFinding, parse_version

__all__ = [
    "ENVELOPE_FIELD_VERSIONS",
    "ENVELOPE_SCHEMA_VERSION_DEFAULT",
    "ENVELOPE_SCHEMA_VERSION_OLDEST",
    "MAX_DOCUMENT_DEPTH",
    "PROBLEM_LINE_CHARS",
    "Capability",
    "Contract",
    "ExecutableDocument",
    "ExecutableDocumentError",
    "ExecutableEnvelope",
    "ExecutableError",
    "FsmBody",
    "FsmEnvelope",
    "InstanceIdentityMismatchError",
    "InstantiationError",
    "ProcedureBody",
    "ProcedureEnvelope",
    "Requires",
    "StrategyBody",
    "StrategyDecl",
    "StrategyEnvelope",
    "canonical_digest",
    "executable_json_schema",
    "identity_of",
    "instantiate_envelope",
    "newer_envelope_fields",
    "parse_executable",
    "rebuild_instance",
    "scan_placeholders",
    "schema_text",
    "substitute_variables",
    "variable_problems",
]

#: The format version an omitted ``schema_version`` takes, and the oldest the table knows. Their
#: own pair, never the FSM definition version (``schema.SCHEMA_VERSION_DEFAULT``).
ENVELOPE_SCHEMA_VERSION_DEFAULT = "1.0"
ENVELOPE_SCHEMA_VERSION_OLDEST = "1.0"

#: Deepest container nesting a document may have. A valid FSM body nests about 100 levels
#: (a JsonLogic maximum of 50 operators) below the header. The identity walks the free-form
#: body itself (``_json_tree``), so no serializer depth guard of the platform applies (D-056).
MAX_DOCUMENT_DEPTH = 200

#: Longest problem line of a refused document (and of every ``load_executable`` message), and
#: the longest key or path part inside one: a line carries document text, so it is bounded.
PROBLEM_LINE_CHARS = 300
_PROBLEM_PART_CHARS = 60

#: The shipped JSON Schema, generated from the models (package data).
SCHEMA_PATH = Path(__file__).with_name("schemas") / "executable.schema.json"

_KINDS = ("fsm", "strategy", "procedure")
_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_MAX_ITEMS = 256
_Name = Annotated[str, Field(min_length=1, max_length=128)]
_Names = Annotated[list[_Name], Field(max_length=_MAX_ITEMS)]
_VariableName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]


class ExecutableError(FSMError):
    """Root of the executable format errors."""


class ExecutableDocumentError(ExecutableError, ValueError):
    """V1 refused a document; ``problems`` holds one line per refusal."""

    def __init__(
        self,
        message: str,
        *,
        problems: Sequence[str] = (),
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.problems = tuple(problems)


class InstantiationError(ExecutableError, ValueError):
    """Variables, options or the instantiated body were refused."""


class InstanceIdentityMismatchError(InstantiationError):
    """A rebuilt instance does not hash to the identity that was logged."""

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.expected = expected
        self.actual = actual


class Capability(str, Enum):
    """What an executable claims to do (a claim, never authorization)."""

    TOOLS = "tools"
    HITL = "hitl"
    STREAMING = "streaming"
    STRATEGY = "strategy"  # selectable by reasoning
    CONVERSATIONAL = "conversational"  # talks to a user

    @classmethod
    def from_string(cls, value: Capability | str) -> Capability:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        try:
            return cls(value)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(f"unknown capability {value!r}; valid: {valid}") from None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Contract(_Model):
    """The keys an executable reads and writes (checked at load against the flow analysis)."""

    inputs: _Names = Field(default_factory=list, description="Keys the caller supplies")
    outputs: _Names = Field(default_factory=list, description="Keys the run returns")
    handler_only_keys: _Names = Field(
        default_factory=list, description="Keys only a handler may write"
    )
    input_templates: dict[_Name, Annotated[str, Field(max_length=2000)]] = Field(
        default_factory=dict,
        description=(
            "input name -> text with {{placeholders}}, rendered from caller "
            "arguments into that input's VALUE, never into instruction text"
        ),
    )


class Requires(_Model):
    """Every name the body uses, resolved at load. ``tools`` may be a bare claim."""

    tools: _Names | Literal["none", "some"] = Field(default_factory=list)
    handlers: _Names = Field(default_factory=list)
    callables: _Names = Field(default_factory=list)
    executables: _Names = Field(default_factory=list)


class StrategyDecl(_Model):
    """How reasoning selects and reads a ``strategy`` executable."""

    aliases: _Names = Field(default_factory=list)
    results: dict[_Name, _Name] = Field(
        default_factory=dict, description="orchestrator key -> key read from the FSM"
    )
    solution_key: _Name


class FsmBody(_Model):
    """``{fsm}``. The FSM is an object here; V2 loads and validates it as ``API`` would."""

    fsm: dict[str, Any] = Field(
        description="An FSM definition, checked by the body loader and not by this schema"
    )


class StrategyBody(FsmBody):
    strategy: StrategyDecl


class ProcedureBody(FsmBody):
    provenance: dict[str, str | list[str] | None] = Field(default_factory=dict)


class ExecutableEnvelope(_Model):
    """The common header. Use a per-kind class (``FsmEnvelope`` ...) or ``parse_executable``.

    ``schema_`` is stored under that name because ``schema`` shadows a ``BaseModel``
    attribute; its alias is the wire key ``schema``. ``kind`` and ``body`` are narrowed by the
    kind classes; the identities are methods of this module, never fields.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["fsm-llm/executable"] = Field(
        alias="schema", description="The envelope format id"
    )
    schema_version: str = Field(
        ENVELOPE_SCHEMA_VERSION_DEFAULT,
        pattern=r"^[0-9]+\.[0-9]+$",
        description="The envelope format version (not the FSM definition version)",
    )
    kind: str
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$", max_length=128)
    version: str = Field("1", min_length=1, max_length=64, description="Author's label")
    name: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)
    contract: Contract = Field(default_factory=Contract)
    requires: Requires = Field(default_factory=Requires)
    capabilities: list[Capability] = Field(default_factory=list, max_length=_MAX_ITEMS)
    lifecycle: Lifecycle = Lifecycle.DRAFT
    variables: Annotated[list[_VariableName], Field(max_length=_MAX_ITEMS)] = Field(
        default_factory=list,
        description="Operator-filled {{name}} placeholders; never a caller's input",
    )
    body: FsmBody
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _readable_version(cls, value: str) -> str:
        parsed = parse_version(value)
        newest = parse_version(ENVELOPE_SCHEMA_VERSION_DEFAULT)
        oldest = parse_version(ENVELOPE_SCHEMA_VERSION_OLDEST)
        if parsed is None or newest is None or oldest is None:
            raise ValueError(f"{value!r} is not an N.M version")
        if parsed[0] != newest[0] or parsed < oldest:
            raise ValueError(
                f"version {value!r} is not readable (this release reads "
                f"{ENVELOPE_SCHEMA_VERSION_OLDEST} to {newest[0]}.x)"
            )
        return value

    @model_validator(mode="after")
    def _caller_input_is_data(self) -> Self:
        """Principle 14: operator variables are never named like a caller's input."""
        inputs = set(self.contract.inputs)
        templates = self.contract.input_templates
        if stray := sorted(set(templates) - inputs):
            raise ValueError(f"input_templates name undeclared inputs: {stray}")
        arguments = scan_placeholders(list(templates.values()))[0]
        if clash := sorted(set(self.variables) & (inputs | arguments)):
            raise ValueError(
                f"variables {clash} are named like a contract input or an input "
                "template placeholder; a caller's input is never a variable"
            )
        return self


class FsmEnvelope(ExecutableEnvelope):
    kind: Literal["fsm"]
    body: FsmBody


class StrategyEnvelope(ExecutableEnvelope):
    kind: Literal["strategy"]
    body: StrategyBody


class ProcedureEnvelope(ExecutableEnvelope):
    kind: Literal["procedure"]
    body: ProcedureBody


#: The union V1 parses: one class per kind, discriminated on the header field ``kind``.
ExecutableDocument = Annotated[
    FsmEnvelope | StrategyEnvelope | ProcedureEnvelope, Field(discriminator="kind")
]
_ADAPTER: TypeAdapter[ExecutableEnvelope] = TypeAdapter(ExecutableDocument)


def _error_where(loc: Sequence[Any]) -> str:
    """The path of an error as plain text: each part control-stripped and capped."""
    parts = (capped_line(str(part), _PROBLEM_PART_CHARS) for part in loc)
    return ".".join(parts) or "document"


def _problems(error: ValidationError) -> tuple[str, ...]:
    """One line per refusal. Key names and messages come from the document (the discriminator
    error echoes the ``kind`` value), so each part and each line is control-stripped and capped."""
    lines = []
    for item in error.errors(
        include_url=False, include_context=False, include_input=False
    ):
        loc = item["loc"]
        if loc and loc[0] in _KINDS:  # the union's tag, not a path
            loc = loc[1:]
        where = _error_where(loc)
        message = item["msg"].removeprefix("Value error, ")
        lines.append(capped_line(f"{where}: {message}", PROBLEM_LINE_CHARS))
    return tuple(lines)


def parse_executable(data: object) -> ExecutableEnvelope:
    """V1: *data* as the envelope its ``kind`` names, or ``ExecutableDocumentError``."""
    if not isinstance(data, Mapping):
        raise ExecutableDocumentError(
            "an executable document must be a JSON object",
            problems=("document: must be a JSON object",),
        )
    kind = data.get("kind")
    if isinstance(kind, str) and kind in ("pattern", "workflow"):
        line = (
            f"kind: {kind!r} is not supported by this version "
            f"(known kinds: {', '.join(_KINDS)})"
        )
        raise ExecutableDocumentError(line, problems=(line,))
    try:
        if nests_deeper_than(data, MAX_DOCUMENT_DEPTH):
            raise RecursionError
        return _ADAPTER.validate_python(data)
    except RecursionError:
        line = f"document: nests deeper than {MAX_DOCUMENT_DEPTH} levels"
        raise ExecutableDocumentError(line, problems=(line,)) from None
    except ValidationError as error:
        problems = _problems(error)
        raise ExecutableDocumentError(
            f"not a valid executable document: {problems[0]}", problems=problems
        ) from None


# ---- the envelope version table (its own, like `schema.FIELD_VERSIONS`) ----------------------

# model -> {introduced envelope version -> its field names, space separated}
_ENVELOPE_TABLE: dict[str, dict[str, str]] = {
    "ExecutableEnvelope": {
        "1.0": (
            "schema_ schema_version kind id version name description contract requires "
            "capabilities lifecycle variables body metadata"
        )
    },
    "Contract": {"1.0": "inputs outputs handler_only_keys input_templates"},
    "Requires": {"1.0": "tools handlers callables executables"},
    "FsmBody": {"1.0": "fsm"},
    "StrategyBody": {"1.0": "strategy"},
    "StrategyDecl": {"1.0": "aliases results solution_key"},
    "ProcedureBody": {"1.0": "provenance"},
}

#: "Model.field" -> the envelope version that introduced it. A kind class inherits the entries
#: of its bases; a test walks the models and fails on a field that has none.
ENVELOPE_FIELD_VERSIONS: dict[str, str] = {
    f"{model}.{name}": version
    for model, tiers in _ENVELOPE_TABLE.items()
    for version, names in tiers.items()
    for name in names.split()
}


def newer_envelope_fields(envelope: ExecutableEnvelope) -> tuple[SchemaFinding, ...]:
    """The fields *envelope* sets that its declared ``schema_version`` predates.

    Reads ``schema_version`` (never the author label ``version``) and walks the header models;
    ``body.fsm`` is an FSM and keeps its own ``newer_fields``. Never raises on a parsed envelope.
    """
    text = envelope.schema_version
    declared = parse_version(text)
    found: dict[tuple[str, str], SchemaFinding] = {}
    if declared is not None:
        _walk(envelope, text, declared, found)
    return tuple(found.values())


def _walk(
    model: BaseModel,
    text: str,
    declared: tuple[int, int],
    found: dict[tuple[str, str], SchemaFinding],
) -> None:
    for name in type(model).model_fields:
        field = type(model).model_fields[name]
        if name not in model.model_fields_set:
            continue
        value = getattr(model, name)
        if value is None or (
            not field.is_required()
            and value == field.get_default(call_default_factory=True)
        ):
            continue
        owner, introduced = next(
            (
                (klass.__name__, ENVELOPE_FIELD_VERSIONS[f"{klass.__name__}.{name}"])
                for klass in type(model).__mro__
                if f"{klass.__name__}.{name}" in ENVELOPE_FIELD_VERSIONS
            ),
            (type(model).__name__, None),
        )
        parsed = parse_version(introduced)
        if parsed is not None and parsed > declared:
            found.setdefault(
                (owner, name), SchemaFinding(owner, name, str(introduced), text)
            )
            continue
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, BaseModel):
                _walk(child, text, declared, found)


# ---- the generated schema -------------------------------------------------------------------


def executable_json_schema() -> dict[str, Any]:
    """The JSON Schema of ``ExecutableDocument`` (a ``oneOf`` over the kinds)."""
    return _ADAPTER.json_schema()


def schema_text() -> str:
    """The canonical text of the shipped schema: sorted keys, indent 2, ASCII, one newline."""
    return (
        json.dumps(
            executable_json_schema(), indent=2, sort_keys=True, ensure_ascii=True
        )
        + "\n"
    )


# ---- identities and instantiation -----------------------------------------------------------


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """The 64-hex sha256 of ``json.dumps(payload, sort_keys=True)``: the one non-FSM recipe.

    The same two lines as ``definition_canonical`` (which carries the byte-pinned D-030 anchor on
    the hot path of every conversation id and cannot import this lazy module); the test
    ``test_the_canonical_digest_of_an_fsm_dump_is_the_fsm_recipe`` holds them equal.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _instance_digest(instance: FSMDefinition | Mapping[str, Any]) -> str:
    if isinstance(instance, FSMDefinition):
        return definition_digest(instance)
    try:
        return definition_digest(FSMDefinition(**instance))
    except (ValueError, TypeError, RecursionError):  # not an FSM: hash it as written
        return canonical_digest({str(k): v for k, v in instance.items()})


def _json_tree(value: Any) -> Any:
    """A JSON-safe copy of a free-form tree, walked with an explicit stack (no recursion).

    A ``Mapping`` becomes a dict with str keys, a non-text ``Sequence`` a list; a leaf goes
    through ``to_jsonable_python`` (what ``model_dump(mode="json")`` does to it) and sees no
    container. Raises ``ExecutableDocumentError`` below ``MAX_DOCUMENT_DEPTH`` levels, which
    also stops a cyclic structure.
    """
    holder: list[Any] = [None]
    stack: list[tuple[Any, Any, Any, int]] = [(value, holder, 0, 1)]
    while stack:
        node, parent, slot, level = stack.pop()
        if isinstance(node, Mapping):
            children: Any = node.items()
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            children = enumerate(node)
        else:
            parent[slot] = to_jsonable_python(node)
            continue
        if level > MAX_DOCUMENT_DEPTH:
            line = f"document: nests deeper than {MAX_DOCUMENT_DEPTH} levels"
            raise ExecutableDocumentError(line, problems=(line,))
        target: Any
        if isinstance(node, Mapping):
            target = parent[slot] = {}
            for key, child in children:
                name = next(iter(to_jsonable_python({key: None})))
                target[name] = None
                stack.append((child, target, name, level + 1))
        else:
            target = parent[slot] = []
            for index, child in children:
                target.append(None)
                stack.append((child, target, index, level + 1))
    return holder[0]


def identity_of(
    template: ExecutableEnvelope,
    instance: FSMDefinition | Mapping[str, Any] | None = None,
) -> Identity:
    """Both identities: of *template* (its canonical dump) and, when given, of *instance*.

    The instance identity of an FSM body is ``definition_digest`` of its ``FSMDefinition``
    whether the caller holds the definition or a dict that loads as one (D-054: one instance, one
    identity, so a log keyed by either form matches). Only a mapping that does not load as an FSM
    (the workflow body) is hashed as written by ``canonical_digest``; ``None`` when no instance is
    given. Never raises on a mapping.
    """
    # DECISION plan-2026-11-20T055240-ace7a8c8/D-056: pydantic dumps only the shallow header;
    # `metadata` and `body.fsm` (free-form `dict[str, Any]`) go through `_json_tree`. Do NOT dump
    # them with `model_dump`: its recursion guard is a per-platform constant (Windows refuses
    # what Linux dumps, and a small stack crashes the process), so an identity would depend on
    # the machine. The result is byte equal to the old whole-model dump.
    dump = template.model_dump(
        mode="json",
        by_alias=True,
        exclude_defaults=True,
        exclude={"metadata": True, "body": {"fsm": True}},
    )
    dump.setdefault("body", {})["fsm"] = _json_tree(template.body.fsm)
    if template.metadata:
        dump["metadata"] = _json_tree(template.metadata)
    digest = None if instance is None else _instance_digest(instance)
    return Identity(template_identity=canonical_digest(dump), instance_identity=digest)


def substitute_variables(node: Any, values: Mapping[str, str]) -> Any:
    """A copy of *node* with ``{{name}}`` replaced by ``values[name]`` in every string LEAF.

    Keys are left alone, an unknown ``{{name}}`` stays as text and a replacement is never
    rescanned. A Mapping becomes a dict, a list stays a list and a tuple a tuple. Recursive: the
    caller bounds the depth (``MAX_DOCUMENT_DEPTH``).
    """
    if isinstance(node, str):
        return _PLACEHOLDER.sub(lambda m: values.get(m[1], m[0]), node)
    if isinstance(node, Mapping):
        return {key: substitute_variables(item, values) for key, item in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(substitute_variables(item, values) for item in node)
    return node


def scan_placeholders(node: Any) -> tuple[set[str], set[str]]:
    """``(names in string leaves, names in dict keys)`` of the ``{{name}}`` placeholders in *node*."""
    leaves: set[str] = set()
    keys: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            leaves.update(_PLACEHOLDER.findall(current))
        elif isinstance(current, Mapping):
            for key, item in current.items():
                if isinstance(key, str):
                    keys.update(_PLACEHOLDER.findall(key))
                stack.append(item)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return leaves, keys


def variable_problems(
    envelope: ExecutableEnvelope, values: Mapping[str, Any]
) -> list[str]:
    """Every reason *values* cannot fill the declared variables of *envelope* (empty: they can).

    A value for an undeclared name (a caller's input is never one), a declared name without a
    value and a value that is not text. Messages name variables only, never a value.
    """
    declared = set(envelope.variables)
    problems = []
    if extra := sorted(set(values) - declared):
        problems.append(f"variables {extra} are not declared by this executable")
    if missing := sorted(declared - set(values)):
        problems.append(f"declared variables {missing} have no value")
    if bad := sorted(
        n for n, v in values.items() if n in declared and not isinstance(v, str)
    ):
        problems.append(f"variables {bad} must be text")
    return problems


def instantiate_envelope(
    envelope: ExecutableEnvelope,
    *,
    options: Mapping[str, Any] | None = None,
    variables: Mapping[str, str] | None = None,
) -> tuple[FSMDefinition, dict[str, Any]]:
    """The FSM body of *envelope* with its variables filled, and the instantiation record."""
    values = dict(variables or {})
    if options:
        raise InstantiationError(
            "options are not accepted: no executable kind has an option schema yet"
        )
    if problems := variable_problems(envelope, values):
        raise InstantiationError("; ".join(problems))
    if nests_deeper_than(envelope.body.fsm, MAX_DOCUMENT_DEPTH):
        raise InstantiationError(
            f"the body nests deeper than {MAX_DOCUMENT_DEPTH} levels"
        )
    body = substitute_variables(envelope.body.fsm, values)
    try:
        definition = FSMDefinition(**body)
    except ValidationError as error:
        # locations and error types only: a message or an input value can carry a variable VALUE
        items = error.errors(
            include_url=False, include_context=False, include_input=False
        )
        shown = "; ".join(
            _error_where(item["loc"]) + f": {item['type']}" for item in items[:5]
        )
        raise InstantiationError(
            f"the instantiated body is not a valid FSM definition ({shown})"
        ) from None
    except (ValueError, TypeError, RecursionError) as error:
        raise InstantiationError(
            "the instantiated body is not a valid FSM definition "
            f"({type(error).__name__})"
        ) from None
    return definition, {"options": {}, "variables": dict(sorted(values.items()))}


def rebuild_instance(
    envelope: ExecutableEnvelope,
    record: Mapping[str, Any],
    expected_instance_identity: str,
) -> FSMDefinition:
    """Re-instantiate from a logged *record*; the digest must equal the logged identity."""
    if (
        not isinstance(record, Mapping)
        or set(record) - {"options", "variables"}
        or not isinstance(record.get("options") or {}, Mapping)
        or not isinstance(record.get("variables") or {}, Mapping)
    ):
        raise InstantiationError(
            "an instantiation record is {options: {...}, variables: {...}}"
        )
    definition, _ = instantiate_envelope(
        envelope, options=record.get("options"), variables=record.get("variables")
    )
    actual = definition_digest(definition)
    if actual != expected_instance_identity:
        raise InstanceIdentityMismatchError(
            "the rebuilt instance does not hash to the logged instance identity",
            expected=expected_instance_identity,
            actual=actual,
        )
    return definition
