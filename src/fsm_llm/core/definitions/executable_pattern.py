"""The body of a ``pattern`` executable: an FSM template with holes, variants and a run block.

A pattern document (roadmap E4.1 and E4.2, plan a555dcfe D-011) is an envelope of kind
``pattern`` whose ``body`` is ``PatternBody``: exactly one of ``fsm`` (an object, the template)
and ``generator`` (a name), ``handlers`` (bindings, resolved by name later), ``options_schema``
and ``options`` (the typed values a caller may give, and their defaults), ``variants`` (named
RFC 6902 patches) and ``run`` (how an agent runs the FSM). This module holds the models and the
pure functions that turn a body and its options into a fresh FSM dict; the envelope classes
live in ``executable``, which imports this module, and the instantiation entry points live in
``executable_instance``, which converts ``PatternBodyError`` into ``InstantiationError``.

The model of a template:

- A HOLE is the leaf mapping ``{"$hole": "<option name>"}``. It stands for one whole JSON value of
  any option type, replaces the leaf in place and is deep-copied. It is never a key, never beside
  other keys, never inside another hole and never the root; its value is inserted verbatim and
  never rescanned for ``{{name}}`` text (a text variable is a different channel).
- ``options_schema`` is a closed subset (``OptionSpec``: ``string``, ``integer``, ``number``,
  ``boolean`` and ``array`` of those or of FLAT objects (D-016: ``items.type = object`` with scalar
  ``properties``, every property required, no extra key) or of ``json`` entries (D-021: any NESTED
  JSON value, at most ``MAX_JSON_DEPTH`` levels and ``MAX_JSON_NODES`` nodes, the shape of a tool
  schema); ``minimum``, ``maximum``, ``maxLength``,
  ``maxItems``, ``enum``; no ``$ref``, no top-level ``object``), each value checked by a pydantic
  ``TypeAdapter`` (strict: a bool is no integer); an option with no entry in ``options`` has no
  default and is required.
- ``variants`` are chosen only by the caller of ``instantiate_body`` (never by an option), applied
  one ``apply_json_patch`` each in DECLARATION order, before the holes are filled.

Interface contract: ``scan_holes(node, where) -> (names, problems)`` (iterative; one line per
malformed hole); ``resolve_options(body, given) -> dict`` (every option, defaults first, validated;
never mutates ``given``); ``apply_variants(body, selected) -> dict`` (the patched FSM, a new dict
when any variant ran, else ``body.fsm`` itself: read only); ``fill_holes(node, values, render_text)
-> copy`` (iterative, key order kept, a hole becomes a copy of ``values[name]``, every text leaf
goes through ``render_text``). The functions raise ``PatternBodyError`` (an ``FSMError`` and a
``ValueError``) with a message that names options, variants and paths, never a value. Imports no
``executable`` module. Lazy: nothing at ``import fsm_llm`` loads it.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.text import capped_line

from .errors import FSMError
from .limits import EXECUTABLE_MAX_ITEMS, EXECUTABLE_NAME_MAX_LENGTH
from .patch import JsonPatchError, apply_json_patch

__all__ = [
    "APPROVAL_GATE_HANDLER",
    "APPROVAL_STATE",
    "HOLE_KEY",
    "PATTERN_FIELD_VERSIONS",
    "HandlerBinding",
    "HookBinding",
    "ItemSpec",
    "OptionSpec",
    "OptionsSchema",
    "PatternBody",
    "PatternBodyError",
    "PropertySpec",
    "RunBlock",
    "apply_variants",
    "fill_holes",
    "resolve_options",
    "scan_holes",
]

#: The one key of a hole leaf.
HOLE_KEY = "$hole"

#: The two names the ``hitl`` rule of the verifier holds equal (D-012): the library handler that
#: gates a tool on a grant, and the state a run waits in for one. ``APPROVAL_STATE`` equals
#: ``fsm_llm.agents.constants.AgentStates.AWAIT_APPROVAL`` (a test holds it: core may not import
#: agents), and the handler library of the agents package registers ``APPROVAL_GATE_HANDLER``.
APPROVAL_GATE_HANDLER = "agents.approval_gate"
APPROVAL_STATE = "await_approval"

MAX_OPTIONS = 64
MAX_VARIANTS = 16
MAX_OPTION_TEXT_CHARS = 10_000  # longest string option, and the cap of ``maxLength``
MAX_OPTION_ITEMS = 256  # most entries of a list option, and the cap of ``maxItems``
MAX_ITEM_PROPERTIES = (
    16  # most properties of the object entries of a list option (D-016)
)
MAX_JSON_DEPTH = 24  # levels of one ``json`` entry (D-021)
MAX_JSON_NODES = 20_000  # values of one ``json`` entry, containers included
MAX_VARIANT_OPS = 64
MAX_VARIANT_CHARS = 200_000
_MAX_PROBLEMS = 10  # lines one scan reports
_SHOWN_CHARS = 60

_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_OPTION_NAME = re.compile(_NAME_PATTERN)
_Name = Annotated[str, Field(min_length=1, max_length=EXECUTABLE_NAME_MAX_LENGTH)]
_Names = Annotated[list[_Name], Field(max_length=EXECUTABLE_MAX_ITEMS)]
_OptionName = Annotated[str, Field(pattern=_NAME_PATTERN, max_length=64)]
_Scalar = str | int | float | bool | None


class PatternBodyError(FSMError, ValueError):
    """A pattern template, option or variant was refused (the message never holds a value)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _shown(text: object) -> str:
    return repr(capped_line(text, _SHOWN_CHARS))


# ---- the options subset ---------------------------------------------------------------------


class _ScalarSpec(_Model):
    """The bounds one scalar value may carry; which apply depends on ``type``."""

    minimum: StrictInt | StrictFloat | None = Field(None, allow_inf_nan=False)
    maximum: StrictInt | StrictFloat | None = Field(None, allow_inf_nan=False)
    maxLength: int | None = Field(None, ge=0, le=MAX_OPTION_TEXT_CHARS)
    enum: list[StrictStr | StrictInt | StrictFloat] | None = Field(
        None, min_length=1, max_length=EXECUTABLE_MAX_ITEMS
    )

    def _check_bounds(self, kind: str) -> None:
        numeric = kind in ("integer", "number")
        if (self.minimum is not None or self.maximum is not None) and not numeric:
            raise ValueError("minimum and maximum bound an integer or a number")
        if self.maxLength is not None and kind != "string":
            raise ValueError("maxLength bounds a string")
        if self.enum is not None and kind == "boolean":
            raise ValueError("enum does not apply to a boolean")
        low, high = self.minimum, self.maximum
        if low is not None and high is not None and low > high:
            raise ValueError("minimum is above maximum")

    def scalar_annotation(self, kind: str) -> Any:
        base: Any
        if kind == "string":
            base = Annotated[
                StrictStr, Field(max_length=self.maxLength or MAX_OPTION_TEXT_CHARS)
            ]
        elif kind == "boolean":
            base = StrictBool
        else:
            number = StrictInt if kind == "integer" else StrictFloat | StrictInt
            base = Annotated[number, Field(ge=self.minimum, le=self.maximum)]
        if self.enum is not None:
            allowed = self.enum

            def _listed(value: Any) -> Any:
                if not any(type(value) is type(a) and value == a for a in allowed):
                    raise ValueError("not one of the listed values")
                return value

            base = Annotated[base, AfterValidator(_listed)]
        return base


class PropertySpec(_ScalarSpec):
    """One property of an object entry (D-016): a scalar type and its bounds."""

    type: Literal["string", "integer", "number", "boolean"]

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        self._check_bounds(self.type)
        return self


def _json_entry(value: Any) -> Any:
    """*value* as a deep copy when it is bounded JSON (D-021), else ``ValueError``.

    Object keys are text, numbers are finite, nothing but dict, list, text, number, boolean and
    null; at most ``MAX_JSON_DEPTH`` levels and ``MAX_JSON_NODES`` values. One iterative walk.
    """
    stack = [(value, 1)]
    nodes = 0
    while stack:
        node, level = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError("a json entry has too many values")
        if isinstance(node, (dict, list)):
            if level > MAX_JSON_DEPTH:
                raise ValueError("a json entry nests too deep")
            if isinstance(node, dict):
                if any(type(key) is not str for key in node):
                    raise ValueError("a json object has text keys")
                node = list(node.values())
            stack.extend((child, level + 1) for child in node)
        elif isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError("a json number is finite")
        elif not (node is None or isinstance(node, (str, int))):
            raise ValueError("not a json value")
    return copy.deepcopy(value)


class ItemSpec(_ScalarSpec):
    """The entries of an ``array`` option: one scalar type and its bounds, a flat object or JSON.

    ``type = "object"`` declares ``properties`` (at most ``MAX_ITEM_PROPERTIES``, scalar only, no
    nesting): an entry is an object with exactly those keys, each required (D-016). ``type =
    "json"`` declares nothing: an entry is any bounded nested JSON value (D-021).
    """

    type: Literal["string", "integer", "number", "boolean", "object", "json"]
    properties: dict[_OptionName, PropertySpec] = Field(
        default_factory=dict, max_length=MAX_ITEM_PROPERTIES
    )

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.type in ("object", "json"):
            if self.type == "object" and not self.properties:
                raise ValueError("an object entry declares its properties")
            if self.type == "json" and self.properties:
                raise ValueError("properties belong to an object entry")
            if (
                self.minimum is not None
                or self.maximum is not None
                or self.maxLength is not None
                or self.enum is not None
            ):
                raise ValueError("bounds of an object's values go in its properties")
        else:
            if self.properties:
                raise ValueError("properties belong to an object entry")
            self._check_bounds(self.type)
        return self

    def entry_annotation(self) -> Any:
        """The annotation that checks one entry: a scalar, a dict with exactly the properties, or JSON."""
        if self.type == "json":
            return Annotated[Any, AfterValidator(_json_entry)]
        if self.type != "object":
            return self.scalar_annotation(self.type)
        checks = {
            name: TypeAdapter(spec.scalar_annotation(spec.type))
            for name, spec in self.properties.items()
        }

        def _flat(entry: dict[str, Any]) -> dict[str, Any]:
            if set(entry) != set(checks):
                raise ValueError("an entry has exactly the declared properties")
            for name, check in checks.items():
                check.validate_python(entry[name])
            return entry

        return Annotated[dict[str, Any], AfterValidator(_flat)]


class OptionSpec(_ScalarSpec):
    """One option: a scalar, or an ``array`` of scalars (``items``)."""

    type: Literal["string", "integer", "number", "boolean", "array"]
    items: ItemSpec | None = None
    maxItems: int | None = Field(None, ge=0, le=MAX_OPTION_ITEMS)
    description: str = Field("", max_length=500)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.type == "array":
            if self.items is None:
                raise ValueError("an array option declares its items")
            if (
                self.minimum is not None
                or self.maximum is not None
                or self.maxLength is not None
                or self.enum is not None
            ):
                raise ValueError("bounds of an array's entries go in items")
        else:
            if self.items is not None or self.maxItems is not None:
                raise ValueError("items and maxItems belong to an array option")
            self._check_bounds(self.type)
        return self

    def adapter(self) -> TypeAdapter[Any]:
        """The ``TypeAdapter`` that checks a value of this option (strict, bounded)."""
        if self.type == "array":
            assert self.items is not None
            entry = self.items.entry_annotation()
            cap = self.maxItems if self.maxItems is not None else MAX_OPTION_ITEMS
            listed = Annotated[list[entry], Field(max_length=cap)]  # type: ignore[valid-type]
            return TypeAdapter(listed)
        return TypeAdapter(self.scalar_annotation(self.type))


class OptionsSchema(_Model):
    """``{type: "object", properties: {name: OptionSpec}}``: the closed subset."""

    type: Literal["object"] = "object"
    properties: dict[_OptionName, OptionSpec] = Field(
        default_factory=dict, max_length=MAX_OPTIONS
    )


def check_option(name: str, spec: OptionSpec, value: object) -> Any:
    """*value* as the option *name* takes it (a copy), or ``PatternBodyError`` naming the option."""
    try:
        return spec.adapter().validate_python(value)
    except ValidationError as error:
        kinds = sorted({e["type"] for e in error.errors(include_input=False)})
        raise PatternBodyError(
            f"option {_shown(name)} is not valid ({', '.join(kinds)})"
        ) from None


# ---- bindings and the run block -------------------------------------------------------------


class HandlerBinding(_Model):
    """One handler a pattern binds: its library name, where it fires and what it is told."""

    handler: _Name
    on: Literal["entry", "exit", "state", "context_update"]
    states: _Names = Field(default_factory=list)
    keys: _Names | None = None
    priority: int | None = Field(None, ge=0, le=100_000)
    params: dict[_Name, _Scalar] = Field(default_factory=dict, max_length=64)


class HookBinding(_Model):
    """One run hook of a pattern, bound by name (the seam is built later; a field here)."""

    hook: _Name
    at: Literal["before_run", "on_iteration", "after_run"]
    params: dict[_Name, _Scalar] = Field(default_factory=dict, max_length=64)


class RunBlock(_Model):
    """How an agent runs the FSM: which keys carry the answer, the proof and the outputs."""

    agent_type: _Name | None = None
    answer_keys: _Names = Field(default_factory=list)
    success_keys: _Names = Field(default_factory=list)
    evidence_keys: _Names = Field(default_factory=list)
    judged: bool = False
    run_output_keys: _Names = Field(default_factory=list)
    context_seed: dict[_Name, _Scalar | list[_Scalar]] = Field(
        default_factory=dict, max_length=EXECUTABLE_MAX_ITEMS
    )
    step_budget: int | None = Field(None, ge=1, le=10_000)
    seconds_exempt: _Names = Field(default_factory=list)
    hooks: list[HookBinding] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def _seed_is_public_data(self) -> Self:
        """A file seeds data, never a driver key or a secret (the grant is internal-prefixed)."""
        for key, value in self.context_seed.items():
            if has_internal_prefix(key) or is_forbidden_context_entry(key, value):
                raise ValueError(
                    f"context_seed key {_shown(key)} is internal or secret-looking"
                )
        return self


# ---- the body -------------------------------------------------------------------------------


class PatternBody(_Model):
    """``{fsm | generator | extends, handlers, options_schema, options, variants, run}`` (D-011, D-033)."""

    fsm: dict[str, Any] | None = Field(
        None, description="The FSM template; holes and variants are applied to it"
    )
    generator: _Name | None = Field(
        None, description="A generator name, instead of fsm (not instantiable yet)"
    )
    extends: _Name | None = Field(
        None, description="A preset: the id of the pattern it builds on, instead of fsm"
    )
    handlers: list[HandlerBinding] = Field(
        default_factory=list, max_length=EXECUTABLE_MAX_ITEMS
    )
    options_schema: OptionsSchema = Field(default_factory=OptionsSchema)
    options: dict[_OptionName, Any] = Field(
        default_factory=dict,
        description="The default value of each option that has one",
        max_length=MAX_OPTIONS,
    )
    variants: dict[_OptionName, list[dict[str, Any]]] = Field(
        default_factory=dict, max_length=MAX_VARIANTS
    )
    run: RunBlock = Field(default_factory=RunBlock)

    @model_validator(mode="after")
    def _template_is_sound(self) -> Self:
        if sum(x is not None for x in (self.fsm, self.generator, self.extends)) != 1:
            raise ValueError(
                "a pattern body has exactly one of fsm, generator and extends"
            )
        properties = self.options_schema.properties
        if "model" in properties or "model" in self.options:
            raise ValueError("an option named 'model' is refused: one model resolver")
        if self.extends is not None:  # a preset (D-033): the base owns the template
            if self.handlers or self.variants or self.run.model_fields_set - {"hooks"}:
                raise ValueError("a preset adds only options and hooks")
            for name in set(self.options) & set(properties):
                check_option(name, properties[name], self.options[name])
            return self
        if unknown := sorted(set(self.options) - set(properties)):
            raise ValueError(f"options give defaults to undeclared options {unknown}")
        for name, value in self.options.items():
            check_option(name, properties[name], value)
        for name, ops in self.variants.items():
            if len(ops) > MAX_VARIANT_OPS:
                raise ValueError(f"variant {_shown(name)} has too many ops")
        if self.generator is not None:
            if self.variants:
                raise ValueError("a generator builds the whole definition: no variants")
            return self
        assert self.fsm is not None
        names, problems = scan_holes(self.fsm, "fsm")
        for name, ops in self.variants.items():
            found, more = scan_holes(ops, f"variants/{name}")
            names |= found
            problems += more
        if HOLE_KEY in self.fsm:
            problems.append("fsm: the root of the template is never a hole")
        if problems:
            raise ValueError("; ".join(problems[:_MAX_PROBLEMS]))
        if undeclared := sorted(names - set(properties)):
            raise ValueError(f"holes name undeclared options {undeclared}")
        return self


#: model -> {envelope version it was introduced in -> its field names}: merged into the envelope
#: table of ``executable`` (the pattern kind is part of format 1.0, which reserved it).
PATTERN_FIELD_VERSIONS: dict[str, dict[str, str]] = {
    "PatternBody": {
        "1.0": "fsm generator extends handlers options_schema options variants run"
    },
    "HandlerBinding": {"1.0": "handler on states keys priority params"},
    "HookBinding": {"1.0": "hook at params"},
    "RunBlock": {
        "1.0": (
            "agent_type answer_keys success_keys evidence_keys judged run_output_keys "
            "context_seed step_budget seconds_exempt hooks"
        )
    },
    "OptionsSchema": {"1.0": "type properties"},
    "OptionSpec": {
        "1.0": "type items minimum maximum maxLength maxItems enum description"
    },
    "ItemSpec": {"1.0": "type properties minimum maximum maxLength enum"},
    "PropertySpec": {"1.0": "type minimum maximum maxLength enum"},
}


# ---- holes, options, variants ---------------------------------------------------------------


def _hole_problem(hole: Mapping[str, Any]) -> str | None:
    """Why the mapping carrying ``$hole`` is no hole (``None``: it is one)."""
    name = hole[HOLE_KEY]
    if len(hole) != 1:
        return "a hole is a whole value, never a key and never beside other keys"
    if isinstance(name, Mapping) and HOLE_KEY in name:
        return "a hole is never inside another hole"
    if not (isinstance(name, str) and _OPTION_NAME.fullmatch(name)):
        return "a hole holds one option name"
    return None


def scan_holes(node: Any, where: str) -> tuple[set[str], list[str]]:
    """``(option names the holes of *node* use, one problem line per malformed hole)``.

    Iterative, in document order; never descends into a malformed hole (one line each) and
    stops at ten lines. *where* prefixes the path shown in a line.
    """
    names: set[str] = set()
    problems: list[str] = []
    stack: list[tuple[Any, str]] = [(node, where)]
    while stack and len(problems) < _MAX_PROBLEMS:
        current, path = stack.pop()
        if isinstance(current, Mapping):
            if HOLE_KEY in current:
                if problem := _hole_problem(current):
                    problems.append(f"{capped_line(path, 120)}: {problem}")
                else:
                    names.add(current[HOLE_KEY])
                continue
            stack.extend(
                (child, f"{path}/{key}")
                for key, child in reversed(list(current.items()))
            )
        elif isinstance(current, list):
            stack.extend(
                (child, f"{path}/{i}")
                for i, child in reversed(list(enumerate(current)))
            )
    return names, problems


def resolve_options(
    body: PatternBody, given: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Every option of *body*, in schema order: given value, else default, each validated.

    Raises ``PatternBodyError`` for an option that is not declared, an invalid value and an
    option with neither a value nor a default. *given* is never mutated; every value returned
    is a copy.
    """
    properties = body.options_schema.properties
    values = dict(given or {})
    if unknown := sorted(n for n in values if n not in properties):
        raise PatternBodyError(
            f"unknown option(s) {', '.join(_shown(n) for n in unknown[:5])}"
        )
    out: dict[str, Any] = {}
    missing = []
    for name, spec in properties.items():
        if name in values:
            out[name] = check_option(name, spec, values[name])
        elif name in body.options:
            out[name] = check_option(name, spec, body.options[name])
        else:
            missing.append(name)
    if missing:
        raise PatternBodyError(f"option(s) {missing} have no value and no default")
    return out


def apply_variants(body: PatternBody, selected: Sequence[str]) -> dict[str, Any]:
    """The FSM of *body* with the *selected* variants applied in declaration order.

    One ``apply_json_patch`` per variant. Returns ``body.fsm`` itself when nothing is selected
    (read only: ``fill_holes`` copies). Raises ``PatternBodyError`` for an undeclared name or a
    refused patch (the patch code and the variant name, never text of the patch).
    """
    assert body.fsm is not None
    wanted = set(selected)
    if unknown := sorted(wanted - set(body.variants)):
        raise PatternBodyError(f"unknown variant(s) {[_shown(n) for n in unknown[:5]]}")
    document = body.fsm
    for name, ops in body.variants.items():
        if name in wanted:
            try:
                document = apply_json_patch(
                    document, ops, max_ops=MAX_VARIANT_OPS, max_chars=MAX_VARIANT_CHARS
                )
            except JsonPatchError as error:
                at = "" if error.op is None else f" at op {error.op}"
                raise PatternBodyError(
                    f"variant {_shown(name)} was refused: {error.code}{at}"
                ) from None
    return document


def fill_holes(
    node: Any, values: Mapping[str, Any], render_text: Callable[[str], str]
) -> Any:
    """A deep copy of *node*: each hole becomes a copy of ``values[name]``, each text leaf is
    passed through *render_text*, keys are never touched.

    One iterative pass (no recursion, no pydantic dump of the tree, so no platform stack
    limit applies) that creates every container with its keys in document order, so insertion
    order, and with it the bytes of ``json.dumps``, equal the template's. A hole value is put
    in place as it is and never walked or rendered. Raises ``PatternBodyError`` for a malformed
    hole or a hole of an option with no value (a variant may have introduced either).
    """
    # DECISION plan-2026-11-24T043150-a555dcfe/D-011: ONE pass, in document order. Do NOT
    # `sorted` the keys (the E0.1 fixtures compare text, so order is identity), do NOT fill
    # holes and then render text over the result (a hole value holding `{{x}}` would be
    # rescanned), do NOT accept a hole in a key position or beside other keys, and do NOT
    # use `model_dump` or recursion here (the Windows small-stack lesson).
    holder: list[Any] = [None]
    stack: list[tuple[Any, Any, Any]] = [(node, holder, 0)]
    while stack:
        current, parent, slot = stack.pop()
        if isinstance(current, Mapping):
            if HOLE_KEY in current:
                name = current[HOLE_KEY]
                if _hole_problem(current) or name not in values:
                    raise PatternBodyError("the template holds a malformed hole")
                parent[slot] = copy.deepcopy(values[name])
                continue
            target: Any
            target = parent[slot] = {}
            for key, child in current.items():
                target[key] = None
                stack.append((child, target, key))
        elif isinstance(current, list):
            target = parent[slot] = [None] * len(current)
            stack.extend((child, target, i) for i, child in enumerate(current))
        elif isinstance(current, str):
            parent[slot] = render_text(current)
        else:
            parent[slot] = current
    return holder[0]
