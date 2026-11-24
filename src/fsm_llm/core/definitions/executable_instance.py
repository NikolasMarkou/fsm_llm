"""Identities and instantiation of an executable envelope (roadmap E1.1, plan ace7a8c8 C7).

Split from ``executable.py`` by a pure move (plan a555dcfe step 1): the two identities of
``canonical_digest`` and ``identity_of``, the ``{{name}}`` substitution and the instantiation of
an envelope into an ``FSMDefinition`` (``instantiate_envelope``, ``rebuild_instance``). The
models, the errors, ``parse_executable`` and ``scan_placeholders`` (the envelope validator reads
it) stay in ``executable.py``, which this module imports; the reverse import never happens.

A ``pattern`` envelope (plan a555dcfe D-011) instantiates through ``instantiate_body(envelope,
options=None, *, variants=(), variables=None) -> dict``: a fresh FSM DICT (so key order is
observable), built from the variants the caller selected (declaration order), the options (given,
else default, validated) and the text variables, with the record ``{"options": resolved,
"variables": {...}, "variants": [selected]}`` that ``instantiate_envelope`` returns beside the
loaded ``FSMDefinition`` and ``rebuild_instance`` replays. A generator body (D-019) is run through
the caller's ``generators`` lookup (``name -> callable | None``) on the resolved options; with no
lookup, or a name it does not know, it raises ``GeneratorUnavailableError``. Template identity never
reads options or variants chosen; instance identity is the digest of the instantiated FSM.

Interface contract: as the ``executable`` module docstring states it for ``identity_of``,
``instantiate_envelope`` and ``rebuild_instance``. ``canonical_digest(payload) -> str`` is the
64-hex sha256 of the sorted-key JSON; ``substitute_variables(node, values)`` is a pure copy;
``variable_problems(envelope, values) -> list[str]`` is empty when the values fill the declared
variables.

Imported by name (``from fsm_llm.core.definitions.executable_instance import identity_of``);
neither ``fsm_llm/__init__.py`` nor ``core/definitions/__init__.py`` imports it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from fsm_llm.core.results import Identity

from .executable import (
    _PLACEHOLDER,
    MAX_DOCUMENT_DEPTH,
    ExecutableDocumentError,
    ExecutableEnvelope,
    InstanceIdentityMismatchError,
    InstantiationError,
    _error_where,
)
from .executable_pattern import (
    PatternBody,
    PatternBodyError,
    apply_variants,
    fill_holes,
    resolve_options,
)
from .fsm import FSMDefinition, definition_digest
from .limits import nests_deeper_than

__all__ = [
    "GeneratorLookup",
    "GeneratorUnavailableError",
    "canonical_digest",
    "identity_of",
    "instantiate_body",
    "instantiate_envelope",
    "rebuild_instance",
    "substitute_variables",
    "variable_problems",
]


#: ``name -> generator`` (``None`` for an unknown name); a generator maps the resolved options of a
#: pattern to its FSM dict. The caller passes ``GENERATORS.get`` (core imports no namespace).
GeneratorLookup = Callable[
    [str], Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
]


class GeneratorUnavailableError(InstantiationError):
    """A pattern declares a generator and the caller gave no lookup that knows its name."""


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
        exclude={"metadata": True, "body": {"fsm": True, "variants": True}},
    )
    body = dump.setdefault("body", {})
    if template.body.fsm is not None:  # a pattern with a generator has none
        body["fsm"] = _json_tree(template.body.fsm)
    if isinstance(template.body, PatternBody) and template.body.variants:
        body["variants"] = _json_tree(template.body.variants)
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


def _load_definition(body: Mapping[str, Any]) -> FSMDefinition:
    """The ``FSMDefinition`` of an instantiated body, or ``InstantiationError`` (no value shown)."""
    try:
        return FSMDefinition(**body)
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


def _generated(
    body: PatternBody, resolved: Mapping[str, Any], generators: GeneratorLookup | None
) -> dict[str, Any]:
    """The checked FSM dict a generator writes from the resolved options (D-019)."""
    # DECISION plan-2026-11-24T043150-a555dcfe/D-019
    # A generator's output is run only after it is checked: a mapping, within the depth cap and a
    # valid FSM definition, handed back as a deep copy. Do NOT hand it the task, a variant or the text
    # variables, do NOT skip the FSMDefinition check (the loader would then accept what V2 never saw),
    # and do NOT put a generator's message or an option value in an error: a step text must not leak.
    name = body.generator or ""
    make = None if generators is None else generators(name)
    if make is None:
        raise GeneratorUnavailableError(
            f"the generator {name!r} cannot run: the caller gave no lookup that knows it"
        )
    try:
        written = make(resolved)  # resolve_options returns copies
    except (
        Exception
    ) as error:  # a generator is operator code: report its type, never its text
        raise InstantiationError(
            f"the generator {name!r} failed ({type(error).__name__})"
        ) from None
    if not isinstance(written, Mapping) or nests_deeper_than(
        written, MAX_DOCUMENT_DEPTH
    ):
        raise InstantiationError(
            f"the generator {name!r} wrote no mapping of at most "
            f"{MAX_DOCUMENT_DEPTH} levels"
        )
    out = copy.deepcopy(dict(written))
    _load_definition(out)
    return out


def _pattern_instance(
    envelope: ExecutableEnvelope,
    options: Mapping[str, Any] | None,
    variables: Mapping[str, str] | None,
    variants: Sequence[str],
    generators: GeneratorLookup | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The instantiated FSM dict of a pattern and its record (the work of ``instantiate_body``)."""
    body = envelope.body
    if not isinstance(body, PatternBody):
        raise InstantiationError(
            "options and variants belong to a pattern: this envelope has none"
        )
    values = dict(variables or {})
    if problems := variable_problems(envelope, values):
        raise InstantiationError("; ".join(problems))
    if body.fsm is None and variants:
        raise InstantiationError("a generator body takes no variants")
    try:
        resolved = resolve_options(body, options)
        # the record is taken BEFORE the fill pass, from values the pass only copies
        record = {
            "options": copy.deepcopy(resolved),
            "variables": dict(sorted(values.items())),
            "variants": [name for name in body.variants if name in set(variants)],
        }
        if body.fsm is None:
            return _generated(body, resolved, generators), record
        patched = apply_variants(body, variants)
        if nests_deeper_than(patched, MAX_DOCUMENT_DEPTH):
            raise InstantiationError(
                f"the body nests deeper than {MAX_DOCUMENT_DEPTH} levels"
            )
        filled = fill_holes(
            patched, resolved, lambda text: substitute_variables(text, values)
        )
    except PatternBodyError as error:
        raise InstantiationError(str(error)) from None
    return filled, record


def instantiate_body(
    envelope: ExecutableEnvelope,
    options: Mapping[str, Any] | None = None,
    *,
    variants: Sequence[str] = (),
    variables: Mapping[str, str] | None = None,
    generators: GeneratorLookup | None = None,
) -> dict[str, Any]:
    """The FSM of a pattern *envelope* as a fresh dict: variants, then holes and text.

    *variants* are names the caller (the adapter) selects, applied in declaration order; an
    option can never select one. *options* are checked against the options schema and fall back
    to the defaults of the body. A generator body is written by ``generators(name)`` from the resolved
    options (no variants; ``GeneratorUnavailableError`` when the lookup does not know it). Two calls
    return equal dicts that share nothing with each other or with the envelope.
    """
    return _pattern_instance(envelope, options, variables, variants, generators)[0]


def instantiate_envelope(
    envelope: ExecutableEnvelope,
    *,
    options: Mapping[str, Any] | None = None,
    variables: Mapping[str, str] | None = None,
    variants: Sequence[str] = (),
    generators: GeneratorLookup | None = None,
) -> tuple[FSMDefinition, dict[str, Any]]:
    """The FSM body of *envelope* with its variables filled, and the instantiation record.

    A pattern also takes *options* and *variants* (see ``instantiate_body``) and records both.
    """
    if isinstance(envelope.body, PatternBody):
        filled, record = _pattern_instance(
            envelope, options, variables, variants, generators
        )
        return _load_definition(filled), record
    values = dict(variables or {})
    if options or variants:
        raise InstantiationError(
            "options and variants are not accepted: only a pattern has an option schema"
        )
    if problems := variable_problems(envelope, values):
        raise InstantiationError("; ".join(problems))
    if nests_deeper_than(envelope.body.fsm, MAX_DOCUMENT_DEPTH):
        raise InstantiationError(
            f"the body nests deeper than {MAX_DOCUMENT_DEPTH} levels"
        )
    definition = _load_definition(substitute_variables(envelope.body.fsm, values))
    return definition, {"options": {}, "variables": dict(sorted(values.items()))}


def rebuild_instance(
    envelope: ExecutableEnvelope,
    record: Mapping[str, Any],
    expected_instance_identity: str,
    generators: GeneratorLookup | None = None,
) -> FSMDefinition:
    """Re-instantiate from a logged *record*; the digest must equal the logged identity."""
    if (
        not isinstance(record, Mapping)
        or set(record) - {"options", "variables", "variants"}
        or not isinstance(record.get("options") or {}, Mapping)
        or not isinstance(record.get("variables") or {}, Mapping)
        or not isinstance(record.get("variants") or [], list)
    ):
        raise InstantiationError(
            "an instantiation record is {options: {...}, variables: {...}} "
            "(a pattern adds variants: [...])"
        )
    definition, _ = instantiate_envelope(
        envelope,
        options=record.get("options"),
        variables=record.get("variables"),
        variants=record.get("variants") or (),
        generators=generators,
    )
    actual = definition_digest(definition)
    if actual != expected_instance_identity:
        raise InstanceIdentityMismatchError(
            "the rebuilt instance does not hash to the logged instance identity",
            expected=expected_instance_identity,
            actual=actual,
        )
    return definition
