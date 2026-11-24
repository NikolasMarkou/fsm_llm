"""Presets: a ``pattern`` document that ``extends`` another one (plan a555dcfe step 22, D-033).

A preset names a base pattern by id and adds options, option defaults and run hooks; it carries no
template. ``resolve_extends`` walks the chain through a caller-supplied lookup and returns the FLAT
document (the base's body with the preset applied), which the caller parses and checks like any
pattern: no consumer of an envelope learns about ``extends``.

Interface contract: ``resolve_extends(preset, lookup) -> dict`` (a document dict, deep copies only;
``lookup(id) -> PatternEnvelope | None`` returns the base as written or already flat; the chain has
at most ``MAX_EXTENDS_DEPTH`` links). ``PresetError`` (a ``PatternBodyError``: an ``FSMError`` and a
``ValueError``) names the chain for a cycle, a chain that is too deep, a base that is unknown or has
no template, and an option the preset redeclares; it never holds a value. Header fields (``id``,
``name``, ``description``, ``version``, ``lifecycle``, ``metadata``) are the preset's own;
``contract`` fields and ``requires.tools`` the preset wrote replace the base's; ``requires.handlers``,
``callables``, ``executables``, ``capabilities``, ``variables`` and ``run.hooks`` are the base's then
the preset's, and ``extends`` joins ``requires.executables``. Pure, imports no engine module. Lazy:
nothing at ``import fsm_llm`` loads it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

from fsm_llm.text import capped_line

from .executable import PatternEnvelope
from .executable_pattern import PatternBodyError

__all__ = ["MAX_EXTENDS_DEPTH", "PresetError", "resolve_extends"]

#: The most ``extends`` links one chain may hold (preset to preset to ... to a template).
MAX_EXTENDS_DEPTH = 4

_OWN_HEADER = (
    "schema",
    "schema_version",
    "id",
    "version",
    "name",
    "description",
    "lifecycle",
    "metadata",
)

BaseLookup = Callable[[str], PatternEnvelope | None]


class PresetError(PatternBodyError):
    """A preset could not be resolved (the message names ids, never a value)."""


def _shown(name: object) -> str:
    return repr(capped_line(name, 60))


def _merged(*lists: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(item for items in lists for item in items))


def _overlay(base: dict[str, Any], top: PatternEnvelope) -> dict[str, Any]:
    """The flat document of *top* on the flat document *base* (see the module text)."""
    written = top.model_dump(mode="json", by_alias=True, exclude_unset=True)
    full = top.model_dump(mode="json", by_alias=True)
    doc = copy.deepcopy(base)
    doc.update({key: full[key] for key in _OWN_HEADER})
    doc["contract"] = {**base["contract"], **written.get("contract", {})}
    requires = {
        key: _merged(base["requires"][key], full["requires"][key])
        for key in ("handlers", "callables", "executables")
    }
    requires["executables"] = _merged(requires["executables"], [top.body.extends])
    requires["tools"] = written.get("requires", {}).get(
        "tools", base["requires"]["tools"]
    )
    doc["requires"] = requires
    doc["capabilities"] = _merged(base["capabilities"], full["capabilities"])
    doc["variables"] = _merged(base["variables"], full["variables"])
    body, added = doc["body"], full["body"]
    properties = body["options_schema"]["properties"]
    if clash := sorted(set(added["options_schema"]["properties"]) & set(properties)):
        raise PresetError(
            f"preset {_shown(top.id)} redeclares the base options {clash}"
        )
    properties.update(added["options_schema"]["properties"])
    body["options"].update(added["options"])
    body["run"]["hooks"] += added["run"]["hooks"]
    return doc


# DECISION plan-2026-11-24T043150-a555dcfe/D-033
# A preset is resolved ONCE, here, at load, to a flat document that the one reader (`parse_executable`
# and the rest of `load_executable`) then checks; the base's template, handlers and variants come
# from the base alone. Do NOT read `extends` anywhere else (an overlay at run time is a second
# reader of the document), do NOT let a preset replace what the base binds, do NOT lift the depth
# bound or the cycle check, and do NOT resolve a preset without a lookup (a base that cannot be found
# is refused, never guessed).
def resolve_extends(preset: PatternEnvelope, lookup: BaseLookup) -> dict[str, Any]:
    """The flat document of *preset*, its chain read through *lookup* (see the module text)."""
    chain = [preset]
    while (name := chain[-1].body.extends) is not None:
        path = " -> ".join(_shown(c.id) for c in chain)
        if any(c.id == name for c in chain):
            raise PresetError(f"extends cycle: {path} -> {_shown(name)}")
        if len(chain) > MAX_EXTENDS_DEPTH:
            raise PresetError(
                f"extends chain deeper than {MAX_EXTENDS_DEPTH} links: {path} -> {_shown(name)}"
            )
        base = lookup(name)
        if base is None:
            raise PresetError(
                f"{_shown(chain[-1].id)} extends {_shown(name)}, which is no pattern with a "
                "template that can be found"
            )
        chain.append(base)
    doc = chain[-1].model_dump(mode="json", by_alias=True)
    for top in reversed(chain[:-1]):
        doc = _overlay(doc, top)
    return doc
