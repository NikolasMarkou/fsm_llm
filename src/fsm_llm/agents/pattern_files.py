"""The shipped pattern files (``agents/patterns/<name>.json``), loaded once (plan a555dcfe step 9, D-015).

A pattern is an executable document of kind ``pattern`` (``fsm_llm.core.definitions.executable``,
``executable_pattern``): a static FSM template with typed holes, named variants, handler bindings
and a run block. This module reads the shipped ones as package data (``importlib.resources``, UTF-8,
never imported as code) and hands out what an agent and a registry need.

Interface contract:

- ``PATTERN_ORDER``: the names of the shipped files, in the order the ``agents`` source serves
  them. The directory is read by name from this tuple, never listed, so a stray file is not served;
  a test holds the tuple equal to the files on disk.
- ``load_document(name)``: a deep copy of the parsed document (``KeyError`` for a name not in
  ``PATTERN_ORDER``; a name is never joined into a path unchecked). The text of every file is
  canonical (``json.dumps(indent=2, ensure_ascii=True)``, LF, one final newline).
- ``load_envelope(name)``: the parsed, LOADED ``PatternEnvelope`` (``load_executable`` with the handler
  resolver, so the default instance and the every-variant instance are checked as FSMs), cached for
  the process and shared: read it, never mutate it. A shipped file that does not load raises
  ``ExecutableError`` naming the file and its first problems: that is a packaging bug, not an input.

Lazy: ``import fsm_llm.agents`` does not import this module (an agent loads it on construction), so
the packaged files cost nothing until a pattern is built.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from importlib import resources
from typing import Any

from fsm_llm.core.definitions.executable import ExecutableError, PatternEnvelope
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables.namespaces import namespace_resolver

__all__ = ["PATTERN_ORDER", "load_document", "load_envelope"]

PATTERN_ORDER: tuple[str, ...] = ("react", "reflexion")


# DECISION plan-2026-11-24T043150-a555dcfe/D-015
# The shape rules of the shipped pattern files are held by tests/test_fsm_llm_agents/
# test_pattern_shapes.py and a JSON file carries no anchor. Do NOT edit a file so that a rule stops
# holding (the unconditional think -> act edge at priority 300, the approval edge at 150 with one
# predicate, conclude on evidence, a silent await_approval that extracts nothing, strict === flags,
# `any` for artifact fields, no None context_keys, no state-level bulk call: the classification
# variant of `think` keeps its bulk fill, because tool_name is classification-owned there and relies
# on it when the classifier declines), do NOT drop the shape test when a file replaces a builder, and
# do NOT let an option select the `approval` variant: the agent selects it from its one HITL
# predicate (BaseAgent._hitl_active). See decisions.md D-015.
@cache
def _document(name: str) -> dict[str, Any]:
    """The parsed document of *name*, shared: never hand it out, copy it."""
    path = resources.files("fsm_llm.agents").joinpath("patterns", f"{name}.json")
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def load_document(name: str) -> dict[str, Any]:
    """The document of the shipped pattern *name* as a deep copy (``KeyError`` when unknown)."""
    if name not in PATTERN_ORDER:
        raise KeyError(name)
    return copy.deepcopy(_document(name))


@cache
def load_envelope(name: str) -> PatternEnvelope:
    """The loaded envelope of the shipped pattern *name* (cached; ``KeyError`` when unknown)."""
    document = load_document(name)
    envelope, report = load_executable(document, resolver=namespace_resolver())
    if not isinstance(envelope, PatternEnvelope) or not report.is_valid:
        raise ExecutableError(
            f"the shipped pattern file {name}.json does not load: "
            + "; ".join(report.errors[:5])
        )
    return envelope
