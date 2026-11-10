"""
Synthesis policy: the limits, the G5 policy gate and abort injection.

A compile is model-written JSON, so it is untrusted. This module holds the
closed rules every compile must meet (``check_policy``, gate G5), the one
decision about when a human must approve it (``needs_approval``), and the
silent ``abort`` terminal the runner relies on (``inject_abort`` and its exact
inverse ``strip_abort``). Pure functions over plain dicts: no LLM, no I/O, no
logging. Nothing here raises on a hostile compile except ``inject_abort``,
which is only called after the gates passed.

# DECISION plan-2026-11-10T194139-2cefeb61/D-007
# The abort path is the one thing a compiled procedure must not be able to
# forge. The key lives in `handler_only_keys` (the pipeline mints no
# extraction for it, so no model reply can set it) and the edge has priority 0
# (compiled edges need priority 1 or more, code `priority-reserved`), so it
# wins outright and never ties into the classifier. Do NOT gate the abort edge
# on a state or a value a model can write, do NOT give it the same priority
# as a compiled edge, and do NOT run G3 or G4 on the injected form (they would
# see the abort edge as a path). The text rules are one generic walk over the
# whole document on purpose: a per-field list would miss the next field.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fsm_llm.agents.exceptions import require_int_limit
from fsm_llm.constants import CONTEXT_KEY_AGENT_TRACE, RESERVED_CONTEXT_KEYS
from fsm_llm.definitions import (
    FSMDefinition,
    State,
    checked_key_names,
    logic_referenced_keys,
)
from fsm_llm.okf import capped_line, has_lone_surrogate, strip_unsafe_text
from fsm_llm.security import has_internal_prefix, is_forbidden_context_entry

from .constants import ABORT_CONTEXT_KEY, ABORT_STATE_ID, RESULT_CONTEXT_KEY
from .exceptions import SynthesisError

#: Integer limits and their minimum. ``max_repair_rounds`` may be 0 (no repair).
_INT_LIMITS = {
    "max_states": 1,
    "max_transitions_per_state": 1,
    "max_text_chars": 1,
    "max_synthesis_depth": 1,
    "max_repair_rounds": 0,
    "max_turns": 1,
    "max_compile_tokens": 1,
    "max_synthesis_seconds": 1,
}

#: Names a compile may never declare or read: the framework keys and ours.
_RESERVED_NAMES = frozenset(
    {ABORT_CONTEXT_KEY, RESULT_CONTEXT_KEY, CONTEXT_KEY_AGENT_TRACE}
) | frozenset(RESERVED_CONTEXT_KEYS)


@dataclass(frozen=True)
class SynthesisPolicy:
    """The closed rules for one synthesizer.

    Interface contract (callers: the compiler, G5, the runner, the tools):
        Every integer limit must be an ``int`` (not a ``bool``, ``float`` or
        ``str``) from its minimum (1; ``max_repair_rounds`` 0) to ``2**63 - 1``.
        ``reuse_min_score`` is an ``int`` or ``float`` from 0 to 1.
        ``trust_after_successes`` is ``None`` or an ``int`` from 1.
        ``allowed_tools`` is a collection of non-blank names (a bare string is
        refused); it is stored as a ``frozenset``. ``require_approval`` is a
        ``bool``. A bad value raises ``SynthesisError`` at construction and is
        never clamped; ``dataclasses.replace`` validates again.
    """

    max_states: int = 12
    max_transitions_per_state: int = 6
    max_text_chars: int = 600
    max_synthesis_depth: int = 1
    max_repair_rounds: int = 3
    max_turns: int = 40
    max_compile_tokens: int = 8192
    max_synthesis_seconds: int = 300
    reuse_min_score: float = 0.8
    trust_after_successes: int | None = None
    allowed_tools: frozenset[str] = frozenset()
    require_approval: bool = False

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        def refuse(name: str, expect: str) -> SynthesisError:
            shown = capped_line(repr(getattr(self, name)), 60)
            return SynthesisError(f"policy.{name} must be {expect}, got {shown}")

        for name, minimum in _INT_LIMITS.items():
            put(
                name,
                require_int_limit(
                    f"policy.{name}",
                    getattr(self, name),
                    minimum=minimum,
                    error=SynthesisError,
                ),
            )
        score = self.reuse_min_score
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1  # False for NaN; an int is compared exactly
        ):
            raise refuse("reuse_min_score", "a number from 0 to 1")
        put("reuse_min_score", float(score))
        if self.trust_after_successes is not None:
            put(
                "trust_after_successes",
                require_int_limit(
                    "policy.trust_after_successes",
                    self.trust_after_successes,
                    minimum=1,
                    error=SynthesisError,
                ),
            )
        try:
            tools = frozenset(
                checked_key_names(self.allowed_tools, argument="allowed_tools")
            )
        except ValueError:
            raise refuse("allowed_tools", "a collection of tool names") from None
        if not all(name.strip() for name in tools):
            raise refuse("allowed_tools", "a collection of non-blank names")
        put("allowed_tools", tools)
        if not isinstance(self.require_approval, bool):
            raise refuse("require_approval", "a bool")


@dataclass(frozen=True)
class PolicyFinding:
    """One G5 defect: plain data, no logging.

    ``code`` is one of the fixed codes of ``check_policy``; ``state`` is the
    state id it sits at (``""`` for the document); ``message`` quotes ids and
    key names, each capped. All of it may be model-authored: a caller defangs
    it before it reaches a prompt.
    """

    code: str
    state: str
    message: str
    gate: str = field(default="G5")


def _q(value: object) -> str:
    """A hostile name or id, control-stripped and capped, for a message."""
    return f"'{capped_line(str(value), 60)}'"


def _items(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _strs(value: object) -> Iterator[str]:
    return (item for item in _items(value) if isinstance(item, str))


def _states(data: object) -> Iterator[tuple[str, dict[str, Any]]]:
    states = data.get("states") if isinstance(data, dict) else None
    for sid, state in (states if isinstance(states, dict) else {}).items():
        if isinstance(state, dict):
            yield str(sid), state


def _edges(state: dict[str, Any]) -> Iterator[dict[str, Any]]:
    return (t for t in _items(state.get("transitions")) if isinstance(t, dict))


def _key_names(state: dict[str, Any]) -> Iterator[str]:
    """Every context-key name a state declares or reads, at every position."""
    for extraction in _items(state.get("field_extractions")):
        if isinstance(extraction, dict):
            yield from _strs([extraction.get("field_name")])
            yield from _strs(extraction.get("context_keys"))
    for extraction in _items(state.get("classification_extractions")):
        if isinstance(extraction, dict):
            yield from _strs([extraction.get("field_name")])
    yield from _strs(state.get("required_context_keys"))
    scope = state.get("context_scope")
    if isinstance(scope, dict):
        yield from _strs(scope.get("read_keys"))
        yield from _strs(scope.get("write_keys"))
    completion = state.get("completion")
    if isinstance(completion, dict):
        yield from _strs([completion.get("result_key")])
    for edge in _edges(state):
        for cond in _items(edge.get("conditions")):
            if not isinstance(cond, dict):
                continue
            yield from _strs(cond.get("requires_context_keys"))
            try:
                yield from logic_referenced_keys(cond.get("logic"))
            except (ValueError, TypeError, RecursionError):
                pass  # malformed logic is gate G1's finding, not G5's


def _text_problems(value: object, limit: int) -> set[str]:
    """Codes for any string under ``value``: over ``limit`` characters, or
    holding a control or format character (a line break and a tab are fine) or
    a lone surrogate. Iterative: a 20,000-deep document must not recurse."""
    found: set[str] = set()
    todo = [value]
    while todo:
        node = todo.pop()
        if isinstance(node, dict):
            todo.extend(node.keys())
            todo.extend(node.values())
        elif isinstance(node, (list, tuple)):
            todo.extend(node)
        elif isinstance(node, str):
            if len(node) > limit:
                found.add("text-too-long")
            if (
                has_lone_surrogate(node)
                or strip_unsafe_text(node, multiline=True) != node
            ):
                found.add("control-text")
    return found


def _completion_tool_names(completion: dict[str, Any]) -> list[str | None]:
    """The function name of each ``completion.tools`` entry; ``None`` for an
    entry that has no readable name."""
    tools = completion.get("tools")
    entries = [] if tools is None else tools if isinstance(tools, list) else [tools]
    names: list[str | None] = []
    for tool in entries:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        names.append(name if isinstance(name, str) and name else None)
    return names


def needs_approval(data: object, policy: SynthesisPolicy) -> bool:
    """Whether a human must approve this compile before it runs: the policy
    demands it, or a ``completion`` state lists tools. Never raises."""
    return policy.require_approval or any(
        _completion_tool_names(state["completion"])
        for _, state in _states(data)
        if isinstance(state.get("completion"), dict)
    )


def check_policy(data: object, policy: SynthesisPolicy) -> list[PolicyFinding]:
    """Gate G5: judge a raw compile against ``policy``.

    Interface contract (callers: the gate runner, the compiler's repair loop):
        data: the compile exactly as parsed, BEFORE any typed load (an earlier
        gate owns typed errors), so any shape is tolerated. Returns the
        findings, in document order, each ``gate="G5"`` with a code from
        ``too-many-states``, ``too-many-transitions``, ``text-too-long``,
        ``unknown-key``, ``persona-set``, ``reserved-state``, ``reserved-key``,
        ``internal-key``, ``secret-key``, ``control-text``,
        ``priority-reserved``, ``completion-tool-not-allowed``,
        ``nested-synthesize`` or ``bad-shape`` (a part that is not the object
        or list it must be); ``[]`` means the compile passes. Never raises,
        never mutates ``data``. The injected abort edge adds one transition to
        a state, so a procedure may end up one over ``max_transitions_per_state``.
    """
    out: list[PolicyFinding] = []
    seen: set[tuple[str, str, str]] = set()

    def add(code: str, state: str, message: str, name: str = "") -> None:
        if (code, state, name) not in seen:
            seen.add((code, state, name))
            out.append(PolicyFinding(code, state, message))

    def judge_names(sid: str, names: Iterator[str]) -> None:
        for name in names:
            if name in _RESERVED_NAMES:
                add("reserved-key", sid, f"key {_q(name)} is reserved", name)
            elif has_internal_prefix(name):
                add("internal-key", sid, f"key {_q(name)} is internal", name)
            if is_forbidden_context_entry(name, ""):
                add("secret-key", sid, f"key {_q(name)} looks like a secret", name)

    if not isinstance(data, dict):
        return [PolicyFinding("bad-shape", "", "the compile is not a JSON object")]
    states = data.get("states")
    if not isinstance(states, dict):
        add("bad-shape", "", "states is not an object")
        states = {}
    top = {k: v for k, v in data.items() if k != "states"}
    for code in sorted(_text_problems(top, policy.max_text_chars)):
        add(code, "", f"a document field breaks {code}")
    unknown = sorted(str(k) for k in data if k not in FSMDefinition.model_fields)
    if unknown:
        add("unknown-key", "", f"unknown top-level keys: {_q(', '.join(unknown))}")
    if data.get("persona") is not None:
        add("persona-set", "", "a compile may not set a persona")
    if len(states) > policy.max_states:
        add(
            "too-many-states",
            "",
            f"{len(states)} states, at most {policy.max_states}",
        )
    judge_names("", _strs(data.get("handler_only_keys")))

    for sid, state in states.items():
        sid = str(sid)
        if not isinstance(state, dict):
            add("bad-shape", sid, f"state {_q(sid)} is not an object")
            continue
        if ABORT_STATE_ID in (sid, state.get("id")):
            add("reserved-state", sid, f"state id '{ABORT_STATE_ID}' is reserved")
        unknown = sorted(str(k) for k in state if k not in State.model_fields)
        if unknown:
            add("unknown-key", sid, f"unknown state keys: {_q(', '.join(unknown))}")
        for code in sorted(_text_problems(state, policy.max_text_chars)):
            add(code, sid, f"state {_q(sid)} breaks {code}")
        if "synthesize" in state and policy.max_synthesis_depth <= 1:
            add("nested-synthesize", sid, "a nested synthesize is refused at depth 1")
        if "transitions" in state and not isinstance(state["transitions"], list):
            add("bad-shape", sid, f"transitions of {_q(sid)} is not a list")
        if len(_items(state.get("transitions"))) > policy.max_transitions_per_state:
            add(
                "too-many-transitions",
                sid,
                f"state {_q(sid)} has more than "
                f"{policy.max_transitions_per_state} transitions",
            )
        for edge in _edges(state):
            if edge.get("target_state") == ABORT_STATE_ID:
                add("reserved-state", sid, f"an edge targets '{ABORT_STATE_ID}'")
            priority = edge.get("priority")
            if isinstance(priority, int) and not isinstance(priority, bool):
                if priority < 1:
                    add(
                        "priority-reserved",
                        sid,
                        f"an edge of {_q(sid)} has priority {priority}; "
                        "0 is reserved for the abort edge",
                    )
        completion = state.get("completion")
        if "completion" in state and not isinstance(completion, dict):
            add("bad-shape", sid, f"completion of {_q(sid)} is not an object")
        elif isinstance(completion, dict):
            for tool in _completion_tool_names(completion):
                if tool is None or tool not in policy.allowed_tools:
                    add(
                        "completion-tool-not-allowed",
                        sid,
                        f"tool {_q(tool)} is not allowed",
                        str(tool),
                    )
        judge_names(sid, _key_names(state))
    return out


# --- Abort injection -----------------------------------------------------

_ABORT_STATE: dict[str, Any] = {
    "id": ABORT_STATE_ID,
    "description": "Silent terminal state the runner reaches when it aborts a run",
    "purpose": "End the procedure without a reply",
    "response_instructions": "",
}

_ABORT_EDGE: dict[str, Any] = {
    "target_state": ABORT_STATE_ID,
    "description": "The runner aborted the procedure",
    "priority": 0,
    "conditions": [
        {
            "description": "The driver set the abort flag",
            "requires_context_keys": [ABORT_CONTEXT_KEY],
        }
    ],
}


def _injectable(state: object) -> bool:
    """A state object whose ``transitions`` is absent or a list of objects."""
    edges = state.get("transitions", []) if isinstance(state, dict) else None
    return isinstance(edges, list) and all(isinstance(e, dict) for e in edges)


def inject_abort(data: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the compile with the abort path added.

    Interface contract (callers: the synthesizer after G1 to G5 passed, the
    library when it re-checks a stored definition):
        data: a compile that passed G5, so ``states`` is an object of objects,
        each ``transitions`` is a list or absent, ``handler_only_keys`` is a
        list of names or absent, and no state is called ``abort``. Anything
        else raises ``SynthesisError`` (G5 should have caught it).
        Returns a new dict: the silent ``abort`` terminal state, a priority-0
        edge to it, gated on ``requires_context_keys=["procedure_abort"]`` and
        appended as the last edge of EVERY non-terminal state, and
        ``procedure_abort`` in ``handler_only_keys`` (once), so no model reply
        can set it. ``data`` is never mutated. ``strip_abort`` inverts this.
    """
    states = data.get("states") if isinstance(data, dict) else None
    handler_only = data.get("handler_only_keys", []) if isinstance(data, dict) else None
    if (
        not isinstance(states, dict)
        or ABORT_STATE_ID in states
        or not isinstance(handler_only, list)
        or not all(isinstance(k, str) for k in handler_only)
        or not all(_injectable(state) for state in states.values())
    ):
        raise SynthesisError("cannot inject abort into a compile that failed G5")
    out = copy.deepcopy(data)
    for state in out["states"].values():
        if state.get("transitions"):
            state["transitions"].append(copy.deepcopy(_ABORT_EDGE))
    out["states"][ABORT_STATE_ID] = copy.deepcopy(_ABORT_STATE)
    if ABORT_CONTEXT_KEY not in handler_only:
        out["handler_only_keys"] = [*handler_only, ABORT_CONTEXT_KEY]
    return out


def strip_abort(data: object) -> dict[str, Any] | None:
    """The exact inverse of ``inject_abort``; ``None`` for anything else.

    Interface contract (callers: the library, before it re-verifies a stored
    definition): returns a new dict equal to what ``inject_abort`` was given,
    or ``None`` when ``data`` is not exactly the canonical injected shape: the
    ``abort`` state altered or with transitions, the abort edge differing,
    absent from a non-terminal state, repeated, not last or on a state that
    had no other edge, ``procedure_abort`` not listed exactly once in
    ``handler_only_keys``, or ``initial_state`` naming ``abort``. A
    ``handler_only_keys`` left empty is dropped, so an explicit empty list in
    the original reads back as an absent key (equal in the ``exclude_defaults``
    form the library stores). Never mutates ``data``, never raises.
    """
    if not isinstance(data, dict) or not isinstance(data.get("states"), dict):
        return None
    out = copy.deepcopy(data)
    abort = out["states"].pop(ABORT_STATE_ID, None)
    if not isinstance(abort, dict) or out.get("initial_state") == ABORT_STATE_ID:
        return None
    if {
        k: v for k, v in abort.items() if (k, v) != ("transitions", [])
    } != _ABORT_STATE:
        return None
    for state in out["states"].values():
        if not isinstance(state, dict):
            return None
        edges = state.get("transitions")
        if edges is None or edges == []:
            continue  # a terminal state carries no abort edge
        if not isinstance(edges, list) or edges.pop() != _ABORT_EDGE or not edges:
            return None
        if any(
            isinstance(e, dict) and e.get("target_state") == ABORT_STATE_ID
            for e in edges
        ):
            return None
    keys = out.get("handler_only_keys")
    if not isinstance(keys, list) or keys.count(ABORT_CONTEXT_KEY) != 1:
        return None
    keys.remove(ABORT_CONTEXT_KEY)
    if not keys:
        del out["handler_only_keys"]
    return out
