"""
The reply judge of the synthesizer: one model reply to one gate report.

``_judge`` reads a compile reply (cut off or not JSON is one G1 finding, a wrong
closing bracket is repaired once, D-097; extraction instructions are named, D-096)
and runs the gates. Moved out of ``synthesizer.py`` unchanged (plan ace7a8c8 D-004);
every function here is synchronous and makes no model call.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fsm_llm.core.definitions import CompletionResponse, State, logic_referenced_keys
from fsm_llm.llm.json_extraction import extract_json_from_text

from .gates import GateFinding, GateReport, run_gates
from .policy import SynthesisPolicy
from .prompts import KEY_NAMING, REPLY_FAILURES

#: The stdlib json messages that carry no input text; any other is dropped.
_PARSER_MESSAGES = frozenset(
    (
        "Expecting value|Expecting ',' delimiter|Expecting ':' delimiter|"
        "Expecting property name enclosed in double quotes|Extra data|"
        "Unterminated string starting at|Invalid control character at|"
        "Invalid \\escape|Invalid \\uXXXX escape"
    ).split("|")
)


def _parser_detail(text: str) -> str:
    """`` (parser: <msg> at line L column C); fix that syntax`` when the reply's
    first-``{``-to-last-``}`` span fails ``json.loads``, else ``""``: a
    ``_PARSER_MESSAGES`` phrase and two integers, never reply text."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return ""
    try:
        json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        if e.msg in _PARSER_MESSAGES:
            return f" (parser: {e.msg} at line {e.lineno} column {e.colno}); fix that syntax"
    except RecursionError:
        pass
    return ""


# DECISION plan-2026-11-10T194139-2cefeb61/D-097
# The strong model's habit (live: 8 first-trial compiles) is to close the
# `states` object with `]`. The slip is named exactly, so it is repaired here,
# once, instead of asked again at about 70 s a round. ONLY the kind of a wrong
# closer changes: a missing closer is NOT appended and a stray one is not
# dropped (no invented content), and the repaired text goes through every gate
# like any reply. Do NOT repair a reply that already parses or one that was cut
# off, do NOT skip G1 for a repaired reply, and do NOT make the scan ignore
# strings (a `]` inside a string would flip the stack).
_CLOSERS = {"{": "}", "[": "]"}
_MAX_NESTING = 100


def _repair_brackets(text: str) -> str | None:
    """The first-``{``-to-last-``}`` span of ``text`` with every closer of the
    wrong kind replaced, or ``None``.

    Interface contract (caller: ``_judge``, after ``extract_json_from_text``
    found nothing): ``text`` is a reply, any shape. Returns the span only when
    it parses with ``json.loads`` after the swaps; ``None`` for a span that is
    missing, has a stray closer, an unclosed container or string, nests deeper
    than ``_MAX_NESTING``, or still is not JSON. Pure; one linear scan, never
    raises.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    chars = list(text[start : end + 1])
    expected: list[str] = []
    quoted = escaped = False
    for i, ch in enumerate(chars):
        if quoted:
            quoted = ch != '"' or escaped
            escaped = ch == "\\" and not escaped
        elif ch == '"':
            quoted = True
        elif ch in _CLOSERS:
            if len(expected) >= _MAX_NESTING:
                return None
            expected.append(_CLOSERS[ch])
        elif ch in "}]":
            if not expected:
                return None
            chars[i] = expected.pop()
    if expected or quoted:
        return None
    fixed = "".join(chars)
    try:
        json.loads(fixed)
    except (json.JSONDecodeError, RecursionError):
        return None
    return fixed


# DECISION plan-2026-11-10T194139-2cefeb61/D-096
# The bulk pass is the only channel for a later-turn correction (per-field
# extraction is skip-if-set) and answers under the name its instructions give:
# a compile that gates on a key it extracts, without naming that key at state
# level, can never correct a refused value (live: `group_size` for `party_size`).
# So the sentence is added HERE, on the parsed reply before `run_gates`, and the
# stored definition is the normalised one. Do NOT change `logic`, do NOT add a
# field extraction, do NOT run this after the gates, do NOT replace the model's
# text, do NOT test "mentions" as a substring, and do NOT exceed the policy's text
# cap (G5 `text-too-long` would reject a compile that passed before).
_INSTRUCTION_CAP = next(
    m.max_length
    for m in State.model_fields["extraction_instructions"].metadata
    if hasattr(m, "max_length")
)


def _naming_text(state: Any, limit: int) -> str | None:
    """The state-level extraction text ``state`` should have, ``None`` to leave it."""
    if (
        not isinstance(state, dict)
        or state.get("completion")
        or state.get("synthesize")
    ):
        return None
    text = state.get("extraction_instructions") or ""
    fields, edges = state.get("field_extractions"), state.get("transitions")
    if not (
        isinstance(text, str) and isinstance(fields, list) and isinstance(edges, list)
    ):
        return None
    read: set[str] = set()
    for edge in edges:
        conds = edge.get("conditions") if isinstance(edge, dict) else None
        for cond in conds if isinstance(conds, list) else ():
            try:
                read |= logic_referenced_keys(
                    cond.get("logic") if isinstance(cond, dict) else None
                )
            except ValueError:
                pass  # malformed logic: G1 reports it
    names = dict.fromkeys(
        f["field_name"]
        for f in fields
        if isinstance(f, dict) and f.get("field_name") in read
    )
    missing = [
        k
        for k in names
        if not re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text, re.IGNORECASE)
    ]
    if not missing:
        return None
    added = " ".join(KEY_NAMING.format(key=k) for k in missing)
    out = f"{text} {added}" if text.strip() else added
    return out if len(out) <= limit else None


def normalise_extraction(data: Any, max_chars: int) -> Any:
    """Name, in the state-level ``extraction_instructions``, every key a state
    extracts and gates on (D-096).

    Interface contract (caller: ``_judge``, once per compile reply):
        data: the parsed reply, any shape (never mutated). ``max_chars`` is the
        policy's text cap. A state with ``field_extractions`` and an edge whose
        ``logic`` reads such a key gets one ``KEY_NAMING`` sentence per key its
        text does not already name (whole identifier, any case), appended after
        the model's own text; a state with no such gate, a terminal,
        ``completion`` or ``synthesize`` state, a classification key, a result
        over the cap or any malformed part is left as it is. Returns ``data``
        itself when nothing changes (so its digest is the same), else a shallow
        copy with only the changed states replaced. Pure, idempotent, never raises.
    """
    states = data.get("states") if isinstance(data, dict) else None
    if not isinstance(states, dict):
        return data
    limit = min(max_chars, _INSTRUCTION_CAP)
    fixed = {
        sid: {**state, "extraction_instructions": text}
        for sid, state in states.items()
        if (text := _naming_text(state, limit)) is not None
    }
    return {**data, "states": {**states, **fixed}} if fixed else data


def _judge(
    reply: CompletionResponse,
    policy: SynthesisPolicy,
    inputs: tuple[str, ...],
    return_keys: tuple[str, ...],
) -> GateReport:
    """One reply to one report: a reply that is not a readable JSON object
    fails as one G1 finding (truncated first, never parsed), else the gates run."""
    code = (
        "reply-truncated"
        if reply.truncated
        else "reply-malformed"
        if reply.kind != "final"
        else ""
    )
    data = None if code else extract_json_from_text(reply.text or "")
    if data is None and not code:
        fixed = _repair_brackets(reply.text or "")
        data = json.loads(fixed) if fixed else None
    if data is None:
        detail = "" if code else _parser_detail(reply.text or "")
        code = code or "reply-not-json-object"
        return GateReport((GateFinding("G1", code, "", REPLY_FAILURES[code] + detail),))
    data = normalise_extraction(data, policy.max_text_chars)
    return run_gates(data, policy, inputs=inputs, return_keys=return_keys)
