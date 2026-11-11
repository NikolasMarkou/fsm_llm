"""
Repair: ask the strong model for a patch to the diagnosed states.

``propose_patch`` only RETURNS raw ops; it never applies, counts or checks
them (``apply_patch`` and the gates do, D-038).

# DECISION plan-2026-11-10T194139-2cefeb61/D-052
# The request holds codes, state ids, key names and numbers in the diagnosis
# lines and ONLY the diagnosed states' JSON, in the one fence of ``prompts``.
# Nothing model-authored reaches the system message, a state id or key that is
# not an identifier is never sent, and an unreadable reply is an empty patch.
# Do NOT send the whole definition, do NOT parse a truncated or non-final
# reply, do NOT apply or count ops here, and do NOT write a second fence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from fsm_llm.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    CompletionRequest,
    LLMResponseError,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.utilities import extract_json_from_text

from .constants import (
    ABORT_STATE_ID,
    CODE_NOT_PROCEDURE_FAULT,
    DIAGNOSIS_CODES,
    PATCH_MAX_TOKENS,
)
from .diagnose import Diagnosis
from .exceptions import SynthesisError
from .policy import SynthesisPolicy
from .prompts import patch_messages


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(ASCII_IDENTIFIER_PATTERN, value)
    )


def propose_patch(
    diagnoses: Sequence[Diagnosis],
    definition: dict[str, Any],
    policy: SynthesisPolicy,
    llm: LLMInterface,
) -> list[dict[str, Any]]:
    """Ask ``llm`` for a JSON Patch limited to the diagnosed states.

    Interface contract: ``definition`` is the stripped canonical FSM dict
    (never mutated). A diagnosis qualifies when its code is a fault code (not
    ``not-procedure-fault``) and its state is an identifier, not ``abort`` and
    a state of ``definition``; one that does not is ignored, and key names that
    are not identifiers are left out of its line. No qualifying diagnosis means
    no call and ``[]``. Otherwise exactly one ``llm.complete`` call
    (``PATCH_MAX_TOKENS``) and the reply's ``{"patch": [...]}`` ops, raw. A cut
    reply, ``kind != "final"``, a non-object, a missing or non-list ``patch`` or
    an op that is not an object is ``[]``, never an exception. A provider
    outage raises ``SynthesisError`` chained from the ``LLMResponseError``, with
    no model text and no retry.
    """
    states = definition.get("states")
    states = states if isinstance(states, dict) else {}
    lines: list[str] = []
    shown: dict[str, Any] = {}
    for d in diagnoses:
        if (
            d.code in DIAGNOSIS_CODES
            and d.code != CODE_NOT_PROCEDURE_FAULT
            and _name(d.state)
            and d.state != ABORT_STATE_ID
            and d.state in states
        ):
            shown[d.state] = states[d.state]
            keys = ",".join(k for k in d.keys if _name(k))
            lines.append(f"- {d.code} state={d.state} keys={keys} runs={d.runs}")
    if not shown:
        return []
    request = CompletionRequest(
        messages=patch_messages(lines, shown, policy), max_tokens=PATCH_MAX_TOKENS
    )
    try:
        reply = llm.complete(request)
    except LLMResponseError as e:
        raise SynthesisError("the strong model call failed") from e
    data = None
    if not reply.truncated and reply.kind == "final":
        data = extract_json_from_text(reply.text or "")
    ops = data.get("patch") if data else None
    if isinstance(ops, list) and all(type(op) is dict for op in ops):
        return ops
    return []
