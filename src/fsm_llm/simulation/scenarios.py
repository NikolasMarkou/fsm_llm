"""The scenario writer call: one request to the judge, its reply parsed and cleaned."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from fsm_llm.core.codes import Purpose
from fsm_llm.core.definitions import CompletionRequest, LLMResponseError
from fsm_llm.llm import LLMInterface
from fsm_llm.llm.json_extraction import extract_json_from_text
from fsm_llm.llm.purpose import purpose_scope
from fsm_llm.text import safe_text


def write_scenarios(
    llm: LLMInterface,
    *,
    system: str,
    user: str,
    kinds: Collection[str],
    max_scenarios: int,
    max_turns: int,
    turn_chars: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """The scenarios ``llm`` writes: ``{"kind", "turns"}`` dicts, at most ``max_scenarios``.

    Interface contract (callers: ``synthesis.simulation.run_g6b``): ``system`` and
    ``user`` are the two messages, built by the caller (the caller fences any
    untrusted text in ``user``; this function adds none). One ``complete`` call under
    ``Purpose.SIMULATE``. Each turn is control-stripped, defanged by ``safe_text`` and
    cut at ``turn_chars``. A cut or non-final reply, a malformed one, a failing call
    (``LLMResponseError``) is ``[]``; a scenario that is odd in any way (a kind outside
    ``kinds``, no turns, more than ``max_turns``, a non-string turn, a turn that cleans
    to empty) is dropped whole, never repaired. No text of the reply goes anywhere else.
    """
    request = CompletionRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    try:
        with purpose_scope(Purpose.SIMULATE):
            reply = llm.complete(request)
    except LLMResponseError:
        return []
    data = (
        None
        if reply.truncated or reply.kind != "final"
        else extract_json_from_text(reply.text or "")
    )
    rows = data.get("scenarios") if isinstance(data, dict) else None
    kept: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        kind = row.get("kind") if isinstance(row, dict) else None
        turns = row.get("turns") if isinstance(row, dict) else None
        if (
            kind not in kinds
            or not isinstance(turns, list)
            or not 0 < len(turns) <= max_turns
            or not all(isinstance(t, str) for t in turns)
        ):
            continue
        clean = [safe_text(t, turn_chars) for t in turns]
        if all(clean):
            kept.append({"kind": kind, "turns": clean})
        if len(kept) == max_scenarios:
            break
    return kept
