"""JSON extraction from model replies: balanced-brace spans and the fallback strategies.

Layer ``llm`` (plan d454fd62, step 72): split out of the old ``fsm_llm.utilities``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fsm_llm.logging import logger

# --------------------------------------------------------------
# JSON Processing Utilities
# --------------------------------------------------------------


def _match_brace_partners(text: str, brace_positions: list[int]) -> dict[int, int]:
    """Map each ``{`` position in *text* to the index of the ``}`` balancing it.

    Contract:
        - ``brace_positions`` must be the ascending list of every ``{`` index in
          ``text`` (i.e. ``[m.start() for m in re.finditer(r"\\{", text)]``).
        - Returns ``{start: end}`` for every start whose balanced-brace scan
          terminates. Starts that never balance before end-of-text are ABSENT
          from the mapping (callers must use ``.get()``).
        - Each span is the one a JSON-string/escape-aware scan begun *at that
          start position with a fresh in-string state* would find — not the one
          a single global scan from index 0 would find. The two differ whenever
          an earlier lone ``"`` shifts the global in-string parity, e.g.
          ``'he said " {"a": 1}'``; the per-start reading is the historical
          behavior and is load-bearing (see the D-023/D-002 notes below).
        - Linear rather than quadratic: spans resolve innermost-first
          (descending start order) so an outer scan jumps over an already
          resolved nested span instead of re-walking it.
        - Never raises for any ``str`` input.
    """
    partner: dict[int, int] = {}
    text_len = len(text)

    # Innermost-first: every nested span an outer scan can meet is already known.
    for start_pos in reversed(brace_positions):
        i = start_pos + 1
        in_string = False
        escape_next = False

        while i < text_len:
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == "\\":
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                i += 1
                continue

            if not in_string:
                if char == "{":
                    nested_end = partner.get(i)
                    if nested_end is None:
                        # The nested span never balances, so neither can this
                        # one — the original scan would have run to end-of-text.
                        break
                    # Resume just past the nested span. Its closing `}` is
                    # outside a string, so the string/escape state is clean.
                    i = nested_end + 1
                    continue
                if char == "}":
                    partner[start_pos] = i
                    break

            i += 1

    return partner


def balanced_json_objects(text: str) -> list[str]:
    """The outermost balanced ``{...}`` spans of *text* that parse as JSON.

    Interface contract (caller: ``llm.py`` reasoning-trace recovery):
        - *text*: any ``str`` (anything else gives ``[]``).
        - Returns the source text of each outermost span (one that is not
          inside an earlier returned span), in order of appearance, that
          ``json.loads`` accepts; the objects inside a span that does not
          parse are skipped with it (as ``extract_json_from_text`` does).
          ``[]`` when there is none.
        - String and escape aware (a ``}`` inside a JSON string does not end
          a span): it reads spans through ``_match_brace_partners``.
        - Never raises.
    """
    if not isinstance(text, str):
        return []
    positions = [m.start() for m in re.finditer(r"\{", text)]
    partner = _match_brace_partners(text, positions)
    spans: list[str] = []
    skip_until = -1
    for start in positions:
        end = partner.get(start)
        if start <= skip_until or end is None:
            continue
        span = text[start : end + 1]
        skip_until = end
        try:
            json.loads(span)
        except (json.JSONDecodeError, RecursionError):
            continue
        spans.append(span)
    return spans


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Enhanced JSON extraction from text with multiple fallback strategies.

    This function handles various formats of JSON that might be returned
    by LLMs, including code blocks, partial JSON, and embedded structures.

    Args:
        text: Text potentially containing JSON data

    Returns:
        Extracted JSON object (always a ``dict``) or ``None`` if extraction
        fails. A JSON value that is not an object (``42``, ``[1, 2]``,
        ``true``, ``"hi"``, ``null``) is NOT an extraction result and yields
        ``None``; callers may rely on ``isinstance(result, dict)`` without a
        further check.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    logger.debug("Attempting enhanced JSON extraction from text")

    # Strategy 1: Direct JSON parsing
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # A text that parses cleanly to a non-object returns None immediately; it
    # does NOT fall through to the brace scan. Falling through would recover
    # the first object inside a top-level array (`[{"a":1}]` -> `{"a":1}`), a
    # "recover more" change, though a top-level array is not a payload.
    # Do NOT return the list/scalar either: the `dict | None` annotation was a
    # lie and Classifier._parse_single died on it with AttributeError.
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: a deeply nested payload is undecodable, not fatal
        pass

    # Strategy 2: Extract from code blocks
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # The first fenced block is located with two `str.find` calls, NOT the regex
    # ``` ```(?:json)?\s*([\s\S]*?)\s*``` ```: with an unclosed fence followed by a
    # long whitespace run that pattern retries `\s*```` from every lazy-group
    # position and is O(n^2) (50,000 spaces took minutes) on provider text. The
    # body is everything between the opener (plus an optional `json` tag) and the
    # next fence, stripped; identical to the regex's group(1).strip().
    scan_text = text
    fence_open = text.find("```")
    if fence_open != -1:
        body_start = fence_open + 3
        if text.startswith("json", body_start):
            body_start += 4
        fence_close = text.find("```", body_start)
        if fence_close != -1:
            try:
                json_str = text[body_start:fence_close].strip()
                logger.debug("Found JSON in code block")
                result = json.loads(json_str)
                # D-010: a fenced non-object (`[1,2]`) is treated exactly like
                # an undecodable fence and falls through to Strategy 3.
                if isinstance(result, dict):
                    return result
                # RA-04: the fenced span is skipped, otherwise the brace scan
                # would return the object inside a fenced array (`[{"a":1}]`).
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-029
                # The span is BLANKED in place (positions preserved), not cut
                # off with `text[fence_close + 3:]`: truncation lost an object
                # BEFORE the fence. Strategy 4 reads this same string (set
                # below); do NOT let it read the original `text`, that merged
                # keys from the fenced array with an earlier object.
                scan_text = (
                    text[:fence_open]
                    + " " * (fence_close + 3 - fence_open)
                    + text[fence_close + 3 :]
                )
            except (json.JSONDecodeError, RecursionError):
                logger.debug("Code block JSON parsing failed")

    # Strategy 3: Find balanced JSON objects
    closing_index: dict[int, int] = {}  # Strategy 4 reads it as well
    try:
        # Find all potential JSON start positions
        brace_positions = [m.start() for m in re.finditer(r"\{", scan_text)]

        # DECISION plan-2026-07-19T191147-4b664252/D-002 [STALE]
        # The closing partner of every `{` is precomputed ONCE, innermost-first,
        # instead of rescanning text[start_pos:] from every start position. That
        # rescan was O(n^2): 20,000 bare `{` characters took 18.2s of CPU on
        # text that arrives straight from the LLM provider, while the caller
        # holds the per-conversation lock (a real DoS vector).
        #
        # This changed the COMPLEXITY ONLY. Two things below are load-bearing
        # and must NOT be "simplified":
        #   1. The loop stays FIRST-wins — see the D-023 block immediately
        #      below, and decisions.md D-023. A last-wins flip was shipped and
        #      REVERTED once already.
        #   2. `_match_brace_partners` deliberately resolves each span with a
        #      scan begun AT that start position with a fresh in-string state.
        #      Do NOT replace it with one global left-to-right stack pass from
        #      index 0. That looks equivalent and is not: a single earlier lone
        #      `"` flips the global in-string parity, so a real object gets
        #      classified as string content and dropped. Probe that regresses:
        #        'x " {"a":1} " y'  ->  must return {'a': 1}, not None.
        # See decisions.md D-002.
        closing_index = _match_brace_partners(scan_text, brace_positions)

        skip_until = -1
        # DECISION plan-2026-07-18T162030-a02151fe/D-023 [STALE]
        # This strategy is FIRST-wins: it returns on the first successful parse.
        # That disagrees with Strategy 4 below and with
        # llm.py::_extract_content_from_thinking, both of which prefer the LAST
        # object. The divergence is REAL and still OPEN — which helper you hit
        # depends only on whether the provider split the reasoning trace into a
        # separate `thinking` field.
        #
        # Do NOT "fix" it by flipping this loop to last-wins. That was tried
        # (D-021) and REVERTED, because it does not work:
        #   - Strategy 2 (code fence, above) runs FIRST and is also first-match,
        #     so the fenced draft-then-final case that motivated the flip stayed
        #     broken. Only the unfenced variant changed.
        #   - It actively broke the answer-then-example shape:
        #     'The intent is {"intent": "buy"}. Schema: {"intent": "<name>"}'
        #     started returning the EXAMPLE. For classification.py that silently
        #     degrades a correct intent to `fallback_intent`.
        # Resolving this means deciding what these three helpers are FOR (is a
        # trailing object a correction, or a restated schema?), and changing
        # Strategy 2 in step with whatever is chosen. That is a design question,
        # not a one-line tie-break. See decisions.md D-023.
        for start_pos in brace_positions:
            if start_pos <= skip_until:
                continue  # Skip positions inside a previously scanned span

            end_pos = closing_index.get(start_pos)
            if end_pos is None:
                continue  # Never balances before end of text — try next start

            # Found complete JSON object
            json_str = scan_text[start_pos : end_pos + 1]
            try:
                brace_result: dict[str, Any] = json.loads(json_str)
                logger.debug(
                    "Successfully extracted JSON using balanced brace matching"
                )
                return brace_result
            except json.JSONDecodeError:
                # Skip nested positions inside this failed span
                skip_until = end_pos

    except Exception as e:
        logger.debug(f"Error during balanced brace JSON extraction: {e}")

    # Strategy 4: Extract key-value pairs using regex (fallback)
    text = scan_text  # same string as Strategy 3 (D-029)
    try:
        # Patterns for simple string values
        string_patterns = [
            r'"message"\s*:\s*"([^"]*)"',
            r'"selected_transition"\s*:\s*"([^"]*)"',
            r'"reasoning"\s*:\s*"([^"]*)"',
            # `intent` is a quoted string value, so it rides the same
            # last-match-wins string loop below. Without it, a recoverable
            # classification intent in garbled text silently degrades to
            # `fallback_intent` even though `meaningful_keys` claims to
            # cover it. See findings G4.
            r'"intent"\s*:\s*"([^"]*)"',
        ]

        extracted = {}

        for pattern in string_patterns:
            # Prefer the LAST match: a <think> trace often quotes these keys
            # before the real JSON, and the trailing occurrence is the
            # authoritative value.
            matches = re.findall(pattern, text)
            if matches:
                key = pattern.split('"')[1]
                extracted[key] = matches[-1]

        # `confidence` is an UNQUOTED number, so the quoted-string loop above
        # cannot capture it. Extract it separately, same last-match-wins
        # rationale. The pattern consumes a FULL numeric token — optional sign,
        # fraction, AND scientific-notation exponent — so `1e-3` is read as
        # 0.001, not silently truncated to "1" -> 1.0 (the G6-class silent
        # max-certainty defect; see findings CF2). The trailing negative
        # lookahead `(?![0-9.eE])` rejects malformed multi-dot / dangling-`e`
        # tokens (e.g. "1.2.3", "1e") outright rather than capturing a partial
        # prefix. The float() guard stays as a belt-and-suspenders skip. See
        # findings G4/CF2.
        confidence_matches = re.findall(
            r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![0-9.eE])',
            text,
        )
        if confidence_matches:
            try:
                extracted["confidence"] = float(confidence_matches[-1])
            except ValueError:
                pass

        # For extracted_data, find the key and then use balanced braces
        ed_match = re.search(r'"extracted_data"\s*:\s*\{', text)
        if ed_match:
            # The object's closing brace is the one Strategy 3 already matched
            # (`closing_index`, string/escape aware, same `scan_text`).
            brace_start = text.index("{", ed_match.start())
            end_pos = closing_index.get(brace_start)
            if end_pos is not None:
                try:
                    extracted["extracted_data"] = json.loads(
                        text[brace_start : end_pos + 1]
                    )
                except json.JSONDecodeError:
                    extracted["extracted_data"] = {}

        # CF5: `intent`/`confidence` are AUXILIARY classification keys. G4 added
        # their capture here, which flipped a previously-None result into a dict
        # for garbled free text that merely mentions one of them — a real
        # cross-package hazard for non-classification callers (e.g. a lenient
        # all-optional structured-output schema in fsm_llm/agents/base.py would
        # then build a partial model from a stray `"confidence": 0.8` substring).
        # Keep a lone auxiliary key ONLY when a co-occurring PRIMARY payload key
        # is present IN THE TEXT — message/selected_transition/value/
        # extracted_data, or the OTHER member of the intent/confidence pair
        # (mutual reinforcement: a genuine classification payload carries both).
        # A malformed-but-present counterpart key (e.g. `"confidence": 1.2.3`
        # that fails to parse) still counts as co-occurring, so a recoverable
        # intent survives. See findings CF5.
        def _key_present_in_text(key: str) -> bool:
            return re.search(rf'"{key}"\s*:', text) is not None

        _primary_present = any(
            _key_present_in_text(k)
            for k in ("message", "selected_transition", "value", "extracted_data")
        )
        if not _primary_present:
            if "intent" in extracted and not _key_present_in_text("confidence"):
                extracted.pop("intent", None)
            if "confidence" in extracted and not _key_present_in_text("intent"):
                extracted.pop("confidence", None)

        # Only return if we have structurally meaningful keys, not just auxiliary ones.
        # Includes keys used by classification (intent, confidence) and response
        # generation (message, reasoning) callers — not just data extraction.
        meaningful_keys = {
            "selected_transition",
            "extracted_data",
            "message",
            "reasoning",
            "intent",
            "confidence",
        }
        if extracted and (meaningful_keys & extracted.keys()):
            logger.debug(
                f"Extracted JSON using regex fallback: {list(extracted.keys())}"
            )
            return extracted
        elif extracted:
            logger.debug(
                f"Regex fallback found only auxiliary keys {list(extracted.keys())}, treating as failed"
            )

    except Exception as e:
        logger.debug(f"Regex fallback extraction failed: {e}")

    logger.warning("All JSON extraction strategies failed")
    return None
