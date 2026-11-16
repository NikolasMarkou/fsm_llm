"""Reply parsing: the shared reading helpers (finish reason, cut-off flag, envelope salvage)
and ``ParseMixin``, the reply parsers of ``LiteLLMInterface``.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypeVar

from fsm_llm.constants import TRUNCATED_SALVAGE_CONFIDENCE
from fsm_llm.core.coerce import (
    coerce_confidence,
    coerce_float,
    coerce_int,
    parse_bool_word,
)
from fsm_llm.core.definitions import (
    FINISH_REASON_LENGTH,
    RESPONSE_TEXT_MAX_LENGTH,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.logging import logger

from .json_extraction import balanced_json_objects, extract_json_from_text
from .text import remove_think_blocks, strip_think_and_fences

# Shown to the END USER when no human-readable text can be recovered from a
# model response.  Referenced from two places in
# _parse_response_generation_response (the non-text-content branch and the
# terminal rung's envelope guard) — one string, one place to change it.
_GENERIC_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't generate a proper response. Please try again."
)


# Shown to the END USER when a reply that opened as a JSON envelope was cut off
# by max_tokens (provider finish_reason "length") and so cannot be parsed. A
# different text from the generic fallback on purpose: the apology retry of
# generate_response fires on the generic message only, and a retry at the same
# max_tokens would be cut at the same place (D-024).
_CUT_OFF_FALLBACK_MESSAGE = (
    "I'm sorry, my answer was cut off before it was finished. "
    "Please try again or ask for a shorter answer."
)


# Top-level keys of the single-field extraction envelope
# (`{"field_name": ..., "value": ..., "confidence": ..., "reasoning": ...}`)
# besides `value`. In an envelope-shaped reply (one carrying `value` or
# `field_name`) a field with one of these names never reads that key as its
# value in `_field_value` (plan-2026-09-29T103145-06a5ec0a D-035).
_FIELD_ENVELOPE_KEYS = frozenset({"field_name", "confidence", "reasoning"})


def _field_value(data: dict[str, Any], field_name: str) -> Any:
    """Read a single-field extraction value from a parsed reply.

    Returns ``data["value"]`` unless it is absent or ``None``, else
    ``data[field_name]`` (``None`` when neither is set). ``dict.get``'s default
    only covers an ABSENT key, so ``{"value": null, "<field>": "a@b"}`` used to
    return ``None`` (audit D11). Never raises for a ``dict``.

    A field named after a single-field envelope key (``_FIELD_ENVELOPE_KEYS``)
    reads ``data["value"]`` only when the reply is envelope-shaped (it carries
    ``value`` or ``field_name``); a flat ``{"confidence": 0.8}`` for a field
    named ``confidence`` still returns ``0.8``.
    """
    value = data.get("value")
    if value is not None:
        return value
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-035: do NOT fall back to
    # data[field_name] when field_name is an envelope key AND the reply is
    # envelope-shaped: there that key is the model's own explanation (or score,
    # or echoed name), not the field's value (it fed meta-commentary back into
    # later prompts). Do NOT widen the guard to flat replies: a flat
    # `{"confidence": 0.8}` has no envelope, so the key is the value (fix
    # 24.1). The D11 fallback stays for every other name. See decisions.md.
    enveloped = "value" in data or "field_name" in data
    if enveloped and field_name in _FIELD_ENVELOPE_KEYS:
        return None
    return data.get(field_name)


# The opening of a single-field extraction envelope, `{"field_name": "<name>",
# "value": ` (whitespace-tolerant), anchored at the start of the reply.
_FIELD_ENVELOPE_PREFIX = re.compile(
    r'\{\s*"field_name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"value"\s*:\s*', re.DOTALL
)


# One backslash escape, scanned left to right so `\\` is consumed as a pair:
# group 1 is set for an escape JSON defines, unset for one it does not (`\d` in a
# regex, `\U` in `C:\Users`, `\u` without four hex digits).
_JSON_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')


def _keep_undefined_escapes(text: str) -> str:
    """Double every backslash that starts an escape JSON does not define."""
    return _JSON_ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", text)


def _decode_json_string_prefix(body: str) -> tuple[str, bool]:
    """Decode the body of a JSON string literal that may be cut off.

    Args:
        body: the text after the opening quote.

    Returns:
        ``(text, complete)``. ``text`` is the unescaped text up to the closing
        quote, or up to the cut when there is none (a dangling ``\\``, a
        partial ``\\uXXXX`` and a trailing unpaired high surrogate are
        dropped). An escape JSON does not define (``\\d``, ``\\U``) is kept
        literally, and text that still cannot be decoded is returned raw, so
        the model's characters are never lost. ``complete`` is True when the
        closing quote was found.

    Failure mode: none, never raises.
    """
    end = len(body)
    complete = False
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            step = 6 if body[i + 1 : i + 2] == "u" else 2
            if i + step > len(body):
                end = i  # escape cut in half
                break
            i += step
            continue
        if ch == '"':
            end = i
            complete = True
            break
        i += 1
    kept = body[:end]
    decoder = json.JSONDecoder(strict=False)
    decoded: Any = None
    for candidate in (kept, _keep_undefined_escapes(kept)):
        try:
            decoded = decoder.decode(f'"{candidate}"')
            break
        except json.JSONDecodeError:
            continue
    text = decoded if isinstance(decoded, str) else kept
    if not complete and text and "\ud800" <= text[-1] <= "\udbff":
        # The cut fell between the two halves of a surrogate pair; a lone
        # surrogate cannot be UTF-8 encoded by any writer downstream.
        text = text[:-1]
    return text, complete


def _salvage_envelope_value(raw: str, field_name: str) -> tuple[bool, Any, bool]:
    """Read the value out of raw text that is this field's extraction envelope.

    Args:
        raw: stripped reply text that the JSON rungs could not parse.
        field_name: the field being extracted.

    Returns:
        ``(False, None, False)`` when ``raw`` does not open with
        ``{"field_name": "<field_name>", "value": ``. Otherwise ``(True,
        value, truncated)``: the fully decoded value when it is complete
        (a complete string with escapes JSON does not define keeps them
        literally), else the decoded prefix of a cut-off string value, else
        the raw text of a cut-off object/array/literal; ``None`` when nothing
        usable is left. ``truncated`` is True when the value itself was cut
        off (a number that runs to the end of the reply counts as cut).

    Failure mode: none, never raises.
    """
    match = _FIELD_ENVELOPE_PREFIX.match(raw)
    if match is None or _decode_json_string_prefix(match.group(1))[0] != field_name:
        return False, None, False
    rest = raw[match.end() :]
    truncated = False
    try:
        value, end = json.JSONDecoder(strict=False).raw_decode(rest)
        if isinstance(value, int | float) and not rest[end:].strip():
            truncated = True
    except json.JSONDecodeError:
        if rest.startswith('"'):
            value, complete = _decode_json_string_prefix(rest[1:])
            truncated = not complete
        else:
            value = rest.rstrip()
            truncated = True
    if isinstance(value, str) and not value.strip():
        value = None
    return True, value, truncated


def _finish_reason(response: Any) -> str | None:
    """The provider's ``finish_reason`` of a reply: ``choices[0].finish_reason``.

    Contract: ``response`` is a raw provider reply (any shape). Returns the
    reason as a non-empty ``str``, or ``None`` when the reply has no choices,
    no such attribute, or a value that is not a ``str``. Never raises. This is
    the one reader; every ``finish_reason`` of this module comes from here.
    """
    try:
        reason = response.choices[0].finish_reason
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return reason if isinstance(reason, str) and reason else None


def cut_off_note(max_tokens: int | None) -> str:
    """The sentence that names a reply cut off at ``max_tokens``.

    Contract: ``max_tokens`` is the limit that was in force (``None`` when
    unknown). Returns one sentence, with advice, for an error message or a
    ``validation_error``. Never raises.
    """
    limit = f"max_tokens={max_tokens}" if max_tokens is not None else "max_tokens"
    return (
        f"the reply was cut off at {limit} (finish_reason=length); raise "
        "max_tokens (LLM_MAX_TOKENS) or ask for a shorter answer"
    )


_CutOffResponse = TypeVar(
    "_CutOffResponse", ResponseGenerationResponse, FieldExtractionResponse
)


def _flag_cut_off(parsed: _CutOffResponse, response: Any) -> _CutOffResponse:
    """``parsed`` with ``truncated=True`` when the provider cut ``response`` off.

    Contract: ``parsed`` is the ``ResponseGenerationResponse`` or
    ``FieldExtractionResponse`` read from the raw provider ``response``. Returns
    ``parsed`` itself when ``_finish_reason(response)`` is not ``length`` (or it
    is flagged already), else a copy with ``truncated=True``. Never raises. The
    one place the flag is set, for both response kinds; the pipeline reads it
    (plan 0068b055 D-025).
    """
    if _finish_reason(response) == FINISH_REASON_LENGTH and not parsed.truncated:
        return parsed.model_copy(update={"truncated": True})
    return parsed


def _safe_str(value: Any) -> str | None:
    """Coerce a MODEL-SUPPLIED value into a value the capped models will accept.

    Args:
        value: any value read off a parsed LLM JSON payload.

    Returns:
        ``value`` truncated to ``RESPONSE_TEXT_MAX_LENGTH`` when it is a ``str``;
        ``None`` for every other type.

    Failure mode: none — total function, never raises.  It exists so that an
    optional ``str | None`` field carrying ``max_length=5000``
    (``ResponseGenerationResponse.reasoning``, ``FieldExtractionResponse.reasoning``)
    or an unannotated-but-typed field (``message_type``) cannot fail model
    construction just because the model padded it.  Callers that need a
    REQUIRED ``str`` field slice directly instead, so mypy keeps the non-optional
    type.
    """
    return value[:RESPONSE_TEXT_MAX_LENGTH] if isinstance(value, str) else None


class ParseMixin:
    """The reply parsers of ``LiteLLMInterface`` (plan d454fd62, step 73; bodies unchanged).

    Mixed into ``LiteLLMInterface``; ``self`` is that interface.
    """

    if TYPE_CHECKING:
        # Provided by ``LiteLLMInterface`` (declared for mypy only).
        max_tokens: int

        @staticmethod
        def _find_nested_key(data: dict, key: str, max_depth: int = 3) -> Any: ...

    @staticmethod
    def _recover_content_from_trace(trace: str) -> str | None:
        """Recover the answer string from an already-resolved reasoning trace.

        Shared JSON-scan body of ``_extract_content_from_thinking``. Kept as a
        separate static helper so BOTH the non-streaming reader (which resolves a
        full ``Message``) and the streaming tail guard (which joins per-chunk
        ``reasoning_content`` fragments accumulated across the stream) recover
        content through one identical code path.

        Contract:
            - Parameter: ``trace`` — the raw reasoning-trace string (already
              resolved via ``resolve_reasoning_trace``; a non-empty ``str``).
            - Returns the last balanced JSON object found in ``trace`` (preferred,
              as the final answer), else the last non-empty line, else ``None``.
            - Never raises; a span that is not JSON is skipped.
        """
        # The objects come from the one string-aware scanner
        # (`llm.json_extraction.balanced_json_objects`), outermost spans only: a line
        # scan plus a string-blind brace count returned an inner one-line
        # fragment or a lone `}` for a pretty-printed object. The last object
        # wins here (the first-wins split with `extract_json_from_text` is
        # the open D-023 and stays as it is).
        objects = balanced_json_objects(trace)
        if objects:
            return objects[-1]
        # Fallback: use the last substantial line
        lines = [line.strip() for line in trace.strip().split("\n") if line.strip()]
        return lines[-1] if lines else None

    def _parse_response_generation_response(
        self, response, structured: bool = False
    ) -> ResponseGenerationResponse:
        """
        Parse LLM response for response generation.

        Handles both structured JSON and unstructured text responses.
        ``structured`` is True when the caller sent a ``response_format``; a
        JSON object with no ``message`` key is then the reply itself.
        """
        content = response.choices[0].message.content

        # Handle structured response (JSON)
        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object, got array or primitive")

                # DECISION plan-2026-09-21T203800-8a03483a/D-039
                # `reasoning` is NEVER the user-facing reply. Do NOT restore the
                # empty-`message` -> `reasoning` fallback: it showed the model's
                # internal chain of thought to the end user verbatim. An empty
                # message now degrades to the apology (and the pipeline's
                # one-shot retry); a structured reply without a `message` key
                # is still the caller's schema (D-020 below, and
                # plan-2026-09-19T175721-21cd7f8e/D-028). See decisions.md D-039.
                message = data.get("message")
                if not isinstance(message, str) or not message.strip():
                    # DECISION plan-2026-09-19T175721-21cd7f8e/D-020: a caller-
                    # requested schema without a `message` key makes the JSON
                    # text the reply. Do NOT unwrap it in the pipeline (it never
                    # sees the raw content) and do NOT apply this when no schema
                    # was sent: that is the D-022 shape ambiguity. Over the
                    # 5000-char cap it degrades to the ladder below.
                    as_text = json.dumps(data, ensure_ascii=False, default=str)
                    if (
                        structured
                        and data
                        and "message" not in data
                        and len(as_text) <= RESPONSE_TEXT_MAX_LENGTH
                    ):
                        return ResponseGenerationResponse(
                            message=as_text, message_type="response"
                        )
                    raise ValueError("No usable message in response")
                return ResponseGenerationResponse(
                    message=message[:RESPONSE_TEXT_MAX_LENGTH],
                    message_type=_safe_str(data.get("message_type")) or "response",
                    reasoning=_safe_str(data.get("reasoning")),
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse structured response generation response: {e}"
                )

        # Fallback: extract JSON embedded in text.
        # DECISION plan-2026-09-21T203800-8a03483a/D-009
        # Strip <think> blocks BEFORE the scan: Strategy 3 is first-wins
        # (utilities D-023, unchanged), so a JSON draft inside a reasoning
        # trace used to beat the real answer after it. Do NOT flip the scan to
        # last-wins instead, and do NOT feed it raw `content` again. When only
        # a think block exists nothing is parsed here; the terminal rung below
        # decides (D-030). See decisions.md D-009.
        if isinstance(content, str):
            data = extract_json_from_text(remove_think_blocks(content))
            if isinstance(data, dict) and "message" in data:
                message = data["message"]
                if isinstance(message, str) and message.strip():
                    logger.debug("Extracted response JSON via fallback")
                    try:
                        return ResponseGenerationResponse(
                            message=message[:RESPONSE_TEXT_MAX_LENGTH],
                            message_type=_safe_str(data.get("message_type"))
                            or "response",
                            reasoning=_safe_str(data.get("reasoning")),
                        )
                    except (ValueError, TypeError) as e:
                        # Fall THROUGH to the terminal raw-text rung below — do not
                        # swallow into a success. A legitimately oversized message
                        # (>5000 chars) or a non-str `reasoning` must degrade, not
                        # fail the turn.
                        logger.warning(
                            f"Embedded-JSON response fallback failed validation: {e}"
                        )

        # Extract message from dict content if possible
        if isinstance(content, dict) and "message" in content:
            content = str(content["message"])
        elif isinstance(content, dict) or isinstance(content, list):
            # Don't expose raw JSON structures as user-facing messages
            logger.error(
                f"Response generation returned non-text content "
                f"(type={type(content).__name__}): {str(content)[:200]}; "
                f"using generic fallback message"
            )
            content = _GENERIC_FALLBACK_MESSAGE
        elif not isinstance(content, str):
            content = str(content)

        # Handle unstructured response (plain text)
        # In this case, use the entire content as the message
        #
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-016 [STALE]: this is the TERMINAL rung of the
        # degradation ladder (structured -> embedded-JSON fallback -> raw text) and
        # it MUST remain construct-safe — there is nothing below it to fall through
        # to, so anything this construction raises escapes _parse_* and fails the
        # whole turn. It builds the SAME max_length=5000-capped model as the rungs
        # above, so guarding only those would have RELOCATED the crash here rather
        # than fixing it. The slice is what makes the guarantee hold; do not remove
        # it, and do not add an uncapped or non-literal field to this construction
        # without re-checking every constraint on the model.
        #
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-020 [STALE]: the terminal rung is the LAST
        # LINE OF DEFENCE and must be BOTH construct-safe (above) AND
        # ENVELOPE-SAFE (below).  Construct-safety alone is not enough, and
        # shipping only half of it caused a real user-facing regression:
        # D-016 capped `message` but not `reasoning`, which carries the SAME
        # max_length=5000 (core/definitions/pass_models.py).  So
        # {"message": "Your booking is confirmed.", "reasoning": "R"*9000} failed
        # construction one rung up and landed here, and `content` was still the
        # whole serialized payload — the END USER was shown
        # `{"message": "Your booking is confirmed.", "reasoning": "RRRR…`.
        # Pre-D-016 that raised a catchable exception; post-D-016 it was a
        # silently-wrong success, which is WORSE than the defect being fixed.
        # Never emit a serialized envelope as user-facing text: recover the
        # human-readable `message` out of it, or say something generic.
        #
        # DECISION plan-2026-07-18T162030-a02151fe/D-022 [STALE]
        # RESIDUAL LEAK, DELIBERATELY LEFT OPEN. `_looks_like_json` requires the
        # text to BOTH start and end with a brace/bracket pair, so three envelope
        # shapes still pass through verbatim: prose-PREFIXED (`Here you go:
        # {...}`), prose-SUFFIXED, and markdown-FENCED (```json ... ```).
        #
        # This is a deliberate acceptance, not an oversight, and it was measured
        # rather than argued. D-016 widened this guard to also fire on "any
        # parseable non-empty JSON object appears ANYWHERE in the text" and that
        # widening was REVERTED here, because it destroyed ordinary assistant
        # replies. All four of these reached the user verbatim before the
        # widening and were replaced by _GENERIC_FALLBACK_MESSAGE after it:
        #   'Sure! To create a user, POST to /api/users with a body like
        #    {"name": "Alice", "role": "admin"}.'
        #   'The server returned {"error": "not_found"} which means the record
        #    does not exist.'
        #   'In Python you would write d = {"key": "value"} and then access
        #    d["key"].'
        #   'Here is the JSON you asked me to write: ```json\n{"order_id": 123}```'
        #
        # Do NOT re-attempt this with a smarter regex or a longer negative-case
        # list. The reason is structural: the leak `{"note": "...", "status":
        # "ok"}` and the legitimate quoted content `{"error": "not_found"}` are
        # the SAME STRING SHAPE. No text-shape discriminator can separate "an
        # envelope the model emitted by mistake" from "prose that legitimately
        # contains JSON", so an over-firing guard silently eats correct replies —
        # strictly worse than showing an obviously-wrong one. Closing this needs a
        # DIFFERENT SIGNAL entirely: schema provenance, or a response-format flag
        # recording that structured output was requested for this call.
        # That signal now exists for the structured branch (`structured`, D-020),
        # but this raw-text rung is still schema-blind. For a reply the provider
        # CUT OFF the different signal is finish_reason "length"
        # (plan-2026-11-06T093029-0068b055/D-024, below).
        # See decisions.md D-022.
        # D-030 (below) narrows the over-firing part: brace-shaped text that does
        # not PARSE as JSON is prose and is no longer replaced.
        #
        # Reuses this class's own `_looks_like_json` and the module's
        # `extract_json_from_text`; core must not import the equivalent
        # `_is_extraction_envelope` from fsm_llm/agents/adapt.py.
        #
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-030 (supersedes the shape-only
        # test above; D-022's accepted `{"a": 1}` case is unchanged). The braces
        # decide nothing alone: the text is replaced by the message or the
        # apology ONLY when it PARSES as JSON (object or array), so brace-shaped prose
        # ('{name}, welcome! ...', '{1, 2, 3}') reaches the user and D-020's retry
        # no longer doubles the apology. <think> blocks are dropped first (a
        # provider may inline them); keep the original when nothing else is left.
        # Do NOT use strip_think_and_fences here: it would strip code fences from
        # a reply that is legitimately prose. Trade-off: a brace-shaped but
        # malformed envelope is now shown as text, unless the provider cut it
        # off at max_tokens (D-024 below). See decisions.md D-030.
        content = remove_think_blocks(content).strip() or content
        parsed: Any = None
        if self._looks_like_json(content):
            parsed = extract_json_from_text(content)
            if parsed is None:
                try:
                    parsed = json.loads(content)  # a valid array envelope
                except (ValueError, RecursionError):
                    pass
            if parsed is not None:
                recovered = (
                    _safe_str(parsed.get("message"))
                    if isinstance(parsed, dict)
                    else None
                )
                content = (recovered or "").strip() or _GENERIC_FALLBACK_MESSAGE
        # DECISION plan-2026-11-06T093029-0068b055/D-024: finish_reason is the
        # DIFFERENT signal D-022 asked for, for the TRUNCATED case only. A reply
        # the provider cut off at max_tokens (finish_reason "length") that
        # opens as a JSON envelope and does NOT parse is a half envelope
        # (live: `{ "message": "CONSOLIDATED ... Recommend`, shipped with
        # success=True): it is replaced by _CUT_OFF_FALLBACK_MESSAGE. The
        # decision is finish_reason AND unparsability, never the text shape
        # alone (D-022 still binds): a cut PROSE reply is shipped as is (the
        # WARNING in `_send` names it), and a whole reply is untouched. Do NOT
        # salvage the fragment's prefix here (a half answer shown as whole) and
        # do NOT reuse the generic message (D-020 would retry at the same
        # max_tokens and be cut at the same place). See decisions.md D-024.
        if (
            parsed is None
            and _finish_reason(response) == FINISH_REASON_LENGTH
            and content.startswith(("{", "["))
        ):
            logger.warning(
                "Response generation was cut off at "
                f"max_tokens={self.max_tokens} inside a JSON envelope; "
                "not shown as the reply"
            )
            content = _CUT_OFF_FALLBACK_MESSAGE
        return ResponseGenerationResponse(
            message=content[:RESPONSE_TEXT_MAX_LENGTH],
            message_type="response",
            reasoning="Unstructured response - used entire content as message",
        )

    def _parse_field_extraction_response(
        self, response, request: FieldExtractionRequest
    ) -> FieldExtractionResponse:
        """Parse LLM response for single-field extraction."""
        content = response.choices[0].message.content

        # Strip <think>...</think> tags and markdown code fences that some
        # models (e.g. Qwen) emit — shared with extract_bulk_data via
        # strip_think_and_fences (llm/text.py) so the two readers cannot
        # re-diverge.
        if isinstance(content, str):
            content = strip_think_and_fences(content)

        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")

                value = _field_value(data, request.field_name)
                # Handle extracted_data wrapper: some models nest the value
                if value is None and "extracted_data" in data:
                    ed = data["extracted_data"]
                    if isinstance(ed, dict):
                        value = ed.get(request.field_name)
                # coerce_confidence maps NaN/±inf → default (1.0) and clamps to
                # [0,1]; a `{...}`/`null` still raises TypeError/ValueError for
                # the ladder below (D-001, core/coerce.py).
                try:
                    confidence = coerce_confidence(data.get("confidence", 1.0), 1.0)
                except (TypeError, ValueError):
                    # DECISION plan-2026-09-19T175721-21cd7f8e/D-028
                    # An unreadable score keeps the value the model returned at
                    # 0.5 ("not scored"). Do NOT let it raise: the ladder then
                    # reaches the unstructured rung, which stores the WHOLE JSON
                    # TEXT as a valid str/any value. Do NOT default to 1.0 (it
                    # would bypass an author `confidence_threshold`) or 0.0
                    # (D-016 drops exact 0.0). See decisions.md D-028.
                    confidence = 0.5
                # D-020: `reasoning` carries max_length=5000 here too
                # (core/definitions/pass_models.py): same trapdoor class as
                # ResponseGenerationResponse.reasoning.
                reasoning = _safe_str(data.get("reasoning"))

                return FieldExtractionResponse(
                    field_name=request.field_name,
                    value=value,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            # D-016: TypeError added by the step-4 sweep, which found this PRIMARY
            # rung escaping too — `float(data.get("confidence", 1.0))` raises
            # TypeError (not ValueError) on a model-supplied
            # `"confidence": {...}` / `null`, so the ladder was breached one rung
            # above the two reported fallback branches. Same defect class, so it
            # is fixed in the same step.
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to parse field extraction response: {e}. "
                    f"Content preview: {str(content)[:200]}"
                )

        # Fallback: extract JSON embedded in text (mirrors response gen path)
        if isinstance(content, str):
            data = extract_json_from_text(content)
            if isinstance(data, dict):
                value = _field_value(data, request.field_name)
                # Nested key search: look through nested dicts (depth ≤ 3)
                if value is None:
                    value = self._find_nested_key(data, request.field_name, max_depth=3)
                try:
                    # `coerce_confidence(...)` is INSIDE the guard on purpose: a
                    # model-supplied `"confidence": {...}` raises TypeError, and a
                    # non-str `reasoning` (or one over max_length=5000) raises
                    # ValidationError — both would otherwise escape the ladder from
                    # this rung just as the unguarded construction did. NaN/±inf
                    # is now mapped to the 0.95 default first (D-001, core/coerce.py).
                    try:
                        confidence = coerce_confidence(
                            data.get("confidence", 0.95), 0.95
                        )
                    except (TypeError, ValueError):
                        # DECISION plan-2026-09-19T175721-21cd7f8e/D-028
                        # Same rule as the primary rung: keep the value, 0.5.
                        confidence = 0.5
                    reasoning = _safe_str(data.get("reasoning"))  # D-020
                    if value is not None:
                        logger.debug("Extracted field JSON via fallback")
                        return FieldExtractionResponse(
                            field_name=request.field_name,
                            value=value,
                            confidence=confidence,
                            reasoning=reasoning,
                        )
                except (ValueError, TypeError) as e:
                    # Fall THROUGH to the unstructured-coercion and terminal rungs.
                    logger.warning(f"Field extraction fallback failed validation: {e}")

        # Unstructured fallback — try to coerce raw content to expected type
        if isinstance(content, str) and content.strip():
            raw = content.strip()
            coerced_value: Any = None
            coerced_reasoning = "Unstructured response coerced to expected type"
            coerced_confidence = 0.5
            # DECISION plan-2026-09-29T103145-06a5ec0a/D-050
            # Raw text that opens this field's own `{"field_name": ..., "value":`
            # envelope (it reaches this rung when max_tokens cut it off) yields
            # the salvaged value, or nothing. Do NOT store the envelope text: a
            # str/any field then carried `{"field_name": "generated_output",
            # "value": "...` into the agent answer. Do NOT widen this to prose
            # that merely contains JSON (D-022 above still binds): only a reply
            # that STARTS with the envelope prefix is unwrapped. On Ollama the
            # `any` grammar has no object branch, so for a JSON artifact this
            # salvage, not the field type, is the protection (D-056).
            # Do NOT return None for a COMPLETE value the strict decoder
            # rejects (`\d`, `C:\Users`): its characters are kept (D-022's
            # "key never lands" class). Do NOT pass a cut-off value off as
            # whole: it is logged as truncated and returned at
            # TRUNCATED_SALVAGE_CONFIDENCE, so a field threshold can drop it.
            is_envelope, salvaged, truncated = False, None, False
            if request.field_type in ("str", "any"):
                is_envelope, salvaged, truncated = _salvage_envelope_value(
                    raw, request.field_name
                )
            if is_envelope:
                if request.field_type == "str" and isinstance(salvaged, dict | list):
                    salvaged = json.dumps(salvaged, ensure_ascii=False)
                elif request.field_type == "str" and salvaged is not None:
                    salvaged = str(salvaged)
                coerced_value = salvaged
                if truncated and salvaged is not None:
                    coerced_confidence = TRUNCATED_SALVAGE_CONFIDENCE
                    coerced_reasoning = (
                        "Value TRUNCATED: salvaged prefix of a cut-off "
                        "extraction envelope"
                    )
                    logger.warning(
                        f"Field '{request.field_name}': the value was cut off "
                        "(likely by max_tokens); kept a truncated prefix of "
                        f"{len(str(salvaged))} chars at confidence "
                        f"{TRUNCATED_SALVAGE_CONFIDENCE}"
                    )
                else:
                    coerced_reasoning = (
                        "Value salvaged from an unparseable extraction envelope"
                    )
                    logger.warning(
                        f"Field '{request.field_name}': the reply was an "
                        "unparseable extraction envelope; kept its value only"
                    )
            elif request.field_type == "str":
                # DECISION plan-2026-07-18T162030-a02151fe/D-022 [STALE]
                # This rung hands the raw text back as the field value even when
                # that text is a prose-wrapped envelope. D-016 guarded it with
                # `None if is_envelope(raw) else raw`; that guard is REVERTED,
                # for the same reason as the response-generation rung above and
                # with a WORSE blast radius here. `None` means the key never
                # lands in context, so a state listing it in
                # `required_context_keys` never satisfies and the conversation
                # silently loops re-asking. Measured: 'The API returned
                # {"error": "not_found"} when I tried to save.' became `None`.
                # A free-text `complaint` / `error_message` field is the
                # canonical victim. See decisions.md D-022.
                coerced_value = raw
            elif request.field_type in ("int", "float"):
                # The pipeline's number rules (core.coerce.coerce_int/_float):
                # no bool, no 3.7 as 3, no inf or nan.
                number = coerce_int if request.field_type == "int" else coerce_float
                try:
                    coerced_value = number(raw)
                except ValueError:
                    pass
            elif request.field_type == "bool":
                # Same closed vocabulary as `engine.kinds.conversation.fields._coerce_bool`
                # (core.coerce.parse_bool_word, plan-2026-11-05T134105-5ed71cf2/D-008).
                coerced_value = parse_bool_word(raw)
            elif request.field_type in ("dict", "list"):
                # Strict: only accept JSON-parsed values of the correct type
                try:
                    parsed = json.loads(raw)
                    expected = dict if request.field_type == "dict" else list
                    if isinstance(parsed, expected):
                        coerced_value = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            elif request.field_type == "any":
                try:
                    coerced_value = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting a quoted value first
                    quoted = re.search(r'"([^"]+)"', raw)
                    if quoted:
                        coerced_value = quoted.group(1)
                    else:
                        coerced_value = raw
            if coerced_value is not None:
                return FieldExtractionResponse(
                    field_name=request.field_name,
                    value=coerced_value,
                    confidence=coerced_confidence,
                    reasoning=coerced_reasoning,
                )

        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="Failed to extract field from LLM response",
            is_valid=False,
            validation_error=(
                "Extraction produced no usable value: " + cut_off_note(self.max_tokens)
                if _finish_reason(response) == FINISH_REASON_LENGTH
                else "Extraction produced no usable value"
            ),
        )

    def _looks_like_json(self, text: str) -> bool:
        """Check if text appears to be JSON format."""
        if not isinstance(text, str):
            return False

        text = text.strip()
        return (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        )
