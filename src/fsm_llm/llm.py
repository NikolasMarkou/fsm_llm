"""
LLM Interface Module for FSM-Driven Conversational AI.

This module provides the core interface and implementation for Large Language Model (LLM)
communication within the fsm-llm library's 2-pass architecture. It defines how
FSM-driven applications interact with various LLM providers while maintaining clear
separation of concerns between data extraction, response generation, and transition decisions.

Architecture Overview
---------------------
The module implements a 2-pass architecture that separates LLM operations into distinct phases:

1. **Data Extraction Pass**: Extract and understand information from user input without
   generating any user-facing content. This prevents premature response generation and
   ensures all necessary data is captured before state transitions.

2. **Response Generation Pass**: Generate appropriate user-facing messages based on the
   final state context after all data extraction and transition evaluation is complete.

Key Components
--------------
LLMInterface : abc.ABC
    Abstract base class defining the contract for LLM communication with support for
    the 2-pass architecture. All LLM implementations must inherit from this interface.

LiteLLMInterface : LLMInterface
    Concrete implementation using LiteLLM for multi-provider support. Handles OpenAI,
    Anthropic, and other popular LLM providers through a unified interface.

Core Methods
------------
generate_response(request: ResponseGenerationRequest) -> ResponseGenerationResponse
    Generate user-facing messages based on final state context and extracted data.

extract_field(request: FieldExtractionRequest) -> FieldExtractionResponse
    Extract a single specific field from user input with custom instructions.

Integration with FSM System
---------------------------
This module integrates with the broader fsm-llm system:

- **FSMManager** (`fsm.py`) orchestrates the overall FSM execution and calls these
  methods at appropriate points in the conversation flow.
- **PromptBuilder** (`prompts.py`) constructs the specialized prompts for each pass
  based on current FSM state, context, and conversation history.
- **API** (`api.py`) provides the high-level interface that developers use, internally
  coordinating between FSM management and LLM communication.
"""

from __future__ import annotations

import abc
import json
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from litellm import completion, embedding, get_supported_openai_params

from .constants import (
    DEFAULT_TEMPERATURE,
    EMPTY_USER_MESSAGE_TURN,
    MALFORMED_TOOL_CALL_MARKERS,
    NEUTRAL_USER_TURN,
    RESERVED_EMBEDDING_CALL_KWARGS,
    RESERVED_LLM_CALL_KWARGS,
    TRUNCATED_SALVAGE_CONFIDENCE,
    USAGE_KIND_BY_CALL_TYPE,
    USAGE_KIND_COMPLETE,
    USAGE_KIND_EMBED,
    USAGE_KIND_STREAM,
)
from .definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMCallCounts,
    LLMResponseError,
    LLMUsage,
    ModelToolCall,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger
from .ollama import (
    apply_ollama_params,
    build_ollama_response_format,
    is_ollama_model,
    prepare_ollama_messages,
)
from .utilities import (
    _remove_think_blocks,
    _resolve_reasoning_trace,
    coerce_confidence,
    extract_json_from_text,
    strip_think_and_fences,
)

# Mirrors ``ResponseGenerationResponse.message``'s ``max_length`` constraint
# (definitions.py:154).  Duplicated deliberately rather than introspected out of
# the pydantic model: the terminal raw-text rung of _parse_response_generation_
# response must be able to truncate WITHOUT importing model internals.  The two
# literals are kept in lockstep by a machine-checked cap-drift assertion in
# tests/test_fsm_llm/test_llm_parse_fallback_seam.py — if you change one and not
# the other, that test fails.
_RESPONSE_MESSAGE_MAX_LEN = 5000

# Shown to the END USER when no human-readable text can be recovered from a
# model response.  Referenced from two places in
# _parse_response_generation_response (the non-text-content branch and the
# terminal rung's envelope guard) — one string, one place to change it.
# DECISION plan-2026-09-22T080837-8b258a25/D-035: guards apology_retry_count.
# Do NOT move it into __init__: tests build LiteLLMInterface via __new__, and the
# D-020 branch would then raise AttributeError. Increments are rare, so one
# process-wide lock costs nothing.
_APOLOGY_COUNT_LOCK = threading.Lock()

_GENERIC_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't generate a proper response. Please try again."
)

# Top-level keys of the bulk-extraction envelope the pipeline asks for
# (`{"extracted_data": {...}, "confidence": ..., "reasoning": ...}`). Dropped
# from a FLAT reply's data in `extract_bulk_data` (D-009, audit D4).
_BULK_ENVELOPE_KEYS = frozenset({"confidence", "reasoning"})

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


# Call types whose reply is parsed as JSON: on Ollama they run at temperature 0.
_STRUCTURED_CALL_TYPES = frozenset(
    {"data_extraction", "field_extraction", "classification"}
)

# Call types whose reply is free text even when a response_format is sent (the
# Pass-2 ``output_schema`` case keeps the user's temperature on Ollama).
_FREE_TEXT_CALL_TYPES = frozenset({"response_generation"})


def _fill_empty_user_turns(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give every user turn without text a provider-safe content.

    Args:
        messages: the provider message list of one request. A ``user``
            content of ``None`` means there is no user message; a string is
            the user's message, empty or not.

    Returns:
        A new list of new dicts. A ``user`` content of ``None`` becomes
        ``NEUTRAL_USER_TURN``, a ``""`` or whitespace-only one becomes
        ``EMPTY_USER_MESSAGE_TURN``; every other message is copied unchanged
        (an ``assistant`` tool-call message keeps ``content: None`` and its
        ``tool_calls``). The input is not mutated.

    Failure mode: none, never raises for a list of dicts.
    """
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-044: ``None`` and an empty
    # string are different facts. Do NOT collapse them (`content or ""`, one
    # constant for both): an instruction-shaped turn sent for a real empty
    # message is read by the model as the user's utterance. See decisions.md
    # D-044.
    filled: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        if message.get("role") == "user":
            content = message.get("content")
            if content is None:
                copy["content"] = NEUTRAL_USER_TURN
            elif isinstance(content, str) and not content.strip():
                copy["content"] = EMPTY_USER_MESSAGE_TURN
        filled.append(copy)
    return filled


def is_malformed_tool_call_error(exc: BaseException, *, tools_sent: bool) -> bool:
    """Whether a provider error is a garbled TOOL CALL rather than an outage.

    Interface contract (callers: ``LiteLLMInterface.complete``, and native_fc's
    own loop until it runs on ``complete``):
        - ``tools_sent``: whether the failed request carried tools. A request
          without tools cannot garble a tool call, so it is never one.
        - Returns ``True`` only when the error text contains one of
          ``MALFORMED_TOOL_CALL_MARKERS`` (case-insensitive); everything not
          positively identified is ``False`` (fail closed: an unrecognised
          failure stays a failure).
        - Never raises.
    """
    if not tools_sent:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in MALFORMED_TOOL_CALL_MARKERS)


def decode_tool_arguments(raw: Any) -> dict[str, Any] | None:
    """The arguments of one tool call as a dict, or ``None`` when malformed.

    Interface contract (callers: the ``complete`` reply normaliser, and
    native_fc's own loop until it runs on ``complete``):
        - A dict is returned as-is. A JSON string is decoded and must decode
          to an object. ``None`` or blank text means "no arguments": ``{}``.
        - Anything else (undecodable text; a JSON array, number, string or
          ``null``; any other type) returns ``None``: the call must not run.
        - Never raises.
    """
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError):
            return None
    return raw if isinstance(raw, dict) else None


_TRANSCRIPT_ROLES = frozenset({"system", "user", "assistant", "tool"})


def tool_exchange(
    content: str | None,
    calls: Sequence[ModelToolCall | Mapping[str, Any]],
    results: Sequence[str],
) -> list[dict[str, Any]]:
    """The transcript messages of one tool round, paired.

    Interface contract (callers: consumers of a completion state that run the
    model's calls, e.g. a tool-running handler appending to the transcript):
        - ``content``: the text the model wrote beside its calls; empty or
          ``None`` gives ``content: None`` (the provider's tool-call shape).
        - ``calls``: the round's calls, as ``ModelToolCall`` or as its JSON
          dict ``{"id", "name", "arguments"}`` (the ``calls`` entries of a
          completion state's result read back from context); at least one.
        - ``results``: one tool-result text per call, in the same order.
        - Returns new dicts: the assistant message ``{"role": "assistant",
          "content", "tool_calls": [{"id", "type": "function", "function":
          {"name", "arguments": <JSON text>}}]}`` followed by one
          ``{"role": "tool", "tool_call_id", "content"}`` per call. This is
          exactly the shape ``check_tool_transcript`` accepts as paired.
        - Raises ``ValueError`` when ``calls`` is empty, the two lengths
          differ, a call is not a valid ``ModelToolCall`` or a result is not
          a ``str``.
    """
    if not calls:
        raise ValueError("tool_exchange needs at least one call")
    if len(calls) != len(results):
        raise ValueError(
            f"tool_exchange got {len(calls)} calls but {len(results)} results"
        )
    parsed = [
        call if isinstance(call, ModelToolCall) else ModelToolCall.model_validate(call)
        for call in calls
    ]
    for index, result in enumerate(results):
        if not isinstance(result, str):
            raise ValueError(
                f"tool_exchange results[{index}] is {type(result).__name__}, not str"
            )
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in parsed
        ],
    }
    return [
        assistant,
        *(
            {"role": "tool", "tool_call_id": call.id, "content": result}
            for call, result in zip(parsed, results, strict=True)
        ),
    ]


def check_tool_transcript(messages: Sequence[Any]) -> None:
    """Refuse a transcript a provider must not receive.

    Interface contract (caller: the completion-state turn of the pipeline,
    before it sends a consumer-owned transcript):
        - Every entry is a dict whose ``role`` is ``system``, ``user``,
          ``assistant`` or ``tool``.
        - An assistant message with a non-empty ``tool_calls`` list is
          followed directly by exactly one ``tool`` message per call, matched
          by ``tool_call_id`` (any order); a ``tool`` message anywhere else is
          an orphan. ``tool_exchange`` builds this shape.
        - Returns ``None`` for a valid transcript (an empty one included);
          raises ``LLMResponseError`` naming the first bad entry otherwise.
    """
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role") if isinstance(message, dict) else None
        if role not in _TRANSCRIPT_ROLES:
            raise LLMResponseError(
                f"Transcript entry {index} is not a chat message with a known role"
            )
        if role == "tool":
            raise LLMResponseError(
                f"Transcript entry {index} is a tool result with no assistant "
                "tool call before it"
            )
        calls = message.get("tool_calls") if role == "assistant" else None
        index += 1
        if not calls:
            continue
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            raise LLMResponseError(
                f"Transcript entry {index - 1} has tool_calls that are not a list "
                "of call objects"
            )
        expected = sorted(str(call.get("id", "")) for call in calls)
        answered: list[str] = []
        while index < len(messages):
            following = messages[index]
            if not isinstance(following, dict) or following.get("role") != "tool":
                break
            answered.append(str(following.get("tool_call_id", "")))
            index += 1
        if sorted(answered) != expected:
            raise LLMResponseError(
                f"Transcript entry {index - len(answered) - 1} is an assistant "
                f"tool-call message whose {len(expected)} call(s) are not each "
                f"answered by one tool result (got {len(answered)}); refusing to "
                "send an unpaired transcript"
            )


def _completion_response(response: Any) -> CompletionResponse:
    """Normalise one provider reply into a ``CompletionResponse``.

    Contract:
        - No reply or no ``choices``: ``LLMResponseError("Empty response from
          LLM")``. A choice without ``message`` or a message without
          ``content`` (or another unreadable shape): ``LLMResponseError``
          ("Malformed LLM response shape: ...") chained from the error.
        - ``content`` is the text when it is a non-empty ``str``; a ``dict``
          content is sent back as its JSON text; empty content is ``None``.
          With no tool call, an empty content is recovered from the reasoning
          trace (``_resolve_reasoning_trace``); with tool calls it is not
          (``content: None`` is their normal shape).
        - Tool calls: every call needs a function name and arguments that
          decode to a JSON object (``decode_tool_arguments``); if ANY call
          fails that, the turn is ``malformed`` with no calls, the valid ones
          included, and its text kept.
    """
    choices = getattr(response, "choices", None) if response else None
    if not choices:
        raise LLMResponseError("Empty response from LLM")
    try:
        message = choices[0].message
        content = message.content
        raw_calls = getattr(message, "tool_calls", None) or []
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        raise LLMResponseError(f"Malformed LLM response shape: {e}") from e

    if isinstance(content, dict):
        text: str | None = json.dumps(content)
    elif isinstance(content, str):
        text = content or None
    elif content is None:
        text = None
    else:
        raise LLMResponseError(
            f"Malformed LLM response shape: content is {type(content).__name__}"
        )

    calls: list[ModelToolCall] = []
    for raw_call in raw_calls:
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", None)
        arguments = decode_tool_arguments(getattr(function, "arguments", None))
        if not isinstance(name, str) or not name or arguments is None:
            logger.warning(
                "Model tool-call turn is malformed (a call with no tool name or "
                "with arguments that are not a JSON object); none of its "
                "calls is returned"
            )
            return CompletionResponse(kind="malformed", text=text)
        call_id = getattr(raw_call, "id", None)
        calls.append(
            ModelToolCall(
                id=call_id if isinstance(call_id, str) else "",
                name=name,
                arguments=arguments,
            )
        )
    if calls:
        return CompletionResponse(kind="calls", text=text, calls=tuple(calls))
    if text is None:
        text = _resolve_reasoning_trace(message) or None
    return CompletionResponse(kind="final", text=text)


def _safe_str(value: Any) -> str | None:
    """Coerce a MODEL-SUPPLIED value into a value the capped models will accept.

    Args:
        value: any value read off a parsed LLM JSON payload.

    Returns:
        ``value`` truncated to ``_RESPONSE_MESSAGE_MAX_LEN`` when it is a ``str``;
        ``None`` for every other type.

    Failure mode: none — total function, never raises.  It exists so that an
    optional ``str | None`` field carrying ``max_length=5000``
    (``ResponseGenerationResponse.reasoning``, ``FieldExtractionResponse.reasoning``)
    or an unannotated-but-typed field (``message_type``) cannot fail model
    construction just because the model padded it.  Callers that need a
    REQUIRED ``str`` field slice directly instead, so mypy keeps the non-optional
    type.
    """
    return value[:_RESPONSE_MESSAGE_MAX_LEN] if isinstance(value, str) else None


# --------------------------------------------------------------
# Usage counters
# --------------------------------------------------------------

_COUNTER_FIELDS = (
    "calls",
    "errors",
    "usage_missing",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _reply_field(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` otherwise; ``None`` when absent."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _token_count(value: Any) -> int:
    """A token count as an ``int``; anything that is not a non-negative int is 0."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _usage_of(response: Any) -> tuple[int, int, int] | None:
    """(prompt, completion, total) tokens of a provider reply; ``None`` if absent.

    Read defensively (the rules of ``scripts/agents_bench.py``'s reader):
    ``usage`` may be a dict or an object, may be missing (streamed replies
    carry none), and its fields may be missing or not ints. A missing
    ``total_tokens`` is the sum of the other two.
    """
    usage = _reply_field(response, "usage")
    if usage is None:
        return None
    prompt = _token_count(_reply_field(usage, "prompt_tokens"))
    completion_tokens = _token_count(_reply_field(usage, "completion_tokens"))
    total = _token_count(_reply_field(usage, "total_tokens"))
    return prompt, completion_tokens, total or prompt + completion_tokens


class _UsageMeter:
    """Per-kind provider-call counters of one interface instance.

    Interface contract (shared by every provider call site of ``llm.py``):
    - ``record(kind, response)``: one answered call of ``kind``; its token
      usage is read with ``_usage_of`` (a streamed call passes ``None`` and
      counts as usage-missing). Never raises on a strange reply shape.
    - ``record_error(kind)``: one call of ``kind`` that raised.
    - ``snapshot(reset=...)``: a frozen ``LLMUsage`` copy, totals plus
      ``by_kind``; with ``reset=True`` the counters are cleared in the same
      critical section, so no call is lost between the read and the reset.
    Every method holds one lock: safe when one interface is shared by
    concurrent conversations. The lock is never held across a provider call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}

    def _bump(self, kind: str, **deltas: int) -> None:
        with self._lock:
            counts = self._counts.setdefault(kind, dict.fromkeys(_COUNTER_FIELDS, 0))
            for name, delta in deltas.items():
                counts[name] += delta

    def record(self, kind: str, response: Any) -> None:
        usage = _usage_of(response)
        if usage is None:
            self._bump(kind, calls=1, usage_missing=1)
            return
        prompt, completion_tokens, total = usage
        self._bump(
            kind,
            calls=1,
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )

    def record_error(self, kind: str) -> None:
        self._bump(kind, calls=1, errors=1)

    def snapshot(self, *, reset: bool = False) -> LLMUsage:
        with self._lock:
            per_kind = {kind: dict(counts) for kind, counts in self._counts.items()}
            if reset:
                self._counts = {}
        totals = {
            name: sum(counts[name] for counts in per_kind.values())
            for name in _COUNTER_FIELDS
        }
        return LLMUsage(
            **totals,
            by_kind={
                kind: LLMCallCounts(**counts)
                for kind, counts in sorted(per_kind.items())
            },
        )


# Guards the lazy creation of a meter, so two threads making an instance's
# first calls at once share one meter instead of each creating its own.
_METER_CREATION_LOCK = threading.Lock()


def _meter_of(owner: Any) -> _UsageMeter:
    """The ``_usage_meter`` of ``owner``, created on first use.

    Lazy so that an instance built with ``__new__`` (no ``__init__``) still
    counts. ``owner`` is any object that declares a class attribute
    ``_usage_meter: _UsageMeter | None = None``.
    """
    meter: _UsageMeter | None = owner._usage_meter
    if meter is None:
        with _METER_CREATION_LOCK:
            meter = owner._usage_meter
            if meter is None:
                meter = _UsageMeter()
                owner._usage_meter = meter
    return meter


def _connection_params(
    model: str,
    kwargs: dict[str, Any],
    *,
    timeout: float | None,
    retries: int,
    reserved: frozenset[str],
) -> dict[str, Any]:
    """The connection part of a provider request: model, kwargs, timeout, retries.

    Interface contract (shared by ``LiteLLMInterface._build_call_params`` and
    ``LiteLLMEmbedder.embed``):
    - ``kwargs``: the owner's constructor kwargs (``api_key``, ``api_base``,
      ...); keys in ``reserved`` are dropped, the rest go first so the
      explicit params below cannot be overridden by them.
    - ``timeout``: sent when not ``None``. ``retries``: sent as the SDK's
      ``max_retries`` when above 0, otherwise omitted (SDK default).
    - Returns a new dict; the caller adds its call-specific params. Never
      raises.
    """
    params: dict[str, Any] = {
        **{k: v for k, v in kwargs.items() if k not in reserved},
        "model": model,
    }
    if timeout is not None:
        params["timeout"] = timeout

    # DECISION plan-2026-07-19T075908-70b6bdec/D-007 [STALE]
    # Retries are delegated to the provider SDK's own retry layer via
    # `max_retries`. Do NOT change this to `num_retries`: that key routes to
    # litellm's tenacity layer, which sits ON TOP of the SDK layer (giving
    # 2N+1 requests, not N+1) and retries EVERYTHING, including 400/401 —
    # deterministic failures that can never succeed. `max_retries` is one
    # layer with correct error classification. Do NOT hand-roll a retry loop
    # here either. Now shared by every request builder via this one helper.
    # See decisions.md D-007 (historical) / D-015 (this plan's extraction).
    if retries > 0:
        params["max_retries"] = retries
    return params


# --------------------------------------------------------------
# Abstract Interface
# --------------------------------------------------------------


class LLMInterface(abc.ABC):
    """
    Abstract interface for LLM communication supporting the 2-pass architecture.

    This interface defines methods for data extraction, response generation, and
    transition decision making, allowing implementations to optimize for different use cases.
    """

    @abc.abstractmethod
    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        """
        Generate user-facing response based on final state context.

        This method generates the actual message shown to users after all data
        extraction and transition evaluation are complete.

        Args:
            request: Response generation request with final state context.
                Never sent for a silent state (empty
                ``response_instructions``): the pipeline makes no call there.

        Returns:
            Response generation response with user-facing message

        Raises:
            LLMResponseError: If response generation fails
        """
        pass

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        """Stream response tokens for Pass 2 (response generation).

        Yields individual text chunks as the LLM produces them.
        The default implementation falls back to ``generate_response``
        and yields the full message as a single chunk.

        Args:
            request: Response generation request with final state context.

        Yields:
            String chunks of the response as they arrive.
        """
        response = self.generate_response(request)
        yield response.message

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """
        Extract a single specific field from user input.

        This method performs targeted extraction of one named field with
        custom instructions, dynamic context, and validation.  Called by
        the engine after bulk ``extract_data`` completes.

        The default implementation raises ``NotImplementedError`` so
        existing subclasses that do not need field extraction remain
        compatible.

        Args:
            request: Field extraction request with focused instructions.
                ``request.context`` and ``request.validation_rules`` are a
                contract for third-party interfaces: the pipeline fills them,
                but ``LiteLLMInterface`` does not read them (the prompt
                already carries the context; the pipeline applies the rules).

        Returns:
            Field extraction response with typed value and confidence

        Raises:
            LLMResponseError: If field extraction fails
            NotImplementedError: If the subclass does not implement this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_field. "
            "Override this method to support targeted field extraction."
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        """
        Extract free-form key/value data per a state's ``extraction_instructions``.

        This method performs untargeted bulk extraction from a single prompt,
        for states that have ``extraction_instructions`` but no per-field
        schema (no ``required_context_keys``/``field_extractions``). Called by
        the pipeline's additive bulk-extraction pass.

        The default implementation raises ``NotImplementedError`` so
        existing subclasses that do not need bulk extraction remain
        compatible.

        Args:
            request: Bulk extraction request with the extraction prompt

        Returns:
            Data extraction response with the extracted key/value data

        Raises:
            LLMResponseError: If bulk extraction fails
            NotImplementedError: If the subclass does not implement this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_bulk_data. "
            "Override this method to support bulk data extraction."
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Send one completion: tool calling, plain text or structured output.

        The one request primitive besides the Pass-1/Pass-2 methods: a caller
        hands over the whole message list (``request.messages``) and reads a
        typed reply. The default implementation raises ``NotImplementedError``
        so existing subclasses that do not need it remain compatible.

        Args:
            request: The messages plus either ``tools`` (with ``tool_choice``)
                or a ``response_format``, and optional per-call
                ``temperature``/``max_tokens``.

        Returns:
            ``kind="calls"`` with decoded calls, ``"final"`` with the reply
            text, or ``"malformed"`` (a tool-call turn that cannot be run;
            no calls).

        Raises:
            LLMResponseError: If the provider call fails (an outage is not a
                reply) or the reply cannot be read.
            NotImplementedError: If the subclass does not implement this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement complete. "
            "Override this method to support completion requests."
        )


# --------------------------------------------------------------
# LiteLLM Implementation
# --------------------------------------------------------------


class LiteLLMInterface(LLMInterface):
    """
    LiteLLM-based implementation supporting multiple providers.

    This implementation uses LiteLLM to communicate with various LLM providers
    while maintaining the 2-pass architecture interface.
    """

    # (model, result) of the last successful get_supported_openai_params call.
    _supported_params_memo: tuple[str, list[str] | None] | None = None
    # Times the D-020 apology retry fired on this instance (read-only for callers).
    apology_retry_count: int = 0
    # Provider-call counters, created on first use (see _meter_of).
    _usage_meter: _UsageMeter | None = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 1000,
        timeout: float | None = 120.0,
        retries: int = 0,
        **kwargs,
    ):
        """
        Initialize LiteLLM interface with configuration.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus")
            api_key: Optional API key (uses environment if not provided)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens for responses
            timeout: Timeout in seconds for LLM API calls (None for no timeout)
            retries: Number of ADDITIONAL attempts the provider SDK makes after a
                failed call, i.e. `retries=N` yields at most N+1 provider requests.
                Only TRANSIENT failures are retried (connection errors, timeouts,
                429, 5xx). Deterministic client errors are NOT retried: a 400 bad
                request or a 401 auth failure costs exactly one request and fails
                immediately, so a malformed prompt or a wrong API key does not
                multiply in cost.

                This parameter REPLACES the provider SDK's own retry count rather
                than adding to it. Retry is NOT off by default: with `retries=0`
                the parameter is omitted entirely and the SDK's built-in default
                (2 additional attempts on transient errors, with exponential
                backoff) still applies — byte-for-byte the historical behavior.
                The practical consequence is that `retries=1` LOWERS resilience
                below the default; only values >= 3 increase it. Measured against
                a local server returning a persistent 429: omitted -> 3 requests,
                retries=1 -> 2, retries=2 -> 3, retries=4 -> 5.

                PROVIDER-DEPENDENT: this parameter is honored by providers routed
                through the OpenAI SDK. It is a NO-OP for providers that are not
                (measured: `ollama_chat/*` and `ollama/*` make exactly 1 request
                regardless of `retries`). Setting it against ollama neither helps
                nor hurts; it is silently ignored.

                Cost: retries multiply worst-case wall clock per turn on transient
                failures, roughly (N+1) x timeout plus the SDK's backoff waits.
                Backoff behavior is the SDK's own and is not configured here.

                Values <= 0 are treated as "leave the SDK default alone" (no
                validation ceremony).
            **kwargs: Additional LiteLLM parameters
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.kwargs = kwargs
        dropped = sorted(RESERVED_LLM_CALL_KWARGS.intersection(kwargs))
        if dropped:
            logger.warning(
                f"LiteLLMInterface ignores reserved call kwargs {dropped}; "
                "the framework sets them per call"
            )

        # Configure API keys based on model type
        self._configure_api_keys(api_key)

        logger.info(f"Initialized LiteLLM interface with model: {model}")

    def _configure_api_keys(self, api_key: str | None) -> None:
        """Configure API keys via kwargs (avoids global os.environ mutation)."""
        if not api_key:
            logger.debug("No API key provided, using environment variables")
            return

        self.kwargs["api_key"] = api_key
        logger.debug("Set API key in kwargs")

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        """
        Generate response using LLM focused on final state context.

        This method creates prompts that generate appropriate user-facing responses
        based on the final state context and all extracted information.
        """
        try:
            start_time = time.time()

            logger.debug(f"Generating response with {self.model}")
            logger.debug(
                f"Final state context: current state, transition: {request.transition_occurred}"
            )

            # Prepare messages for response generation
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Get LLM response
            response = self._make_llm_call(
                messages,
                "response_generation",
                response_format=request.response_format,
            )
            response_time = time.time() - start_time

            logger.debug(f"Response generation completed in {response_time:.2f}s")

            # Parse response for response generation
            parsed = self._parse_response_generation_response(
                response, structured=request.response_format is not None
            )
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-020: the apology means
            # the model produced nothing usable (live: `{"message": ""}` on a
            # greeting, 1/18). Retry ONCE and return that result whatever it is:
            # no loop, and an error on the retry keeps the first apology instead
            # of failing the turn. Do NOT retry on any other outcome.
            if parsed.message == _GENERIC_FALLBACK_MESSAGE:
                logger.warning(
                    "Response generation yielded no usable text; retrying once"
                )
                with _APOLOGY_COUNT_LOCK:
                    self.apology_retry_count += 1
                try:
                    return self._parse_response_generation_response(
                        self._make_llm_call(
                            messages,
                            "response_generation",
                            response_format=request.response_format,
                        ),
                        structured=request.response_format is not None,
                    )
                except Exception as retry_error:
                    logger.warning(f"Response generation retry failed: {retry_error!s}")
            return parsed

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary.
            error_msg = f"Response generation failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        """Stream response tokens for Pass 2 using LiteLLM streaming.

        Pass 1 (extraction) is never streamed — it must complete fully
        for transition evaluation.  This method only streams Pass 2
        (user-facing response generation).
        """
        try:
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Build call params via the shared builder (DECISION
            # plan-2026-09-12T135914-45a654de/D-015 — this used to be a
            # second, independently-maintained ~40-line copy of
            # _make_llm_call's builder; see _build_call_params's own D-015
            # anchor for the full rationale and the D-007 max_retries
            # constraint it still preserves). `stream=True` adds
            # `"stream": True`; the forced-structured-output branch never
            # fires here because `call_type="response_generation"`.
            call_params = self._build_call_params(
                messages,
                "response_generation",
                response_format=request.response_format,
                stream=True,
            )

            response = self._send(call_params, call_type="response_generation")

            accumulated: list[str] = []
            reasoning_parts: list[str] = []
            last_delta = None
            for chunk in response:
                # DECISION plan-2026-09-20T114608-a8e47b88/D-022
                # `getattr(chunk, "choices", None)` replaces
                # `hasattr(chunk, "choices") and chunk.choices` -- a "not
                # falsy" check either way, so an ABSENT attribute (getattr's
                # None default) and a PRESENT-but-falsy one (`None`/`[]`)
                # both `continue` identically to before. Safe here because
                # nothing downstream distinguishes "attribute missing" from
                # "attribute present but falsy" -- see decisions.md D-022 for
                # the ONE flagged call site (llm.py content check) where that
                # distinction IS load-bearing and was deliberately left as
                # `hasattr()`.
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                last_delta = delta
                # Accumulate reasoning the same way content is accumulated: a
                # provider may stream `reasoning_content` as one fragment per
                # chunk. Reuse the shared resolver so a Delta is read via the
                # same field precedence as a Message — never hand-roll the
                # `getattr(..., "thinking")` lookup here. See decisions.md D-002.
                frag = _resolve_reasoning_trace(delta)
                if frag:
                    reasoning_parts.append(frag)
                content = getattr(delta, "content", None)
                if content is not None:
                    accumulated.append(content)
                    yield content

            # Same two tail guards _make_llm_call applies: a stream whose every
            # delta.content is "" is a FAILURE, not a silent empty reply. Recover
            # the answer from the reasoning trace ACCUMULATED across all chunks
            # (matching the non-streaming path, which sees the full assembled
            # message) — not just the final delta; otherwise a provider that
            # streams reasoning incrementally leaves only the last fragment
            # available and recovery fails where the non-streaming path succeeds.
            if not "".join(accumulated):
                full_trace = "".join(reasoning_parts)
                if full_trace:
                    recovered = self._recover_content_from_trace(full_trace)
                else:
                    # Defensive: a provider that puts the whole trace only on the
                    # final delta object (not in per-chunk reasoning_content).
                    recovered = (
                        self._extract_content_from_thinking(last_delta)
                        if last_delta is not None
                        else None
                    )
                if not recovered:
                    raise LLMResponseError("LLM returned empty content")
                yield recovered

        except LLMResponseError:
            raise
        except Exception as e:
            error_msg = f"Streaming response generation failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """Extract a single specific field from user input."""
        try:
            start_time = time.time()

            logger.debug(
                f"Extracting field '{request.field_name}' "
                f"(type={request.field_type}) with {self.model}"
            )

            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Ollama only: a field-typed grammar so the model cannot wrap the
            # answer in an object (LV-01, decisions.md D-001). Other providers
            # keep the generic `json_object` format.
            typed_format = (
                build_ollama_response_format("field_extraction", request.field_type)
                if is_ollama_model(self.model)
                else None
            )
            response = self._make_llm_call(
                messages, "field_extraction", response_format=typed_format
            )
            response_time = time.time() - start_time

            logger.debug(
                f"Field extraction for '{request.field_name}' "
                f"completed in {response_time:.2f}s"
            )

            return self._parse_field_extraction_response(response, request)

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary.
            error_msg = f"Field extraction failed for '{request.field_name}': {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        """Extract free-form key/value data per a state's extraction_instructions.

        Interface contract (mirrors ``extract_field``'s error boundary):
            - Parses whatever the LLM returns via ``_make_llm_call``, reusing
              the same ``<think>``/fence-stripping and JSON-parsing logic as
              ``_parse_field_extraction_response``.
            - A response with no ``extracted_data``/dict-shaped payload
              (content is neither a ``str`` nor a ``dict``, or the parsed
              ``extracted_data``/top-level value isn't a ``dict``) resolves to
              an EMPTY ``DataExtractionResponse`` — this is "the model found
              nothing", not a failure, so it is NOT raised as
              ``LLMResponseError``.
            - Non-object JSON (``[1, 2]``, ``null``, ``42``) is the same
              "found nothing" case and resolves to an empty response.
            - When ``json.loads`` fails on the stripped text, the shared
              ``extract_json_from_text`` ladder is tried ONCE (prose around a
              JSON object, e.g. ``Sure! {"a": 1}``) before giving up.
            - Any actual failure (unparseable text, transport/parsing error)
              raises ``LLMResponseError``, wrapping the underlying cause.
            - Filtering of ``None``/empty-string/empty-dict values out of
              ``extracted_data`` is the CALLER's job (``pipeline.py``'s merge
              logic), not this method's — this method returns the extraction
              verbatim, except that a FLAT reply (no ``extracted_data``
              wrapper) loses its top-level ``confidence``/``reasoning``
              envelope keys (the confidence is still read from it).
        """
        try:
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            response = self._make_llm_call(messages, "data_extraction")
            content = response.choices[0].message.content

            if isinstance(content, str):
                content = strip_think_and_fences(content)
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # One recovery rung (D-010): prose around a JSON object.
                    # `extract_json_from_text` returns a dict or None, so a
                    # None means there is genuinely nothing parseable and the
                    # original decode error is raised below as before.
                    data = extract_json_from_text(content)
                    if data is None:
                        raise
            elif isinstance(content, dict):
                data = content
            else:
                return DataExtractionResponse(extracted_data={})

            if not isinstance(data, dict):
                # Valid JSON that is not an object (`[1,2]`, `null`, `42`):
                # "the model found nothing", per the contract above.
                return DataExtractionResponse(extracted_data={})

            # DECISION plan-2026-09-21T203800-8a03483a/D-009 (D4): a flat reply
            # (no `extracted_data` wrapper) carries the envelope's `confidence`
            # and `reasoning` at top level; they are metadata, not user data,
            # so they never enter context. A wrapped reply keeps whatever its
            # `extracted_data` holds. Do NOT return the flat object verbatim.
            if "extracted_data" in data:
                extracted = data["extracted_data"]
            else:
                extracted = {
                    k: v for k, v in data.items() if k not in _BULK_ENVELOPE_KEYS
                }
            if not isinstance(extracted, dict):
                return DataExtractionResponse(extracted_data={})

            try:
                confidence = coerce_confidence(data.get("confidence", 1.0), 1.0)
            except (TypeError, ValueError):
                # An uncoercible score (`"high"`, `null`, `{...}`) must not
                # discard the extracted data: fall back to the field default
                # (D-023). The signal is lost, the data is not.
                confidence = 1.0
            return DataExtractionResponse(
                extracted_data=extracted, confidence=confidence
            )

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary, matching
            # extract_field's error boundary above.
            error_msg = f"Bulk data extraction failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send one completion through the shared builder and read the reply.

        The request is built by ``_build_call_params`` like every other call:
        connection kwargs, timeout, the filled user turns, ``tools`` (never
        gated on litellm's supported-params list) or a ``response_format``
        (sent only when the model supports it, else a WARNING), per-call
        temperature and max_tokens, and Ollama preparation (thinking off,
        ``/nothink`` on the last user turn, the schema echoed into it; a
        ``response_format`` call runs at temperature 0). A ``seed`` is sent
        only when this interface was built with one.

        Returns:
            The normalised reply (see ``CompletionResponse``). A provider error
            that names a garbled tool call, on a request with tools, is a
            ``malformed`` reply rather than an error.

        Raises:
            LLMResponseError: the provider call failed, chained from its
                error, or the reply could not be read.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-002: one request primitive
        # (supersedes 07ad3f8c/D-022's classifier-only method). Do NOT add a
        # second builder, a second `completion` binding or a per-kind method
        # (one for tools, one for text, one for schemas): every request
        # rule (tools XOR response_format, the Ollama gate, seed absent unless
        # set, user-turn filling) lives once in _build_call_params, and every
        # send goes through _send. A malformed tool turn is data
        # (kind="malformed", no calls), an outage raises. See decisions.md D-002.
        call_params = self._build_call_params(
            request.messages,
            request.call_type,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        try:
            response = self._send(call_params, call_type=request.call_type)
        except Exception as e:
            # Broad catch is intentional: the provider boundary. A garbled
            # tool call is a model behaviour, every other failure an outage.
            if is_malformed_tool_call_error(e, tools_sent=request.tools is not None):
                logger.warning(f"Provider rejected a garbled tool call: {e!s}")
                return CompletionResponse(kind="malformed")
            error_msg = f"Completion call failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e
        return _completion_response(response)

    def _send(self, call_params: dict[str, Any], *, call_type: str) -> Any:
        """Send one built request to the provider: the one ``completion`` call.

        Every provider request of this interface (Pass 1, Pass 2, stream,
        ``complete``) goes through here; the raw provider reply (a stream
        iterator when ``call_params["stream"]``) is returned and errors
        propagate unchanged. Each request is counted once on this instance's
        usage meter, under the kind of ``call_type`` (``stream`` for a
        streamed call): a raising request as a call and an error, a streamed
        one as a call with usage missing (its chunks carry no usage on
        Ollama, and the reply is not read here).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-004: the meter is per
        # instance and fed here only. Do NOT add a process-global counter
        # (shared state across conversations and tests), do NOT count in the
        # public methods (a second count site drifts from the one send path),
        # and do NOT read usage off a stream (its chunks carry none on Ollama).
        # Readers that want counts own (inject) the interface. See D-004.
        stream = bool(call_params.get("stream"))
        kind = (
            USAGE_KIND_STREAM
            if stream
            else USAGE_KIND_BY_CALL_TYPE.get(call_type, USAGE_KIND_COMPLETE)
        )
        meter = _meter_of(self)
        try:
            response = completion(**call_params)
        except Exception:
            # Broad catch is intentional: count the failed request, re-raise
            # it unchanged for the caller's own error boundary.
            meter.record_error(kind)
            raise
        meter.record(kind, None if stream else response)
        return response

    def usage(self) -> LLMUsage:
        """A frozen snapshot of this interface's provider-call counters.

        Counts every provider request this instance sent since it was built
        or since the last ``reset_usage``: totals plus ``by_kind`` (see
        ``LLMUsage``). Safe to call while other threads use the interface.
        """
        return _meter_of(self).snapshot()

    def reset_usage(self) -> LLMUsage:
        """Clear the counters and return the snapshot they held.

        The read and the clear are one atomic step, so a request made
        concurrently is counted either in the returned snapshot or in the
        next one, never lost.
        """
        return _meter_of(self).snapshot(reset=True)

    def _supported_openai_params(self) -> list[str] | None:
        """Return litellm's supported-param list for ``self.model``, memoised.

        Only a successful result is memoised (a ``None`` "unknown model" answer
        included); an exception propagates and the next call retries. The memo
        is keyed by the model string, so reassigning ``self.model`` refreshes it.
        """
        memo = self._supported_params_memo
        if memo is not None and memo[0] == self.model:
            return memo[1]
        result = get_supported_openai_params(model=self.model)
        self._supported_params_memo = (self.model, result)
        return result

    def _build_call_params(
        self,
        messages: list[dict[str, Any]],
        call_type: str,
        *,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Build the ``litellm.completion(**call_params)`` kwargs shared by
        ``_make_llm_call`` (non-streaming), ``generate_response_stream`` and
        ``complete``. A user turn without text is filled by
        ``_fill_empty_user_turns`` (``None``: the neutral instruction; empty
        string: the empty-message placeholder); no other message is touched
        and the caller's list is not mutated.

        # DECISION plan-2026-09-12T135914-45a654de/D-015
        # Extracted from two independently-maintained ~40-line builders
        # (_make_llm_call and generate_response_stream) that the codebase's
        # own D-007 comments (plan-2026-07-19T075908-70b6bdec, now [STALE] but
        # NOT resolved — see findings/core-reasoning-harness-fixes.md item 3)
        # already flagged as duplicated and asked to be kept in sync by hand.
        # Do NOT re-split this back into two builders "for clarity" — that is
        # exactly the shape that was drifting. Do NOT change `max_retries` to
        # `num_retries`: see the retry-layer rationale at _connection_params: this
        # is the SAME constraint D-007 documented, now enforced in one place
        # instead of two. See decisions.md D-015 (this plan) and D-007 (prior
        # plan, historical context only — not re-litigated here).
        #
        # The two deltas between callers are parameters, not hardcoded
        # branches: (a) the structured-output/response_format branch below is
        # gated on `call_type in ["data_extraction", "field_extraction"]`,
        # which is NEVER true for a streaming call (streaming call_type is
        # always "response_generation") — so passing `stream=True` alone does
        # not need its own extra guard on that branch; (b) `stream=True`
        # literally adds `"stream": True` to the dict, nothing else.
        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).
            stream: Whether this call is a streaming call (adds
                ``"stream": True``; never applies the forced-structured-output
                branch, though that branch's own ``call_type`` gate already
                makes it a no-op for the streaming caller's call_type).
            tools: OpenAI function schemas (``complete`` only). Sent as given,
                with ``tool_choice`` (``"auto"`` when ``None``), and never
                gated on litellm's supported-params list: it omits tools for
                models that call them natively (``ollama/``). The caller never
                passes ``response_format`` with tools (``CompletionRequest``
                refuses it).
            tool_choice: see ``tools``.
            temperature: per-call temperature; ``None`` keeps the interface's.
            max_tokens: per-call max tokens; ``None`` keeps the interface's.

        Returns:
            The kwargs dict ready to pass to ``litellm.completion(**...)``.
        """
        # Check for structured output support
        supported_params = self._supported_openai_params()

        # The connection part (kwargs first, then model, timeout, max_retries;
        # retry-layer rationale at _connection_params) is shared with
        # LiteLLMEmbedder.
        call_params = _connection_params(
            self.model,
            self.kwargs,
            timeout=self.timeout,
            retries=self.retries,
            reserved=RESERVED_LLM_CALL_KWARGS,
        )
        call_params["messages"] = _fill_empty_user_turns(messages)
        call_params["temperature"] = (
            self.temperature if temperature is None else temperature
        )
        call_params["max_tokens"] = (
            self.max_tokens if max_tokens is None else max_tokens
        )
        if stream:
            call_params["stream"] = True
        if tools is not None:
            call_params["tools"] = tools
            call_params["tool_choice"] = "auto" if tool_choice is None else tool_choice

        # Add structured output if supported and beneficial.
        # Do NOT force structured output for response_generation — the
        # response is user-facing natural language, not structured data
        # UNLESS an explicit response_format was provided (e.g., for
        # schema-enforced agent output). This branch never fires for a
        # streaming call: streaming's call_type is always
        # "response_generation", never "data_extraction"/"field_extraction".
        if (
            supported_params
            and "response_format" in supported_params
            and call_type in ["data_extraction", "field_extraction"]
        ):
            if is_ollama_model(self.model):
                # Ollama: use json_schema with explicit schema for
                # grammar-constrained output.
                ollama_fmt = build_ollama_response_format(call_type)
                if ollama_fmt is not None:
                    call_params["response_format"] = ollama_fmt
            else:
                call_params["response_format"] = {"type": "json_object"}

        # Apply explicit response_format override (e.g., from output_schema).
        # This allows schema-enforced output for response_generation calls
        # when the caller provides a JSON schema.
        if (
            response_format is not None
            and supported_params
            and "response_format" in supported_params
        ):
            call_params["response_format"] = response_format
        elif response_format is not None:
            logger.warning(
                f"response_format requested but not supported by "
                f"model '{self.model}'; output may not match schema"
            )

        self._apply_model_specific_params(
            call_params, call_type, response_format=response_format
        )

        # Ollama: prepend /nothink and embed schema in prompt
        call_params["messages"] = prepare_ollama_messages(
            call_params["messages"],
            self.model,
            call_params.get("response_format"),
        )

        return call_params

    def _make_llm_call(
        self,
        messages: list[dict[str, str | None]],
        call_type: str,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """
        Make LLM API call with appropriate configuration.

        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).

        Returns:
            Raw LLM response
        """
        call_params = self._build_call_params(
            messages, call_type, response_format=response_format
        )

        # Make the API call
        response = self._send(call_params, call_type=call_type)

        # Validate response structure. D-022: `getattr(response, "choices", None)`
        # replaces `hasattr(response, "choices") and response.choices` -- see the
        # streaming loop above for why this "not falsy" shape is safe to convert.
        choices = getattr(response, "choices", None) if response else None
        if not choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = choices[0]
        # D-022: `getattr(choice, "message", None) is not None` replaces
        # `hasattr(choice, "message")` -- safe, since a `message` attribute
        # present-but-`None` falls through to the SAME raise via the `content`
        # check below either way (`hasattr(None, "content")` is `False`).
        #
        # The SECOND check is deliberately LEFT as `hasattr()`, not converted.
        # DECISION plan-2026-09-20T114608-a8e47b88/D-022
        # `hasattr(choice.message, "content")` checks only that the `content`
        # ATTRIBUTE EXISTS, regardless of its value -- and a real litellm
        # `Message` legitimately carries `content=None` for an Ollama
        # reasoning-only reply (H3, `test_make_llm_call_recovers_from_none_content`).
        # `getattr(choice.message, "content", None) is not None` CANNOT
        # replicate this: it cannot distinguish "attribute absent" from
        # "attribute present, value None" (both collapse to the same `None`),
        # so it would raise "Response missing message content" for that
        # legitimate None-content case, before `_extract_content_from_thinking`
        # ever runs -- a real regression, not a style nit. Confirmed empirically
        # (not just reasoned): temporarily forcing this conversion during EXECUTE
        # made `test_make_llm_call_recovers_from_none_content` fail. This is the
        # `hasattr()` call D-003 already reserved: "falls back to a documented
        # exception... only if that test cannot be made to pass." See decisions.md
        # D-022 (this call site) and D-003 (the original judgment call).
        message = getattr(choice, "message", None)
        if message is None or not hasattr(message, "content"):
            raise LLMResponseError("Response missing message content")

        content = message.content
        if not content:
            # H3: `content` is falsy — empty string OR None. Ollama reasoning-
            # only replies arrive as content=None (not ""), so this must fire
            # on None too, else C2 recovery never runs in production.
            content = self._extract_content_from_thinking(choice.message)
            if content is not None:
                choice.message.content = content

        if not content:
            raise LLMResponseError("LLM returned empty content")

        return response

    def _apply_model_specific_params(
        self,
        call_params: dict,
        call_type: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        """Apply model-specific parameters to the LLM call.

        Handles quirks of specific model providers (e.g. Ollama's thinking
        mode) by mutating *call_params* in place. A call is structured
        (temperature 0 on Ollama) when its call type parses JSON, or when it
        sends a ``response_format`` and its call type is not free text (Pass
        2 keeps the user's temperature with an ``output_schema``).
        """
        # Ollama: disable thinking mode and force deterministic output
        # for structured calls (data extraction, classification, a
        # response_format completion).
        is_structured = call_type in _STRUCTURED_CALL_TYPES or (
            response_format is not None and call_type not in _FREE_TEXT_CALL_TYPES
        )
        apply_ollama_params(call_params, self.model, structured=is_structured)

    @staticmethod
    def _extract_content_from_thinking(message) -> str | None:
        """Extract structured content from a model's reasoning/thinking trace.

        Some models (e.g. Qwen 3.5 via Ollama) place the actual answer in a
        reasoning field and leave ``content`` empty or ``None``. This helper
        tries to recover the last JSON object from that trace, falling back to
        the last non-empty line.

        Returns the extracted content string, or ``None`` if no reasoning
        trace is present.
        """
        # DECISION plan-2026-07-21T045419-9925aa3a/D-002
        # The reasoning-trace field-name resolution (reasoning_content ->
        # legacy thinking -> thinking_blocks) now lives in the SINGLE shared
        # `_resolve_reasoning_trace` (utilities.py) so this reader and
        # classification.py::_extract_response can never re-diverge. Do NOT
        # revert to a `hasattr(message, "thinking")`/`.thinking`-only read:
        # installed litellm RENAMES `thinking` to `reasoning_content` and
        # DELETES `thinking` before building the Message/Delta object, making a
        # `.thinking`-only read dead code for the project's own
        # DEFAULT_LLM_MODEL. Rationale + anti-simplification note live at the
        # resolver's D-002 anchor. See decisions.md D-001/D-002.
        trace = _resolve_reasoning_trace(message)
        if not trace:
            return None
        logger.debug("Content empty but reasoning trace present, extracting from it")
        return LiteLLMInterface._recover_content_from_trace(trace)

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
              resolved via ``_resolve_reasoning_trace``; a non-empty ``str``).
            - Returns the last balanced JSON object found in ``trace`` (preferred,
              as the final answer), else the last non-empty line, else ``None``.
            - Never raises; JSON parse failures are swallowed per candidate.
        """
        thinking = trace
        # Find JSON objects using proper parsing — supports multi-line JSON
        json_candidates: list[str] = []
        # First try line-by-line for single-line JSON
        for line in thinking.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    json.loads(line)
                    json_candidates.append(line)
                except json.JSONDecodeError:
                    pass
        # If no single-line JSON found, try extracting multi-line JSON blocks
        if not json_candidates:
            depth = 0
            start = -1
            for i, ch in enumerate(thinking):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidate = thinking[start : i + 1]
                        try:
                            json.loads(candidate)
                            json_candidates.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        start = -1
        if json_candidates:
            # Prefer the last JSON object (most likely the final answer)
            return json_candidates[-1]
        # Fallback: use the last substantial line
        lines = [line.strip() for line in thinking.strip().split("\n") if line.strip()]
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
                        and len(as_text) <= _RESPONSE_MESSAGE_MAX_LEN
                    ):
                        return ResponseGenerationResponse(
                            message=as_text, message_type="response"
                        )
                    raise ValueError("No usable message in response")
                return ResponseGenerationResponse(
                    message=message[:_RESPONSE_MESSAGE_MAX_LEN],
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
            data = extract_json_from_text(_remove_think_blocks(content))
            if isinstance(data, dict) and "message" in data:
                message = data["message"]
                if isinstance(message, str) and message.strip():
                    logger.debug("Extracted response JSON via fallback")
                    try:
                        return ResponseGenerationResponse(
                            message=message[:_RESPONSE_MESSAGE_MAX_LEN],
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
        # max_length=5000 (definitions.py:157).  So
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
        # but this raw-text rung is still schema-blind.
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
        # malformed envelope is now shown as text. See decisions.md D-030.
        content = _remove_think_blocks(content).strip() or content
        if self._looks_like_json(content):
            parsed: Any = extract_json_from_text(content)
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
        return ResponseGenerationResponse(
            message=content[:_RESPONSE_MESSAGE_MAX_LEN],
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
        # strip_think_and_fences (utilities.py) so the two readers cannot
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
                # the ladder below (D-001, utilities.py).
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
                # (definitions.py:347) — same trapdoor class as
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
                    # is now mapped to the 0.95 default first (D-001, utilities.py).
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
            elif request.field_type == "int":
                try:
                    coerced_value = int(raw)
                except ValueError:
                    pass
            elif request.field_type == "float":
                try:
                    coerced_value = float(raw)
                except ValueError:
                    pass
            elif request.field_type == "bool":
                if raw.lower() in ("true", "yes", "1"):
                    coerced_value = True
                elif raw.lower() in ("false", "no", "0"):
                    coerced_value = False
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
            validation_error="Extraction produced no usable value",
        )

    @staticmethod
    def _find_nested_key(data: dict, key: str, max_depth: int = 3) -> Any:
        """Search nested dicts for a key, returning first match."""
        if max_depth <= 0:
            return None
        for v in data.values():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
                found = LiteLLMInterface._find_nested_key(v, key, max_depth - 1)
                if found is not None:
                    return found
        return None

    def _looks_like_json(self, text: str) -> bool:
        """Check if text appears to be JSON format."""
        if not isinstance(text, str):
            return False

        text = text.strip()
        return (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        )


# --------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------


class LiteLLMEmbedder:
    """Text embeddings through LiteLLM: the one ``embedding`` call of fsm_llm.

    An embedder has its own model and connection kwargs (``api_key``,
    ``api_base``, ...), separate from any chat interface, and its own usage
    counters (kind ``embed``). Ollama models (``ollama/...``) need nothing
    extra: litellm reads ``OLLAMA_API_BASE`` (default
    ``http://localhost:11434``) itself.

    Example::

        embedder = LiteLLMEmbedder("ollama/qwen3-embedding:0.6b")
        vectors = embedder.embed(["first text", "second text"])
    """

    # Provider-call counters, created on first use (see _meter_of).
    _usage_meter: _UsageMeter | None = None

    # DECISION plan-2026-10-01T093600-944e2692/D-005: one concrete embedder
    # owns the one ``embedding`` binding. Do NOT add an ``embed`` method to
    # ``LLMInterface``/``LiteLLMInterface`` (the embedding model differs from
    # the chat model and would inherit the chat model's credentials), do NOT
    # add an Embedder ABC (one implementation; consumers plug custom backends
    # in through a callable ``embed_fn``), and do NOT call
    # ``litellm.embedding`` anywhere else. See D-005.
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float | None = 120.0,
        retries: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Embedding model identifier (e.g. ``"ollama/qwen3-embedding:0.6b"``,
                ``"text-embedding-3-small"``).
            api_key: Optional API key (the environment is used when omitted).
            timeout: Seconds per provider request (``None`` for no timeout).
            retries: The SDK's ``max_retries``; same semantics as
                ``LiteLLMInterface`` (0 leaves the SDK default; a no-op on Ollama).
            **kwargs: Additional litellm embedding parameters (``api_base``,
                ``dimensions``, ...). Keys in
                ``constants.RESERVED_EMBEDDING_CALL_KWARGS`` are ignored with a
                WARNING.

        Raises:
            ValueError: ``model`` is empty.
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.kwargs: dict[str, Any] = dict(kwargs)
        dropped = sorted(RESERVED_EMBEDDING_CALL_KWARGS.intersection(kwargs))
        if dropped:
            logger.warning(
                f"LiteLLMEmbedder ignores reserved call kwargs {dropped}; "
                "the framework sets them per call"
            )
        if api_key:
            self.kwargs["api_key"] = api_key

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in one provider request, one vector per text, in order.

        No texts means no request and ``[]``. Provider errors propagate
        unchanged (callers keep their own degrade paths); each request is
        counted once on this embedder's meter, a raising one as an error.

        Raises:
            TypeError: ``texts`` is a single string or holds a non-string.
            LLMResponseError: the reply does not hold one numeric vector per
                text.
        """
        if isinstance(texts, str) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a sequence of strings")
        batch = list(texts)
        if not batch:
            return []
        params = _connection_params(
            self.model,
            self.kwargs,
            timeout=self.timeout,
            retries=self.retries,
            reserved=RESERVED_EMBEDDING_CALL_KWARGS,
        )
        params["input"] = batch
        meter = _meter_of(self)
        try:
            response = embedding(**params)
        except Exception:
            # Broad catch is intentional: count the failed request, re-raise
            # it unchanged for the caller's own error boundary.
            meter.record_error(USAGE_KIND_EMBED)
            raise
        meter.record(USAGE_KIND_EMBED, response)
        return _embedding_vectors(response, len(batch))

    def usage(self) -> LLMUsage:
        """A frozen snapshot of this embedder's provider-call counters."""
        return _meter_of(self).snapshot()

    def reset_usage(self) -> LLMUsage:
        """Clear the counters and return the snapshot they held (atomically)."""
        return _meter_of(self).snapshot(reset=True)


def _embedding_vectors(response: Any, expected: int) -> list[list[float]]:
    """The vectors of an embedding reply, ordered by their ``index``.

    ``data`` items may be dicts or objects. Raises ``LLMResponseError`` when
    the reply does not hold exactly ``expected`` numeric vectors.
    """
    data = _reply_field(response, "data")
    if not isinstance(data, list) or len(data) != expected:
        got = len(data) if isinstance(data, list) else type(data).__name__
        raise LLMResponseError(
            f"Embedding reply holds {got} vectors for {expected} texts"
        )
    indexed: list[tuple[int, list[float]]] = []
    for position, item in enumerate(data):
        index = _reply_field(item, "index")
        vector = _reply_field(item, "embedding")
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector
        ):
            raise LLMResponseError(f"Embedding reply item {position} is not a vector")
        order = (
            index
            if isinstance(index, int) and not isinstance(index, bool)
            else position
        )
        indexed.append((order, [float(x) for x in vector]))
    if sorted(order for order, _ in indexed) != list(range(expected)):
        raise LLMResponseError("Embedding reply indexes do not match the texts")
    return [vector for _, vector in sorted(indexed, key=lambda pair: pair[0])]
