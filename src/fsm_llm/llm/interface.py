"""The provider-neutral LLM interface: ``LLMInterface``, ``tool_exchange``,
``check_tool_transcript``, ``resolve_model``, the ``complete`` response reader and the connection defaults.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import aclosing
from typing import Any

from fsm_llm.aio import aiterate
from fsm_llm.constants import ENV_LLM_MODEL, MALFORMED_TOOL_CALL_MARKERS
from fsm_llm.core.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ModelNotConfiguredError,
    ModelToolCall,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.logging import logger

from .parsing import _finish_reason
from .text import resolve_reasoning_trace

# Defaults of the provider connection, defined once here (the one LLM layer) and
# read by the API, the classifier, the CLI runner and the agents.
DEFAULT_MAX_TOKENS = 1000


DEFAULT_LLM_TIMEOUT_SECONDS = 120.0


# The embedding model ``LiteLLMEmbedder`` consumers default to (a local Ollama
# model; no key needed).
DEFAULT_EMBEDDING_MODEL = "ollama/embeddinggemma-2:740m"


def is_malformed_tool_call_error(exc: BaseException, *, tools_sent: bool) -> bool:
    """Whether a provider error is a garbled TOOL CALL rather than an outage.

    Interface contract (caller: ``LiteLLMInterface.complete``):
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

    Interface contract (caller: the ``complete`` reply normaliser):
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


def implements_complete(llm: Any) -> bool:
    """Whether ``llm`` is an ``LLMInterface`` whose class overrides ``complete``.

    Interface contract (callers: ``Classifier(llm_interface=...)`` and ``API``, which
    refuse an interface the classifier cannot send through):
        - ``llm``: any object.
        - Returns ``False`` for a non-``LLMInterface`` and for an interface
          whose ``complete`` is the base one (which only raises
          ``NotImplementedError``); ``True`` otherwise: a subclass override,
          a callable assigned on the instance, or a
          ``Mock(spec=LLMInterface)`` attribute.
        - Never raises.
    """
    if not isinstance(llm, LLMInterface):
        return False
    method = getattr(llm, "complete", None)
    return getattr(method, "__func__", None) is not LLMInterface.complete


PLACEHOLDER_MODEL = "unset/placeholder"
"""A model id that names nothing and is never sent to a provider.

A config that must hold a model string although an injected interface owns the
model (``agents.AgentConfig`` for an agent given only ``llm_interface=``) holds
this one. :func:`resolve_model` counts it as unset, so a config that outlives its
interface fails with the no-model error where a model is needed, never with the
provider's "LLM Provider NOT provided"."""


def resolve_model(model: str | None = None) -> str:
    """The model name to use: *model*, else env ``LLM_MODEL``; never a default.

    Interface contract (callers: ``API`` and ``Classifier`` (each only when no
    ``llm_interface`` is injected), ``ReasoningEngine``, ``agents.resolve_agent_model``,
    ``eval.resolve_model``, the ``fsm-llm`` runner and the monitor launch routes):
        - *model*: an explicit name or ``None``.
        - Returns the first usable candidate, stripped: *model*, then
          ``$LLM_MODEL`` (read now, never at import). A blank value (``""`` or
          only whitespace, e.g. an empty ``LLM_MODEL=`` line in a ``.env``) and
          ``PLACEHOLDER_MODEL`` count as unset at either tier. Always a
          non-empty ``str``.
        - Raises ``ModelNotConfiguredError`` (an ``FSMError`` and a
          ``ValueError``) when both are unset: the library has no built-in
          model, so a fresh install fails with a message that says how to set
          one. A caller that holds an injected interface does not call this: the
          ``Classifier`` and ``API`` take the model from that interface and read
          no environment.
    """
    # DECISION plan-2026-11-12T073150-d454fd62/D-005
    # Do NOT fall back to a built-in model here (the old built-in default): a
    # silent default sent a fresh install to a 27B local model it never chose.
    # DECISION plan-2026-11-12T073150-d454fd62/D-028
    # The placeholder is not a model: a config built for an injected interface
    # keeps working until it is used without one, then says "no model" (it used
    # to reach litellm with ``unset/placeholder``). Do NOT send it to a provider
    # and do NOT keep a second copy of the string in ``agents``.
    for candidate in (model, os.environ.get(ENV_LLM_MODEL)):
        if candidate is not None:
            name = candidate.strip()
            if name and name != PLACEHOLDER_MODEL:
                return name
    raise ModelNotConfiguredError()


def interface_model(llm: Any) -> str | None:
    """The model name an injected interface states, if it states one.

    Interface contract (callers: ``Classifier(llm_interface=...)`` and ``API``, which
    refuse a ``model`` beside an injected interface unless it equals this):
        - ``llm``: any object.
        - Returns ``llm.model`` when it is a ``str``; ``None`` otherwise
          (absent, ``None``, a mock attribute, any other type).
        - Never raises.
    """
    model = getattr(llm, "model", None)
    return model if isinstance(model, str) else None


def _checked_call_id(call: dict[str, Any], position: int) -> str:
    """The ``id`` of one transcript tool call, after checking its shape.

    Raises ``LLMResponseError`` (naming transcript entry ``position``) when
    the id is not a non-empty ``str`` or the call has no ``function`` with a
    non-empty ``str`` name and ``str`` arguments.
    """
    call_id = call.get("id")
    function = call.get("function")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(function, dict)
        or not isinstance(function.get("name"), str)
        or not function["name"]
        or not isinstance(function.get("arguments"), str)
    ):
        raise LLMResponseError(
            f"Transcript entry {position} has a tool call without a non-empty "
            "str id and a function with a str name and str arguments"
        )
    return call_id


def check_tool_transcript(messages: Sequence[Any]) -> None:
    """Refuse a transcript a provider must not receive.

    Interface contract (caller: the completion-state turn of the pipeline,
    before it sends a consumer-owned transcript):
        - Every entry is a dict whose ``role`` is ``system``, ``user``,
          ``assistant`` or ``tool``; a ``user`` entry has non-``None``
          content (the builder would otherwise send an instruction-shaped
          neutral turn as the user's words).
        - An assistant message with a non-empty ``tool_calls`` list is
          followed directly by exactly one ``tool`` message per call, matched
          by ``tool_call_id`` (any order); a ``tool`` message anywhere else is
          an orphan. Every call has a non-empty ``str`` ``id``, unique within
          its turn, and a ``function`` with a non-empty ``str`` ``name`` and
          ``str`` ``arguments``; every result's ``tool_call_id`` equals one of
          them exactly (no ``str()`` coercion) and its ``content`` is a
          ``str``. ``tool_exchange`` builds this shape.
        - Returns ``None`` for a valid transcript (an empty one included);
          raises ``LLMResponseError`` naming the first bad entry otherwise.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-021: refuse, never repair.
    # Do NOT drop orphans, coerce ids with str(), pair by position or accept
    # an id-less call: a provider that omits ids pairs two calls of one turn
    # by position only, which nothing could detect (D-029). See D-021, D-029.
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
        if role == "user" and message.get("content") is None:
            raise LLMResponseError(
                f"Transcript entry {index} is a user message with no content"
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
        expected = [_checked_call_id(call, index - 1) for call in calls]
        if len(set(expected)) != len(expected):
            raise LLMResponseError(
                f"Transcript entry {index - 1} repeats a tool call id"
            )
        answered: list[str] = []
        while index < len(messages):
            following = messages[index]
            if not isinstance(following, dict) or following.get("role") != "tool":
                break
            call_id = following.get("tool_call_id")
            if not isinstance(call_id, str) or not isinstance(
                following.get("content"), str
            ):
                raise LLMResponseError(
                    f"Transcript entry {index} is a tool result without a str "
                    "tool_call_id and str content"
                )
            answered.append(call_id)
            index += 1
        if sorted(answered) != sorted(expected):
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
          trace (``resolve_reasoning_trace``); with tool calls it is not
          (``content: None`` is their normal shape).
        - ``tool_calls`` is absent, ``None``, empty, or an iterable of calls
          (a list in every litellm reply), read once into a list; a ``str``,
          ``bytes``, mapping or non-iterable is ``LLMResponseError``
          ("Malformed LLM response shape: ...").
        - Tool calls: every call needs a non-empty ``str`` id, unique within
          the turn, a function name and arguments that decode to a JSON object
          (``decode_tool_arguments``); if ANY call fails that, the turn is
          ``malformed`` with no calls, the valid ones included, and its text
          kept.
        - Total: returns a ``CompletionResponse`` or raises
          ``LLMResponseError``, never another exception, whatever the reply's
          shape (``complete`` also wraps anything unforeseen raised here).
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
    # DECISION plan-2026-10-01T093600-944e2692/D-039: `complete` is total. A
    # reply shape that cannot be read raises LLMResponseError here, never a
    # raw TypeError from iterating it: the classifier turns only
    # LLMResponseError into its soft ClassificationError, so any other
    # exception now fails the turn. Do NOT iterate `tool_calls` unchecked:
    # a str, bytes or mapping, or anything that is not iterable, is a shape
    # error; any other iterable is materialised once into a list.
    try:
        if isinstance(raw_calls, (str, bytes, Mapping)):
            raise TypeError("a str, bytes or mapping is not a list of calls")
        raw_calls = list(raw_calls)
    except TypeError as e:
        raise LLMResponseError(
            "Malformed LLM response shape: tool_calls is "
            f"{type(raw_calls).__name__}, not a list ({e})"
        ) from e

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

    # DECISION plan-2026-10-01T093600-944e2692/D-027: a tool turn with ANY
    # call that has no name or non-object arguments is `malformed` with no
    # calls, the valid ones included, so a consumer runs none of them
    # (06a5ec0a/D-025, bf7ffe24/D-016). Do NOT coerce bad arguments to `{}`
    # (a tool ran with parameters the model never sent) or drop only the bad
    # call. The reasoning trace is recovered only without tool calls
    # (bf7ffe24/D-003: `content: None` is their normal shape). See D-027.
    # An id-less call makes the turn malformed too (D-039), and so does an id
    # repeated within the turn: its result could not be paired in the next
    # turn (check_tool_transcript refuses both), so it is refused here,
    # before any tool runs, not after.
    finish_reason = _finish_reason(response)
    calls: list[ModelToolCall] = []
    seen_ids: set[str] = set()
    for raw_call in raw_calls:
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", None)
        arguments = decode_tool_arguments(getattr(function, "arguments", None))
        call_id = getattr(raw_call, "id", None)
        if (
            not isinstance(name, str)
            or not name
            or arguments is None
            or not isinstance(call_id, str)
            or not call_id
            or call_id in seen_ids
        ):
            logger.warning(
                "Model tool-call turn is malformed (a call with no id or a "
                "repeated id, no tool name or arguments that are not a JSON "
                "object); none of its calls is returned"
            )
            return CompletionResponse(
                kind="malformed", text=text, finish_reason=finish_reason
            )
        seen_ids.add(call_id)
        calls.append(ModelToolCall(id=call_id, name=name, arguments=arguments))
    if calls:
        return CompletionResponse(
            kind="calls",
            text=text,
            calls=tuple(calls),
            finish_reason=finish_reason,
        )
    if text is None:
        text = resolve_reasoning_trace(message) or None
    return CompletionResponse(kind="final", text=text, finish_reason=finish_reason)


class LLMInterface(abc.ABC):
    """
    Abstract interface for LLM communication supporting the 2-pass architecture.

    This interface defines methods for data extraction, response generation and the
    ``complete`` request primitive (tool calling, plain and structured completion),
    allowing implementations to optimize for different use cases. Transitions are
    decided by rules, never by a method here.

    Async twins (``acomplete``, ``agenerate_response``, ``agenerate_response_stream``,
    ``aextract_field``, ``aextract_bulk_data``) are not abstract: the default of each runs
    the sync method on a worker thread (``asyncio.to_thread``; the stream steps its
    generator one ``next`` per worker call and closes it on early exit). A cancelled await
    returns at once but does NOT stop that thread, which runs the call to its end, so a
    late result can land after the caller gave up. An interface that implements only the
    ``a*`` methods cannot serve sync callers: the sync methods stay the abstract or
    ``NotImplementedError`` ones. ``LiteLLMInterface`` and the ``fsm_llm.testing`` doubles
    override the twins natively. A ``Mock(spec=LLMInterface)`` needs an ``AsyncMock`` for
    each ``a*`` method it should answer.
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
                A non-``None`` ``request.temperature`` is this call's
                sampling temperature and must be honoured (it replaces the
                interface's own for this request only).

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

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """The async twin of ``complete``: the default runs it on a worker thread."""
        return await asyncio.to_thread(self.complete, request)

    async def agenerate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        """The async twin of ``generate_response``: the default runs it on a worker thread."""
        return await asyncio.to_thread(self.generate_response, request)

    async def agenerate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> AsyncGenerator[str, None]:
        """The async twin of ``generate_response_stream``.

        The default steps the sync generator one ``next`` per worker-thread call and
        closes it when the consumer stops early (``aclose``) or the stream ends.
        """
        chunks = await asyncio.to_thread(self.generate_response_stream, request)
        async with aclosing(aiterate(chunks)) as steps:
            async for chunk in steps:
                yield chunk

    async def aextract_field(
        self, request: FieldExtractionRequest
    ) -> FieldExtractionResponse:
        """The async twin of ``extract_field``: the default runs it on a worker thread."""
        return await asyncio.to_thread(self.extract_field, request)

    async def aextract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        """The async twin of ``extract_bulk_data``: the default runs it on a worker thread."""
        return await asyncio.to_thread(self.extract_bulk_data, request)
