"""
Fake ``LLMInterface`` implementations and a fake embedder for offline tests.

None of them reaches a network, a clock or a random source, and none is a
``unittest.mock`` object: they are plain classes (and one pure function) whose
attributes a test may read, change between turns and subclass.

- ``MockLLM2Interface`` answers every field from a fixed dict and every reply
  with one fixed text, whatever the prompt says.
- ``PromptGroundedLLM`` knows a fact only when the prompt shows the evidence
  for it, like a real model reading its prompt.
- ``ScriptedLLM`` answers ``complete`` from a queue, fields from a dict and the
  Pass-2 reply per state, keeps every request, meters its calls like the real
  interface, and records a call nothing scripted instead of hiding it.
- ``state_in_prompt`` reads the state a response prompt was written for.
- ``hash_embed`` is a deterministic embedder for semantic-memory tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator
from typing import NoReturn

from fsm_llm.constants import (
    USAGE_KIND_BY_CALL_TYPE,
    USAGE_KIND_COMPLETE,
    USAGE_KIND_STREAM,
)
from fsm_llm.core.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.llm.usage import _meter_of, _Metered

from .builders import final_reply

#: Where the response prompt names the state it answers for (``context/layout/response.py``).
_CURRENT_STATE_TAG = re.compile(r"<current_state>([^<]+)</current_state>")

#: One item of the ``ScriptedLLM`` reply queue: text, a response or an exception to raise.
ScriptedReply = str | CompletionResponse | BaseException

# Usage-meter kinds of the Pass-1 and Pass-2 call paths, from the one map core owns.
_KIND_GENERATE = USAGE_KIND_BY_CALL_TYPE["response_generation"]
_KIND_FIELD = USAGE_KIND_BY_CALL_TYPE["field_extraction"]
_KIND_BULK = USAGE_KIND_BY_CALL_TYPE["data_extraction"]


def state_in_prompt(prompt: str) -> str | None:
    """The state id a response prompt was written for, or ``None``.

    Interface contract (the one reader of the ``current_state`` tag; the
    response prompt builder in ``fsm_llm.context.layout.response`` writes it, a drift test
    pins the two together):
        - ``prompt``: the ``system_prompt`` of a ``ResponseGenerationRequest``
          (any text is accepted).
        - Returns the id inside the first ``current_state`` tag, or ``None``
          when the text holds none (an extraction prompt, a hand-written one).
        - Never raises.
    """
    match = _CURRENT_STATE_TAG.search(prompt)
    return match.group(1) if match else None


class MockLLM2Interface(LLMInterface):
    """A fixed-answer LLM double for the 2-pass pipeline.

    Not ``unittest.mock``: a plain ``LLMInterface`` subclass with public,
    mutable attributes. It answers Pass 1 field extraction from a dict and
    Pass 2 with one text, so it cannot tell a context-free prompt from a
    grounded one (see ``PromptGroundedLLM`` for that).

    Interface contract (subclassed by the repo's own tests, which override
    ``generate_response``/``extract_field`` and call ``super()``):
        - Parameters: ``extraction_data`` is ``{field_name: value}`` (``None``
          or empty means no field is known); ``response_text`` is the Pass-2
          reply; ``cut_reply`` flags the reply as cut at ``max_tokens``;
          ``cut_fields`` names fields that come back unusable and cut.
        - Attributes, all plain and changeable between turns:
          ``extraction_data`` (dict), ``response_text``, ``cut_reply``,
          ``cut_fields`` (a ``set``, built from the argument) and
          ``call_history``, a list a test may read or ``.clear()``.
        - ``extract_field``: value ``extraction_data.get(field_name)`` with
          confidence 1.0 and ``is_valid=True``; a missing name gives value
          ``None``, confidence 0.0 and ``is_valid=False``; a name in
          ``cut_fields`` gives ``None`` and ``truncated=True``. Never raises.
        - ``generate_response``: ``ResponseGenerationResponse`` carrying
          ``response_text`` with ``truncated=cut_reply``. Never raises.
        - ``call_history`` records only those two calls, in call order, as
          ``(method_name, request)``.
        - Not implemented: ``extract_bulk_data`` (the base raises
          ``NotImplementedError``, which the pipeline turns into a failed bulk
          pass), ``complete`` (a FSM that classifies is refused), ``usage()``
          and ``model``.
    """

    def __init__(
        self,
        extraction_data: dict[str, object] | None = None,
        response_text: str = "Hello! How can I help you?",
        cut_reply: bool = False,
        cut_fields: tuple[str, ...] | set[str] | list[str] = (),
    ) -> None:
        self.extraction_data = extraction_data or {}
        self.response_text = response_text
        self.call_history: list[tuple[str, object]] = []
        # A provider cut-off at max_tokens (plan 0068b055 D-025): when True the
        # Pass-2 reply carries `truncated=True`; a field named in `cut_fields`
        # comes back unusable (value None) with `truncated=True`. Both are plain
        # attributes, so a test can change them between turns.
        self.cut_reply = cut_reply
        self.cut_fields = set(cut_fields)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.call_history.append(("generate_response", request))
        return ResponseGenerationResponse(
            message=self.response_text,
            message_type="response",
            reasoning="Mock response",
            truncated=self.cut_reply,
        )

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.call_history.append(("extract_field", request))
        cut = request.field_name in self.cut_fields
        value = None if cut else self.extraction_data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
            truncated=cut,
        )


class PromptGroundedLLM(LLMInterface):
    """Fake LLM that knows a fact only when the prompt shows its evidence.

    ``MockLLM2Interface`` answers every field from a fixed dict whatever the
    prompt says, so it cannot tell a context-free prompt from a grounded one.
    This fake can: a value comes back only when the text it would ground on
    is in the request, like a real model reading its prompt.

    Interface contract (shared by the agents Phase-1 tests):
        - ``facts``: ``{field_name: (value, evidence)}``. ``evidence`` is a
          substring; the field resolves to ``value`` only when it occurs in
          the request text.
        - ``extract_field``: the text is ``system_prompt``, ``user_message``
          and the JSON-dumped ``context``. Returns ``value`` (``is_valid=True``)
          or ``None`` (``is_valid=False``, confidence 0) for a missing fact or
          absent evidence.
        - ``extract_bulk_data``: returns only facts whose name the prompt asks
          for (``"name"`` quoted in ``system_prompt``) and whose evidence is in
          ``system_prompt`` or ``user_message``. A context-free prompt
          therefore yields ``{}``.
        - ``generate_response``: ``responses[state]`` for the state named by
          the prompt's ``<current_state>`` tag, else ``default_response``.
        - ``complete``: scripts no reply; raises ``LLMResponseError``, so a
          classifier call fails soft exactly like a provider outage (core
          refuses an interface without ``complete`` for an FSM that
          classifies, D-029 of plan 944e2692). Not recorded.
        - ``requests``: every other call as ``(kind, request)`` in call
          order, kind being the method name. Only ``complete`` raises.
        - ``model`` is the plain attribute ``"prompt-grounded-fake"``;
          ``facts``, ``responses`` and ``default_response`` are plain
          attributes (copies of the arguments), changeable between turns.
    """

    def __init__(
        self,
        facts: dict[str, tuple[object, str]] | None = None,
        responses: dict[str, str] | None = None,
        default_response: str = "ok",
    ) -> None:
        self.model = "prompt-grounded-fake"
        self.facts = dict(facts or {})
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.requests: list[tuple[str, object]] = []

    def calls(self, kind: str) -> list:
        """The recorded requests of one kind, in call order."""
        return [request for k, request in self.requests if k == kind]

    def _grounded(self, name: str, text: str) -> object | None:
        value, evidence = self.facts.get(name, (None, ""))
        return value if evidence and evidence in text else None

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.requests.append(("extract_field", request))
        text = "\n".join(
            [
                request.system_prompt,
                request.user_message or "",
                json.dumps(request.context or {}, default=str),
            ]
        )
        value = self._grounded(request.field_name, text)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="prompt-grounded fake",
            is_valid=value is not None,
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.requests.append(("extract_bulk_data", request))
        text = f"{request.system_prompt}\n{request.user_message or ''}"
        data = {}
        for name in self.facts:
            if f'"{name}"' not in request.system_prompt:
                continue
            value = self._grounded(name, text)
            if value is not None:
                data[name] = value
        return DataExtractionResponse(extracted_data=data)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.requests.append(("generate_response", request))
        state = state_in_prompt(request.system_prompt)
        message = self.responses.get(state or "", self.default_response)
        return ResponseGenerationResponse(
            message=message, message_type="response", reasoning="prompt-grounded fake"
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise LLMResponseError("PromptGroundedLLM scripts no completion replies")


class UnscriptedCallError(AssertionError):
    """A ``ScriptedLLM`` call that nothing in the script covers.

    Interface contract:
        - Raised by ``ScriptedLLM`` itself, and recorded in its ``unexpected``
          list first. An ``AssertionError``, so a test that lets it escape
          fails with the message (which names the call and what is missing).
        - The pipeline swallows any exception raised inside field and bulk
          extraction, so there the error does not reach the test: call
          ``ScriptedLLM.assert_exhausted()`` at the end of the test.
    """


class ScriptedLLM(_Metered, LLMInterface):
    """A queue-driven ``LLMInterface`` for one test: script the replies, run, assert.

    Not ``unittest.mock``: a plain subclass with public, mutable attributes.
    ``complete`` answers from an ordered queue, ``extract_field`` and
    ``extract_bulk_data`` from a dict, ``generate_response`` (Pass 2) from a
    per-state dict. Whatever the script does not cover is an unexpected call:
    recorded, counted as an error and raised, never answered with an invented
    reply.

    Interface contract (the double for the completion, classification,
    agent, synthesis and meta-builder tests; subclass it to add routing):
        - ``replies``: the ``complete`` queue, consumed in order. An item is a
          text (a ``final_reply``), a ``CompletionResponse`` (build one with
          ``final_reply``, ``tool_call_reply``, ``malformed_reply``) or an
          exception INSTANCE that the call raises. ``__init__`` takes any
          iterable except a single string and copies it; ``replies`` is a
          plain list a test may extend later. A reply of another type raises
          ``TypeError`` (at construction, or when taken from the queue).
        - ``extraction_data``: ``{field_name: value}``. ``None`` (the default)
          means no extraction is expected, so any extraction call is
          unscripted. A dict, even ``{}``, answers ``extract_field`` (a name it
          lacks comes back not extracted, which is not an unexpected call) and
          ``extract_bulk_data`` (a copy of the whole dict).
        - ``responses``: ``{state_id: reply_text}`` for ``generate_response``,
          the state read from the prompt with ``state_in_prompt``.
          ``default_response`` is the reply for a state ``responses`` does not
          name (``None``: such a call is unscripted).
        - ``stream_chunks``: what ``generate_response_stream`` yields, the same
          chunks every time; empty (the default) streams the one scripted reply
          as a single chunk.
        - ``model``: a plain attribute, ``None`` by default. A classification
          entry of an FSM that names a different ``model`` builds a real
          interface, so leave it ``None`` unless a test needs the attribute.
        - Records: ``requests`` (every ``CompletionRequest`` that reached
          ``complete``), ``call_history`` (every call of any method, in order,
          as ``(method_name, request)``), ``calls(method_name)`` (the requests
          of one method) and ``unexpected`` (the unscripted ones, same shape).
        - ``assert_exhausted()``: raises ``AssertionError`` when ``unexpected``
          is not empty or ``replies`` still holds an unused item. Call it at
          the end of a test; it is the only thing that fails a test whose
          unscripted extraction call the pipeline swallowed.
        - ``usage()`` / ``reset_usage()``: the real interface's counters
          (``LLMUsage``, kinds ``generate``, ``stream``, ``extract``,
          ``classify``, ``complete``). Every call counts; a call that raises,
          a scripted exception or an unscripted call, also counts an error.
          Answered calls count as ``usage_missing`` (no token usage). A
          ``Simulator`` and the synthesis runner read these.
        - Failure modes: an unscripted call raises ``UnscriptedCallError``;
          a scripted exception propagates as itself. The double never sleeps,
          reads a clock or touches a network.
    """

    def __init__(
        self,
        replies: Iterable[ScriptedReply] = (),
        *,
        extraction_data: dict[str, object] | None = None,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
        stream_chunks: Iterable[str] = (),
        model: str | None = None,
    ) -> None:
        if isinstance(replies, str):
            raise TypeError(
                "replies must be an iterable of replies, not one string: "
                "pass [text] for a single reply"
            )
        self.replies: list[ScriptedReply] = [self._checked(item) for item in replies]
        self.extraction_data: dict[str, object] | None = (
            None if extraction_data is None else dict(extraction_data)
        )
        self.responses: dict[str, str] = dict(responses or {})
        self.default_response = default_response
        self.stream_chunks: tuple[str, ...] = tuple(stream_chunks)
        self.model = model
        self.requests: list[CompletionRequest] = []
        self.call_history: list[tuple[str, object]] = []
        self.unexpected: list[tuple[str, object]] = []

    @staticmethod
    def _checked(item: object) -> ScriptedReply:
        if isinstance(item, (str, CompletionResponse, BaseException)):
            return item
        hint = (
            " (pass an exception instance, not the class)"
            if isinstance(item, type) and issubclass(item, BaseException)
            else ""
        )
        raise TypeError(
            "a scripted reply is a str, a CompletionResponse or an exception "
            f"instance, got {type(item).__name__}{hint}"
        )

    def calls(self, method_name: str) -> list:
        """The recorded requests of one method (``"complete"``, ``"extract_field"``...)."""
        return [r for name, r in self.call_history if name == method_name]

    def assert_exhausted(self) -> None:
        """Fail unless every scripted reply was used and no call was unscripted."""
        __tracebackhide__ = True
        problems: list[str] = []
        if self.unexpected:
            names = ", ".join(sorted({name for name, _ in self.unexpected}))
            problems.append(
                f"{len(self.unexpected)} unscripted call(s) ({names}); see `unexpected`"
            )
        if self.replies:
            shown = ", ".join(repr(item)[:80] for item in self.replies[:3])
            more = ", ..." if len(self.replies) > 3 else ""
            problems.append(
                f"{len(self.replies)} scripted reply(ies) never used (next: {shown}{more})"
            )
        if problems:
            raise AssertionError(f"{type(self).__name__}: " + "; ".join(problems))

    def _unscripted(
        self, kind: str, method: str, request: object, why: str
    ) -> NoReturn:
        self.unexpected.append((method, request))
        _meter_of(self).record_error(kind)
        raise UnscriptedCallError(
            f"{type(self).__name__}.{method} was called with nothing scripted for "
            f"it: {why}. The call is recorded in `unexpected`; script it or "
            "fix the code under test."
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kind = USAGE_KIND_BY_CALL_TYPE.get(request.call_type, USAGE_KIND_COMPLETE)
        self.requests.append(request)
        self.call_history.append(("complete", request))
        if not self.replies:
            self._unscripted(kind, "complete", request, "the reply queue is empty")
        item = self._checked(self.replies.pop(0))
        if isinstance(item, BaseException):
            _meter_of(self).record_error(kind)
            raise item
        response = final_reply(item) if isinstance(item, str) else item
        _meter_of(self).record(kind, response)
        return response

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.call_history.append(("extract_field", request))
        if self.extraction_data is None:
            self._unscripted(
                _KIND_FIELD, "extract_field", request, "extraction_data is None"
            )
        value = self.extraction_data.get(request.field_name)
        response = FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="scripted field extraction",
            is_valid=value is not None,
        )
        _meter_of(self).record(_KIND_FIELD, response)
        return response

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.call_history.append(("extract_bulk_data", request))
        if self.extraction_data is None:
            self._unscripted(
                _KIND_BULK, "extract_bulk_data", request, "extraction_data is None"
            )
        response = DataExtractionResponse(extracted_data=dict(self.extraction_data))
        _meter_of(self).record(_KIND_BULK, response)
        return response

    def _scripted_text(self, request: ResponseGenerationRequest) -> str | None:
        state = state_in_prompt(request.system_prompt)
        if state is not None and state in self.responses:
            return self.responses[state]
        return self.default_response

    def _no_reply_why(self, request: ResponseGenerationRequest) -> str:
        state = state_in_prompt(request.system_prompt)
        return (
            f"`responses` has no entry for state {state!r} and default_response is None"
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.call_history.append(("generate_response", request))
        text = self._scripted_text(request)
        if text is None:
            self._unscripted(
                _KIND_GENERATE,
                "generate_response",
                request,
                self._no_reply_why(request),
            )
        response = ResponseGenerationResponse(
            message=text, message_type="response", reasoning="scripted reply"
        )
        _meter_of(self).record(_KIND_GENERATE, response)
        return response

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        self.call_history.append(("generate_response_stream", request))
        chunks = self.stream_chunks
        if not chunks:
            text = self._scripted_text(request)
            if text is None:
                self._unscripted(
                    USAGE_KIND_STREAM,
                    "generate_response_stream",
                    request,
                    self._no_reply_why(request),
                )
            chunks = (text,)
        _meter_of(self).record(USAGE_KIND_STREAM, None)
        yield from chunks


def hash_embed(dim: int = 8) -> Callable[[str], list[float]]:
    """A deterministic embedder: the same text always gives the same vector.

    Interface contract (an ``EmbedFn``, ``Callable[[str], list[float]]``, the
    seam of the agents' semantic memory; no class, because the package has no
    embedder base class by decision):
        - ``dim``: the vector length, a positive ``int`` (default 8). A larger
          one separates more words; 8 is enough to tell a few phrases apart.
        - Returns a pure function ``embed(text) -> list[float]``: a
          feature-hashed bag of lowercase words (``\\w+``), counted into ``dim``
          buckets by SHA-256 and scaled to unit length. Texts that share words
          are closer (cosine) than texts that do not; case and punctuation do
          not matter. No clock, no randomness, no state between calls.
        - Text with no word (empty, only punctuation) is hashed whole, so no
          vector is ever all zeros.
        - Raises ``ValueError`` for a ``dim`` that is not a positive ``int``.
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}")

    def embed(text: str) -> list[float]:
        lowered = text.lower()
        words = re.findall(r"\w+", lowered) or [lowered]
        counts = [0.0] * dim
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8", "surrogatepass")).digest()
            counts[int.from_bytes(digest[:8], "big") % dim] += 1.0
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm for c in counts]

    return embed
