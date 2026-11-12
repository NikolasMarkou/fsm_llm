"""
Fake ``LLMInterface`` implementations for offline tests.

Two doubles, promoted unchanged (apart from one dead parameter) from the
repository's own test configuration. Neither reaches a network, a clock or a
random source, and neither is a ``unittest.mock`` object: they are plain
classes whose attributes a test may read, change between turns and subclass.

- ``MockLLM2Interface`` answers every field from a fixed dict and every reply
  with one fixed text, whatever the prompt says.
- ``PromptGroundedLLM`` knows a fact only when the prompt shows the evidence
  for it, like a real model reading its prompt.
"""

from __future__ import annotations

import json
import re

from fsm_llm.definitions import (
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

#: Where the response prompt names the state it answers for (``prompts.py``).
_CURRENT_STATE_TAG = re.compile(r"<current_state>([^<]+)</current_state>")


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
        match = _CURRENT_STATE_TAG.search(request.system_prompt)
        state = match.group(1) if match else None
        message = self.responses.get(state or "", self.default_response)
        return ResponseGenerationResponse(
            message=message, message_type="response", reasoning="prompt-grounded fake"
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise LLMResponseError("PromptGroundedLLM scripts no completion replies")
