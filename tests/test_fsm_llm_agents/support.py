"""Helpers shared by several test files of this suite. Not collected (no test_ prefix)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.core.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface


def _bug_fixes_search(params: dict) -> str:
    return f"Results for: {params.get('query', '')}"


def _bug_fixes_make_registry() -> ToolRegistry:
    """Create a registry with a dummy search tool."""
    registry = ToolRegistry()
    registry.register_function(
        _bug_fixes_search, name="search", description="Search the web"
    )
    return registry


class SequenceMockLLM(LLMInterface):
    """Mock LLM that replays pre-programmed extraction sequences.

    Each call to ``extract_field`` looks up the requested field in the current
    extraction sequence entry.  ``generate_response`` always returns a
    canned string.
    """

    def __init__(
        self,
        extraction_sequence: list[dict[str, Any]],
        response_text: str = "Agent response.",
    ) -> None:
        self._extractions = list(extraction_sequence)
        self._call_index = 0
        self._response_text = response_text
        self._asked_this_cycle: set[str] = set()
        self.model = "mock-model"

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """Per-field extraction using the current extraction sequence entry.

        The pipeline calls extract_field once per typed field of a state.
        We look up the requested field_name in the current extraction dict.
        The index advances once per extracting step: when a field already
        asked in the current cycle is asked again (a silent step makes no
        ``generate_response`` call to key on).
        """
        if request.field_name in self._asked_this_cycle:
            self._call_index += 1
            self._asked_this_cycle = set()
        self._asked_this_cycle.add(request.field_name)

        if self._call_index < len(self._extractions):
            data = self._extractions[self._call_index]
        else:
            data = {}

        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock field extraction",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message=self._response_text,
            message_type="response",
            reasoning="mock response",
        )


class _Out(BaseModel):
    title: str = "unset"
    answer: str = ""
