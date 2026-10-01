"""One engine: every agent pattern's model calls go through its LLMInterface.

Plan 944e2692 step 16 (success criterion W2): every ``create_agent`` pattern
except ``swarm`` (it only runs member agents) is built with an injected
scripted ``LLMInterface`` while the provider bindings raise. The interface
must receive every model call of the run, and no binding may be reached: no
pattern keeps a private provider path (native_fc's own litellm loop was the
last one; on the parent commit ced9eb5 its case fails).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import (
    AgentConfig,
    ChainStep,
    EvaluationResult,
    ToolRegistry,
    create_agent,
)
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.meta_builder import MetaBuilderResult
from fsm_llm.agents.semantic_memory import SemanticMemoryStore
from fsm_llm.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface

_TYPED_VALUES: dict[str, Any] = {
    "bool": True,
    "int": 1,
    "float": 1.0,
    "list": [],
    "dict": {},
}


class _EveryCallLLM(LLMInterface):
    """Answers every kind of model call with a plausible value; counts them.

    Field extractions get a value of the requested type (``"done"`` for
    text), so loops conclude; ``complete`` answers a final text (a JSON
    object for a structured turn).
    """

    model = "scripted/model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.calls.append("extract_field")
        value = _TYPED_VALUES.get(str(request.field_type), "done")
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0,
            reasoning="scripted",
            is_valid=True,
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.calls.append("extract_bulk_data")
        return DataExtractionResponse(extracted_data={}, confidence=1.0)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.calls.append("generate_response")
        return ResponseGenerationResponse(
            message="Answer: done", message_type="response", reasoning="scripted"
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append("complete")
        text = "{}" if request.response_format is not None else "done"
        return CompletionResponse(kind="final", text=text)


def _lookup(query: str) -> str:
    """Look something up."""
    return f"found {query}"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_lookup, name="lookup", description="Look up")
    return registry


def _pass(output: str, context: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(passed=True, score=1.0, feedback="ok")


def _no_embedding(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("no embedding model in this test")


def _pattern_kwargs(pattern: str) -> dict[str, Any]:
    """The extra constructor arguments a pattern needs to run offline."""
    tools = {"tools": _registry()}
    return {
        "react": tools,
        "rewoo": tools,
        "plan_execute": tools,
        "parallel_react": tools,
        "native_fc": tools,
        "verified_react": tools,
        "reflexion": tools,
        "reasoning_react": tools,
        "adapt": tools,
        "auto_memory": {
            **tools,
            "memory": SemanticMemoryStore(embed_fn=_no_embedding),
        },
        "debate": {"num_rounds": 1},
        "prompt_chain": {
            "chain": [
                ChainStep(
                    step_id="draft",
                    name="Draft",
                    extraction_instructions="Draft the answer.",
                    response_instructions="Give the answer.",
                )
            ]
        },
        "self_consistency": {"num_samples": 2},
        "orchestrator": {},
        "evaluator_optimizer": {"evaluation_fn": _pass},
        "maker_checker": {
            "maker_instructions": "Write one line.",
            "checker_instructions": "Check the line.",
        },
        "meta_builder": {},
    }[pattern]


_PATTERNS = [
    "react",
    "rewoo",
    "debate",
    "plan_execute",
    "prompt_chain",
    "self_consistency",
    "orchestrator",
    "adapt",
    "evaluator_optimizer",
    "maker_checker",
    "reflexion",
    "meta_builder",
    "parallel_react",
    "native_fc",
    "verified_react",
    "auto_memory",
    "reasoning_react",
]


class TestEveryPatternUsesItsInterface:
    def test_the_pattern_list_is_every_pattern_but_swarm(self):
        from fsm_llm.agents import _PATTERNS as registered

        assert sorted(_PATTERNS) == sorted(set(registered) - {"swarm"})

    @pytest.mark.parametrize("pattern", _PATTERNS)
    def test_every_model_call_reaches_the_injected_interface(
        self, pattern: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached: list[str] = []

        def binding(name: str) -> Any:
            def refuse(*args: Any, **kwargs: Any) -> Any:
                reached.append(name)
                raise AssertionError(f"provider binding {name} was called")

            return refuse

        for name in (
            "fsm_llm.llm.completion",
            "fsm_llm.llm.embedding",
            "litellm.completion",
            "litellm.embedding",
        ):
            monkeypatch.setattr(name, binding(name))

        llm = _EveryCallLLM()
        kwargs = _pattern_kwargs(pattern)
        tools = kwargs.pop("tools", None)
        config = AgentConfig(model="scripted/model", max_iterations=3)
        if pattern == "meta_builder":
            from fsm_llm.agents.definitions import MetaBuilderConfig

            config = MetaBuilderConfig(model="scripted/model")
        agent = create_agent(pattern, tools, config=config, llm_interface=llm, **kwargs)

        result = agent.run("Find the capital of France.")

        assert isinstance(result, AgentResult | MetaBuilderResult)
        assert reached == []
        assert llm.calls, f"{pattern} made no model call through its interface"
