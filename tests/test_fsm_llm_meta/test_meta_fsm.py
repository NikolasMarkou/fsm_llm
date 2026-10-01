"""The meta-builder FSM definition (plan 944e2692, step 10, D-011).

``build_meta_builder_fsm()`` is the definition core runs for the
meta-builder: ``classify`` (initial, silent, artifact-type classification)
-> ``collect`` (speaking) -> ``build`` (silent structured completion state)
-> ``done`` (terminal) or ``build_failed`` (silent). These tests pin the
structure, the routing on the driver-written keys, the build state's
completion config and the build request pieces (response format with the
c1d5bfbc/D-010 enum, the agent-pattern enum, the build prompt), and drive the
definition through core with a scripted ``LLMInterface``. The driver itself
(``MetaBuilderAgent``) is wired in step 11.

On the parent commit (b44f79c) none of ``build_meta_builder_fsm``,
``MetaBuilderStates``, ``MetaContextKeys``, ``build_response_format`` or
``build_collect_response_instructions`` exists, so every test fails there.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents.constants import (
    META_AGENT_PATTERN_INTENTS,
    META_ARTIFACT_TYPE_INTENTS,
    META_HANDLER_ONLY_KEYS,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
    MetaDefaults,
)
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.fsm_definitions import build_meta_builder_fsm
from fsm_llm.agents.meta_builders import AgentBuilder, WorkflowBuilder
from fsm_llm.agents.meta_prompts import (
    artifact_schema,
    build_artifact_prompt,
    build_collect_response_instructions,
    build_response_format,
)
from fsm_llm.constants import has_internal_prefix
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    FSMContext,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.llm import LLMInterface
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm.validator import FSMValidator
from fsm_llm.validator import main as validate_main
from fsm_llm.visualizer import build_fsm_graph, to_mermaid

S = MetaBuilderStates
K = MetaContextKeys

_SILENT = (S.CLASSIFY, S.BUILD, S.BUILD_FAILED, S.DONE)

# (source, target, priority) of every edge, the documented routing.
_EDGES = {
    (S.CLASSIFY, S.BUILD, 10),
    (S.CLASSIFY, S.COLLECT, 900),
    (S.COLLECT, S.BUILD, 10),
    (S.COLLECT, S.CLASSIFY, 20),
    (S.BUILD, S.DONE, 10),
    (S.BUILD, S.BUILD_FAILED, 900),
    (S.BUILD_FAILED, S.BUILD, 10),
    (S.BUILD_FAILED, S.CLASSIFY, 20),
    (S.BUILD_FAILED, S.COLLECT, 900),
}


@pytest.fixture
def fsm_dict() -> dict[str, Any]:
    return build_meta_builder_fsm()


@pytest.fixture
def fsm(fsm_dict: dict[str, Any]) -> FSMDefinition:
    return FSMDefinition(**fsm_dict)


def _targets(fsm: FSMDefinition, state: str) -> set[str]:
    return {t.target_state for t in fsm.states[state].transitions}


class TestStructure:
    def test_loads_as_fsm_definition_and_through_api(self, fsm_dict):
        definition, fsm_id = API.process_fsm_definition(fsm_dict)
        assert definition.initial_state == S.CLASSIFY
        assert set(definition.states) == {
            S.CLASSIFY,
            S.COLLECT,
            S.BUILD,
            S.BUILD_FAILED,
            S.DONE,
        }
        assert fsm_id.startswith("fsm_meta_builder_")

    def test_definition_is_fresh_per_call(self):
        first = build_meta_builder_fsm()
        first["states"][S.COLLECT]["transitions"].clear()
        assert build_meta_builder_fsm()["states"][S.COLLECT]["transitions"]

    def test_validator_reports_no_error_and_no_warning(self, fsm_dict):
        result = FSMValidator(fsm_dict).validate()
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_fsm_llm_validate_cli_exits_ok(self, fsm_dict, tmp_path, capsys):
        path = tmp_path / "meta_builder.json"
        path.write_text(json.dumps(fsm_dict))
        assert validate_main(str(path)) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is True
        assert report["errors"] == []
        assert report["warnings"] == []

    def test_graph_export_has_exactly_the_documented_edges(self, fsm_dict):
        graph = build_fsm_graph(fsm_dict)
        assert graph.initial_state == S.CLASSIFY
        assert {(e.source, e.target, e.priority) for e in graph.edges} == _EDGES
        mermaid = to_mermaid(graph)
        assert mermaid.startswith("stateDiagram-v2")
        for source, target, priority in _EDGES:
            assert f"{source} --> {target} : P{priority} " in mermaid
        assert f"{S.DONE} --> [*]" in mermaid

    def test_every_state_reachable_and_done_reachable_from_every_state(self, fsm):
        seen = {fsm.initial_state}
        queue = deque([fsm.initial_state])
        while queue:
            for target in _targets(fsm, queue.popleft()):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        assert seen == set(fsm.states)
        for state in fsm.states:
            reach = {state}
            queue = deque([state])
            while queue:
                for target in _targets(fsm, queue.popleft()):
                    if target not in reach:
                        reach.add(target)
                        queue.append(target)
            assert S.DONE in reach, state

    def test_only_done_is_terminal(self, fsm):
        terminal = {sid for sid, st in fsm.states.items() if not st.transitions}
        assert terminal == {S.DONE}

    def test_priorities_distinct_per_state(self, fsm):
        for state_id, state in fsm.states.items():
            priorities = [t.priority for t in state.transitions]
            assert len(priorities) == len(set(priorities)), state_id

    def test_silent_states_and_the_one_speaking_state(self, fsm):
        for state_id in _SILENT:
            assert fsm.states[state_id].response_instructions == "", state_id
        collect = fsm.states[S.COLLECT].response_instructions
        assert collect == build_collect_response_instructions()
        assert "say 'build it' when you're ready" in collect
        for key in (K.ARTIFACT_TYPE, K.REQUIREMENTS, K.VALIDATION_ERRORS):
            assert key in collect

    def test_no_state_extracts_free_text(self, fsm):
        for state_id, state in fsm.states.items():
            assert not state.extraction_instructions, state_id
            assert not state.field_extractions, state_id
            assert not state.required_context_keys, state_id
        with_classification = {
            sid for sid, st in fsm.states.items() if st.classification_extractions
        }
        assert with_classification == {S.CLASSIFY}

    def test_collect_prompt_sees_only_its_three_keys(self, fsm):
        scope = fsm.states[S.COLLECT].context_scope
        assert scope is not None
        assert scope.read_keys == [K.ARTIFACT_TYPE, K.REQUIREMENTS, K.VALIDATION_ERRORS]


class TestClassification:
    def test_artifact_type_classification(self, fsm):
        (config,) = fsm.states[S.CLASSIFY].classification_extractions
        assert config.field_name == K.ARTIFACT_TYPE
        assert [i.name for i in config.intents] == [t.value for t in ArtifactType]
        assert [(i.name, i.description) for i in config.intents] == list(
            META_ARTIFACT_TYPE_INTENTS
        )
        assert config.fallback_intent == ArtifactType.FSM.value
        assert config.confidence_threshold == MetaDefaults.TYPE_CONFIDENCE_THRESHOLD
        assert config.confidence_threshold == 0.4
        assert config.context_keys == [K.LATEST_REQUEST]


class TestHandlerOnlyKeys:
    def test_handler_only_keys_are_the_driver_and_handler_keys(self, fsm):
        assert fsm.handler_only_keys == list(META_HANDLER_ONLY_KEYS)
        for key in (
            K.REQUIREMENTS,
            K.LATEST_REQUEST,
            K.BUILD_REQUESTED,
            K.TYPE_SWITCH,
            K.KEYWORD_TYPE,
            K.BUILD_OUTCOME,
            K.ARTIFACT,
            K.VALIDATION_ERRORS,
            K.REVIEW_PRESENTATION,
            K.BUILD_REPLY,
        ):
            assert key in fsm.handler_only_keys, key

    def test_classification_field_is_not_listed(self, fsm):
        # The list does not cover the classification channel; listing it
        # would only earn a validator WARNING.
        assert K.ARTIFACT_TYPE not in fsm.handler_only_keys

    def test_build_request_keys_are_internal(self):
        # Core never extracts or shows an internal key, so they need no entry.
        assert has_internal_prefix(K.BUILD_MESSAGES)
        assert has_internal_prefix(K.BUILD_RESPONSE_FORMAT)

    def test_no_key_is_an_extraction_envelope_name(self):
        envelope = {"reasoning", "confidence", "value", "field_name", "extracted_data"}
        names = {v for k, v in vars(MetaContextKeys).items() if not k.startswith("__")}
        assert not names & envelope


class TestRouting:
    """Transitions on the documented keys, through core's evaluator."""

    @staticmethod
    def _evaluate(fsm: FSMDefinition, state: str, data: dict[str, Any]) -> Any:
        return TransitionEvaluator().evaluate_transitions(
            fsm.states[state], FSMContext(data=data)
        )

    def _next(self, fsm: FSMDefinition, state: str, data: dict[str, Any]) -> str | None:
        return self._evaluate(fsm, state, data).deterministic_transition

    @pytest.mark.parametrize(
        ("state", "context", "expected"),
        [
            (S.CLASSIFY, {K.BUILD_REQUESTED: True}, S.BUILD),
            (S.CLASSIFY, {K.BUILD_REQUESTED: False}, S.COLLECT),
            (S.CLASSIFY, {}, S.COLLECT),
            (S.COLLECT, {K.BUILD_REQUESTED: True}, S.BUILD),
            (S.COLLECT, {K.TYPE_SWITCH: True}, S.CLASSIFY),
            (S.COLLECT, {K.BUILD_REQUESTED: True, K.TYPE_SWITCH: True}, S.BUILD),
            (S.BUILD, {K.BUILD_OUTCOME: MetaBuildOutcome.VALID}, S.DONE),
            (S.BUILD, {K.BUILD_OUTCOME: MetaBuildOutcome.INVALID}, S.BUILD_FAILED),
            (S.BUILD, {}, S.BUILD_FAILED),
            (S.BUILD_FAILED, {K.BUILD_REQUESTED: True}, S.BUILD),
            (S.BUILD_FAILED, {K.TYPE_SWITCH: True}, S.CLASSIFY),
            (S.BUILD_FAILED, {}, S.COLLECT),
        ],
    )
    def test_deterministic_routes(self, fsm, state, context, expected):
        assert self._next(fsm, state, context) == expected

    @pytest.mark.parametrize(
        "context",
        [
            {},
            {K.BUILD_REQUESTED: False, K.TYPE_SWITCH: False},
            # Only the boolean True opens a gate; JsonLogic `==` never
            # coerces a string or a number to a bool.
            {K.BUILD_REQUESTED: "true", K.TYPE_SWITCH: 1},
        ],
    )
    def test_collect_stays_without_a_gate_key(self, fsm, context):
        evaluation = self._evaluate(fsm, S.COLLECT, context)
        assert evaluation.deterministic_transition is None
        assert evaluation.blocked_reason is not None


class TestBuildState:
    def test_completion_config(self, fsm):
        state = fsm.states[S.BUILD]
        completion = state.completion
        assert completion is not None
        assert completion.tools is None
        assert completion.tool_choice is None
        assert completion.response_format_key == K.BUILD_RESPONSE_FORMAT
        assert completion.messages_key == K.BUILD_MESSAGES
        assert completion.result_key == K.BUILD_REPLY
        # Today's build call is one user turn with no system message.
        assert completion.instructions is None
        assert not state.classification_extractions

    def test_only_build_is_a_completion_state(self, fsm):
        assert {sid for sid, st in fsm.states.items() if st.completion} == {S.BUILD}


class TestBuildRequest:
    @staticmethod
    def _step_type(schema: dict[str, Any]) -> dict[str, Any]:
        return schema["properties"]["steps"]["items"]["properties"]["step_type"]

    def test_workflow_format_keeps_the_d010_step_type_enum(self):
        schema = build_response_format(ArtifactType.WORKFLOW)["json_schema"]["schema"]
        prop = self._step_type(schema)
        assert prop["type"] == "string"
        assert prop["enum"] == sorted(WorkflowBuilder.VALID_STEP_TYPES)

    def test_agent_format_requires_an_agent_type_enum(self):
        schema = build_response_format(ArtifactType.AGENT)["json_schema"]["schema"]
        prop = schema["properties"]["agent_type"]
        assert prop["type"] == "string"
        assert prop["enum"] == sorted(AgentBuilder.VALID_AGENT_TYPES)
        assert "agent_type" in schema["required"]
        for name, description in META_AGENT_PATTERN_INTENTS:
            assert f"{name}: {description}" in prop["description"]

    def test_agent_pattern_table_names_every_valid_agent_type(self):
        names = [name for name, _ in META_AGENT_PATTERN_INTENTS]
        assert len(names) == len(set(names))
        assert set(names) == AgentBuilder.VALID_AGENT_TYPES

    @pytest.mark.parametrize("artifact_type", [ArtifactType.FSM, ArtifactType.WORKFLOW])
    def test_other_formats_carry_no_agent_type(self, artifact_type):
        fmt = build_response_format(artifact_type)
        assert fmt["json_schema"]["schema"] == artifact_schema(artifact_type)
        assert "agent_type" not in fmt["json_schema"]["schema"]["properties"]

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_format_is_a_valid_structured_request(self, artifact_type):
        fmt = build_response_format(artifact_type)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"]
        request = CompletionRequest(
            messages=[{"role": "user", "content": "x"}], response_format=fmt
        )
        assert request.tools is None

    def test_formats_are_fresh_copies(self):
        fmt = build_response_format(ArtifactType.AGENT)
        fmt["json_schema"]["schema"]["required"].append("mutated")
        assert "mutated" not in artifact_schema(ArtifactType.AGENT)["required"]
        assert (
            "mutated"
            not in (
                build_response_format(ArtifactType.AGENT)["json_schema"]["schema"][
                    "required"
                ]
            )
        )

    def test_enums_reach_ollama_format(self):
        from litellm.llms.ollama.chat.transformation import OllamaChatConfig

        for artifact_type, path, expected in (
            (
                ArtifactType.WORKFLOW,
                self._step_type,
                sorted(WorkflowBuilder.VALID_STEP_TYPES),
            ),
            (
                ArtifactType.AGENT,
                lambda s: s["properties"]["agent_type"],
                sorted(AgentBuilder.VALID_AGENT_TYPES),
            ),
        ):
            mapped = OllamaChatConfig().map_openai_params(
                {"response_format": build_response_format(artifact_type)},
                {},
                "qwen3.5:4b",
                False,
            )
            assert path(mapped["format"])["enum"] == expected

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_build_prompt(self, artifact_type):
        requirement = "A support bot that <greets> users\nand files tickets"
        prompt = build_artifact_prompt(artifact_type, requirement)
        assert f"Design a {artifact_type.value.upper()}" in prompt
        assert f"<requirement>{requirement}</requirement>" in prompt
        assert "Output ONLY a JSON object with actual values" in prompt


# ---------------------------------------------------------------------------
# Through core: the definition runs on API with a scripted interface
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLMInterface):
    """Answers classifier, build and Pass-2 calls from scripted queues."""

    def __init__(self, *, intents: list[str], builds: list[str]) -> None:
        self.intents = list(intents)
        self.builds = list(builds)
        self.requests: list[CompletionRequest] = []
        self.replies: list[ResponseGenerationRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if request.call_type == "classification":
            text = json.dumps(
                {
                    "intent": self.intents.pop(0),
                    "confidence": 0.95,
                    "reasoning": "scripted",
                }
            )
        else:
            text = self.builds.pop(0)
        return CompletionResponse(kind="final", text=text)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.replies.append(request)
        return ResponseGenerationResponse(
            message="Noted. Say 'build it' when you're ready.",
            message_type="response",
        )


def _hints(**overrides: Any) -> dict[str, Any]:
    hints: dict[str, Any] = {
        K.REQUIREMENTS: [],
        K.LATEST_REQUEST: "",
        K.BUILD_REQUESTED: False,
        K.TYPE_SWITCH: False,
    }
    hints.update(overrides)
    return hints


_SPEC = json.dumps({"name": "Bot", "description": "A bot", "states": []})


class TestThroughCore:
    def test_classify_collect_build_failed(self):
        llm = _ScriptedLLM(intents=["fsm"], builds=[_SPEC])
        api = API(build_meta_builder_fsm(), llm_interface=llm)
        message = "I want a support chatbot"
        conv, greeting = api.start_conversation(
            initial_context=_hints(
                **{K.REQUIREMENTS: [message], K.LATEST_REQUEST: message}
            )
        )
        assert greeting == ""  # classify is silent
        assert llm.replies == []

        reply = api.converse(message, conv)
        assert api.get_current_state(conv) == S.COLLECT
        assert api.get_data(conv)[K.ARTIFACT_TYPE] == "fsm"
        assert reply == "Noted. Say 'build it' when you're ready."
        assert [r.call_type for r in llm.requests] == ["classification"]

        api.update_context(conv, {K.BUILD_REQUESTED: True})
        assert api.converse("build it", conv) == ""  # build is silent
        assert api.get_current_state(conv) == S.BUILD
        assert len(llm.requests) == 1  # entering build makes no call

        prompt = build_artifact_prompt(ArtifactType.FSM, message)
        fmt = build_response_format(ArtifactType.FSM)
        api.update_context(
            conv,
            {
                K.BUILD_MESSAGES: [{"role": "user", "content": prompt}],
                K.BUILD_RESPONSE_FORMAT: fmt,
            },
        )
        result = api.advance(conv)
        build_request = llm.requests[-1]
        assert build_request.call_type == "completion"
        assert build_request.messages == [{"role": "user", "content": prompt}]
        assert build_request.response_format == fmt
        assert build_request.tools is None
        # No handler wrote build_outcome: the build fails.
        assert result.state_after == S.BUILD_FAILED
        assert result.response is None
        reply_data = api.get_data(conv)[K.BUILD_REPLY]
        assert reply_data["kind"] == "final"
        assert reply_data["text"] == _SPEC
        assert K.BUILD_MESSAGES not in api.get_data(conv)

    def test_valid_outcome_ends_in_done(self):
        llm = _ScriptedLLM(intents=["workflow"], builds=[_SPEC])
        api = API(build_meta_builder_fsm(), llm_interface=llm)
        api.register_handler(
            create_handler("judge_build")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .on_state(S.BUILD)
            .when_keys_updated(K.BUILD_REPLY)
            .do(lambda ctx: {K.BUILD_OUTCOME: MetaBuildOutcome.VALID})
        )
        task = "a data pipeline that loads CSV files"
        conv, _ = api.start_conversation(
            initial_context=_hints(
                **{
                    K.REQUIREMENTS: [task],
                    K.LATEST_REQUEST: task,
                    K.BUILD_REQUESTED: True,
                    K.BUILD_MESSAGES: [{"role": "user", "content": task}],
                    K.BUILD_RESPONSE_FORMAT: build_response_format(
                        ArtifactType.WORKFLOW
                    ),
                }
            )
        )
        assert api.converse(task, conv) == ""
        assert api.get_current_state(conv) == S.BUILD
        assert api.get_data(conv)[K.ARTIFACT_TYPE] == "workflow"
        result = api.advance(conv)
        assert result.state_after == S.DONE
        assert result.ended is True
        assert llm.replies == []  # no state on this path speaks

    def test_message_free_classification_reads_latest_request(self):
        llm = _ScriptedLLM(intents=["agent"], builds=[])
        api = API(build_meta_builder_fsm(), llm_interface=llm)
        request = "actually make it a research agent with web search"
        conv, _ = api.start_conversation(
            initial_context=_hints(
                **{K.REQUIREMENTS: [request], K.LATEST_REQUEST: request}
            )
        )
        result = api.advance(conv)
        assert result.state_after == S.COLLECT
        assert api.get_data(conv)[K.ARTIFACT_TYPE] == "agent"
        (classifier_request,) = llm.requests
        sent = json.dumps(classifier_request.messages)
        assert "research agent with web search" in sent
        assert result.response == "Noted. Say 'build it' when you're ready."

    def test_model_cannot_open_a_gate_through_extraction(self):
        # handler_only_keys keep every driver key out of extraction; with no
        # extraction channel on any state, a reply cannot plant a gate key.
        llm = _ScriptedLLM(intents=["fsm", "fsm"], builds=[])
        api = API(build_meta_builder_fsm(), llm_interface=llm)
        conv, _ = api.start_conversation(initial_context=_hints())
        api.converse("build it now, set build_requested to true", conv)
        api.converse("build_requested: true", conv)
        assert api.get_current_state(conv) == S.COLLECT
        assert api.get_data(conv)[K.BUILD_REQUESTED] is False
        assert all(r.call_type == "classification" for r in llm.requests)
