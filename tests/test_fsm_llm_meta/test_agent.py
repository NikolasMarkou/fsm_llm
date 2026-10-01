from __future__ import annotations

"""Tests for MetaBuilderAgent — agentic architecture."""

import json
import socket

import pytest

from fsm_llm.agents.definitions import (
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from fsm_llm.agents.exceptions import (
    AgentError,
    BuilderError,
    MetaBuilderError,
    MetaValidationError,
)
from fsm_llm.agents.meta_builder import MetaBuilderAgent
from fsm_llm.agents.meta_prompts import artifact_schema
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface


class _ScriptedLLM(LLMInterface):
    """Answers the meta FSM's calls: classifier intents, build replies, replies.

    ``intents`` items are an intent name (confidence 0.95) or an
    ``(intent, confidence)`` pair; ``builds`` items are reply texts, or an
    exception instance the build call raises. Every request is recorded.
    """

    def __init__(
        self, *, intents: list | None = None, builds: list | None = None
    ) -> None:
        self.intents = list(intents or [])
        self.builds = list(builds or [])
        self.requests: list[CompletionRequest] = []
        self.replies: list[ResponseGenerationRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if request.call_type == "classification":
            intent = self.intents.pop(0) if self.intents else "fsm"
            confidence = 0.95
            if isinstance(intent, tuple):
                intent, confidence = intent
            return CompletionResponse(
                kind="final",
                text=json.dumps(
                    {"intent": intent, "confidence": confidence, "reasoning": "x"}
                ),
            )
        build = self.builds.pop(0)
        if isinstance(build, BaseException):
            raise build
        return CompletionResponse(kind="final", text=build)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.replies.append(request)
        return ResponseGenerationResponse(
            message="Noted. Say 'build it' when you're ready.",
            message_type="response",
        )

    def build_requests(self) -> list[CompletionRequest]:
        return [r for r in self.requests if r.call_type != "classification"]


class TestOfflineNetworkGuard:
    """E4: the meta conftest refuses TCP, loopback included."""

    def test_connect_to_loopback_refused(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ConnectionRefusedError, match="network blocked"):
                client.connect(server.getsockname())
        finally:
            client.close()
            server.close()


class TestSchemaEchoRejection:
    """DECISION plan_2026-05-30_26c9510a/D-001 [STALE] — a JSON-schema echo must be
    rejected, not silently assembled into an empty 'Unnamed FSM' stub."""

    def test_detects_schema_echo(self):
        # The exact shape observed in the meta_from_spec eval failure.
        assert MetaBuilderAgent._is_schema_echo(
            {"type": "object", "properties": {}, "required": []}
        )
        assert MetaBuilderAgent._is_schema_echo({"properties": {"name": {}}})
        assert MetaBuilderAgent._is_schema_echo({"$schema": "...", "type": "object"})

    def test_concrete_specs_are_not_echoes(self):
        assert not MetaBuilderAgent._is_schema_echo(
            {"name": "Bot", "states": [{"state_id": "start"}]}
        )
        assert not MetaBuilderAgent._is_schema_echo(
            {"name": "Flow", "steps": [{"step_id": "s1"}]}
        )
        assert not MetaBuilderAgent._is_schema_echo(
            {"name": "Agent", "tools": [{"name": "search"}]}
        )

    def test_run_raises_on_schema_echo(self):
        # The build call echoes a schema.
        echo = json.dumps({"type": "object", "properties": {}, "required": ["name"]})
        agent = MetaBuilderAgent(llm_interface=_ScriptedLLM(builds=[echo]))
        with pytest.raises(MetaValidationError, match="JSON schema"):
            agent.run("build a bot")


class TestMetaAgentInit:
    def test_default_config(self):
        agent = MetaBuilderAgent()
        assert agent.meta_config.temperature == 0.7
        assert agent.meta_config.max_turns == 50

    def test_custom_config(self):
        config = MetaBuilderConfig(model="gpt-4o", max_turns=10)
        agent = MetaBuilderAgent(config=config)
        assert agent.meta_config.model == "gpt-4o"

    def test_removed_output_path_is_rejected(self):
        """output_path was removed; since D-012 of plan 06a5ec0a the configs are
        extra="forbid", so an old caller gets an error naming the key."""
        from pydantic import ValidationError

        assert "output_path" not in MetaBuilderConfig.model_fields
        with pytest.raises(ValidationError, match="output_path"):
            MetaBuilderConfig(output_path="x")

    def test_initial_state(self):
        agent = MetaBuilderAgent()
        assert not agent.is_complete()
        assert agent._builder is None
        assert agent._started is False


class TestMetaAgentLifecycle:
    def test_send_before_start_raises(self):
        agent = MetaBuilderAgent()
        with pytest.raises(MetaBuilderError, match="not been started"):
            agent.send("hello")

    def test_get_result_before_complete_raises(self):
        agent = MetaBuilderAgent()
        with pytest.raises(MetaBuilderError, match="not complete"):
            agent.get_result()

    def test_double_start_raises(self):
        agent = MetaBuilderAgent()
        agent._started = True
        with pytest.raises(MetaBuilderError, match="already been started"):
            agent.start()

    def test_max_turns_exceeded(self):
        config = MetaBuilderConfig(max_turns=1)
        agent = MetaBuilderAgent(config=config)
        agent._started = True
        agent._turn_count = 1
        with pytest.raises(MetaBuilderError, match="Maximum turns"):
            agent.send("hello")


class TestTypeDetection:
    """The keyword type the driver writes as ``keyword_type`` (used when the
    classification gives no type)."""

    def test_fsm_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build a chatbot",
            "create an FSM",
            "conversation bot",
            "state machine for support",
        ]:
            result = agent._detect_type_fallback(text)
            assert result == ArtifactType.FSM, f"'{text}' should resolve to FSM"

    def test_workflow_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build a workflow",
            "data pipeline",
            "automation process",
            "ETL steps",
        ]:
            result = agent._detect_type_fallback(text)
            assert result == ArtifactType.WORKFLOW, (
                f"'{text}' should resolve to WORKFLOW"
            )

    def test_agent_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build an agent with tools",
            "react pattern",
            "research agent",
        ]:
            result = agent._detect_type_fallback(text)
            assert result == ArtifactType.AGENT, f"'{text}' should resolve to AGENT"

    def test_unknown_defaults_to_fsm(self):
        agent = MetaBuilderAgent()
        assert (
            agent._detect_type_fallback("build something amazing") == ArtifactType.FSM
        )

    def test_just_build_defaults_to_fsm(self):
        agent = MetaBuilderAgent()
        assert agent._detect_type_fallback("just build it") == ArtifactType.FSM


class TestBuildTrigger:
    """Test that build trigger phrases are detected."""

    def test_build_triggers(self):
        for phrase in ["build it", "go", "done", "approve", "yes", "lgtm"]:
            assert MetaBuilderAgent._is_build_trigger(phrase), (
                f"'{phrase}' should trigger build"
            )

    def test_non_triggers(self):
        for phrase in ["add a state", "I want 3 states", "change the name"]:
            assert not MetaBuilderAgent._is_build_trigger(phrase), (
                f"'{phrase}' should not trigger build"
            )


class TestInternalState:
    def test_initial_state(self):
        agent = MetaBuilderAgent()
        state = agent.get_internal_state()
        assert state["phase"] == "collecting"
        assert state["turn_count"] == 0
        assert state["builder_summary"] is None

    def test_state_with_builder(self):
        from fsm_llm.agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "Desc")
        agent._builder = builder

        state = agent.get_internal_state()
        assert state["artifact_type"] == "fsm"
        assert state["builder_summary"] is not None


class TestMetaAgentOutput:
    def test_output_module_imports(self):
        from fsm_llm.agents.meta_output import (
            format_artifact_json,
            format_summary,
            save_artifact,
        )

        assert callable(format_artifact_json)
        assert callable(format_summary)
        assert callable(save_artifact)

    def test_format_artifact_json(self):
        from fsm_llm.agents.meta_output import format_artifact_json

        result = format_artifact_json({"name": "test", "states": {}})
        assert '"name": "test"' in result

    def test_format_summary(self):
        from fsm_llm.agents.meta_output import format_summary

        result = MetaBuilderResult(
            artifact_type=ArtifactType.FSM,
            artifact={"name": "Bot"},
            is_valid=True,
            conversation_turns=5,
        )
        summary = format_summary(result)
        assert "fsm" in summary
        assert "Bot" in summary
        assert "5" in summary

    def test_save_artifact(self, tmp_path):
        from fsm_llm.agents.meta_output import save_artifact

        artifact = {"name": "test", "states": {}}
        path = save_artifact(artifact, tmp_path / "test.json")
        assert path.exists()
        content = path.read_text()
        assert '"name": "test"' in content

    def test_save_artifact_resolves_dotdot_segments(self, tmp_path):
        """save_artifact trusts its caller and writes to the resolved path."""
        from fsm_llm.agents.meta_output import save_artifact

        artifact = {"name": "test", "states": {}}
        raw = tmp_path / "a" / "b" / ".." / ".." / "out" / "test.json"
        path = save_artifact(artifact, raw)
        expected = (tmp_path / "out" / "test.json").resolve()
        assert path == expected
        assert ".." not in path.parts
        assert expected.exists()
        assert '"name": "test"' in expected.read_text()


class TestMetaAgentImports:
    def test_main_imports(self):
        import fsm_llm.agents

        assert hasattr(fsm_llm.agents, "MetaBuilderAgent")
        assert hasattr(fsm_llm.agents, "FSMBuilder")
        assert hasattr(fsm_llm.agents, "WorkflowBuilder")
        assert hasattr(fsm_llm.agents, "AgentBuilder")
        assert hasattr(fsm_llm.agents, "ArtifactType")
        assert hasattr(fsm_llm.agents, "MetaBuilderConfig")
        assert hasattr(fsm_llm.agents, "MetaBuilderResult")
        assert hasattr(fsm_llm.agents, "MetaBuilderError")
        assert hasattr(fsm_llm.agents, "create_builder_tools")
        assert hasattr(fsm_llm.agents, "create_fsm_tools")

    def test_version(self):
        from fsm_llm.agents import __version__

        assert isinstance(__version__, str)


class TestBuildResult:
    def test_build_result_with_valid_builder(self):
        from fsm_llm.agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("start", "Start", "Begin")
        builder.set_initial_state("start")
        agent._builder = builder

        agent._build_result()
        assert agent._result is not None
        assert agent._result.artifact_type == ArtifactType.FSM
        assert "Bot" in agent._result.artifact_json
        assert agent._result.final_context["artifact_type"] == "fsm"

    def test_build_result_with_no_builder(self):
        agent = MetaBuilderAgent()
        agent._build_result()
        assert agent._result is not None
        assert agent._result.success is False
        assert agent._result.is_valid is False


@pytest.mark.usefixtures("offline_llm")
class TestStartSendFlow:
    """Test the turn-by-turn conversation flow."""

    def test_start_with_message_detects_type(self):
        agent = MetaBuilderAgent()
        response = agent.start("build me a chatbot")
        assert "FSM" in response
        assert agent._artifact_type == ArtifactType.FSM
        assert agent._started is True

    def test_start_without_message_shows_welcome(self):
        agent = MetaBuilderAgent()
        response = agent.start()
        assert "FSM" in response
        assert "Workflow" in response
        assert "Agent" in response

    def test_send_accumulates_messages(self):
        agent = MetaBuilderAgent()
        agent.start("build a chatbot")
        agent.send("with greeting, help, and goodbye states")
        assert len(agent._messages) == 2

    def test_send_after_complete_raises(self):
        agent = MetaBuilderAgent()
        agent._started = True
        agent._complete = True
        with pytest.raises(MetaBuilderError, match="already completed"):
            agent.send("hello")


class TestCreateBuilder:
    def test_creates_fsm_builder(self):
        from fsm_llm.agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.FSM)
        assert isinstance(builder, FSMBuilder)

    def test_creates_workflow_builder(self):
        from fsm_llm.agents.meta_builders import WorkflowBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.WORKFLOW)
        assert isinstance(builder, WorkflowBuilder)

    def test_creates_agent_builder(self):
        from fsm_llm.agents.meta_builders import AgentBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.AGENT)
        assert isinstance(builder, AgentBuilder)


class TestFewShotFSMExample:
    """Step 25 (META-06): the FSM few-shot example in the extraction prompt is
    a valid FSM. A model that copies it must get a loadable definition with
    its transition kept (the example once pointed at an undeclared state)."""

    def test_copied_example_builds_a_loadable_fsm(self):
        import re

        from fsm_llm.definitions import FSMDefinition

        class _EchoExample(_ScriptedLLM):
            def complete(self, request):
                if request.call_type == "classification":
                    return super().complete(request)
                self.requests.append(request)
                prompt = request.messages[-1]["content"]
                match = re.search(r"Example:\n(\{.*\})\n", prompt)
                assert match, prompt
                return CompletionResponse(kind="final", text=match.group(1))

        llm = _EchoExample()
        result = MetaBuilderAgent(llm_interface=llm).run("a greeter")

        prompt = llm.build_requests()[0].messages[-1]["content"]
        spec = json.loads(re.search(r"Example:\n(\{.*\})\n", prompt).group(1))
        declared = {s["state_id"] for s in spec["states"]}
        for trans in spec["transitions"]:
            assert trans["from_state"] in declared
            assert trans["target_state"] in declared

        assert result.is_valid
        assert result.validation_errors == []
        definition = FSMDefinition.model_validate(result.artifact)
        edges = [
            (sid, t.target_state)
            for sid, state in definition.states.items()
            for t in state.transitions
        ]
        assert edges == [
            (t["from_state"], t["target_state"]) for t in spec["transitions"]
        ]


class TestLlmCallProviderFailure:
    """F-03 / SC-10 — a provider failure of the build call used to be turned
    into ``""``, making a total outage indistinguishable from a model that
    answered with nothing. It must raise ``BuilderError`` (an ``AgentError``
    subclass) chained from core's ``LLMResponseError`` (which chains the
    provider exception).

    DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE].
    """

    def test_provider_failure_raises_builder_error_chained(self, monkeypatch):
        provider_error = RuntimeError("provider unreachable")

        def _explode(**kwargs):
            raise provider_error

        # The one send binding of core's LLM layer: classifier and build call.
        monkeypatch.setattr("fsm_llm.llm.completion", _explode)
        agent = MetaBuilderAgent(config=MetaBuilderConfig(model="gpt-4o-mini"))
        with pytest.raises(BuilderError) as excinfo:
            agent.run("design an FSM")

        # Wraps to the package root, never to core's FSMError surface (I-5).
        assert isinstance(excinfo.value, AgentError)
        assert isinstance(excinfo.value.__cause__, LLMResponseError)
        assert excinfo.value.__cause__.__cause__ is provider_error

    def test_provider_failure_does_not_return_a_result(self):
        """The exact inverse defect: the un-fixed source returned a stub."""
        llm = _ScriptedLLM(builds=[LLMResponseError("provider unreachable")])
        agent = MetaBuilderAgent(llm_interface=llm)
        result = None
        try:
            result = agent.run("design an FSM")
        except BuilderError:
            pass
        assert result is None, "provider failure must not be reported as an answer"
        assert not agent.is_complete()

    def test_empty_answer_is_an_invalid_build_not_an_error(self):
        """`""` keeps its one true meaning — the model answered with nothing.
        Pins that the fix did not turn a legitimate empty answer into an error."""
        agent = MetaBuilderAgent(llm_interface=_ScriptedLLM(builds=[""]))
        result = agent.run("say nothing")
        assert result.is_valid is False
        assert agent.is_complete()

    def test_send_keeps_the_session_open_after_a_build_outage(self):
        """In turn-by-turn mode an outage of the build call is a failed build:
        the reply lists it, the session stays open and a second "build it"
        calls the model again."""
        spec = json.dumps(
            {
                "name": "Bot",
                "description": "A bot",
                "states": [{"state_id": "start", "description": "S", "purpose": "P"}],
            }
        )
        llm = _ScriptedLLM(builds=[LLMResponseError("provider unreachable"), spec])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support bot")
        reply = agent.send("build it")
        assert "The build call failed" in reply
        assert "say 'build it' again" in reply
        assert not agent.is_complete()
        assert "Build complete!" in agent.send("build it")
        assert agent.is_complete()
        assert len(llm.build_requests()) == 2

    @pytest.mark.usefixtures("offline_llm")
    def test_collect_response_fallback_survives_the_raise(self):
        """A-5 census pin (decisions.md D-007 of 876e7164). A provider failure of
        the collect reply keeps the session open with the canned reply."""
        agent = MetaBuilderAgent()
        reply = agent.start("I want a support bot")

        assert "Say 'build it' when ready" in reply
        assert reply.strip() != ""
        assert not agent.is_complete()


class TestWorkflowStepTypeEnum:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-010: the workflow extraction
    schema constrains ``step_type`` to ``WorkflowBuilder.VALID_STEP_TYPES`` at
    extraction time, and the enum survives into the ``response_format`` that
    is actually sent (and into Ollama's ``format`` grammar)."""

    @staticmethod
    def _step_type_prop(schema: dict) -> dict:
        return schema["properties"]["steps"]["items"]["properties"]["step_type"]

    def test_schema_step_type_is_the_valid_set(self):
        from fsm_llm.agents.meta_builders import WorkflowBuilder

        prop = self._step_type_prop(artifact_schema(ArtifactType.WORKFLOW))
        assert prop["type"] == "string"
        assert prop["enum"] == sorted(WorkflowBuilder.VALID_STEP_TYPES)

    def test_enum_reaches_response_format_and_ollama_format(self, monkeypatch):
        from litellm.llms.ollama.chat.transformation import OllamaChatConfig

        from fsm_llm.agents.meta_builders import WorkflowBuilder

        sent: list[dict] = []
        spec = {
            "name": "Flow",
            "description": "A flow",
            "workflow_id": "wf1",
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "auto_transition",
                    "name": "Start",
                    "description": "Begin",
                }
            ],
        }

        class _Msg:
            content = json.dumps(spec)

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = (_Choice(),)

        def _completion(**kwargs):
            sent.append(kwargs)
            if kwargs["response_format"]["json_schema"]["name"] != "artifact_spec":
                raise RuntimeError("classifier offline: keyword type is used")
            return _Resp()

        monkeypatch.setattr("fsm_llm.llm.completion", _completion)
        agent = MetaBuilderAgent(
            config=MetaBuilderConfig(model="ollama_chat/qwen3.5:4b")
        )
        result = agent.run("build a flow")
        assert result.artifact_type == ArtifactType.WORKFLOW

        builds = [
            kw
            for kw in sent
            if kw["response_format"]["json_schema"]["name"] == "artifact_spec"
        ]
        assert len(builds) == 1
        response_format = builds[0]["response_format"]
        schema = response_format["json_schema"]["schema"]
        expected = sorted(WorkflowBuilder.VALID_STEP_TYPES)
        assert self._step_type_prop(schema)["enum"] == expected

        mapped = OllamaChatConfig().map_openai_params(
            {"response_format": response_format},
            {},
            "qwen3.5:4b",
            False,
        )
        assert self._step_type_prop(mapped["format"])["enum"] == expected


_FSM_SPEC = json.dumps(
    {
        "name": "Bot",
        "description": "A support bot",
        "states": [
            {"state_id": "start", "description": "Greet", "purpose": "Greet"},
            {"state_id": "end", "description": "Bye", "purpose": "Close"},
        ],
        "transitions": [
            {"from_state": "start", "target_state": "end", "description": "Done"}
        ],
    }
)
_EMPTY_SPEC = json.dumps({"name": "Bot", "description": "A bot", "states": []})


class TestRunsThroughCore:
    """Plan 944e2692 step 11 (D-011): ``MetaBuilderAgent`` drives the meta FSM
    on a core ``API``. Every model call goes through the conversation's LLM
    interface (an injected one included); the agent makes no call of its own.
    On the parent commit an injected ``llm_interface`` was ignored and every
    call went to ``litellm.completion``."""

    @pytest.fixture
    def provider_calls(self, monkeypatch) -> list[dict]:
        calls: list[dict] = []

        def _record(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("no provider call expected")

        monkeypatch.setattr("litellm.completion", _record)
        monkeypatch.setattr("fsm_llm.llm.completion", _record)
        return calls

    def test_run_sends_every_call_to_the_injected_interface(self, provider_calls):
        from fsm_llm.agents.meta_prompts import (
            build_artifact_prompt,
            build_response_format,
        )

        llm = _ScriptedLLM(intents=["fsm"], builds=[_FSM_SPEC])
        task = "A support bot that greets users"
        result = MetaBuilderAgent(llm_interface=llm).run(task)

        assert provider_calls == []
        assert [r.call_type for r in llm.requests] == ["classification", "completion"]
        assert llm.replies == []  # no state on the run path speaks
        assert task in json.dumps(llm.requests[0].messages)
        (build,) = llm.build_requests()
        assert build.messages == [
            {"role": "user", "content": build_artifact_prompt(ArtifactType.FSM, task)}
        ]
        assert build.response_format == build_response_format(ArtifactType.FSM)
        assert build.tools is None
        assert result.is_valid
        assert result.artifact["name"] == "Bot"

    def test_session_sends_every_call_to_the_injected_interface(self, provider_calls):
        llm = _ScriptedLLM(intents=["fsm"], builds=[_FSM_SPEC])
        agent = MetaBuilderAgent(llm_interface=llm)

        reply = agent.start("I want a support chatbot")
        assert reply == "Noted. Say 'build it' when you're ready."
        assert agent.send("with a greeting and a goodbye") == reply
        assert "Build complete!" in agent.send("build it")

        assert provider_calls == []
        assert [r.call_type for r in llm.requests] == ["classification", "completion"]
        assert len(llm.replies) == 2  # the two collect replies (Pass 2)
        assert agent.is_complete()
        assert agent.get_result().is_valid

    def test_build_retry_calls_the_model_again(self, provider_calls):
        """A failed build clears its result on the next ``build`` entry, so the
        retry is a new build call (core skips a completion state whose result
        key is set)."""
        llm = _ScriptedLLM(intents=["fsm"], builds=[_EMPTY_SPEC, _FSM_SPEC])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")

        failed = agent.send("build it")
        assert "At least one state is required" in failed
        assert not agent.is_complete()
        assert "Build complete!" in agent.send("build it")

        assert len(llm.build_requests()) == 2
        assert agent.get_result().is_valid

    def test_type_switch_classifies_once(self, provider_calls):
        """A switch routes ``collect -> classify``; the one ``advance`` then
        classifies ``latest_request`` with no user message (D-023: one
        classifier call per switch, as before the FSM)."""
        llm = _ScriptedLLM(intents=["fsm", "agent"])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        assert agent.get_internal_state()["artifact_type"] == "fsm"

        reply = agent.send("actually make it a research agent")
        assert reply == "Noted. Say 'build it' when you're ready."
        assert agent.get_internal_state()["artifact_type"] == "agent"
        classifications = [r for r in llm.requests if r.call_type == "classification"]
        assert len(classifications) == 2
        assert "research agent" in json.dumps(classifications[1].messages)

    def test_unclassified_type_falls_back_to_the_keyword_type(self, provider_calls):
        """A low-confidence first classification leaves ``artifact_type`` unset;
        ``classify`` exit fills it from the message's keyword type."""
        llm = _ScriptedLLM(intents=[("agent", 0.1)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")
        assert agent.get_internal_state()["artifact_type"] == "workflow"

    def test_agent_pattern_comes_from_the_build_reply(self, provider_calls):
        spec = json.dumps(
            {
                "name": "Researcher",
                "description": "Searches the web",
                "agent_type": "plan_execute",
                "tools": [{"name": "search", "description": "Search the web"}],
            }
        )
        llm = _ScriptedLLM(intents=["agent"], builds=[spec])
        result = MetaBuilderAgent(llm_interface=llm).run("a research agent")

        assert result.artifact_type == ArtifactType.AGENT
        assert result.artifact["agent_type"] == "plan_execute"
        schema = llm.build_requests()[0].response_format["json_schema"]["schema"]
        assert "agent_type" in schema["required"]

    def test_config_timeout_reaches_the_interface(self):
        agent = MetaBuilderAgent(config=MetaBuilderConfig(timeout_seconds=42.0))
        agent.start()
        assert agent._api is not None
        assert agent._api.llm_interface.timeout == 42.0

    @pytest.mark.parametrize("kwarg", ["model", "temperature", "max_tokens", "hitl"])
    def test_config_owned_and_misplaced_kwargs_are_refused(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            MetaBuilderAgent(**{kwarg: 1})

    def test_module_has_no_direct_litellm_use(self):
        import inspect

        from fsm_llm.agents import meta_builder

        assert "litellm" not in inspect.getsource(meta_builder)
        assert not hasattr(MetaBuilderAgent, "_llm_call")
