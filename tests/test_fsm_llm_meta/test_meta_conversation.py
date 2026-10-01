"""MetaBuilderAgent conversations, by behaviour, through the meta FSM and core.

Every test drives the real ``build_meta_builder_fsm()`` on a core ``API``
through the public driver (``start``/``send``/``run``/``run_interactive``)
with a scripted ``LLMInterface`` (``ScriptedMetaLLM``): classifier calls,
the build call and the collect reply (Pass 2) are answered from queues, so
each test states which model calls a conversation makes and what the user
sees. The FSM state is read through the core ``API`` the agent drives.

Path under test (D-011, D-023 of plan 944e2692): ``classify -> collect``
(or ``-> build`` when the first message asks for it), ``collect -> build``
on a build trigger, ``collect -> classify`` on a switch word,
``build -> done`` on a valid artifact else ``-> build_failed``, and
``build_failed -> build | classify | collect``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fsm_llm.agents.constants import (
    META_BUILD_CALL_FAILED,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
)
from fsm_llm.agents.definitions import ArtifactType, MetaBuilderConfig
from fsm_llm.agents.exceptions import BuilderError, MetaBuilderError
from fsm_llm.agents.meta_builder import MetaBuilderAgent
from fsm_llm.agents.meta_builders import AgentBuilder, WorkflowBuilder
from fsm_llm.agents.meta_prompts import build_response_format
from fsm_llm.definitions import FSMDefinition, LLMResponseError

from .conftest import SCRIPTED_REPLY, ScriptedMetaLLM

S = MetaBuilderStates
K = MetaContextKeys

# ---------------------------------------------------------------------------
# Build replies with real shapes (what a model returns for each schema)
# ---------------------------------------------------------------------------

_FSM_SPEC: dict[str, Any] = {
    "name": "SupportBot",
    "description": "Greets users, records their issue and says goodbye",
    "persona": "A patient support agent",
    "states": [
        {
            "state_id": "greeting",
            "description": "Greet the user",
            "purpose": "Welcome the user and ask how to help",
            "response_instructions": "Greet the user warmly",
        },
        {
            "state_id": "collect_issue",
            "description": "Record the issue",
            "purpose": "Find out what went wrong",
            "extraction_instructions": "Extract the issue as 'issue'",
            "response_instructions": "Confirm the issue back to the user",
        },
        {
            "state_id": "farewell",
            "description": "Say goodbye",
            "purpose": "Close the conversation",
            "response_instructions": "Thank the user and say goodbye",
        },
    ],
    "transitions": [
        {
            "from_state": "greeting",
            "target_state": "collect_issue",
            "description": "The user describes a problem",
        },
        {
            "from_state": "collect_issue",
            "target_state": "farewell",
            "description": "The issue is recorded",
        },
    ],
}

_WORKFLOW_SPEC: dict[str, Any] = {
    "workflow_id": "csv_ingest",
    "name": "CSV ingest",
    "description": "Loads CSV files, summarises them and asks for approval",
    "steps": [
        {
            "step_id": "load",
            "step_type": "auto_transition",
            "name": "Load",
            "description": "Read the CSV files",
        },
        {
            "step_id": "summarise",
            "step_type": "llm_processing",
            "name": "Summarise",
            "description": "Summarise the rows",
        },
        {
            "step_id": "approve",
            "step_type": "wait_for_event",
            "name": "Approve",
            "description": "Wait for a reviewer",
        },
    ],
}

_AGENT_SPEC: dict[str, Any] = {
    "name": "Researcher",
    "description": "Answers questions by searching the web and reading pages",
    "agent_type": "react",
    "tools": [
        {"name": "web_search", "description": "Search the web for a query"},
        {"name": "read_page", "description": "Fetch and read one web page"},
    ],
}


def _spec(base: dict[str, Any], **changes: Any) -> str:
    """``base`` with top-level ``changes`` (``None`` drops a key), as JSON."""
    spec = dict(base)
    for key, value in changes.items():
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = value
    return json.dumps(spec)


def _state(agent: MetaBuilderAgent) -> str:
    api, conversation_id = agent._session()
    return api.get_current_state(conversation_id)


def _data(agent: MetaBuilderAgent) -> dict[str, Any]:
    api, conversation_id = agent._session()
    return api.get_data(conversation_id)


def _build_prompt(request: Any) -> str:
    (message,) = request.messages
    assert message["role"] == "user"
    return message["content"]


# ---------------------------------------------------------------------------
# Interactive build of each artifact kind
# ---------------------------------------------------------------------------


class TestInteractiveBuildPerKind:
    """classify -> collect -> (collect) -> build -> done, for each kind."""

    def test_fsm(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[json.dumps(_FSM_SPEC)])
        agent = MetaBuilderAgent(llm_interface=llm)

        assert agent.start("I want a support chatbot") == SCRIPTED_REPLY
        assert _state(agent) == S.COLLECT
        assert agent.send("it greets, records the issue, says goodbye") == (
            SCRIPTED_REPLY
        )
        assert _state(agent) == S.COLLECT
        reply = agent.send("build it")

        assert reply.startswith("Build complete!")
        assert "SupportBot" in reply  # the review presentation
        assert agent.is_complete()
        assert _state(agent) == S.DONE
        # One classification, two collect replies, one build: nothing else.
        assert [r.call_type for r in llm.requests] == ["classification", "completion"]
        assert len(llm.replies) == 2
        prompt = _build_prompt(llm.build_requests()[0])
        assert "I want a support chatbot" in prompt
        assert "records the issue" in prompt

        result = agent.get_result()
        assert result.artifact_type == ArtifactType.FSM
        assert result.is_valid and result.success
        assert result.conversation_turns == 2
        definition = FSMDefinition.model_validate(result.artifact)
        assert definition.initial_state == "greeting"
        assert set(definition.states) == {"greeting", "collect_issue", "farewell"}
        assert [t.target_state for t in definition.states["greeting"].transitions] == [
            "collect_issue"
        ]

    def test_workflow(self):
        llm = ScriptedMetaLLM(intents=["workflow"], builds=[json.dumps(_WORKFLOW_SPEC)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")
        agent.send("then summarise them and wait for approval")
        assert "Build complete!" in agent.send("build it")

        (build,) = llm.build_requests()
        assert build.response_format == build_response_format(ArtifactType.WORKFLOW)
        assert "Design a WORKFLOW" in _build_prompt(build)
        result = agent.get_result()
        assert result.artifact_type == ArtifactType.WORKFLOW
        assert result.is_valid
        steps = result.artifact["steps"]
        assert list(steps) == ["load", "summarise", "approve"]
        # Steps are chained in reply order; the first is the initial step.
        assert result.artifact["initial_step_id"] == "load"
        assert steps["load"]["transitions"][0]["target"] == "summarise"
        assert steps["summarise"]["transitions"][0]["target"] == "approve"
        assert steps["approve"]["transitions"] == []

    def test_agent(self):
        llm = ScriptedMetaLLM(intents=["agent"], builds=[json.dumps(_AGENT_SPEC)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a research agent that answers questions from the web")
        assert "Build complete!" in agent.send("build it")

        result = agent.get_result()
        assert result.artifact_type == ArtifactType.AGENT
        assert result.is_valid
        assert result.artifact["agent_type"] == "react"
        assert [t["name"] for t in result.artifact["tools"]] == [
            "web_search",
            "read_page",
        ]

    def test_start_with_the_type_name_alone(self):
        """The monitor starts a session with the chosen type as the message."""
        llm = ScriptedMetaLLM(intents=["workflow"])
        agent = MetaBuilderAgent(llm_interface=llm)

        assert agent.start("workflow") == SCRIPTED_REPLY
        state = agent.get_internal_state()
        assert state["artifact_type"] == "workflow"
        assert state["message_count"] == 1
        assert state["turn_count"] == 0  # start is not a send
        assert _state(agent) == S.COLLECT
        assert "workflow" in json.dumps(llm.classifier_requests()[0].messages)

    def test_start_without_a_message_makes_no_model_call(self):
        llm = ScriptedMetaLLM()
        agent = MetaBuilderAgent(llm_interface=llm)
        welcome = agent.start()
        for kind in ("FSM", "Workflow", "Agent"):
            assert kind in welcome
        assert llm.requests == [] and llm.replies == []
        assert _state(agent) == S.CLASSIFY

        # The first send is then the classified message.
        llm.intents.append("agent")
        agent.send("a research agent")
        assert agent.get_internal_state()["artifact_type"] == "agent"
        assert _state(agent) == S.COLLECT


# ---------------------------------------------------------------------------
# The agent pattern: an enum of the build schema, not a second classifier
# ---------------------------------------------------------------------------


class TestAgentPatternFromTheBuildSchema:
    def test_one_classifier_call_and_the_enum_in_the_request(self):
        llm = ScriptedMetaLLM(
            intents=["agent"],
            builds=[_spec(_AGENT_SPEC, agent_type="plan_execute")],
        )
        result = MetaBuilderAgent(llm_interface=llm).run(
            "an agent that plans research steps and runs them with web search"
        )

        # Only the artifact type is classified; the pattern rides the build.
        assert len(llm.classifier_requests()) == 1
        (build,) = llm.build_requests()
        schema = build.response_format["json_schema"]["schema"]
        assert schema["properties"]["agent_type"]["enum"] == sorted(
            AgentBuilder.VALID_AGENT_TYPES
        )
        assert "agent_type" in schema["required"]
        assert result.artifact["agent_type"] == "plan_execute"
        assert result.is_valid

    def test_missing_pattern_falls_back_to_the_pattern_named_in_the_request(self):
        llm = ScriptedMetaLLM(
            intents=["agent"], builds=[_spec(_AGENT_SPEC, agent_type=None)]
        )
        result = MetaBuilderAgent(llm_interface=llm).run(
            "a reflexion agent that critiques its own web answers"
        )
        assert result.artifact["agent_type"] == "reflexion"
        assert result.is_valid

    def test_unknown_pattern_falls_back_to_the_pattern_named_in_the_request(self):
        llm = ScriptedMetaLLM(
            intents=["agent"], builds=[_spec(_AGENT_SPEC, agent_type="supervisor")]
        )
        result = MetaBuilderAgent(llm_interface=llm).run(
            "a plan execute agent for web research"
        )
        assert result.artifact["agent_type"] == "plan_execute"

    def test_no_pattern_anywhere_is_an_invalid_build(self):
        llm = ScriptedMetaLLM(
            intents=["agent"], builds=[_spec(_AGENT_SPEC, agent_type=None)]
        )
        result = MetaBuilderAgent(llm_interface=llm).run("an agent for web research")
        assert result.is_valid is False
        assert "Agent type is required" in result.validation_errors


# ---------------------------------------------------------------------------
# Type switch mid-collection
# ---------------------------------------------------------------------------


class TestTypeSwitch:
    def test_switch_reclassifies_once_and_builds_the_new_type(self):
        llm = ScriptedMetaLLM(
            intents=["fsm", "agent"], builds=[json.dumps(_AGENT_SPEC)]
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        assert agent.get_internal_state()["artifact_type"] == "fsm"

        reply = agent.send("actually make it a research agent with web search")
        # collect -> classify, then the message-free step: classify -> collect
        # with the collect reply.
        assert reply == SCRIPTED_REPLY
        assert _state(agent) == S.COLLECT
        assert agent.get_internal_state()["artifact_type"] == "agent"
        classifications = llm.classifier_requests()
        assert len(classifications) == 2
        assert "research agent with web search" in json.dumps(
            classifications[1].messages
        )

        assert "Build complete!" in agent.send("build it")
        (build,) = llm.build_requests()
        assert "Design a AGENT" in _build_prompt(build)
        assert build.response_format == build_response_format(ArtifactType.AGENT)
        assert agent.get_result().artifact_type == ArtifactType.AGENT

    def test_a_failed_switch_classification_keeps_the_previous_type(self):
        """D-023: ``artifact_type`` is not cleared on ``classify`` entry, so a
        low-confidence switch keeps the type the collect reply names."""
        llm = ScriptedMetaLLM(intents=["workflow", ("agent", 0.1)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")
        agent.send("actually, not sure what it should be")
        assert agent.get_internal_state()["artifact_type"] == "workflow"
        assert _state(agent) == S.COLLECT

    def test_switch_after_a_failed_build_drops_the_old_errors(self):
        llm = ScriptedMetaLLM(
            intents=["fsm", "workflow"],
            builds=[_spec(_FSM_SPEC, states=[], transitions=[])],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        failed = agent.send("build it")
        assert "At least one state is required" in failed
        assert _state(agent) == S.BUILD_FAILED

        agent.send("actually make it a data pipeline instead")
        # build_failed -> classify (p20), classify entry clears the errors.
        assert _state(agent) == S.COLLECT
        assert agent.get_internal_state()["artifact_type"] == "workflow"
        assert K.VALIDATION_ERRORS not in _data(agent)
        collect_prompt = llm.replies[-1].system_prompt
        assert "At least one state is required" not in collect_prompt


# ---------------------------------------------------------------------------
# Failed build, more details, second build
# ---------------------------------------------------------------------------


class TestBuildRetry:
    def test_invalid_build_then_details_then_a_valid_build(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"],
            builds=[
                _spec(_FSM_SPEC, states=[], transitions=[]),
                json.dumps(_FSM_SPEC),
            ],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")

        failed = agent.send("build it")
        assert failed.startswith("I couldn't complete the build yet:")
        assert "  - At least one state is required" in failed
        assert "say 'build it' again" in failed
        assert not agent.is_complete()
        assert _state(agent) == S.BUILD_FAILED
        assert _data(agent)[K.BUILD_OUTCOME] == MetaBuildOutcome.INVALID

        # The details go back to collect (build_failed -> collect, p900) and
        # its reply sees the validation errors.
        assert agent.send("greeting, collect_issue and farewell") == SCRIPTED_REPLY
        assert _state(agent) == S.COLLECT
        assert "At least one state is required" in llm.replies[-1].system_prompt

        assert "Build complete!" in agent.send("build it")
        assert _state(agent) == S.DONE
        first, second = llm.build_requests()
        # build entry cleared build_reply: the retry is a new model call, with
        # the details in its requirement.
        assert "collect_issue and farewell" not in _build_prompt(first)
        assert "collect_issue and farewell" in _build_prompt(second)
        assert agent.get_result().is_valid

    def test_build_again_straight_from_build_failed(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"],
            builds=[_spec(_FSM_SPEC, name=""), json.dumps(_FSM_SPEC)],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        assert "FSM name is required" in agent.send("build it")
        # build_failed -> build (p10): no collect reply in between.
        replies_before = len(llm.replies)
        assert "Build complete!" in agent.send("build it")
        assert len(llm.replies) == replies_before
        assert len(llm.build_requests()) == 2

    def test_a_retry_of_the_same_type_keeps_the_agent_builder(self):
        """A second build of the same type reuses the builder: tools from the
        first reply survive a second reply that only adds the pattern."""
        llm = ScriptedMetaLLM(
            intents=["agent"],
            builds=[
                _spec(_AGENT_SPEC, agent_type=None),
                _spec(_AGENT_SPEC, tools=[], agent_type="react"),
            ],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("an agent for web research")
        assert "Agent type is required" in agent.send("build it")
        assert "Build complete!" in agent.send("build it")
        assert [t["name"] for t in agent.get_result().artifact["tools"]] == [
            "web_search",
            "read_page",
        ]


# ---------------------------------------------------------------------------
# Outages
# ---------------------------------------------------------------------------


class TestOutages:
    def test_run_raises_builder_error_chained_from_core(self):
        outage = LLMResponseError("provider unreachable")
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[outage])
        agent = MetaBuilderAgent(llm_interface=llm)
        with pytest.raises(BuilderError, match="provider unreachable") as excinfo:
            agent.run("a support chatbot")
        assert excinfo.value.__cause__ is outage
        assert not agent.is_complete()

    def test_send_records_the_failed_call_and_keeps_the_session_open(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"],
            builds=[LLMResponseError("provider unreachable"), json.dumps(_FSM_SPEC)],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")

        reply = agent.send("build it")
        assert "The build call failed" in reply
        assert "provider unreachable" in reply
        assert "say 'build it' again" in reply
        assert not agent.is_complete()
        # D-023: the failure is the build state's result, so the FSM's own
        # edge took the session to build_failed.
        assert _state(agent) == S.BUILD_FAILED
        data = _data(agent)
        assert data[K.BUILD_REPLY]["kind"] == META_BUILD_CALL_FAILED
        assert data[K.BUILD_REPLY]["calls"] == []
        assert data[K.BUILD_OUTCOME] == MetaBuildOutcome.INVALID
        assert data[K.VALIDATION_ERRORS] == [data[K.BUILD_REPLY]["text"]]

        assert "Build complete!" in agent.send("build it")
        assert len(llm.build_requests()) == 2

    def test_collect_reply_outage_gives_the_canned_reply(self):
        llm = ScriptedMetaLLM(
            intents=["workflow"],
            replies=[SCRIPTED_REPLY, LLMResponseError("provider unreachable")],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")

        reply = agent.send("it also writes a report")
        assert "Say 'build it' when ready" in reply
        assert "WORKFLOW" in reply
        assert not agent.is_complete()
        # The turn was rolled back, the requirement is kept: the next turn
        # carries both details.
        assert _state(agent) == S.COLLECT
        agent.send("and emails it")
        assert _data(agent)[K.REQUIREMENTS] == [
            "a data pipeline that loads CSV files",
            "it also writes a report",
            "and emails it",
        ]

    def test_classifier_outage_uses_the_keyword_type(self):
        llm = ScriptedMetaLLM(
            intents=[RuntimeError("classifier down")],
            builds=[json.dumps(_WORKFLOW_SPEC)],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        assert agent.start("a data pipeline that loads CSV files") == SCRIPTED_REPLY
        # classify exit filled the unset type from the keyword type.
        assert agent.get_internal_state()["artifact_type"] == "workflow"
        assert "Build complete!" in agent.send("build it")
        assert agent.get_result().artifact_type == ArtifactType.WORKFLOW

    def test_classifier_outage_with_no_keyword_is_an_fsm(self):
        llm = ScriptedMetaLLM(intents=[RuntimeError("classifier down")])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("something that helps my team")
        assert agent.get_internal_state()["artifact_type"] == "fsm"


# ---------------------------------------------------------------------------
# Session lifecycle: run vs send vs run_interactive, limits
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_run_builds_at_once_and_completes_even_when_invalid(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"], builds=[_spec(_FSM_SPEC, states=[], transitions=[])]
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        result = agent.run("a support chatbot, build it")

        # classify -> build in one turn: no collect reply, one build call.
        assert llm.replies == []
        assert [r.call_type for r in llm.requests] == ["classification", "completion"]
        assert agent.is_complete()
        assert result.is_valid is False and result.success is False
        assert "At least one state is required" in result.validation_errors
        assert agent.get_result() is result
        # run is not a session: there is nothing to send to afterwards.
        with pytest.raises(MetaBuilderError, match="not been started"):
            agent.send("more")

    def test_send_after_completion_raises(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[json.dumps(_FSM_SPEC)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        agent.send("build it")
        with pytest.raises(MetaBuilderError, match="already completed"):
            agent.send("one more state")
        assert len(llm.build_requests()) == 1

    def test_second_start_raises(self):
        agent = MetaBuilderAgent(llm_interface=ScriptedMetaLLM())
        agent.start()
        with pytest.raises(MetaBuilderError, match="already been started"):
            agent.start("again")

    def test_send_before_start_raises(self):
        llm = ScriptedMetaLLM()
        with pytest.raises(MetaBuilderError, match="not been started"):
            MetaBuilderAgent(llm_interface=llm).send("hello")
        assert llm.requests == []

    def test_max_turns(self):
        llm = ScriptedMetaLLM(intents=["fsm"])
        agent = MetaBuilderAgent(
            config=MetaBuilderConfig(max_turns=2), llm_interface=llm
        )
        agent.start("I want a support chatbot")
        agent.send("it greets users")
        agent.send("and says goodbye")
        replies = len(llm.replies)
        with pytest.raises(MetaBuilderError, match=r"Maximum turns \(2\) exceeded"):
            agent.send("and records issues")
        assert len(llm.replies) == replies  # the refused turn made no call
        assert _data(agent)[K.REQUIREMENTS] == [
            "I want a support chatbot",
            "it greets users",
            "and says goodbye",
        ]

    def test_build_trigger_needs_the_whole_message_or_a_build_phrase(self):
        llm = ScriptedMetaLLM(intents=["fsm"])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        agent.send("ok so it should greet people")  # "ok" inside: no build
        assert _state(agent) == S.COLLECT
        assert llm.build_requests() == []

    def test_run_interactive_until_complete(self, monkeypatch, capsys):
        llm = ScriptedMetaLLM(intents=["workflow"], builds=[json.dumps(_WORKFLOW_SPEC)])
        inputs = iter(["a data pipeline that loads CSV files", "   ", "build it"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        agent = MetaBuilderAgent(llm_interface=llm)

        result = agent.run_interactive()

        out = capsys.readouterr().out
        assert "Workflow" in out  # the welcome text
        assert SCRIPTED_REPLY in out
        assert "Build complete!" in out
        assert result.is_valid
        assert result.artifact_type == ArtifactType.WORKFLOW
        # The blank line was skipped: two sends, not three.
        assert result.conversation_turns == 2

    def test_run_interactive_eof_returns_the_unfinished_result(
        self, monkeypatch, capsys
    ):
        llm = ScriptedMetaLLM(intents=["fsm"])
        inputs = iter(["I want a support chatbot"])

        def _input(prompt: str = "") -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr("builtins.input", _input)
        agent = MetaBuilderAgent(llm_interface=llm)

        result = agent.run_interactive()

        assert "Session ended by user." in capsys.readouterr().out
        assert not agent.is_complete()
        assert result.is_valid is False
        assert result.validation_errors == ["Builder was not initialized"]
        assert llm.build_requests() == []


# ---------------------------------------------------------------------------
# get_internal_state: the keys the monitor's builder page reads
# ---------------------------------------------------------------------------


class TestInternalStateForTheMonitor:
    _KEYS = frozenset(
        {
            "phase",
            "turn_count",
            "is_complete",
            "started",
            "message_count",
            "builder_progress",
            "builder_summary",
            "artifact_preview",
            "validation_errors",
            "is_valid",
        }
    )

    def test_through_a_failed_and_a_valid_build(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"],
            builds=[_spec(_FSM_SPEC, states=[], transitions=[]), json.dumps(_FSM_SPEC)],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")

        state = agent.get_internal_state()
        assert self._KEYS | {"artifact_type"} <= set(state)
        assert state["phase"] == "collecting"
        assert state["artifact_type"] == "fsm"
        assert state["builder_progress"] is None  # no build yet

        agent.send("build it")
        state = agent.get_internal_state()
        assert state["phase"] == "collecting"
        assert state["turn_count"] == 1
        progress = state["builder_progress"]
        assert set(progress) == {
            "percentage",
            "completed",
            "total_required",
            "missing",
            "warnings",
        }
        assert progress["percentage"] < 100
        assert progress["missing"]
        assert "At least one state is required" in state["validation_errors"]
        assert state["is_valid"] is False
        assert state["artifact_preview"]["name"] == "SupportBot"
        assert "SupportBot" in state["builder_summary"]

        agent.send("build it")
        state = agent.get_internal_state()
        assert state["phase"] == "complete"
        assert state["is_complete"] is True
        assert state["is_valid"] is True
        assert state["validation_errors"] == []
        assert state["builder_progress"]["percentage"] == 100
        assert set(state["artifact_preview"]["states"]) == {
            "greeting",
            "collect_issue",
            "farewell",
        }
        # The state is JSON (the monitor sends it over HTTP).
        json.dumps(state)


# ---------------------------------------------------------------------------
# The workflow step-type enum (c1d5bfbc/D-010) on the conversation path
# ---------------------------------------------------------------------------


class TestWorkflowStepTypeEnumInSession:
    def test_enum_reaches_the_session_build_request(self):
        llm = ScriptedMetaLLM(
            intents=["workflow"],
            builds=[
                _spec(
                    _WORKFLOW_SPEC,
                    steps=[
                        {
                            "step_id": "load",
                            "step_type": "teleport",
                            "name": "Load",
                            "description": "Read the CSV files",
                        }
                    ],
                )
            ],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")
        reply = agent.send("build it")

        (build,) = llm.build_requests()
        step_type = build.response_format["json_schema"]["schema"]["properties"][
            "steps"
        ]["items"]["properties"]["step_type"]
        assert step_type["enum"] == sorted(WorkflowBuilder.VALID_STEP_TYPES)
        # A model that ignores the enum gets an invalid build, not a crash.
        assert "teleport" in reply
        assert not agent.is_complete()
