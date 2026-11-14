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

import io
import json
import sys
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents.constants import (
    META_BUILD_CALL_FAILED,
    META_BUILD_PROMPT,
    META_UNKNOWN_ARTIFACT_TYPE,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
)
from fsm_llm.agents.definitions import ArtifactType, MetaBuilderConfig
from fsm_llm.agents.exceptions import (
    BuilderError,
    MetaBuilderError,
    MetaValidationError,
)
from fsm_llm.agents.meta_builder import MetaBuilderAgent
from fsm_llm.agents.meta_builder.session import _with_build_prompt
from fsm_llm.agents.meta_builders import (
    AgentArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from fsm_llm.agents.meta_prompts import build_response_format, build_welcome_message
from fsm_llm.core.definitions import FSMDefinition, LLMResponseError

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
            AgentArtifactBuilder.VALID_AGENT_TYPES
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

    def test_every_build_reply_gets_a_fresh_builder(self):
        """D-029 18.3: a retry assembles the new reply alone; nothing of the
        failed reply survives (``agent_type`` is a required field of every
        agent build reply, D-022, so the old reason for reuse is gone)."""
        llm = ScriptedMetaLLM(
            intents=["agent"],
            builds=[
                _spec(_AGENT_SPEC, agent_type=None),
                _spec(
                    _AGENT_SPEC,
                    tools=[{"name": "lookup", "description": "Look it up"}],
                    agent_type="react",
                ),
            ],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("an agent for web research")
        assert "Agent type is required" in agent.send("build it")
        assert "Build complete!" in agent.send("build it")
        assert [t["name"] for t in agent.get_result().artifact["tools"]] == ["lookup"]


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
        assert reply.endswith(_SENTENCE)
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
            intents=[LLMResponseError("classifier down")],
            builds=[json.dumps(_WORKFLOW_SPEC)],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        assert agent.start("a data pipeline that loads CSV files") == SCRIPTED_REPLY
        # classify exit filled the unset type from the keyword type.
        assert agent.get_internal_state()["artifact_type"] == "workflow"
        assert "Build complete!" in agent.send("build it")
        assert agent.get_result().artifact_type == ArtifactType.WORKFLOW

    def test_classifier_outage_with_no_keyword_is_an_fsm(self):
        llm = ScriptedMetaLLM(intents=[LLMResponseError("classifier down")])
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
        assert step_type["enum"] == sorted(WorkflowArtifactBuilder.VALID_STEP_TYPES)
        # A model that ignores the enum gets an invalid build, not a crash.
        assert "teleport" in reply
        assert not agent.is_complete()


# ---------------------------------------------------------------------------
# The build prompt closes every collect reply (D-024, D-025 of plan 944e2692)
# ---------------------------------------------------------------------------

_SENTENCE = "Say 'build it' when you're ready."
_NO_PROMPT = "A support bot with three states. Should it escalate to a human?"


class TestCollectReplyEndsWithTheBuildPrompt:
    """Step 13 live probe (qwen3.5:4b): 6 of 18 collect replies ended with the
    build prompt even with the sentence last in the instructions; the old
    driver's canned collect text always did. The driver now appends it."""

    def test_a_reply_without_the_sentence_gets_it_once_as_a_final_line(self):
        llm = ScriptedMetaLLM(intents=["fsm"], replies=[_NO_PROMPT, _NO_PROMPT])
        agent = MetaBuilderAgent(llm_interface=llm)
        first = agent.start("I want a support chatbot")
        second = agent.send("it should greet the user first")
        for reply in (first, second):
            assert reply == f"{_NO_PROMPT}\n{_SENTENCE}"
            assert reply.count("build it") == 1
        assert _state(agent) == S.COLLECT

    def test_the_reply_after_a_type_switch_gets_it_too(self):
        # collect -> classify, then the message-free step's collect reply.
        llm = ScriptedMetaLLM(intents=["fsm", "agent"], replies=[_NO_PROMPT] * 2)
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        reply = agent.send("actually make it a research agent with web search")
        assert reply == f"{_NO_PROMPT}\n{_SENTENCE}"

    @pytest.mark.parametrize(
        "reply",
        [
            f"Noted. {_SENTENCE}",
            f"Noted. {_SENTENCE}  \n",
            "Noted. Say 'build it' when you're ready!",
            "Noted. Say ‘build it’ when you’re ready.",
            "Noted. **Say 'build it' when you're ready.**",
            "Noted. say 'build it' when you're  ready",
        ],
    )
    def test_a_reply_that_already_ends_with_it_is_unchanged(self, reply):
        llm = ScriptedMetaLLM(intents=["fsm"], replies=[reply])
        agent = MetaBuilderAgent(llm_interface=llm)
        assert agent.start("I want a support chatbot") == reply

    def test_never_doubled_over_a_session(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"], replies=[_NO_PROMPT, f"Noted. {_SENTENCE}", _NO_PROMPT]
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        replies = [
            agent.start("I want a support chatbot"),
            agent.send("it should greet the user first"),
            agent.send("and record the issue"),
        ]
        for reply in replies:
            assert reply.endswith(_SENTENCE)
            assert reply.count(_SENTENCE) == 1

    def test_build_failed_and_done_replies_are_untouched(self):
        llm = ScriptedMetaLLM(
            intents=["fsm"],
            builds=[_spec(_FSM_SPEC, name=""), json.dumps(_FSM_SPEC)],
            replies=[_NO_PROMPT],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        failed = agent.send("build it")
        assert _state(agent) == S.BUILD_FAILED
        assert failed.startswith("I couldn't complete the build yet:")
        assert failed.endswith("then say 'build it' again.")
        assert _SENTENCE not in failed
        done = agent.send("build it")
        assert _state(agent) == S.DONE
        assert done.startswith("Build complete!")
        assert done.endswith("The artifact JSON has been generated.")
        assert _SENTENCE not in done

    def test_history_keeps_the_model_reply(self):
        """D-025: the appended line is driver UI text; core's history holds
        what the model wrote (no core API rewrites a stored reply)."""
        llm = ScriptedMetaLLM(intents=["fsm"], replies=[_NO_PROMPT, _NO_PROMPT])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("I want a support chatbot")
        agent.send("it should greet the user first")
        api, conversation_id = agent._session()
        system = [
            e["system"]
            for e in api.get_conversation_history(conversation_id)
            if "system" in e
        ]
        assert system == [_NO_PROMPT, _NO_PROMPT]


_QUIZ_SPEC: dict[str, Any] = {
    "name": "Quiz",
    "description": "Asks one question and ends",
    "states": [
        {"state_id": "q1", "description": "Ask", "purpose": "Ask a question"},
        {"state_id": "end", "description": "End", "purpose": "Say goodbye"},
    ],
    "transitions": [
        {"from_state": "q1", "target_state": "end", "description": "Answered"}
    ],
}


_GOOD_JSON = json.dumps(_QUIZ_SPEC)


def _assemble(text: str) -> tuple[str, list[str]]:
    return MetaBuilderAgent._assemble_reply(
        FSMArtifactBuilder(), ArtifactType.FSM, text, "a quiz"
    )


class TestBuildReplyReadsTheCoreJsonLadder:
    """F-AM-03 + F-XD-20: the reply is read by ``extract_json_from_text``."""

    @pytest.mark.parametrize(
        "reply",
        [
            _GOOD_JSON,
            f"Sure, {{here}} you go:\n```json\n{_GOOD_JSON}\n```",
            f"{_GOOD_JSON} {json.dumps({'extra': 1})}",
            f"{_GOOD_JSON}\nLet me know if {{anything}} is wrong.",
        ],
        ids=["bare", "braces-before-fence", "two-objects", "braces-after"],
    )
    def test_a_good_spec_in_prose_is_a_valid_build(self, reply):
        assert _assemble(reply) == ("valid", [])

    @pytest.mark.parametrize(
        "reply",
        ["I could not do that.", 'I think the "message": "hello" is fine', "[1, 2]"],
    )
    def test_text_with_no_spec_is_an_invalid_build(self, reply):
        """Core's last rung may read a phantom ``{"message": ...}``: still invalid."""
        outcome, errors = _assemble(reply)
        assert outcome == "invalid"
        assert errors


class TestAFailedStartLeavesTheSessionUsable:
    """F-AM-07: an internal fault in ``start_conversation`` is retryable."""

    def _fail_once(self, monkeypatch):
        real = API.start_conversation
        calls = {"n": 0}

        def flaky(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(API, "start_conversation", flaky)

    def test_start_can_be_retried(self, monkeypatch):
        self._fail_once(monkeypatch)
        agent = MetaBuilderAgent(llm_interface=ScriptedMetaLLM(intents=["fsm"]))
        with pytest.raises(RuntimeError, match="boom"):
            agent.start()
        assert agent.start() == build_welcome_message()
        assert agent.send("a quiz bot") == _with_build_prompt(SCRIPTED_REPLY)

    def test_send_still_says_not_started(self, monkeypatch):
        self._fail_once(monkeypatch)
        agent = MetaBuilderAgent(llm_interface=ScriptedMetaLLM())
        with pytest.raises(RuntimeError):
            agent.start()
        with pytest.raises(MetaBuilderError, match="not been started"):
            agent.send("hello")

    def test_run_closes_the_api_when_start_fails(self, monkeypatch):
        self._fail_once(monkeypatch)
        closed: list[bool] = []
        real_close = API.close
        monkeypatch.setattr(
            API, "close", lambda self: (closed.append(True), real_close(self))[1]
        )
        agent = MetaBuilderAgent(llm_interface=ScriptedMetaLLM())
        with pytest.raises(RuntimeError, match="boom"):
            agent.run("a quiz bot")
        assert closed == [True]


_GOOD_FSM = json.dumps(
    {
        "name": "Quiz",
        "description": "Asks one question and ends",
        "states": [
            {"state_id": "q1", "description": "Ask", "purpose": "Ask a question"},
            {"state_id": "end", "description": "End", "purpose": "Say goodbye"},
        ],
        "transitions": [
            {"from_state": "q1", "target_state": "end", "description": "Answered"}
        ],
    }
)


_GOOD_WORKFLOW = json.dumps(
    {
        "workflow_id": "ingest",
        "name": "Ingest",
        "description": "Loads files",
        "steps": [
            {
                "step_id": "load",
                "step_type": "auto_transition",
                "name": "Load",
                "description": "Read the files",
            }
        ],
    }
)


_GOOD_AGENT = json.dumps(
    {
        "name": "Researcher",
        "description": "Searches the web",
        "agent_type": "react",
        "tools": [{"name": "web_search", "description": "Search the web"}],
    }
)


_GOOD = {"fsm": _GOOD_FSM, "workflow": _GOOD_WORKFLOW, "agent": _GOOD_AGENT}


# (artifact type, malformed reply, a field the error must name)
_MALFORMED: list[tuple[str, dict[str, Any], str]] = [
    (
        "fsm",
        {"name": "B", "states": [{"state_id": "a"}], "transitions": None},
        "transitions",
    ),
    ("fsm", {"name": "B", "states": None}, "states"),
    ("fsm", {"name": "B", "states": ["greet", "end"], "transitions": []}, "states.0"),
    (
        "fsm",
        {"name": "B", "states": [{"state_id": "a", "description": None}]},
        "states.0.description",
    ),
    ("fsm", {"name": 5, "states": [{"state_id": "a"}]}, "name"),
    (
        "fsm",
        {
            "name": "B",
            "states": [{"state_id": "a"}],
            "transitions": [{"from_state": 1}],
        },
        "transitions.0.from_state",
    ),
    ("workflow", {"name": "W", "workflow_id": "w", "steps": None}, "steps"),
    ("workflow", {"name": "W", "steps": ["load"]}, "steps.0"),
    (
        "workflow",
        {"name": "W", "steps": [{"step_id": "s", "step_type": 3}]},
        "steps.0.step_type",
    ),
    ("agent", {"name": "A", "agent_type": "react", "tools": None}, "tools"),
    ("agent", {"name": "A", "agent_type": "react", "tools": ["search"]}, "tools.0"),
    ("agent", {"name": ["A"], "agent_type": "react", "tools": []}, "name"),
    ("agent", {"name": "A", "agent_type": 7, "tools": []}, "agent_type"),
]


_MALFORMED_IDS = [f"{kind}-{field}" for kind, _, field in _MALFORMED]


class TestMalformedBuildReplyIsAValidationError:
    @pytest.mark.parametrize(("kind", "reply", "field"), _MALFORMED, ids=_MALFORMED_IDS)
    def test_send_lands_in_build_failed_and_the_session_stays_usable(
        self, kind, reply, field
    ):
        llm = ScriptedMetaLLM(intents=[kind], builds=[json.dumps(reply), _GOOD[kind]])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("something to build")

        failed = agent.send("build it")
        assert failed.startswith("I couldn't complete the build yet:")
        assert f"Build reply field '{field}'" in failed
        assert _state(agent) == S.BUILD_FAILED
        assert not agent.is_complete()
        assert _data(agent)[K.BUILD_OUTCOME] == MetaBuildOutcome.MALFORMED

        # A detail is a collect turn (no build call) ...
        assert agent.send("add more detail please") == SCRIPTED_REPLY
        assert _state(agent) == S.COLLECT
        assert len(llm.build_requests()) == 1
        # ... and the next build is a fresh model call that can succeed.
        assert agent.send("build it").startswith("Build complete!")
        assert agent.is_complete() and _state(agent) == S.DONE
        assert agent.get_result().is_valid

    @pytest.mark.parametrize(("kind", "reply", "field"), _MALFORMED, ids=_MALFORMED_IDS)
    def test_run_raises_its_documented_error(self, kind, reply, field):
        llm = ScriptedMetaLLM(intents=[kind], builds=[json.dumps(reply)])
        agent = MetaBuilderAgent(llm_interface=llm)
        with pytest.raises(MetaValidationError) as excinfo:
            agent.run("something to build")
        assert any(f"'{field}'" in e for e in excinfo.value.errors)
        assert not agent.is_complete()

    def test_an_assembly_bug_routes_to_build_failed_too(self, monkeypatch):
        """Anything the handler raises (here a planted bug) is a failed build
        in ``send`` and ``BuilderError`` in ``run``, never a stuck session."""

        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("assembly bug")

        monkeypatch.setattr(MetaBuilderAgent, "_assemble_reply", broken)
        llm = ScriptedMetaLLM(intents=["fsm", "fsm"], builds=[_GOOD_FSM, _GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        reply = agent.send("build it")
        assert "could not be assembled" in reply
        assert _state(agent) == S.BUILD_FAILED
        assert agent.send("one more state") == SCRIPTED_REPLY

        with pytest.raises(BuilderError, match="assembly bug"):
            MetaBuilderAgent(llm_interface=llm).run("a quiz bot")

    def test_cli_prints_no_traceback(self, monkeypatch, capsys):
        import fsm_llm.agents.meta_cli as cli

        bad = json.dumps(
            {"name": "B", "states": [{"state_id": "a"}], "transitions": None}
        )
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[bad])

        class Injected(MetaBuilderAgent):
            def __init__(self, config: Any = None, **kw: Any) -> None:
                super().__init__(config, llm_interface=llm, **kw)

        monkeypatch.setattr(cli, "MetaBuilderAgent", Injected)
        monkeypatch.setattr(sys, "stdin", io.StringIO("a greeting bot\nbuild it\n"))
        assert cli.main(["--model", "gpt-4o-mini"]) == 1
        captured = capsys.readouterr()
        assert "Build reply field 'transitions'" in captured.out
        assert "validation errors" in captured.err
        assert "Traceback" not in captured.out + captured.err


class TestNoSessionStateBesideTheFSM:
    def test_no_mirrored_fields_on_the_agent(self):
        agent = MetaBuilderAgent(llm_interface=ScriptedMetaLLM(intents=["agent"]))
        agent.start("a research agent")
        for name in ("_builder", "_artifact_type", "_complete", "_build_error"):
            assert not hasattr(agent, name), name

    def test_a_failed_builds_states_do_not_leak_into_the_retry(self):
        """Pass 6 concern 2 (probe3): build 1 returns ask/score with no edges,
        build 2 a valid q1 -> end; the retry used to keep ask/score and fail
        with orphaned states forever."""
        bad = json.dumps(
            {
                "name": "Quiz",
                "description": "d",
                "states": [
                    {"state_id": "ask", "description": "A", "purpose": "A"},
                    {"state_id": "score", "description": "B", "purpose": "B"},
                ],
                "transitions": [],
            }
        )
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[bad, _GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        assert agent.send("build it").startswith("I couldn't complete the build")
        assert agent.send("build it").startswith("Build complete!")
        assert set(agent.get_result().artifact["states"]) == {"q1", "end"}

    def test_completion_is_the_done_state(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        assert not agent.is_complete()
        agent.send("build it")
        assert _state(agent) == S.DONE
        assert agent.is_complete()
        assert agent.get_internal_state()["phase"] == "complete"

    def test_build_outputs_live_in_context(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        agent.send("build it")
        data = _data(agent)
        assert data[K.ARTIFACT_TYPE] == "fsm"
        assert set(data[K.ARTIFACT]["states"]) == {"q1", "end"}
        assert data[K.BUILD_PROGRESS]["percentage"] == 100
        assert "Quiz" in data[K.BUILD_SUMMARY]
        state = agent.get_internal_state()
        assert state["artifact_preview"] == data[K.ARTIFACT]
        assert state["builder_progress"] == data[K.BUILD_PROGRESS]


class TestReclassificationKeepsThePreviousType:
    def _agent_session(self, switch_answer: Any) -> MetaBuilderAgent:
        llm = ScriptedMetaLLM(intents=["agent", switch_answer])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("A research agent with a web search tool")
        assert agent.get_internal_state()["artifact_type"] == "agent"
        agent.send("Actually, change the tool name to web_lookup")
        assert _state(agent) == S.COLLECT
        return agent

    @pytest.mark.parametrize(
        "switch_answer",
        [
            ("fsm", 0.1),  # pass 6 P2: a low-confidence fsm flipped the type
            ("workflow", 0.1),
            META_UNKNOWN_ARTIFACT_TYPE,
            ("nonsense", 0.9),  # unknown intent -> the fallback
            LLMResponseError("classifier down"),
        ],
    )
    def test_a_fallback_or_low_confidence_switch_keeps_the_type(self, switch_answer):
        agent = self._agent_session(switch_answer)
        assert agent.get_internal_state()["artifact_type"] == "agent"
        assert K.PREVIOUS_ARTIFACT_TYPE not in _data(agent)

    def test_a_confident_switch_changes_the_type(self):
        agent = self._agent_session(("fsm", 0.95))
        assert agent.get_internal_state()["artifact_type"] == "fsm"

    def test_a_first_classification_without_a_type_uses_the_keyword_type(self):
        llm = ScriptedMetaLLM(intents=[("fsm", 0.1)])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a data pipeline that loads CSV files")
        assert agent.get_internal_state()["artifact_type"] == "workflow"

    def test_the_switch_drops_the_old_build(self):
        llm = ScriptedMetaLLM(
            intents=["fsm", META_UNKNOWN_ARTIFACT_TYPE],
            builds=[json.dumps({"name": "B", "description": "d", "states": []})],
        )
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        agent.send("build it")
        assert K.ARTIFACT in _data(agent)
        agent.send("actually, add a scoring state")
        assert agent.get_internal_state()["artifact_type"] == "fsm"
        assert K.ARTIFACT not in _data(agent)
        assert agent.get_internal_state()["builder_progress"] is None


class TestStartNeverBuilds:
    @pytest.mark.parametrize(
        "message", ["ok", "build it", "Build a quiz bot and just generate it"]
    )
    def test_start_classifies_and_collects_only(self, message):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        reply = agent.start(message)
        assert reply == SCRIPTED_REPLY
        assert llm.build_requests() == []
        assert _state(agent) == S.COLLECT
        assert not agent.is_complete()
        # The next trigger builds.
        assert agent.send("build it").startswith("Build complete!")

    def test_the_monitor_start_route_returns_no_artifact_and_send_works(
        self, monkeypatch
    ):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        import fsm_llm.agents.meta_builder as meta_module
        from fsm_llm.monitor.server import app, configure, routes_builder

        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])

        class Injected(MetaBuilderAgent):
            def __init__(self, config: Any = None, **kw: Any) -> None:
                super().__init__(config, llm_interface=llm, **kw)

        monkeypatch.setattr(meta_module, "MetaBuilderAgent", Injected)
        configure()
        client = TestClient(app)
        try:
            started = client.post("/api/builder/start", json={"artifact_type": "ok"})
            assert started.status_code == 200
            body = started.json()
            assert body["is_complete"] is False
            assert "artifact" not in body
            assert llm.build_requests() == []

            sent = client.post(
                "/api/builder/send",
                json={"session_id": body["session_id"], "message": "build it"},
            )
            assert sent.status_code == 200
            assert sent.json()["is_complete"] is True
            assert set(sent.json()["artifact"]["states"]) == {"q1", "end"}
        finally:
            routes_builder._builder_sessions.clear()
            routes_builder._builder_busy.clear()


class TestBuildPromptNeverDoubled:
    @pytest.mark.parametrize(
        "reply",
        [
            "Great. Say 'build it' when you're ready.",
            'Great. Say "build it" when you\'re ready.',
            "Great. Say “build it” when you’re ready.",
            "Great. Say ‘build it’ when you’re ready.",
            "Great. Say 'build it' when you are ready.",
            "Great. **Say 'build it' when you're ready.**",
            "Say 'build it' when you're ready.\n\nAlso, which tools should it use?",
            'Say "build it" when you\'re ready. Which tools should it use?',
        ],
    )
    def test_a_reply_carrying_the_sentence_is_unchanged(self, reply):
        assert _with_build_prompt(reply) == reply

    @pytest.mark.parametrize(
        "reply",
        [
            "Which tools should it use?",
            "Say 'build' when you're ready.",  # not the sentence
            "",
        ],
    )
    def test_a_reply_without_it_gets_it_once_as_the_last_line(self, reply):
        out = _with_build_prompt(reply)
        assert out.endswith(META_BUILD_PROMPT)
        assert out.count("build it") == 1

    def test_the_canned_fallback_ends_with_the_one_sentence(self):
        from fsm_llm.core.definitions import LLMResponseError

        llm = ScriptedMetaLLM(intents=["workflow"], replies=[LLMResponseError("down")])
        reply = MetaBuilderAgent(llm_interface=llm).start("a data pipeline")
        assert reply.endswith(META_BUILD_PROMPT)
        assert reply.count("build it") == 1
        assert "WORKFLOW" in reply


class TestKeywordHints:
    @pytest.mark.parametrize(
        ("message", "switch"),
        [
            ("we exchange data with the CRM", False),
            ("not again, add a farewell state", False),
            ("change the name to Helper", True),
            ("No, it should be a workflow", True),
            ("actually make it a workflow", True),
        ],
    )
    def test_switch_words_match_whole_words(self, message, switch):
        assert MetaBuilderAgent._is_type_switch(message.lower()) is switch

    @pytest.mark.parametrize(
        ("message", "build"),
        [
            ("don't build it yet, I want to add a scoring state", False),
            ("Don’t build it yet", False),
            ("do not build", False),
            ("please do not build it now", False),
            ("I don't like the name, build it", True),
            ("great, build it", True),
            ("rebuild it", True),
            ("ok", True),
        ],
    )
    def test_build_trigger_with_a_negation_guard(self, message, build):
        normalized = " ".join(message.replace("’", "'").split()).lower()
        assert MetaBuilderAgent._is_build_trigger(normalized) is build

    def test_a_negated_build_phrase_in_a_session_does_not_build(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        agent.send("Don't build it yet, I want to add a scoring state")
        assert llm.build_requests() == []
        assert _state(agent) == S.COLLECT

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            ("agent", ArtifactType.AGENT),
            ("an agent", ArtifactType.AGENT),
            ("two agents", ArtifactType.AGENT),
            ("a workflow", ArtifactType.WORKFLOW),
            ("a toolkit for robots", ArtifactType.FSM),  # no "tool", no "bot"
            ("a data processing job", ArtifactType.WORKFLOW),
        ],
    )
    def test_type_aliases(self, text, kind):
        assert MetaBuilderAgent._detect_type_fallback(text) == kind

    def test_start_agent_with_the_classifier_down_is_an_agent(self):
        llm = ScriptedMetaLLM(intents=[LLMResponseError("classifier down")])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("agent")
        assert agent.get_internal_state()["artifact_type"] == "agent"

    def test_switch_and_build_in_one_message_builds_the_new_type(self):
        llm = ScriptedMetaLLM(intents=["fsm", "workflow"], builds=[_GOOD_WORKFLOW])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        reply = agent.send("Actually, make it a workflow instead. build it")
        assert reply.startswith("Build complete!")
        assert agent.get_result().artifact_type == ArtifactType.WORKFLOW
        assert len(llm.classifier_requests()) == 2
        (build,) = llm.build_requests()
        assert "Design a WORKFLOW" in build.messages[0]["content"]


class TestNegationGovernsOnlyItsBuildPhrase:
    """Review round 2 (pass 9 W3, D-044 of plan 944e2692; RED on 1be35d7).

    Any negation earlier in the clause blocked the build phrase, so five
    ordinary build requests that built at e1f63a9 stopped building. A
    negation now blocks the phrase only when it governs it.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "it's not perfect but build it",
            "please don't change anything and build it",
            "not a problem build it",
            "don't forget to build it",
            "why not build it",
        ],
    )
    def test_pass9_false_negatives_build_again(self, message):
        normalized = " ".join(message.split()).lower()
        assert MetaBuilderAgent._is_build_trigger(normalized) is True

    @pytest.mark.parametrize(
        "message",
        [
            "don't build it yet",
            "do not build",
            "do not build it",
            "do not ever build it",
            "never build it",
            "please don't just rebuild it",
        ],
    )
    def test_a_governing_negation_still_blocks(self, message):
        normalized = " ".join(message.split()).lower()
        assert MetaBuilderAgent._is_build_trigger(normalized) is False

    def test_a_session_builds_on_a_message_with_an_earlier_negation(self):
        llm = ScriptedMetaLLM(intents=["fsm"], builds=[_GOOD_FSM])
        agent = MetaBuilderAgent(llm_interface=llm)
        agent.start("a quiz bot")
        agent.send("It's not perfect but build it")
        assert len(llm.build_requests()) == 1
        assert agent.is_complete()
