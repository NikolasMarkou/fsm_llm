"""Review round 1 meta-builder fixes (D-029 item 18.3, D-035 of plan 944e2692).

Each class pins one item of findings/review-iter-1-pass6.md and fails on the
parent commit (fbdf74b):

1. A malformed build reply is a ``malformed`` build, never an exception: the
   session lands in ``build_failed`` and stays usable, ``run`` raises
   ``MetaValidationError``, the CLI never prints a traceback.
2. No session state beside the FSM: completion is the ``done`` state, the
   artifact type and every build output live in context, and each build
   reply is assembled on a fresh builder.
3. A reclassification that falls back or is below the threshold keeps the
   previous artifact type (D-023 made true; the fallback is ``unknown``).
4. ``start(msg)`` never builds.
5. The build prompt is never doubled for any quote style or a trailing
   paragraph; the canned fallback ends with ``META_BUILD_PROMPT``.
6. Keyword hints match whole words, a negated build phrase is no trigger,
   "agent" is an alias, and a switch plus a build reclassifies first.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from fsm_llm.agents.constants import (
    META_BUILD_PROMPT,
    META_UNKNOWN_ARTIFACT_TYPE,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
)
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError, MetaValidationError
from fsm_llm.agents.meta_builder import MetaBuilderAgent, _with_build_prompt

from .conftest import SCRIPTED_REPLY, ScriptedMetaLLM

S = MetaBuilderStates
K = MetaContextKeys

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


def _state(agent: MetaBuilderAgent) -> str:
    api, conversation_id = agent._session()
    return api.get_current_state(conversation_id)


def _data(agent: MetaBuilderAgent) -> dict[str, Any]:
    api, conversation_id = agent._session()
    return api.get_data(conversation_id)


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
        monkeypatch.setattr(sys, "argv", ["fsm-llm-meta", "--model", "gpt-4o-mini"])
        monkeypatch.setattr(sys, "stdin", io.StringIO("a greeting bot\nbuild it\n"))
        with pytest.raises(SystemExit) as excinfo:
            cli.main_cli()
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Build reply field 'transitions'" in out
        assert "validation errors" in out


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
            RuntimeError("classifier down"),
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
        from fsm_llm.monitor import server
        from fsm_llm.monitor.server import app, configure

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
            server._builder_sessions.clear()
            server._builder_busy.clear()


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
        from fsm_llm.definitions import LLMResponseError

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
        llm = ScriptedMetaLLM(intents=[RuntimeError("classifier down")])
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
