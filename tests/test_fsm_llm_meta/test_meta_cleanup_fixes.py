"""Step 14 of plan 5980fbb7: meta-builder fixes (F-AM-01, -02, -03, -07, -16).

Each class fails on the parent commit (ea95f3f); the golden class pins the
``ArtifactBuilder`` summary text the shared skeleton must keep byte for byte.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents import meta_cli
from fsm_llm.agents.definitions import ArtifactType, MetaBuilderResult
from fsm_llm.agents.exceptions import MetaBuilderError, OutputError
from fsm_llm.agents.meta_builder import (
    MetaBuilderAgent,
    _normalize_message,
    _with_build_prompt,
)
from fsm_llm.agents.meta_builders import FSMArtifactBuilder
from fsm_llm.agents.meta_output import save_artifact
from fsm_llm.agents.meta_prompts import build_welcome_message

from .conftest import SCRIPTED_REPLY, ScriptedMetaLLM

_GOOD_FSM: dict[str, Any] = {
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


def _trigger(message: str) -> bool:
    return MetaBuilderAgent._is_build_trigger(_normalize_message(message))


class TestTriggersTolerateTrailingPunctuation:
    """F-AM-01: "Done." builds as "done" does."""

    @pytest.mark.parametrize(
        "message",
        ["Done.", "ok!", "make it.", "go!", "ship it!", "finish.", "approve.",
         "Yes!!", "lgtm...", "ok ?", "build now!"],
    )  # fmt: skip
    def test_a_trigger_with_trailing_punctuation_builds(self, message):
        assert _trigger(message)

    @pytest.mark.parametrize(
        "message",
        ["don't build it yet", "done with the first state, thanks.", "no.",
         "okay then", "go on and add a state.", "yes, but add a state!"],
    )  # fmt: skip
    def test_punctuation_does_not_turn_other_messages_into_triggers(self, message):
        assert not _trigger(message)


def _ascii_stream() -> io.TextIOWrapper:
    """A stream that cannot encode non-ASCII text (an ASCII terminal)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)


class TestCliSurvivesAnUnencodableStdout:
    """F-AM-02: an artifact named "Café bot" is printed, not lost."""

    def _install(self, monkeypatch, outcome):
        class _FakeAgent:
            def __init__(self, config=None):
                pass

            def run_interactive(self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        monkeypatch.setattr(meta_cli, "MetaBuilderAgent", _FakeAgent)
        out, err = _ascii_stream(), _ascii_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setattr(sys, "argv", ["fsm-llm-meta"])
        return out, err

    def test_the_artifact_is_printed_with_escapes(self, monkeypatch):
        artifact = {"name": "Café bot", "initial_state": "s", "states": {}}
        out, _ = self._install(
            monkeypatch,
            MetaBuilderResult(
                artifact=artifact,
                artifact_json=json.dumps(artifact, ensure_ascii=False),
                is_valid=True,
            ),
        )
        with pytest.raises(SystemExit) as exit_info:
            meta_cli.main_cli()
        assert exit_info.value.code == 0
        assert "Caf\\xe9 bot" in out.buffer.getvalue().decode("ascii")

    def test_a_domain_error_is_one_stderr_line_and_exit_1(self, monkeypatch):
        out, err = self._install(monkeypatch, MetaBuilderError("Café failed"))
        with pytest.raises(SystemExit) as exit_info:
            meta_cli.main_cli()
        assert exit_info.value.code == 1
        assert out.buffer.getvalue() == b""
        assert "fsm-llm-meta: error: Caf" in err.buffer.getvalue().decode("ascii")

    def test_a_bad_option_value_is_no_traceback(self, monkeypatch):
        _, err = self._install(monkeypatch, MetaBuilderError("unused"))
        monkeypatch.setattr(sys, "argv", ["fsm-llm-meta", "--temperature", "9"])
        assert meta_cli.main() == 1
        assert "fsm-llm-meta: error:" in err.buffer.getvalue().decode("ascii")


_GOOD_JSON = json.dumps(_GOOD_FSM)


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


class TestSaveArtifactIsAtomic:
    """F-AM-16: a failed write leaves the old file whole."""

    def test_a_failed_replace_keeps_the_old_file_and_leaves_no_temp(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "bot.json"
        target.write_text('{"old": true}', encoding="utf-8")

        def refuse(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", refuse)
        with pytest.raises(OutputError):
            save_artifact({"name": "new"}, target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        assert [p.name for p in tmp_path.iterdir()] == ["bot.json"]

    def test_an_unserialisable_artifact_creates_no_directory(self, tmp_path):
        target = tmp_path / "new" / "dir" / "bot.json"
        with pytest.raises(OutputError):
            save_artifact({"name": object()}, target)
        assert not (tmp_path / "new").exists()
