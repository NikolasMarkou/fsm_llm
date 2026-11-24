"""``fsm-llm-meta --executables DIR`` and the agent types beside the built-in ones (plan a555dcfe
step 20, D-031).

The ids of the pattern files of a directory are valid ``agent_type`` values of the meta-builder (the
build schema enum, the builder, the single-shot run); an unknown id is rejected as before; the
built-in set ``VALID_AGENT_TYPES`` is untouched; a file named like a built-in is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.agents import meta_cli
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.agents.meta_builder import MetaBuilderAgent
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.agents.meta_prompts import build_response_format
from tests.test_fsm_llm_agents.test_pattern_user_dir import _loop_doc
from tests.test_fsm_llm_meta.conftest import ScriptedMetaLLM

_USER = "user.loop"


def _spec(agent_type: str) -> str:
    return json.dumps(
        {
            "name": "A",
            "description": "d",
            "agent_type": agent_type,
            "tools": [{"name": "web", "description": "w"}],
        }
    )


class TestTheBuilderAcceptsRegistryIds:
    def test_a_user_id_is_valid_and_the_built_ins_still_are(self):
        builder = AgentArtifactBuilder([_USER])

        builder.set_agent_type(_USER)
        assert builder.to_dict()["agent_type"] == _USER
        builder.set_agent_type("debate")
        assert builder.to_dict()["agent_type"] == "debate"

    def test_an_unknown_id_is_rejected_and_names_the_valid_ones(self):
        builder = AgentArtifactBuilder([_USER])

        with pytest.raises(BuilderError, match="Unknown agent type 'nope'") as raised:
            builder.set_agent_type("nope")
        assert _USER in str(raised.value) and "react" in str(raised.value)

    def test_a_builder_with_no_extra_id_rejects_the_user_id(self):
        with pytest.raises(BuilderError):
            AgentArtifactBuilder().set_agent_type(_USER)

    def test_the_class_set_is_never_extended(self):
        before = set(AgentArtifactBuilder.VALID_AGENT_TYPES)
        AgentArtifactBuilder([_USER])
        assert AgentArtifactBuilder.VALID_AGENT_TYPES == before
        assert _USER not in before

    def test_the_requirement_fallback_sees_a_user_id(self):
        builder = AgentArtifactBuilder(["zeta"])
        MetaBuilderAgent._set_agent_type(builder, "bad", "use the zeta pattern")
        assert builder.to_dict()["agent_type"] == "zeta"


class TestTheBuildRequestAndReply:
    def test_the_schema_enum_lists_the_user_ids_sorted_with_the_built_ins(self):
        enum = build_response_format(ArtifactType.AGENT, [_USER])["json_schema"][
            "schema"
        ]["properties"]["agent_type"]["enum"]
        assert enum == sorted({*AgentArtifactBuilder.VALID_AGENT_TYPES, _USER})

    def test_no_extra_id_leaves_the_request_bytes_as_they_were(self):
        assert build_response_format(ArtifactType.AGENT) == build_response_format(
            ArtifactType.AGENT, ()
        )
        enum = build_response_format(ArtifactType.AGENT)["json_schema"]["schema"][
            "properties"
        ]["agent_type"]["enum"]
        assert enum == sorted(AgentArtifactBuilder.VALID_AGENT_TYPES)

    def test_the_hints_carry_the_ids_only_when_there_are_some(self):
        plain = MetaBuilderAgent(llm_interface=ScriptedMetaLLM())
        extra = MetaBuilderAgent(
            llm_interface=ScriptedMetaLLM(), agent_types=["b.two", "a.one", "b.two"]
        )
        assert "_agent_types" not in plain._turn_hints("m", [], build_requested=False)
        assert extra._turn_hints("m", [], build_requested=False)["_agent_types"] == [
            "a.one",
            "b.two",
        ]

    def test_a_single_shot_run_builds_a_user_pattern_agent(self):
        llm = ScriptedMetaLLM(intents=["agent"], builds=[_spec(_USER)])
        agent = MetaBuilderAgent(llm_interface=llm, agent_types=[_USER])

        result = agent.run("an agent of my own pattern")

        assert result.artifact["agent_type"] == _USER
        (build,) = llm.build_requests()
        enum = build.response_format["json_schema"]["schema"]["properties"][
            "agent_type"
        ]["enum"]
        assert _USER in enum

    def test_without_the_id_the_same_reply_is_rejected(self):
        llm = ScriptedMetaLLM(intents=["agent"], builds=[_spec(_USER)])

        result = MetaBuilderAgent(llm_interface=llm).run("an agent of my own pattern")

        assert result.artifact.get("agent_type") != _USER


class _FakeAgent:
    """Stands in for ``MetaBuilderAgent`` in the CLI: records its keywords."""

    seen: list[dict[str, Any]] = []  # noqa: RUF012

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        type(self).seen.append(kwargs)

    def run_interactive(self) -> Any:
        from fsm_llm.agents.definitions import MetaBuilderResult

        return MetaBuilderResult(artifact={}, artifact_json="{}", is_valid=True)


class TestTheCliFlag:
    @pytest.fixture(autouse=True)
    def _fake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeAgent.seen = []
        monkeypatch.setattr(meta_cli, "MetaBuilderAgent", _FakeAgent)
        monkeypatch.setenv("LLM_MODEL", "x/y")

    def _put(self, directory: Path, name: str, doc: Any, *, bom: bool = False) -> None:
        text = ("﻿" if bom else "") + json.dumps(doc)
        (directory / name).write_text(text, encoding="utf-8")

    def test_no_flag_passes_no_keyword_at_all(self):
        assert meta_cli.main([]) == 0
        assert _FakeAgent.seen == [{}]

    def test_the_ids_of_the_directory_reach_the_agent(self, tmp_path: Path):
        self._put(tmp_path, "a.json", _loop_doc(_USER), bom=True)
        self._put(tmp_path, "B.JSON", _loop_doc("user.other"))

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        assert _FakeAgent.seen == [{"agent_types": [_USER, "user.other"]}]

    def test_a_file_named_like_a_built_in_is_refused_with_one_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        self._put(tmp_path, "react.json", _loop_doc("react"))

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        err = [x for x in capsys.readouterr().err.splitlines() if "warning" in x]
        assert len(err) == 1 and err[0].startswith("fsm-llm-meta: warning: ")
        assert _FakeAgent.seen == [{}]

    def test_a_document_that_is_not_a_pattern_is_no_agent_type(self, tmp_path: Path):
        from tests import executables_corpus as corpus

        case = next(n for n in corpus.fsm_cases() if n.startswith("agents/"))
        self._put(tmp_path, "fsm.json", corpus.wrap(case))
        self._put(tmp_path, "ok.json", _loop_doc(_USER))

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        assert _FakeAgent.seen == [{"agent_types": [_USER]}]

    def test_a_garbage_file_is_one_warning_and_the_rest_loads(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
        self._put(tmp_path, "ok.json", _loop_doc(_USER))

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        assert capsys.readouterr().err.count("bad.json") == 1
        assert _FakeAgent.seen == [{"agent_types": [_USER]}]

    def test_a_missing_directory_is_one_stderr_line_and_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        assert meta_cli.main(["--executables", str(tmp_path / "nope")]) == 1

        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1 and "not an existing directory" in lines[0]
        assert _FakeAgent.seen == []

    def test_the_flag_is_in_the_help(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit):
            meta_cli.main(["--help"])
        assert "--executables DIR" in capsys.readouterr().out


class TestTheShippedPresetsAreKnownToTheFlag:
    """A user preset may extend ``code-review``, ``summarize`` or ``data-extraction`` (step 28.4)."""

    @pytest.fixture(autouse=True)
    def _fake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeAgent.seen = []
        monkeypatch.setattr(meta_cli, "MetaBuilderAgent", _FakeAgent)
        monkeypatch.setenv("LLM_MODEL", "x/y")

    def _preset(self, doc_id: str, base: str) -> dict[str, Any]:
        return {
            "schema": "fsm-llm/executable",
            "kind": "pattern",
            "id": doc_id,
            "name": doc_id,
            "description": "d",
            "body": {"extends": base, "options": {"max_iterations": 4}},
        }

    @pytest.mark.parametrize("base", ["code-review", "summarize", "data-extraction"])
    def test_a_user_preset_over_a_shipped_preset_resolves_and_is_an_agent_type(
        self, tmp_path: Path, base: str
    ):
        (tmp_path / "mine.json").write_text(
            json.dumps(self._preset("user.mine", base)), encoding="utf-8"
        )

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        assert _FakeAgent.seen == [{"agent_types": ["user.mine"]}]

    def test_a_file_named_like_a_shipped_preset_is_refused_and_not_listed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        (tmp_path / "cr.json").write_text(
            json.dumps(_loop_doc("code-review")), encoding="utf-8"
        )

        assert meta_cli.main(["--executables", str(tmp_path)]) == 0

        err = [x for x in capsys.readouterr().err.splitlines() if "warning" in x]
        assert len(err) == 1 and "code-review" in err[0]
        assert _FakeAgent.seen == [{}]
