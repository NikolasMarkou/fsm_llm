"""The default model is gone: reasoning, eval, monitor and the CLIs.

Step 15 of plan d454fd62 (D-005): ``DEFAULT_LLM_MODEL`` and both
``Defaults.MODEL`` are deleted. ``ReasoningEngine``, the eval model resolution
and the monitor launch routes resolve the model only through
``llm.resolve_model`` and raise ``ModelNotConfiguredError`` when nothing is
set; every CLI that needs a model exits 1 with one stderr line naming the
three ways to set it, never a traceback. The CLI tests run in a child process
with a scrubbed environment (``LLM_MODEL`` blank, ``LITELLM_MODE=PRODUCTION``
so no ``.env`` is read) and a temporary working directory.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm import ModelNotConfiguredError
from fsm_llm.constants import ENV_LLM_MODEL

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "fsm_llm"
MESSAGE = ModelNotConfiguredError.DEFAULT_MESSAGE


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_LLM_MODEL, raising=False)


# --------------------------------------------------------------------------
# The constants are gone
# --------------------------------------------------------------------------


class TestTheConstantsAreGone:
    def test_no_default_model_constant_in_core(self):
        from fsm_llm import constants

        assert not hasattr(constants, "DEFAULT_LLM_MODEL")

    def test_no_defaults_model_in_agents_and_reasoning(self):
        from fsm_llm.agents.constants import Defaults as AgentDefaults
        from fsm_llm.reasoning.constants import Defaults as ReasoningDefaults

        assert not hasattr(AgentDefaults, "MODEL")
        assert not hasattr(ReasoningDefaults, "MODEL")

    def test_no_source_file_names_either(self):
        pattern = re.compile(r"DEFAULT_LLM_MODEL|Defaults\.MODEL\b")
        hits = [
            f"{path.relative_to(REPO)}:{number}"
            for path in sorted(SRC.rglob("*.py"))
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            )
            if pattern.search(line)
        ]
        assert hits == []


# --------------------------------------------------------------------------
# Reasoning
# --------------------------------------------------------------------------


class TestReasoningEngine:
    def test_unset_raises(self, unset):
        from fsm_llm.reasoning import ReasoningEngine

        with pytest.raises(ModelNotConfiguredError):
            ReasoningEngine()

    def test_blank_model_counts_as_unset(self, unset):
        from fsm_llm.reasoning import ReasoningEngine

        with pytest.raises(ModelNotConfiguredError):
            ReasoningEngine(model="  ")

    def test_env_model_is_used(self, monkeypatch):
        from fsm_llm.reasoning import ReasoningEngine

        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert ReasoningEngine().model == "env/model"

    def test_argument_beats_env(self, monkeypatch):
        from fsm_llm.reasoning import ReasoningEngine

        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert ReasoningEngine(model="arg/model").model == "arg/model"

    def test_an_injected_interface_needs_no_model(self, unset):
        from fsm_llm.reasoning import ReasoningEngine
        from fsm_llm.testing import MockLLM2Interface

        engine = ReasoningEngine(llm_interface=MockLLM2Interface())
        assert isinstance(engine.model, str)


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------


class TestEvalResolveModel:
    def test_unset_raises(self, unset):
        from fsm_llm.eval import EvalConfig
        from fsm_llm.eval.config import resolve_model

        with pytest.raises(ModelNotConfiguredError):
            resolve_model(EvalConfig())

    def test_config_beats_env(self, monkeypatch):
        from fsm_llm.eval import EvalConfig
        from fsm_llm.eval.config import resolve_model

        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert resolve_model(EvalConfig(model="cfg/model")) == "cfg/model"
        assert resolve_model(EvalConfig()) == "env/model"


# --------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------

_FSM = {
    "name": "m",
    "description": "d",
    "initial_state": "a",
    "states": {
        "a": {
            "id": "a",
            "description": "start",
            "purpose": "p",
            "transitions": [{"target_state": "b", "description": "go", "priority": 1}],
        },
        "b": {"id": "b", "description": "end", "purpose": "p"},
    },
}


class TestMonitorRequests:
    def test_request_models_name_no_model(self):
        from fsm_llm.monitor.definitions import (
            BuilderStartRequest,
            LaunchAgentRequest,
            LaunchFSMRequest,
        )

        assert LaunchFSMRequest(fsm_json=_FSM).model is None
        assert LaunchAgentRequest(task="x").model is None
        assert BuilderStartRequest().model is None

    def test_manager_launch_fsm_unset_raises(self, unset):
        from fsm_llm.monitor.instance_manager import InstanceManager

        with pytest.raises(ModelNotConfiguredError):
            InstanceManager().launch_fsm(fsm_json=_FSM)

    def test_manager_launch_fsm_uses_env(self, monkeypatch):
        from fsm_llm.monitor.instance_manager import InstanceManager

        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        mgr = InstanceManager()
        try:
            assert mgr.launch_fsm(fsm_json=_FSM).instance_id
        finally:
            mgr.close() if hasattr(mgr, "close") else None

    def test_manager_launch_agent_unset_raises(self, unset):
        from fsm_llm.monitor.definitions import StubToolConfig
        from fsm_llm.monitor.instance_manager import InstanceManager

        with pytest.raises(ModelNotConfiguredError):
            InstanceManager().launch_agent(
                task="x",
                tools_config=[StubToolConfig(name="t", description="d")],
            )


class TestMonitorRoutes:
    @staticmethod
    def _client():
        from fastapi.testclient import TestClient

        from fsm_llm.monitor.instance_manager import InstanceManager
        from fsm_llm.monitor.server import app, configure

        configure(manager=InstanceManager())
        return TestClient(app)

    def test_fsm_launch_without_a_model_is_400(self, unset):
        resp = self._client().post("/api/fsm/launch", json={"fsm_json": _FSM})
        assert resp.status_code == 400
        assert resp.json()["detail"] == MESSAGE

    def test_a_blank_model_in_the_body_is_400_too(self, unset):
        resp = self._client().post(
            "/api/fsm/launch", json={"fsm_json": _FSM, "model": ""}
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == MESSAGE

    def test_fsm_launch_with_the_env_model_works(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        resp = self._client().post("/api/fsm/launch", json={"fsm_json": _FSM})
        assert resp.status_code == 200, resp.text

    def test_agent_launch_without_a_model_is_400(self, unset):
        resp = self._client().post(
            "/api/agent/launch",
            json={
                "task": "x",
                "tools": [{"name": "t", "description": "d"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == MESSAGE

    def test_builder_start_without_a_model_is_400(self, unset):
        resp = self._client().post("/api/builder/start", json={})
        assert resp.status_code == 400
        assert resp.json()["detail"] == MESSAGE


# --------------------------------------------------------------------------
# The CLIs, in a child process
# --------------------------------------------------------------------------


def _scrubbed_env() -> dict[str, str]:
    """Only what a child needs; ``LLM_MODEL`` blank, no ``.env`` load."""
    keep = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env[ENV_LLM_MODEL] = ""
    env["LITELLM_MODE"] = "PRODUCTION"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=_scrubbed_env(),
        cwd=cwd,
        check=False,
    )


def _assert_one_line_exit_1(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    lines = [line for line in proc.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, proc.stderr
    assert MESSAGE in lines[0]


class TestCliWithoutAModel:
    def test_fsm_llm_run(self, tmp_path):
        fsm_file = tmp_path / "f.json"
        fsm_file.write_text("{}", encoding="utf-8")
        proc = _run(["-m", "fsm_llm", "--fsm", str(fsm_file)], tmp_path)
        _assert_one_line_exit_1(proc)

    def test_reasoning(self, tmp_path):
        proc = _run(["-m", "fsm_llm.reasoning", "what is 2 + 2"], tmp_path)
        _assert_one_line_exit_1(proc)

    def test_eval_run(self, tmp_path):
        dataset = REPO / "evaluation" / "datasets" / "simple_greeting_cases.json"
        proc = _run(["-m", "fsm_llm.eval", "run", str(dataset)], tmp_path)
        _assert_one_line_exit_1(proc)

    def test_eval_examples(self, tmp_path):
        proc = _run(
            [
                "-m",
                "fsm_llm.eval",
                "examples",
                "--examples-dir",
                str(REPO / "examples"),
            ],
            tmp_path,
        )
        _assert_one_line_exit_1(proc)

    def test_meta(self, tmp_path):
        proc = _run(["-m", "fsm_llm.agents.meta_cli"], tmp_path)
        _assert_one_line_exit_1(proc)

    def test_the_package_import_is_the_fsm_llm_under_test(self):
        # Guards the child-process tests above against a stale install.
        assert Path(fsm_llm.__file__).resolve().parent == SRC
