"""The entry-point sweep: no path names a model the caller did not set (census D11).

There is no built-in model anywhere (plan d454fd62 D-005 for the core; plan
ace7a8c8 D-048, roadmap 0.15, for ``fsm-llm-eval`` and the monitor too):
``llm.resolve_model`` is the one resolver and raises ``ModelNotConfiguredError``
when nothing names a model. The documented rule is "an injected
``llm_interface`` needs no model". The root autouse fixture of
``tests/conftest.py`` sets ``LLM_MODEL`` for every test, which hid two
regressions once (the SOP registry and the eval runners, review gate 1
concerns 1 and 2). Every case below deletes ``LLM_MODEL`` itself and then
builds, runs or launches a public entry point:

- ``TestNoModelNeeded``: with an injected double (or no LLM) the call works;
- ``TestModelStillRequired``: with nothing injected and nothing set the call
  raises ``ModelNotConfiguredError``, the one message that says how to set one
  (the library entries, the eval runners without a factory, the monitor
  launches);
- ``TestEveryCliWithoutAModel``: ``fsm-llm``, reasoning, meta, ``fsm-llm-eval``
  ``run`` and ``examples`` exit 1 with that one stderr line, never a traceback;
- ``TestMonitorRoutes``: the launch routes answer HTTP 400 with the message;
  the dashboard itself starts without a model (``test_env_file_flag.py``);
- ``TestPrecedence`` and ``TestNoPathNamesAModelTheCallerDidNotSet``: the
  argument beats the environment, and no source module names a default.

A new public constructor, runner or launch route that takes ``llm_interface``
or a model gets one row here. Fix a failing row in ``src/``, never here.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API, Classifier, FSMManager, ModelNotConfiguredError
from fsm_llm.agents import (
    AgentConfig,
    AgentError,
    AgentGraphBuilder,
    ChainStep,
    ConfiguredAgentBuilder,
    EvaluationResult,
    MetaBuilderAgent,
    SemanticToolRegistry,
    SwarmAgent,
    create_agent,
    tool,
)
from fsm_llm.agents import sop as sop_module
from fsm_llm.agents.composition import default_llm_judge
from fsm_llm.agents.definitions import default_agent_config
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.constants import ENV_LLM_MODEL
from fsm_llm.core.definitions import ClassificationSchema, IntentDefinition
from fsm_llm.eval import ConversationCase, EvalConfig, run_cases, run_dataset
from fsm_llm.eval import __main__ as eval_cli
from fsm_llm.llm import interface_model, resolve_model
from fsm_llm.monitor import InstanceManager
from fsm_llm.monitor.definitions import StubToolConfig
from fsm_llm.reasoning import ReasoningEngine
from fsm_llm.synthesis import (
    FSMSynthesizer,
    Simulator,
    SynthesizeExecutor,
    create_procedure_tools,
)
from fsm_llm.testing import ScriptedLLM
from fsm_llm.workflows import (
    WorkflowEngine,
    agent_step,
    auto_step,
    create_workflow,
)

_FSM: dict[str, Any] = {
    "name": "NoModel",
    "description": "two silent-ish states",
    "initial_state": "a",
    "states": {
        "a": {
            "id": "a",
            "description": "start",
            "purpose": "start",
            "response_instructions": "Greet.",
            "transitions": [
                {"target_state": "b", "description": "always", "priority": 100}
            ],
        },
        "b": {
            "id": "b",
            "description": "end",
            "purpose": "end",
            "response_instructions": "Say goodbye.",
        },
    },
}

_SCHEMA = ClassificationSchema(
    intents=[
        IntentDefinition(name="yes", description="agrees"),
        IntentDefinition(name="no", description="disagrees"),
    ],
    fallback_intent="no",
)


def _double() -> ScriptedLLM:
    return ScriptedLLM(default_response="ok", extraction_data={})


@tool
def _echo(text: str) -> str:
    """Echo the text."""
    return text


@pytest.fixture(autouse=True)
def _no_model_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete ``LLM_MODEL`` (the root fixture sets it for every test)."""
    monkeypatch.delenv(ENV_LLM_MODEL, raising=False)


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "config": {"trials": 1, "workers": 1},
                "cases": [
                    {
                        "id": "greets",
                        "fsm": _FSM,
                        "turns": ["hello"],
                        "expect": {"final_state": "b"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Paths that need no model
# ---------------------------------------------------------------------------


def _core_api(tmp_path: Path) -> None:
    api = API.from_definition(_FSM, llm_interface=_double())
    conversation_id, _ = api.start_conversation()
    api.converse("hello", conversation_id)
    api.close()


def _core_fsm_manager(tmp_path: Path) -> None:
    from fsm_llm.core.definitions import FSMDefinition

    definition = FSMDefinition.model_validate(_FSM)
    manager = FSMManager(fsm_loader=lambda _id: definition, llm_interface=_double())
    manager.start_conversation("x")


def _core_classifier(tmp_path: Path) -> None:
    Classifier(_SCHEMA, llm_interface=_double())


def _agents_config_injected(tmp_path: Path) -> None:
    assert default_agent_config(_double()).model


def _agents_config_named(tmp_path: Path) -> None:
    AgentConfig(model="provider/named")


def _sop_register_without_model_key(tmp_path: Path) -> None:
    registry = sop_module.SOPRegistry()
    registry.register(
        sop_module.SOPDefinition(name="s", config_overrides={"temperature": 0.3})
    )


def _sop_builtin_registry(tmp_path: Path) -> None:
    assert len(sop_module.load_builtin_sops()) >= 3


def _sop_from_dict_and_file(tmp_path: Path) -> None:
    registry = sop_module.SOPRegistry()
    registry.register_from_dict(
        {"name": "d", "config_overrides": {"max_iterations": 3}}
    )
    path = tmp_path / "f.json"
    path.write_text(
        json.dumps({"name": "f", "config_overrides": {"temperature": 0.1}}),
        encoding="utf-8",
    )
    registry.register_from_file(path)


def _sop_to_agent_config_injected(tmp_path: Path) -> None:
    sop = sop_module.SOPDefinition(name="s", config_overrides={"temperature": 0.3})
    config = sop.to_agent_config(llm_interface=_double())
    assert config.temperature == 0.3 and config.model


def _sop_to_agent_config_named(tmp_path: Path) -> None:
    sop = sop_module.SOPDefinition(name="s", config_overrides={"temperature": 0.3})
    assert sop.to_agent_config(model="provider/named").model == "provider/named"


def _every_pattern(pattern: str) -> Any:
    def build(tmp_path: Path) -> None:
        if pattern in _TOOL_PATTERNS:
            create_agent(pattern, [_echo], llm_interface=_double())
        else:
            create_agent(
                pattern, **_REQUIRED_ARGUMENTS.get(pattern, {}), llm_interface=_double()
            )

    return build


#: Constructor arguments a pattern needs besides the interface.
_REQUIRED_ARGUMENTS: dict[str, dict[str, Any]] = {
    "prompt_chain": {
        "chain": [
            ChainStep(
                step_id="s",
                name="s",
                extraction_instructions="Extract the topic.",
                response_instructions="Answer.",
            )
        ]
    },
    "maker_checker": {
        "maker_instructions": "Write.",
        "checker_instructions": "Check.",
    },
    "evaluator_optimizer": {
        "evaluation_fn": lambda output, context: EvaluationResult(passed=True)
    },
}

_TOOL_PATTERNS = frozenset(
    {
        "react",
        "rewoo",
        "plan_execute",
        "reflexion",
        "parallel_react",
        "native_fc",
        "verified_react",
        "auto_memory",
        "reasoning_react",
    }
)


def _agents_configured_builder(tmp_path: Path) -> None:
    ConfiguredAgentBuilder().set_pattern("react").add_tool(_echo).set_option(
        "llm_interface", _double()
    ).build()


def _eval_run_cases_factory(tmp_path: Path) -> None:
    case = ConversationCase(id="c", fsm=_FSM, turns=["hi"], expect={"final_state": "b"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = EvalConfig(trials=1, workers=1, output_dir=str(run_dir))
    report = run_cases([case], config, run_dir, llm_interface_factory=_double)
    assert report.model


def _eval_run_dataset_factory(tmp_path: Path) -> None:
    report = run_dataset(
        _write_dataset(tmp_path),
        llm_interface_factory=_double,
        output_root=str(tmp_path / "out"),
    )
    assert report.model


def _reasoning_engine_injected(tmp_path: Path) -> None:
    ReasoningEngine(llm_interface=_double())


def _workflows_agent_step(tmp_path: Path) -> None:
    agent = create_agent("react", [_echo], llm_interface=_double())
    definition = (
        create_workflow("w", "w")
        .with_initial_step(agent_step("a", "A", agent=agent, success_step_id="done"))
        .with_step(auto_step("done", "Done", next_step_id="", action=lambda ctx: {}))
    )
    engine = WorkflowEngine()
    engine.register_workflow(definition)


def _monitor_construction(tmp_path: Path) -> None:
    manager = InstanceManager()
    manager.attach_api(API.from_definition(_FSM, llm_interface=_double()))
    manager.launch_workflow(preset_id=sorted(_workflow_presets())[0])


def _workflow_presets() -> list[str]:
    from fsm_llm.monitor.instance_manager import workflow_presets

    return list(workflow_presets._WORKFLOW_PRESETS)


def _synthesis_synthesizer_injected(tmp_path: Path) -> None:
    FSMSynthesizer(_double())


def _synthesis_executor_and_tools(tmp_path: Path) -> None:
    synthesizer = FSMSynthesizer(_double())
    API.from_definition(
        _FSM,
        llm_interface=_double(),
        executors={"synthesize": SynthesizeExecutor(synthesizer)},
    )
    create_procedure_tools(synthesizer, runner_llm=_double())
    Simulator(_double())


def _agents_run_with_a_double(tmp_path: Path) -> None:
    create_agent("react", [_echo], llm_interface=_double()).run("hi")
    create_agent("debate", llm_interface=_double()).run("hi")


def _agents_graph(tmp_path: Path) -> None:
    (
        AgentGraphBuilder()
        .add_node("a", create_agent("debate", llm_interface=_double()))
        .set_entry("a")
        .build()
    )


def _agents_semantic_registry(tmp_path: Path) -> None:
    SemanticToolRegistry(embed_fn=lambda texts: [[0.0, 1.0] for _ in texts])


def _agents_judge_with_a_backend(tmp_path: Path) -> None:
    default_llm_judge(complete_fn=lambda system, user: "{}")


def _meta_agent_starts(tmp_path: Path) -> None:
    assert MetaBuilderAgent(llm_interface=_double()).start()


def _meta_artifact_builder(tmp_path: Path) -> None:
    builder = AgentArtifactBuilder()
    builder.set_overview("a", "an agent")
    builder.set_config(temperature=0.3)
    builder.validate_partial()


_NO_MODEL_NEEDED: dict[str, Any] = {
    "core: API.from_definition(llm_interface=) converses": _core_api,
    "core: FSMManager(llm_interface=)": _core_fsm_manager,
    "core: Classifier(llm_interface=)": _core_classifier,
    "agents: default_agent_config(injected interface)": _agents_config_injected,
    "agents: AgentConfig(model=named)": _agents_config_named,
    "agents: SOPRegistry.register, overrides name no model": (
        _sop_register_without_model_key
    ),
    "agents: load_builtin_sops()": _sop_builtin_registry,
    "agents: SOPRegistry.register_from_dict and register_from_file": (
        _sop_from_dict_and_file
    ),
    "agents: SOPDefinition.to_agent_config(llm_interface=)": (
        _sop_to_agent_config_injected
    ),
    "agents: SOPDefinition.to_agent_config(model=)": _sop_to_agent_config_named,
    "agents: ConfiguredAgentBuilder with an injected interface": (
        _agents_configured_builder
    ),
    "eval: run_cases(llm_interface_factory=)": _eval_run_cases_factory,
    "eval: run_dataset(llm_interface_factory=)": _eval_run_dataset_factory,
    "reasoning: ReasoningEngine(llm_interface=)": _reasoning_engine_injected,
    "workflows: AgentStep around an injected agent": _workflows_agent_step,
    "monitor: InstanceManager, attach_api, launch_workflow": _monitor_construction,
    "synthesis: FSMSynthesizer(injected llm)": _synthesis_synthesizer_injected,
    "synthesis: SynthesizeExecutor, create_procedure_tools, Simulator": (
        _synthesis_executor_and_tools
    ),
    "agents: run() of react and debate on a double": _agents_run_with_a_double,
    "agents: AgentGraphBuilder with an injected-interface node": _agents_graph,
    "agents: SemanticToolRegistry(embed_fn=)": _agents_semantic_registry,
    "agents: default_llm_judge(complete_fn=)": _agents_judge_with_a_backend,
    "meta: MetaBuilderAgent(llm_interface=).start()": _meta_agent_starts,
    "meta: AgentArtifactBuilder.set_config": _meta_artifact_builder,
}

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
    "swarm",
    "parallel_react",
    "native_fc",
    "verified_react",
    "auto_memory",
    "reasoning_react",
]
for _pattern in _PATTERNS:
    if _pattern == "swarm":
        continue  # takes no llm_interface: its own rows below
    _NO_MODEL_NEEDED[f"agents: create_agent({_pattern!r}, llm_interface=)"] = (
        _every_pattern(_pattern)
    )


def _swarm_direct(tmp_path: Path) -> None:
    """The members hold their own interfaces; the swarm owns no model (D-025)."""
    member = create_agent("debate", llm_interface=_double())
    SwarmAgent(agents={"m": member}, entry_agent="m")


def _swarm_through_create_agent(tmp_path: Path) -> None:
    member = create_agent("debate", llm_interface=_double())
    assert isinstance(
        create_agent("swarm", agents={"m": member}, entry_agent="m"), SwarmAgent
    )


_NO_MODEL_NEEDED["agents: SwarmAgent(agents=, entry_agent=) over injected members"] = (
    _swarm_direct
)
_NO_MODEL_NEEDED["agents: create_agent('swarm', agents=, entry_agent=)"] = (
    _swarm_through_create_agent
)


class TestNoModelNeeded:
    """An injected interface (or no LLM at all) needs no model."""

    @pytest.mark.parametrize("name", sorted(_NO_MODEL_NEEDED))
    def test_builds_and_runs_without_llm_model(self, name: str, tmp_path: Path) -> None:
        _NO_MODEL_NEEDED[name](tmp_path)

    def test_every_create_agent_pattern_has_a_row(self) -> None:
        from fsm_llm.engine.executables.namespaces import PATTERN_AGENTS as registered

        assert set(_PATTERNS) == set(registered.list())
        for pattern in set(_PATTERNS) - {"swarm"}:
            assert f"agents: create_agent({pattern!r}, llm_interface=)" in (
                _NO_MODEL_NEEDED
            ), pattern
        assert any("create_agent('swarm'" in name for name in _NO_MODEL_NEEDED)

    def test_create_agent_swarm_takes_no_llm_interface_by_design(self) -> None:
        """Plan 0068b055 D-012 (anchor in ``swarm.py``): ``SwarmAgent`` builds no
        API (its members own theirs) and has no ``**api_kwargs``, so
        ``create_agent('swarm', llm_interface=...)`` is a plain ``TypeError``,
        not an accepted-and-inert keyword. The row for the other patterns is
        ``create_agent(<pattern>, llm_interface=)``; a swarm has none."""
        member = create_agent("debate", llm_interface=_double())
        with pytest.raises(TypeError, match="llm_interface"):
            create_agent(
                "swarm", agents={"m": member}, entry_agent="m", llm_interface=_double()
            )

    def test_the_variable_is_really_unset(self) -> None:
        import os

        assert os.environ.get(ENV_LLM_MODEL) is None


# ---------------------------------------------------------------------------
# Paths that still need a model when nothing is injected
# ---------------------------------------------------------------------------


def _need_api(tmp_path: Path) -> None:
    API.from_definition(_FSM)


def _need_classifier(tmp_path: Path) -> None:
    Classifier(_SCHEMA)


def _need_agent_config(tmp_path: Path) -> None:
    AgentConfig()


def _need_create_agent(tmp_path: Path) -> None:
    create_agent("react", [_echo])


def _need_sop_to_agent_config(tmp_path: Path) -> None:
    sop_module.SOPDefinition(name="s").to_agent_config()


def _need_sop_config_outliving_its_interface(tmp_path: Path) -> None:
    """A config made for an injected interface holds the placeholder; run on its
    own (no interface) the agent must say "no model", not reach a provider."""
    sop = sop_module.SOPDefinition(name="s", config_overrides={"temperature": 0.3})
    config = sop.to_agent_config(llm_interface=_double())
    agent = create_agent("react", [_echo], config=config)
    try:
        agent.run("hi")
    except AgentError as exc:
        raise exc.__cause__ or exc from None


def _need_reasoning_engine(tmp_path: Path) -> None:
    ReasoningEngine()


def _need_judge_without_backend(tmp_path: Path) -> None:
    default_llm_judge()


def _need_synthesizer_by_model(tmp_path: Path) -> None:
    FSMSynthesizer(model=resolve_model())


def _need_eval_run_cases(tmp_path: Path) -> None:
    case = ConversationCase(id="c", fsm=_FSM, turns=["hi"], expect={"final_state": "b"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cases([case], EvalConfig(trials=1, workers=1), run_dir)


def _need_eval_run_dataset(tmp_path: Path) -> None:
    run_dataset(_write_dataset(tmp_path), output_root=str(tmp_path / "out"))


def _need_monitor_launch_fsm(tmp_path: Path) -> None:
    InstanceManager().launch_fsm(fsm_json=_FSM)


def _need_monitor_launch_agent(tmp_path: Path) -> None:
    InstanceManager().launch_agent(
        task="x", tools_config=[StubToolConfig(name="t", description="d")]
    )


_MODEL_REQUIRED: dict[str, Any] = {
    "core: API.from_definition without an interface": _need_api,
    "core: Classifier without an interface": _need_classifier,
    "agents: AgentConfig()": _need_agent_config,
    "agents: create_agent without an interface": _need_create_agent,
    "agents: SOPDefinition.to_agent_config() for running": (_need_sop_to_agent_config),
    "agents: to_agent_config(llm_interface=) config used without the interface": (
        _need_sop_config_outliving_its_interface
    ),
    "reasoning: ReasoningEngine()": _need_reasoning_engine,
    "agents: default_llm_judge() without a backend": _need_judge_without_backend,
    "synthesis: FSMSynthesizer(model=resolve_model())": _need_synthesizer_by_model,
    "eval: run_cases without a factory": _need_eval_run_cases,
    "eval: run_dataset without a factory": _need_eval_run_dataset,
    "monitor: InstanceManager.launch_fsm": _need_monitor_launch_fsm,
    "monitor: InstanceManager.launch_agent": _need_monitor_launch_agent,
}


class TestModelStillRequired:
    """Nothing injected and nothing set: the one 'no model' error, never a hang."""

    @pytest.mark.parametrize("name", sorted(_MODEL_REQUIRED))
    def test_raises_model_not_configured(self, name: str, tmp_path: Path) -> None:
        with pytest.raises(ModelNotConfiguredError):
            _MODEL_REQUIRED[name](tmp_path)

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.parametrize("name", sorted(_MODEL_REQUIRED))
    def test_a_blank_model_counts_as_unset(
        self, name: str, blank: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, blank)
        with pytest.raises(ModelNotConfiguredError):
            _MODEL_REQUIRED[name](tmp_path)

    def test_the_eval_runners_build_no_run_directory_first(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ModelNotConfiguredError):
            _need_eval_run_dataset(tmp_path)
        assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# The CLIs
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]


def _scrubbed_env() -> dict[str, str]:
    """Only what a child needs; ``LLM_MODEL`` blank, no ``.env`` load."""
    keep = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env[ENV_LLM_MODEL] = ""
    env["LITELLM_MODE"] = "PRODUCTION"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(_REPO / "src")
    return env


def _child(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=_scrubbed_env(),
        cwd=cwd,
        check=False,
    )


def _one_line_exit_1(code: int, err: str) -> None:
    assert code == 1, err
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1, err
    assert ModelNotConfiguredError.DEFAULT_MESSAGE in lines[0]


class TestEveryCliWithoutAModel:
    """Exit 1 with the one message: the child processes run the real entry
    modules with a scrubbed environment, the eval rows call ``main_cli``."""

    def test_the_child_runs_the_checkout_under_test(self, tmp_path: Path) -> None:
        proc = _child(["-c", "import fsm_llm;print(fsm_llm.__file__)"], tmp_path)
        assert Path(proc.stdout.strip()).resolve().parent == (
            _REPO / "src" / "fsm_llm"
        ), proc.stderr

    def test_fsm_llm_run(self, tmp_path: Path) -> None:
        fsm_file = tmp_path / "f.json"
        fsm_file.write_text("{}", encoding="utf-8")
        proc = _child(["-m", "fsm_llm", "--fsm", str(fsm_file)], tmp_path)
        _one_line_exit_1(proc.returncode, proc.stderr)

    def test_reasoning(self, tmp_path: Path) -> None:
        proc = _child(["-m", "fsm_llm.reasoning", "what is 2 + 2"], tmp_path)
        _one_line_exit_1(proc.returncode, proc.stderr)

    def test_meta(self, tmp_path: Path) -> None:
        proc = _child(["-m", "fsm_llm.agents.meta_cli"], tmp_path)
        _one_line_exit_1(proc.returncode, proc.stderr)

    def test_eval_run(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        code = eval_cli.main_cli(["run", str(_write_dataset(tmp_path))])
        _one_line_exit_1(
            code, capsys.readouterr().err.replace("fsm-llm-eval: error: ", "")
        )

    def test_eval_examples(self, capsys: pytest.CaptureFixture) -> None:
        argv = [
            "examples",
            "--examples-dir",
            str(_REPO / "examples"),
            "--category",
            "basic",
        ]
        code = eval_cli.main_cli(argv)
        _one_line_exit_1(
            code, capsys.readouterr().err.replace("fsm-llm-eval: error: ", "")
        )

    def test_eval_names_no_default_model_in_its_help(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            eval_cli.main_cli(["run", "--help"])
        help_text = capsys.readouterr().out
        assert "bonsai" not in help_text and "ollama" not in help_text


# ---------------------------------------------------------------------------
# The monitor routes
# ---------------------------------------------------------------------------


class TestMonitorRoutes:
    """A launch with no model is HTTP 400 with the one message; the request
    models name no model themselves."""

    @staticmethod
    def _client() -> Any:
        from fastapi.testclient import TestClient

        from fsm_llm.monitor.server import app, configure

        configure(manager=InstanceManager())
        return TestClient(app)

    def test_request_models_name_no_model(self) -> None:
        from fsm_llm.monitor.definitions import (
            BuilderStartRequest,
            LaunchAgentRequest,
            LaunchFSMRequest,
        )

        assert LaunchFSMRequest(fsm_json=_FSM).model is None
        assert LaunchAgentRequest(task="x").model is None
        assert BuilderStartRequest().model is None

    @pytest.mark.parametrize("model", [None, " "])
    def test_fsm_launch_is_400(self, model: str | None) -> None:
        body: dict[str, Any] = {"fsm_json": _FSM}
        if model is not None:
            body["model"] = model
        resp = self._client().post("/api/fsm/launch", json=body)
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == ModelNotConfiguredError.DEFAULT_MESSAGE

    def test_agent_launch_is_400(self) -> None:
        body = {"task": "x", "tools": [{"name": "t", "description": "d"}]}
        resp = self._client().post("/api/agent/launch", json=body)
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == ModelNotConfiguredError.DEFAULT_MESSAGE

    def test_builder_start_is_400_and_builds_no_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fsm_llm.agents.meta_builder as meta

        built: list[object] = []
        monkeypatch.setattr(
            meta, "MetaBuilderAgent", lambda config: built.append(config)
        )
        resp = self._client().post("/api/builder/start", json={})
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == ModelNotConfiguredError.DEFAULT_MESSAGE
        assert built == []

    def test_a_named_model_still_launches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = self._client().post(
            "/api/fsm/launch", json={"fsm_json": _FSM, "model": "named/model"}
        )
        assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Precedence: the argument beats the environment, the environment beats nothing
# ---------------------------------------------------------------------------


def _api_model(model: str | None) -> str | None:
    return interface_model(API.from_definition(_FSM, model=model).llm_interface)


def _classifier_model(model: str | None) -> str | None:
    return Classifier(_SCHEMA, model=model).model


def _agent_config_model(model: str | None) -> str | None:
    return AgentConfig(model=model).model if model else AgentConfig().model


def _reasoning_model(model: str | None) -> str | None:
    return ReasoningEngine(model=model).model


def _eval_label(model: str | None) -> str:
    from fsm_llm.eval.cases import _run_model_label

    return _run_model_label(EvalConfig(model=model), None)


_PRECEDENCE = {
    "core: API": _api_model,
    "core: Classifier": _classifier_model,
    "agents: AgentConfig": _agent_config_model,
    "reasoning: ReasoningEngine": _reasoning_model,
    "eval: the run label": _eval_label,
}


class TestPrecedence:
    @pytest.mark.parametrize("name", sorted(_PRECEDENCE))
    def test_the_environment_is_used_without_an_argument(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert _PRECEDENCE[name](None) == "env/model"

    @pytest.mark.parametrize("name", sorted(_PRECEDENCE))
    def test_the_argument_beats_the_environment(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert _PRECEDENCE[name]("arg/model") == "arg/model"


# ---------------------------------------------------------------------------
# No source module names a default
# ---------------------------------------------------------------------------

_SRC = _REPO / "src" / "fsm_llm"
_REMOVED_NAMES = frozenset({"resolve_tool_model", "DEFAULT_TOOL_MODEL"})
_REMOVED_DEFAULTS = frozenset({"DEFAULT_LLM_MODEL"})


def _named_in_code(path: Path) -> set[str]:
    """Identifiers of the removed defaults (not docstrings or comments), and any
    code string constant that names the old default model."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.alias):
            name = node.name
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        if name in _REMOVED_NAMES | _REMOVED_DEFAULTS:
            found.add(name)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and "bonsai" in node.value
        ):
            found.add("a string constant naming bonsai")
    return found


class TestNoPathNamesAModelTheCallerDidNotSet:
    def test_no_module_names_a_removed_default_in_code(self) -> None:
        hits = {
            path.relative_to(_SRC).as_posix(): sorted(names)
            for path in sorted(_SRC.rglob("*.py"))
            if (names := _named_in_code(path))
        }
        assert hits == {}

    def test_the_removed_names_are_gone_from_their_modules(self) -> None:
        import fsm_llm.constants
        import fsm_llm.eval
        import fsm_llm.eval.config
        import fsm_llm.llm.interface

        assert not hasattr(fsm_llm.constants, "DEFAULT_TOOL_MODEL")
        assert not hasattr(fsm_llm.constants, "DEFAULT_LLM_MODEL")
        assert not hasattr(fsm_llm.llm.interface, "resolve_tool_model")
        assert not hasattr(fsm_llm.eval, "resolve_model")
        assert not hasattr(fsm_llm.eval.config, "resolve_model")

    def test_the_scan_sees_a_planted_default(self, tmp_path: Path) -> None:
        planted = tmp_path / "m.py"
        planted.write_text(
            'from x import resolve_tool_model\nmodel = "ollama_chat/bonsai-2:27b"\n',
            encoding="utf-8",
        )
        assert _named_in_code(planted) == {
            "resolve_tool_model",
            "a string constant naming bonsai",
        }

    def test_resolve_model_still_raises_with_nothing_set(self) -> None:
        with pytest.raises(ModelNotConfiguredError):
            resolve_model()
