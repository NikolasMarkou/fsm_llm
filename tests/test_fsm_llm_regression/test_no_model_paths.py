"""The no-model paths: what needs a model and what must not (review gate 1).

The library has no built-in model (plan d454fd62 D-005): ``llm.resolve_model``
raises ``ModelNotConfiguredError`` when nothing names one. The documented rule
is "an injected ``llm_interface`` needs no model". The root autouse fixture of
``tests/conftest.py`` sets ``LLM_MODEL`` for every test, which hid two
regressions (the SOP registry and the eval runners, review gate 1 concerns 1
and 2). Every case below deletes ``LLM_MODEL`` itself and then builds or runs a
public entry point of every package that takes an injected interface (or that
needs no LLM at all):

- ``TestNoModelNeeded``: with an injected double (or no LLM) the call works;
- ``TestModelStillRequired``: with nothing injected and nothing set the call
  raises ``ModelNotConfiguredError``, the one message that says how to set one.

A new public constructor or runner that takes ``llm_interface`` gets one row in
``_NO_MODEL_NEEDED`` (and, when it can also build its own interface, one row in
``_MODEL_REQUIRED``). Fix a failing row in ``src/``, never here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API, Classifier, FSMManager, ModelNotConfiguredError
from fsm_llm.agents import (
    AgentConfig,
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
from fsm_llm.definitions import ClassificationSchema, IntentDefinition
from fsm_llm.eval import ConversationCase, EvalConfig, run_cases, run_dataset
from fsm_llm.llm import resolve_model
from fsm_llm.monitor import InstanceManager
from fsm_llm.reasoning import ReasoningEngine
from fsm_llm.synthesis import (
    FSMSynthesizer,
    ProcedureRunner,
    Simulator,
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
    from fsm_llm.definitions import FSMDefinition

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
    from fsm_llm.monitor import instance_manager

    return list(instance_manager._WORKFLOW_PRESETS)


def _synthesis_synthesizer_injected(tmp_path: Path) -> None:
    FSMSynthesizer(_double())


def _synthesis_runner_and_tools(tmp_path: Path) -> None:
    synthesizer = FSMSynthesizer(_double())
    ProcedureRunner(
        API.from_definition(_FSM, llm_interface=_double()), _FSM, synthesizer
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
    "synthesis: ProcedureRunner, create_procedure_tools, Simulator": (
        _synthesis_runner_and_tools
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
        from fsm_llm.agents import _PATTERNS as registered

        assert set(_PATTERNS) == set(registered)
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


def _need_reasoning_engine(tmp_path: Path) -> None:
    ReasoningEngine()


def _need_eval_run_cases(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_cases([], EvalConfig(trials=1, output_dir=str(run_dir)), run_dir)


def _need_eval_run_dataset(tmp_path: Path) -> None:
    run_dataset(_write_dataset(tmp_path), output_root=str(tmp_path / "out"))


def _need_monitor_launch_fsm(tmp_path: Path) -> None:
    InstanceManager().launch_fsm(fsm_json=_FSM)


def _need_judge_without_backend(tmp_path: Path) -> None:
    default_llm_judge()


def _need_synthesizer_by_model(tmp_path: Path) -> None:
    FSMSynthesizer(model=resolve_model())


_MODEL_REQUIRED: dict[str, Any] = {
    "core: API.from_definition without an interface": _need_api,
    "core: Classifier without an interface": _need_classifier,
    "agents: AgentConfig()": _need_agent_config,
    "agents: create_agent without an interface": _need_create_agent,
    "agents: SOPDefinition.to_agent_config() for running": (_need_sop_to_agent_config),
    "reasoning: ReasoningEngine()": _need_reasoning_engine,
    "eval: run_cases without a factory": _need_eval_run_cases,
    "eval: run_dataset without a factory": _need_eval_run_dataset,
    "monitor: InstanceManager.launch_fsm": _need_monitor_launch_fsm,
    "agents: default_llm_judge() without a backend": _need_judge_without_backend,
    "synthesis: FSMSynthesizer(model=resolve_model())": _need_synthesizer_by_model,
}


class TestModelStillRequired:
    """Nothing injected and nothing set: the one 'no model' error, never a hang."""

    @pytest.mark.parametrize("name", sorted(_MODEL_REQUIRED))
    def test_raises_model_not_configured(self, name: str, tmp_path: Path) -> None:
        with pytest.raises(ModelNotConfiguredError):
            _MODEL_REQUIRED[name](tmp_path)
