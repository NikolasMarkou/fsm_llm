"""``create_agent`` through the registry (plan a555dcfe step 19, D-030; roadmap E4.1 and E4.3).

A pattern file held by an ``ExecutableRegistry`` is built by ``create_agent(id, registry=)`` as a
``PatternAgent``; a built-in id builds its class from the ``pattern_agents`` namespace and never
consults the registry; the class table is one object (no private ``_PATTERNS`` anywhere in src).
The fail-closed rules are those of ``PatternAgent`` (census D1) and are asserted here through the
public entry. The directory part (shadowing, the shared loader) is step 20.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.agents import (
    EvaluatorOptimizerAgent,
    ReactAgent,
    create_agent,
)
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.namespaces import PATTERN_AGENTS
from tests import executables_corpus as corpus
from tests import pattern_runs as runs
from tests.conftest import symlink_or_skip

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"
_FIRST_FOUR = [
    "agents.tool_executor",
    "agents.think_turn_limiter",
    "agents.think_fresh_keys",
    "agents.feedback_consumed",
]


def _loop_doc(doc_id: str = "user.loop", *, gate: bool = False) -> dict[str, Any]:
    """The react file under a new id; without the gate it keeps the four loop handlers."""
    doc = pf.load_document("react")
    doc["id"] = doc_id
    if not gate:
        doc["capabilities"] = ["tools"]
        doc["requires"]["handlers"] = list(_FIRST_FOUR)
        doc["body"]["handlers"] = doc["body"]["handlers"][:4]
        doc["body"]["variants"].pop("approval")
    return doc


def _static_doc(doc_id: str = "user.static") -> dict[str, Any]:
    """The evaluator_optimizer file (requires.tools none, one required callable) under a new id."""
    doc = pf.load_document("evaluator_optimizer")
    doc["id"] = doc_id
    return doc


def _lookup(query: str) -> str:
    """Look it up."""
    return "x"


def _registry(*docs: dict[str, Any]) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    for doc in docs:
        registry.add_document(doc)
    return registry


def _make(pattern: str, registry: ExecutableRegistry | None, **kw: Any) -> Any:
    kw.setdefault("config", runs._config())
    kw.setdefault("llm_interface", runs.ScriptedModel())
    return create_agent(pattern, kw.pop("tools", None), registry=registry, **kw)


class TestAUserFileThroughCreateAgent:
    def test_the_id_builds_a_pattern_agent_over_the_file(self):
        registry = _registry(_loop_doc())

        agent = _make("user.loop", registry, tools=runs._tools())

        assert type(agent) is PatternAgent
        assert agent.envelope.id == "user.loop"
        assert agent.tools is not None

    def test_the_agent_runs_a_tool_loop_end_to_end(self):
        executions: list[str] = []
        from tests.test_fsm_llm_agents.test_pattern_agent import _counting_registry

        registry = _registry(_loop_doc())
        agent = _make(
            "user.loop",
            registry,
            tools=_counting_registry(executions),
            llm_interface=runs.ScriptedModel(),
        )

        result = agent.run("find it")

        assert result.success and result.status is Status.OK
        assert executions and set(executions) == {"search"}
        assert result.run_result().detail["tools_used"] == ["search"]

    def test_a_file_in_a_directory_is_found_by_a_registry_built_from_paths(
        self, tmp_path: Path
    ):
        (tmp_path / "user_loop.json").write_text(
            json.dumps(_loop_doc()), encoding="utf-8"
        )

        registry = ExecutableRegistry.from_paths([tmp_path])
        agent = _make("user.loop", registry, tools=runs._tools())

        assert type(agent) is PatternAgent

    def test_a_list_of_functions_and_system_prompt_reach_the_agent(self):
        registry = _registry(_loop_doc())

        agent = _make("user.loop", registry, tools=[_lookup], system_prompt="Be brief.")

        assert agent.config.instructions == "Be brief."
        assert agent.tools.tool_names == ["_lookup"]

    def test_the_caller_options_and_callables_reach_the_file(self):
        registry = _registry(_static_doc())

        agent = _make(
            "user.static",
            registry,
            options={"max_refinements": 1},
            callables={"evaluation_fn": lambda *a: None},
        )

        assert type(agent) is PatternAgent
        assert agent._options == {"max_refinements": 1}


class TestFailClosedThroughTheEntry:
    def test_a_flagged_tool_and_a_file_with_no_gate_is_refused_at_creation(self):
        registry = _registry(_loop_doc())
        model = runs.ScriptedModel()

        with pytest.raises(AgentError, match="no human-in-the-loop approval"):
            _make(
                "user.loop",
                registry,
                tools=runs._tools(flagged=True),
                llm_interface=model,
            )
        assert model.log == []

    def test_a_gate_and_nobody_to_approve_is_refused_at_creation(self):
        registry = _registry(_loop_doc("user.gated", gate=True))

        with pytest.raises(AgentError, match="nobody can approve"):
            _make("user.gated", registry, tools=runs._tools(flagged=True))

    def test_a_gate_with_a_hitl_builds_and_a_hitl_without_a_gate_is_refused(self):
        registry = _registry(_loop_doc("user.gated", gate=True), _loop_doc())

        built = _make(
            "user.gated",
            registry,
            tools=runs._tools(flagged=True),
            hitl=runs._hitl(),
        )
        assert type(built) is PatternAgent
        with pytest.raises(AgentError, match="could never be consulted"):
            _make("user.loop", registry, tools=runs._tools(), hitl=runs._hitl())

    def test_a_tool_free_file_refuses_tools_and_takes_an_empty_list(self):
        registry = _registry(_static_doc())
        callables = {"evaluation_fn": lambda *a: None}

        with pytest.raises(TypeError, match="does not take tools"):
            _make("user.static", registry, tools=[_lookup], callables=callables)
        agent = _make("user.static", registry, tools=[], callables=callables)
        assert type(agent) is PatternAgent and agent.tools is None


class TestBuiltInsAndTheTable:
    def test_a_registry_lacking_react_still_builds_react(self):
        for registry in (None, ExecutableRegistry(), _registry(_loop_doc())):
            agent = _make("react", registry, tools=runs._tools())
            assert type(agent) is ReactAgent

    def test_every_built_in_id_resolves_through_the_namespace(self):
        for name, cls in PATTERN_AGENTS.items():
            assert PATTERN_AGENTS.resolve(name) is cls
        assert PATTERN_AGENTS.get("react") is ReactAgent
        assert len(PATTERN_AGENTS.list()) == len(set(PATTERN_AGENTS.list()))

    def test_normalisation_of_a_built_in_name_is_kept_and_the_registry_is_not_read(
        self,
    ):
        registry = _registry(_static_doc())

        agent = _make(" Evaluator_Optimizer ", registry, evaluation_fn=lambda *a: None)

        assert type(agent) is EvaluatorOptimizerAgent

    def test_a_user_file_named_like_a_built_in_never_replaces_it(self):
        registry = _registry(_loop_doc("react"))

        agent = _make("react", registry, tools=runs._tools())

        assert type(agent) is ReactAgent

    def test_the_unknown_id_error_lists_built_ins_and_the_registry_ids(self):
        registry = _registry(_loop_doc())

        with pytest.raises(ValueError, match=r"user\.loop") as with_registry:
            _make("nope", registry)
        with pytest.raises(ValueError, match="react") as without:
            _make("user.loop", None)

        assert "Unknown pattern 'nope'" in str(with_registry.value)
        assert "user.loop" not in str(without.value).split("Available:")[1]

    def test_an_executable_that_is_not_a_pattern_is_unknown(self):
        case = next(name for name in corpus.fsm_cases() if name.startswith("agents/"))
        registry = _registry(corpus.wrap(case))
        assert registry.get(case).kind != "pattern"

        with pytest.raises(ValueError, match="Unknown pattern"):
            _make(case, registry)


class TestOneClassTable:
    def test_no_private_patterns_name_exists_in_src(self):
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else node.name
                    if isinstance(node, ast.alias)
                    else ""
                )
                if name == "_PATTERNS":
                    offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")
        assert offenders == []

    def test_the_table_keeps_the_order_of_the_dict_it_replaced(self):
        assert [name for name, _ in PATTERN_AGENTS.items()] == [
            "react", "rewoo", "debate", "plan_execute", "prompt_chain",
            "self_consistency", "orchestrator", "adapt", "evaluator_optimizer",
            "maker_checker", "reflexion", "meta_builder", "swarm", "parallel_react",
            "native_fc", "verified_react", "auto_memory", "reasoning_react",
        ]  # fmt: skip

    def test_the_executables_follow_the_same_table(self):
        from fsm_llm.agents.executables import builtin_executables

        assert [e.id for e in builtin_executables()] == [
            name for name, _ in PATTERN_AGENTS.items()
        ]


class TestTheSharedDirectoryHelper:
    """Plan a555dcfe step 20 (D-031): one loader for ``--executables DIR`` (built-ins first)."""

    @staticmethod
    def _load(directory: Path):
        from fsm_llm.agents.executables import builtin_executables
        from fsm_llm.engine.executables.directory import load_directory_registry

        return load_directory_registry(directory, builtin_executables(), prog="prog-x")

    @staticmethod
    def _put(directory: Path, name: str, doc: Any, *, bom: bool = False) -> None:
        text = ("﻿" if bom else "") + json.dumps(doc)
        (directory / name).write_text(text, encoding="utf-8")

    def test_a_bom_file_and_an_upper_case_suffix_load(self, tmp_path: Path):
        self._put(tmp_path, "bom.json", _loop_doc("user.bom"), bom=True)
        self._put(tmp_path, "UP.JSON", _loop_doc("user.up"))

        registry = self._load(tmp_path)

        assert {"user.bom", "user.up"} <= {e.id for e in registry.list(kind="pattern")}

    def test_a_garbage_file_is_skipped_with_one_stderr_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")
        self._put(tmp_path, "good.json", _loop_doc("user.good"))

        registry = self._load(tmp_path)

        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("prog-x: warning: ") and "garbage.json" in lines[0]
        assert registry.get("user.good").kind == "pattern"

    def test_a_file_named_like_a_built_in_is_refused_and_the_built_in_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        shadow = _loop_doc("react")
        self._put(tmp_path, "react.json", shadow)

        registry = self._load(tmp_path)

        err = capsys.readouterr().err.splitlines()
        assert len(err) == 1 and "react.json" in err[0] and "react" in err[0]
        assert (
            registry.get("react").envelope.body.handlers != shadow["body"]["handlers"]
        )
        assert (
            type(create_agent("react", runs._tools(), registry=registry, **_kw()))
            is ReactAgent
        )

    def test_a_file_that_differs_from_a_built_in_only_in_case_is_refused_with_one_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Review of plan 659c07c3 (probe1): ``create_agent`` strips and lowercases a built-in id,
        so a file ``React`` was listed and drawn from the file and built as the built-in."""
        self._put(tmp_path, "React.json", _loop_doc("React"))

        registry = self._load(tmp_path)

        err = capsys.readouterr().err.splitlines()
        assert len(err) == 1 and "React.json" in err[0] and "only in case" in err[0]
        assert "React" not in {e.id for e in registry.list(kind="pattern")}
        assert type(registry.get("react")).__name__ == "PatternExecutable"

    def test_two_files_equal_after_casefold_load_the_first_one_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        self._put(tmp_path, "a_upper.json", _loop_doc("user.Loop"))
        self._put(tmp_path, "b_lower.json", _loop_doc("user.loop"))

        registry = self._load(tmp_path)

        ids = {e.id for e in registry.list(kind="pattern")}
        assert ids >= {"user.Loop"} and "user.loop" not in ids
        err = capsys.readouterr().err.splitlines()
        assert len(err) == 1 and "b_lower.json" in err[0]

    def test_a_symlink_is_skipped_and_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        real = tmp_path / "elsewhere"
        real.mkdir()
        self._put(real, "t.json", _loop_doc("user.t"))
        directory = tmp_path / "dir"
        directory.mkdir()
        symlink_or_skip(directory / "link.json", real / "t.json")

        registry = self._load(directory)

        assert "link.json" in capsys.readouterr().err
        assert "user.t" not in {e.id for e in registry.list(kind="pattern")}

    def test_a_missing_directory_is_a_file_not_found_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="--executables"):
            self._load(tmp_path / "nope")

    def test_a_user_file_loads_as_operator_input_not_as_model_output(
        self, tmp_path: Path
    ):
        doc = _loop_doc("user.operator")
        assert doc["requires"]["handlers"]  # a model-origin document may bind none (V7)
        self._put(tmp_path, "op.json", doc)

        registry = self._load(tmp_path)

        assert registry.get("user.operator").envelope.requires.handlers


def _kw() -> dict[str, Any]:
    return {"config": runs._config(), "llm_interface": runs.ScriptedModel()}
