"""``PatternExecutable.preview_definition()``: the FSM of a pattern by executable id (plan 659c07c3 step 2).

The preview is a fresh dict built with no agent, model, handler, ``HumanInTheLoop``, event log or caller
text (D-005 of this plan; D-061 of plan ace7a8c8 keeps the "no grant" side). The G1 golden of
``tests/fixtures/agent_preview_golden.json`` was written from the PARENT commit, so equality to it is the
differential proof that the seam draws what the file route and the builders drew before it existed.
Doubles only: a patched ``completion`` that raises, no provider, no network.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import build_fsm_graph
from fsm_llm.agents import executables as module
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.executables import (
    PatternExecutable,
    builtin_executables,
    pattern_adapter,
    preset_executables,
)
from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.definitions.executable import InstantiationError, PatternEnvelope
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolRegistry
from tests.test_fsm_llm_agents.agent_preview_cells import graph_of

GOLDEN = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/agent_preview_golden.json"
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["g1"]


def _all() -> dict[str, PatternExecutable]:
    return {e.id: e for e in [*builtin_executables(), *preset_executables()]}


def _graph(definition: dict[str, Any]) -> dict[str, Any]:
    return graph_of(definition)


class TestEveryIdDraws:
    def test_the_set_is_the_18_ids_and_the_3_presets(self):
        assert len(_all()) == 21

    def test_every_id_but_swarm_returns_a_dict_and_swarm_none(self):
        previews = {name: exe.preview_definition() for name, exe in _all().items()}

        assert previews.pop("swarm") is None
        assert len(previews) == 20
        assert all(isinstance(p, dict) for p in previews.values())

    def test_each_preview_loads_through_fsm_definition(self):
        for name, exe in _all().items():
            definition = exe.preview_definition()
            if definition is None:
                continue
            loaded = FSMDefinition.model_validate(definition)
            graph = build_fsm_graph(loaded)
            assert sum(1 for n in graph.nodes if n.is_initial) == 1, name
            assert any(n.is_terminal for n in graph.nodes), name

    def test_the_graph_equals_the_parent_golden_for_every_id(self, golden):
        assert set(golden) == set(_all())
        for name, exe in _all().items():
            definition = exe.preview_definition()
            cell = golden[name]
            if cell["route"] == "none":
                assert definition is None, name
            else:
                assert _graph(definition) == cell["graph"], name

    def test_the_route_of_each_id_is_the_golden_route(self, golden):
        for name, exe in _all().items():
            expected = "file" if exe.envelope is not None else "builder"
            if name == "swarm":
                expected = "none"
            assert golden[name]["route"] == expected, name

    def test_the_placeholders_are_the_variables_in_angle_brackets(self, golden):
        for name, exe in _all().items():
            if exe.envelope is None:
                continue
            text = json.dumps(exe.preview_definition(), sort_keys=True)
            found = sorted(n for n in exe.envelope.variables if f"<{n}>" in text)
            assert found == golden[name]["placeholders"], name

    def test_a_default_run_fsm_equals_the_preview_for_every_comparable_id(self, golden):
        compared = 0
        for name, exe in _all().items():
            real = golden[name]["real"]
            if real is None or name == "native_fc":
                continue
            assert _graph(exe.preview_definition()) == real["graph"], name
            compared += 1
        assert compared == 15

    def test_native_fc_draws_the_file_machine_of_5_states_against_3_in_a_run(
        self, golden
    ):
        preview = _all()["native_fc"].preview_definition()

        assert len(preview["states"]) == 5
        assert golden["native_fc"]["real"]["states"] == 3


class TestAFreshDictPerCall:
    @pytest.mark.parametrize(
        "name", ["react", "prompt_chain", "adapt", "reasoning_react"]
    )
    def test_two_calls_are_equal_and_share_nothing(self, name):
        exe = _all()[name]
        first, second = exe.preview_definition(), exe.preview_definition()

        assert first == second
        first["states"].clear()
        first["name"] = "changed"
        assert exe.preview_definition() == second
        assert second["states"]

    def test_the_envelope_is_not_changed_by_a_preview(self):
        exe = _all()["react"]
        before = exe.envelope.model_dump_json()

        exe.preview_definition()["states"].clear()

        assert exe.envelope.model_dump_json() == before

    def test_the_method_takes_no_argument_so_no_caller_text_can_reach_it(self):
        parameters = inspect.signature(PatternExecutable.preview_definition).parameters

        assert list(parameters) == ["self"]


class TestNothingIsExecuted:
    def test_no_model_agent_hitl_event_log_or_tool_runs(self):
        from fsm_llm.runtime import EventLog

        def refuse(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("a preview must run nothing")

        with (
            mock.patch("fsm_llm.llm.providers.litellm.completion", refuse),
            mock.patch.object(module, "create_agent", refuse),
            mock.patch.object(module, "PatternAgent", refuse),
            mock.patch.object(HumanInTheLoop, "__init__", refuse),
            mock.patch.object(EventLog, "__init__", refuse),
            mock.patch.object(ToolRegistry, "execute", refuse),
        ):
            definitions = [e.preview_definition() for e in _all().values()]

        assert sum(d is not None for d in definitions) == 20

    def test_a_file_executable_reads_no_file_again(self):
        files = [e for e in _all().values() if e.envelope is not None]
        reads: list[str] = []

        def spy(name: str, *_a: Any, **_k: Any) -> Any:
            reads.append(name)
            raise AssertionError("the envelope is already loaded")

        with (
            mock.patch.object(pf, "load_envelope", spy),
            mock.patch.object(pf, "load_document", spy),
            mock.patch.object(module, "load_envelope", spy),
        ):
            assert all(e.preview_definition() for e in files)

        assert len(files) == 15
        assert reads == []

    @pytest.mark.parametrize("name", ["parallel_react", "reasoning_react"])
    def test_the_two_builders_that_need_a_registry_get_one_stub_tool(self, name):
        text = json.dumps(_all()[name].preview_definition())

        assert "lookup" in text

    def test_the_stub_tool_of_the_builders_is_never_flagged(self):
        flags: list[Any] = []
        original = ToolRegistry.register_function

        def spy(self, fn, *args, **kwargs):
            flags.append((kwargs.get("requires_approval", False), args[2:3]))
            return original(self, fn, *args, **kwargs)

        with mock.patch.object(ToolRegistry, "register_function", spy):
            for name in ("parallel_react", "reasoning_react"):
                _all()[name].preview_definition()

        assert flags
        assert not any(requires for requires, _ in flags)


def _user_doc(doc_id: str = "user.loop") -> dict[str, Any]:
    doc = pf.load_document("evaluator_optimizer")
    doc["id"] = doc_id
    return doc


class TestAUserFile:
    def test_a_user_pattern_file_draws_like_a_shipped_one(self):
        envelope = pf.load_envelope("react")
        shipped = pattern_adapter(envelope).preview_definition()
        doc = pf.load_document("react")
        doc["id"] = "user.react"
        user = pattern_adapter(PatternEnvelope.model_validate(doc))

        assert _graph(user.preview_definition()) == _graph(shipped)
        assert user.preview_definition()["states"]

    def test_a_file_that_cannot_instantiate_raises_its_typed_error(self):
        doc = pf.load_document("prompt_chain")
        doc["id"] = "user.chain"
        doc["body"]["generator"] = "user.nowhere"
        exe = pattern_adapter(PatternEnvelope.model_validate(doc))

        with pytest.raises(InstantiationError):
            exe.preview_definition()

    def test_a_third_party_adapter_with_a_static_file_still_draws(self):
        exe = pattern_adapter(PatternEnvelope.model_validate(_user_doc()))

        assert isinstance(exe.preview_definition(), dict)
