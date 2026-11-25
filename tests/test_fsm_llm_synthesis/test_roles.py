"""The roles of a simulated compile: where they appear, where the refusal is raised.

Plan c720601d, step 6 (D-009). Everything runs on scripted doubles.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from fsm_llm.simulation import (
    CODE_JUDGE_NOT_DISTINCT,
    Roles,
    SimulationError,
    Simulator,
)
from fsm_llm.synthesis import (
    FSMLibrary,
    FSMSynthesizer,
    SynthesisPolicy,
    SynthesisRefusedError,
    create_procedure_tools,
    repair_procedure,
    run_gates,
)
from fsm_llm.synthesis.constants import CODE_JUDGE_NOT_DISTINCT as SYNTHESIS_CODE
from fsm_llm.synthesis.simulation import replay_g6b, run_g6b
from fsm_llm.testing import ScriptedLLM
from fsm_llm.tools import ToolCall
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, good_fsm

GOAL = "Collect one answer"
SCENARIOS = json.dumps({"scenarios": [{"kind": "off_topic", "turns": ["hello there"]}]})


def _writer(model: str | None = "judge") -> ScriptedLLM:
    return ScriptedLLM([SCENARIOS] * 4, model=model)


def _synth(valid_fsm: dict[str, Any], tmp_path: Any, **kw: Any) -> FSMSynthesizer:
    compiler = ScriptedLLM([json.dumps(valid_fsm)] * 2, model="compiler")
    sim = Simulator(ScriptedRunnerLLM({"answer": "42"}), _writer())
    return FSMSynthesizer(
        compiler, library=FSMLibrary(tmp_path / "lib"), simulator=sim, **kw
    )


class TestTheCode:
    def test_the_synthesis_code_is_the_simulation_code(self):
        assert SYNTHESIS_CODE == CODE_JUDGE_NOT_DISTINCT == "judge_not_distinct"

    def test_it_is_a_gate_code_and_not_a_reason(self):
        from fsm_llm.synthesis.constants import REASON_COMPILE_FAILED, REASONS
        from fsm_llm.synthesis.procedures import reason_of

        assert CODE_JUDGE_NOT_DISTINCT not in REASONS
        refused = SynthesisRefusedError("x", CODE_JUDGE_NOT_DISTINCT)
        assert reason_of(refused) == REASON_COMPILE_FAILED


class TestWhereTheRolesAppear:
    def test_the_report_the_compile_result_and_the_receipt_name_the_models(
        self, valid_fsm, tmp_path
    ):
        synth = _synth(valid_fsm, tmp_path)
        result = synth.compile(GOAL, return_keys=["answer"])
        named = Roles("compiler", "runner", "judge", False)
        assert result.roles == named
        assert result.simulation is not None
        assert result.simulation.roles == Roles(None, "runner", "judge", False)

        synth = _synth(valid_fsm, tmp_path / "b")
        tools = create_procedure_tools(synth, runner_llm=ScriptedRunnerLLM({}))
        call = ToolCall(
            tool_name="obtain_procedure",
            parameters={"goal": GOAL, "return_keys": ["answer"]},
        )
        text = tools.execute(call).result
        assert isinstance(text, str)
        for line in (
            "compiler: compiler",
            "executor: runner",
            "judge: judge",
            "same_model_allowed: False",
        ):
            assert line in text.splitlines(), text

    def test_the_flag_is_in_the_report(self, valid_fsm, tmp_path):
        sim = Simulator(
            ScriptedRunnerLLM({"answer": "42"}), _writer(None), allow_same_model=True
        )
        compiler = ScriptedLLM([json.dumps(valid_fsm)], model="compiler")
        result = FSMSynthesizer(compiler, simulator=sim).compile(
            GOAL, return_keys=["answer"]
        )
        assert result.simulation is not None
        assert result.simulation.roles == Roles(None, "runner", None, True)

    def test_a_compile_without_a_simulator_has_no_roles(self, valid_fsm):
        compiler = ScriptedLLM([json.dumps(valid_fsm)])
        result = FSMSynthesizer(compiler).compile(GOAL, return_keys=["answer"])
        assert result.roles is None and result.simulation is None

    def test_a_library_hit_has_no_roles_and_the_receipt_no_role_lines(
        self, valid_fsm, tmp_path
    ):
        synth = _synth(valid_fsm, tmp_path)
        synth.obtain(GOAL, return_keys=["answer"])
        hit = synth.obtain(GOAL, return_keys=["answer"])
        assert hit.source == "library" and hit.roles is None

    def test_the_library_record_never_holds_a_role(self, valid_fsm, tmp_path):
        synth = _synth(valid_fsm, tmp_path)
        proc = synth.obtain(GOAL, return_keys=["answer"])
        assert proc.roles == Roles("compiler", "runner", "judge", False)
        assert proc.record is not None
        plain = FSMSynthesizer(
            ScriptedLLM([json.dumps(valid_fsm)], model="compiler"),
            library=FSMLibrary(tmp_path / "plain"),
        ).obtain(GOAL, return_keys=["answer"])
        assert plain.record is not None
        assert set(proc.record.provenance) == set(plain.record.provenance)
        raw = (synth.library.root / "records" / f"{proc.digest}.json").read_text(
            encoding="utf-8"
        )
        assert "judge" not in raw and "runner" not in raw


class TestTheRefusal:
    def test_the_synthesizer_refuses_a_judge_that_is_the_compiler(self, tmp_path):
        judge = ScriptedLLM([], model="same")
        sim = Simulator(ScriptedRunnerLLM(), judge)
        with pytest.raises(SynthesisRefusedError) as info:
            FSMSynthesizer(judge, simulator=sim)
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT
        assert judge.requests == []

    def test_an_unnamed_compiler_fails_closed(self):
        sim = Simulator(ScriptedRunnerLLM(), _writer())
        with pytest.raises(SynthesisRefusedError):
            FSMSynthesizer(ScriptedLLM([]), simulator=sim)

    def test_the_flag_lets_the_synthesizer_through(self):
        judge = ScriptedLLM([], model="same")
        sim = Simulator(ScriptedRunnerLLM(), judge, allow_same_model=True)
        FSMSynthesizer(judge, simulator=sim)

    def test_no_simulator_no_check(self):
        FSMSynthesizer(ScriptedLLM([]))

    def test_the_refusal_is_the_same_in_async_code(self):
        judge = ScriptedLLM([], model="same")
        sim = Simulator(ScriptedRunnerLLM(), judge)

        async def build() -> None:
            FSMSynthesizer(judge, simulator=sim)

        with pytest.raises(SynthesisRefusedError) as info:
            asyncio.run(build())
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT

    def test_repair_refuses_before_any_call(self, tmp_path):
        synth = FSMSynthesizer(ScriptedLLM([], model="same"))
        judge = ScriptedLLM([], model="same")
        runner = ScriptedRunnerLLM()
        sim = Simulator(runner, judge)
        with pytest.raises(SynthesisRefusedError) as info:
            repair_procedure(
                "0" * 64,
                [],
                library=FSMLibrary(tmp_path / "lib"),
                synthesizer=synth,
                simulator=sim,
            )
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT
        assert judge.requests == [] and runner.call_history == []


class TestAJudgeChangedAfterConstructionIsRefusedBeforeAnyCall:
    """Review concern 8 (probe p7): ``Simulator.executor`` and ``judge`` and ``FSMSynthesizer.llm``
    are plain attributes. A refusal that waited for the report came after every model call."""

    @staticmethod
    def _armed() -> tuple[Simulator, ScriptedRunnerLLM, ScriptedLLM]:
        runner, judge = ScriptedRunnerLLM({"answer": "42"}), _writer()
        sim = Simulator(runner, judge)
        sim.judge = runner  # the default the check exists against, set late
        return sim, runner, judge

    @staticmethod
    def _definition() -> Any:
        report = run_gates(
            good_fsm(), SynthesisPolicy(), inputs=("topic",), return_keys=("answer",)
        )
        return report.injected

    def test_run_g6b_refuses_before_the_writer_and_before_any_trial(self):
        sim, runner, judge = self._armed()
        with pytest.raises(SimulationError) as info:
            run_g6b(
                sim,
                self._definition(),
                goal=GOAL,
                inputs=("topic",),
                return_keys=("answer",),
                policy=SynthesisPolicy(),
            )
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT
        assert judge.requests == [] and runner.call_history == []

    def test_replay_g6b_refuses_before_any_trial(self):
        sim, runner, judge = self._armed()
        with pytest.raises(SimulationError) as info:
            replay_g6b(
                sim,
                self._definition(),
                [{"kind": "happy", "turns": ["hello"]}],
                inputs=("topic",),
                return_keys=("answer",),
                trials=1,
                policy=SynthesisPolicy(),
            )
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT
        assert judge.requests == [] and runner.call_history == []

    @pytest.mark.parametrize("use_async", [False, True])
    def test_a_compile_refuses_before_the_strong_model_is_called(
        self, valid_fsm, tmp_path, use_async: bool
    ):
        compiler = ScriptedLLM([json.dumps(valid_fsm)] * 2, model="compiler")
        runner = ScriptedRunnerLLM({"answer": "42"})
        sim = Simulator(runner, _writer())
        synth = FSMSynthesizer(compiler, simulator=sim)
        sim.judge = (
            compiler  # the judge is the compiler, set after the construction check
        )

        def go() -> Any:
            if use_async:
                return asyncio.run(
                    synth.acompile(GOAL, inputs=("topic",), return_keys=("answer",))
                )
            return synth.compile(GOAL, inputs=("topic",), return_keys=("answer",))

        with pytest.raises(SynthesisRefusedError) as info:
            go()
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT
        assert compiler.requests == [] and runner.call_history == []
