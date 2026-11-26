"""RunContext adoption by every runner (plan 7efd5cb4, roadmap row 1.14): the parity class.

Invariant 8 of the plan: with ``context=`` unset every runner is byte for byte what it was at the
plan base 51366ed3. ``tests/run_context_rig.collect_sections`` runs the parity matrix (PG-agent,
PG-reasoning, PG-workflow, PG-adapters: request bytes as sha256 of the normalised JSON, the stable
fields of every event, the results, the errors, the step numbers every ``before_step`` hook
received) and ``tests/fixtures/run_context_parity_golden.json`` holds what the same function
returned on a ``git archive`` of the base. The golden is the PARENT's, written by an out-of-repo
script that imports the archive copy: never regenerate it from the tree to make a row pass (a
difference is a runner that now sends, logs or counts something else; a deliberate change names the
decision and the section it moves).

The comparer is shown able to FAIL: ``TestTheComparerCanFail`` plants a changed request byte, an
extra hook call, an extra event and a dropped result field in a copy of the golden, and every one
is reported.

The file also holds one class per runner (the agent, the reasoning engine) and the cross-runner
acceptance (step 8): ``TestOneLog``, ``TestOneCancel``, ``TestOneBudget``,
``TestOneBudgetOverARunLoop``, ``TestThreeThreadsOneLedger`` and ``TestNoGrant`` run an agent, a
reasoning solve and a workflow instance (the workflow adapter, a step that runs an FSM executable
as a child) under ONE context and pin: one log, one cancellation, one ledger, no grant. The sweep
class at the end requires every run loop of the three packages to pass ``before_step``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from tests import run_context_rig as rig

REPO = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO / "tests" / "fixtures" / "run_context_parity_golden.json"
GOLDEN: dict[str, dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
SECTIONS = ("PG-adapters", "PG-agent", "PG-reasoning", "PG-workflow")


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")


def first_difference(got: Any, want: Any, path: str = "") -> str | None:
    """The path of the first place ``got`` differs from ``want``, or ``None`` when equal."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            if key not in got:
                return f"{path}/{key}: missing from the run"
            if key not in want:
                return f"{path}/{key}: not in the golden"
            found = first_difference(got[key], want[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(got, list) and isinstance(want, list):
        for index, (a, b) in enumerate(zip(got, want, strict=False)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        if len(got) != len(want):
            return f"{path}: {len(got)} items against {len(want)}"
        return None
    return None if got == want else f"{path}: {got!r} against {want!r}"


@pytest.fixture(scope="module")
def sections(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """The whole matrix, run once for the module (about 25 s: every case is a scripted solve)."""
    work = tmp_path_factory.mktemp("parity")
    return json.loads(json.dumps(rig.collect_sections(work)))


class TestTheGoldenIsNotVacuous:
    def test_every_section_is_there_and_holds_cases(self) -> None:
        assert tuple(sorted(GOLDEN)) == SECTIONS
        assert all(len(GOLDEN[name]) >= 5 for name in SECTIONS)

    def test_every_case_holds_a_result_or_an_error(self) -> None:
        for name in SECTIONS:
            for case, record in GOLDEN[name].items():
                assert ("result" in record) != ("error" in record), (name, case)

    def test_the_agent_section_reached_its_stop_paths(self) -> None:
        agent = GOLDEN["PG-agent"]
        assert agent["react_ok"]["result"]["stop_reason"] == "answered"
        assert agent["react_flagged_no_hitl"]["error"]["type"] == "AgentError"
        for case in ("react_max_iterations", "react_timeout", "react_tool_error"):
            assert "result" in agent[case] or "error" in agent[case], case
        assert agent["react_cut_reply"]["requests"]
        assert agent["react_stream"]["result"]["chunks"]
        assert agent["react_ok"]["runs"][0]["hook"] is True

    def test_the_reasoning_section_reached_the_solve_and_both_budget_errors(
        self,
    ) -> None:
        reasoning = GOLDEN["PG-reasoning"]
        assert reasoning["deductive_ok"]["result"]["visited"][-1] == "final_answer"
        assert reasoning["calculator_ok"]["result"]["solution"] == "36"
        assert "within 7 steps" in reasoning["never_ending_7"]["error"]["message"]
        assert reasoning["never_ending_7"]["error"]["details"] == [
            "conversation_id",
            "partial_context",
            "responses_so_far",
        ]
        assert "10 steps" in reasoning["classifier_budget"]["error"]["message"]

    def test_the_solve_loop_has_no_hook_and_its_stream_has_one(self) -> None:
        reasoning = GOLDEN["PG-reasoning"]
        assert [r["hook"] for r in reasoning["deductive_ok"]["runs"]] == [False, False]
        assert reasoning["stream_ok"]["runs"][0]["hook"] is True
        assert reasoning["stream_ok"]["runs"][0]["steps"]

    def test_the_workflow_and_adapter_sections_reached_their_outcomes(self) -> None:
        workflow = GOLDEN["PG-workflow"]
        assert workflow["linear"]["result"]["status"] == "completed"
        assert "Step budget exceeded" in workflow["step_budget"]["result"]["error"]
        adapters = GOLDEN["PG-adapters"]
        assert adapters["fsm_ok"]["result"]["status"] == "ok"
        assert adapters["fsm_budget"]["result"]["status"] == "aborted"
        assert adapters["fsm_cancelled"]["result"]["reason"] == "cancelled"
        assert adapters["solve_budget"]["result"]["reason"] == "budget_spent"
        assert adapters["workflow_budget"]["result"]["reason"] == "budget_spent"
        assert adapters["pattern_react_max_steps"]["error"]["type"] == "ValueError"


class TestTheComparerCanFail:
    """A comparer that cannot fail proves nothing: four planted mutants, each reported."""

    def test_identical_data_is_equal(self) -> None:
        assert first_difference(GOLDEN, copy.deepcopy(GOLDEN)) is None

    def test_a_changed_request_byte(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        row = mutant["PG-agent"]["react_ok"]["requests"][0]
        row["sha256"] = row["sha256"][:-1] + ("0" if row["sha256"][-1] != "0" else "1")
        found = first_difference(mutant, GOLDEN)
        assert found and "requests[0]/sha256" in found

    def test_an_extra_hook_call(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        mutant["PG-agent"]["react_ok"]["runs"][0]["steps"].append(99)
        found = first_difference(mutant, GOLDEN)
        assert found and "runs[0]/steps" in found

    def test_an_extra_event(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        mutant["PG-reasoning"]["deductive_ok"]["events"].append({"type": "extra"})
        found = first_difference(mutant, GOLDEN)
        assert found and "events" in found

    def test_a_dropped_result_field(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        del mutant["PG-adapters"]["fsm_ok"]["result"]["answer"]
        found = first_difference(mutant, GOLDEN)
        assert found and "result/answer" in found


class TestParityWithTheBase:
    """Invariant 8: with ``context=`` unset the tree is the base, case by case."""

    def test_the_run_has_the_golden_sections_and_cases(
        self, sections: dict[str, dict[str, Any]]
    ) -> None:
        assert {name: sorted(cases) for name, cases in sections.items()} == {
            name: sorted(cases) for name, cases in GOLDEN.items()
        }

    @pytest.mark.parametrize(
        ("section", "case"),
        [(name, case) for name in SECTIONS for case in sorted(GOLDEN[name])],
    )
    def test_the_case_equals_the_golden(
        self, sections: dict[str, dict[str, Any]], section: str, case: str
    ) -> None:
        got, want = sections[section][case], GOLDEN[section][case]
        if (section, case) in HOOKED_BY_STEP_6:
            got, want = _hooked(got, hooked=True), _hooked(want, hooked=False)
        found = first_difference(got, want)
        assert found is None, f"{section}/{case}{found}"


#: The ONE deliberate delta from the base (D-009 of plan 7efd5cb4, step 6): `reasoning.solve` under
#: a context hands it to its engine, so both run loops of the solve (orchestrator and classifier)
#: are now hooked. Result and events stay byte equal; only `hook` and `steps` of `runs` move, and
#: the rows of `TestReasoningUnderContext` pin the hook calls (the exact step numbers).
HOOKED_BY_STEP_6 = frozenset(
    {("PG-adapters", "solve_ok"), ("PG-adapters", "solve_budget")}
)
_ONE_CALL_PER_STEP = "<one hook call per step>"


def _hooked(record: dict[str, Any], *, hooked: bool) -> dict[str, Any]:
    """``record`` with the ``runs`` of a hooked solve told apart from an unhooked one by name."""
    runs = []
    for run in record["runs"]:
        steps = run["steps"]
        if hooked and not (
            run["hook"] is True and steps and steps == list(range(1, len(steps) + 1))
        ):
            runs.append(
                run
            )  # not hooked as expected: keep it, so the comparer reports it
        else:
            runs.append({**run, "hook": True, "steps": _ONE_CALL_PER_STEP})
    return {**record, "runs": runs}


# ---- the agent row (plan 7efd5cb4 step 5, D-008) -------------------------------------------------


class _Clock:
    """A stand-in for ``fsm_llm.engine.fsm.budget._monotonic``: moves only when told."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _agent_tools(on_call: Any = None, *, flagged: bool = False) -> Any:
    """The scripted double's tool registry; ``on_call()`` runs inside the tool (a cancel, a clock)."""
    from fsm_llm.tools import ToolRegistry

    def search(params: dict[str, Any]) -> str:
        if on_call is not None:
            on_call()
        return "result"

    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Tool search",
        parameter_schema={"properties": {"x": {"type": "string"}}},
        requires_approval=flagged,
    )
    return registry


class _Rig:
    """One context over a scripted double and an ``EventLog`` OBJECT, and the agents built on it."""

    def __init__(self, tmp_path: Path, **context: Any) -> None:
        from fsm_llm.engine.context import RunContext
        from fsm_llm.runtime import EventLog
        from tests import pattern_runs as pr

        self.model = pr.ScriptedModel()
        self.log = EventLog(tmp_path / "logs")
        self.ctx = RunContext(llm_interface=self.model, event_log=self.log, **context)
        self.pr = pr

    def agent(self, tools: Any = None, **kw: Any) -> Any:
        from fsm_llm import agents as A

        kw.setdefault("config", self.pr._config())
        return A.ReactAgent(tools or _agent_tools(), context=self.ctx, **kw)

    def types(self) -> list[str]:
        return [e["type"] for e in rig.events_of(self.log)]

    def paired(self) -> bool:
        types = self.types()
        ends = types.count("llm/response") + types.count("provider/error")
        return types.count("llm/request") == ends


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)


@pytest.mark.usefixtures("no_model")
class TestAgentUnderContext:
    """ONE log, ONE cancel, ONE budget for ``BaseAgent(context=)``; every refusal by name."""

    @pytest.mark.parametrize("how", ["run", "arun"])
    def test_one_log_and_the_interface_of_the_context(
        self, how: str, tmp_path: Path
    ) -> None:
        r = _Rig(tmp_path)
        agent = r.agent()
        result = (
            agent.run(r.pr._TASK)
            if how == "run"
            else rig.run_async(agent.arun(r.pr._TASK))
        )
        assert (
            result.stop_reason == "answered" and r.model.log
        )  # no LLM_MODEL, the double ran
        assert len(r.log.conversation_ids()) == 1
        assert "llm/request" in r.types() and r.paired()

    @pytest.mark.parametrize("how", ["run", "stream", "arun"])
    def test_a_set_token_raises_before_any_model_call(
        self, how: str, tmp_path: Path
    ) -> None:
        """Kills: the context ignored; the check dropped from `_create_api`."""
        from fsm_llm.engine.context import RunCancelledError

        r = _Rig(tmp_path)
        r.ctx.cancel.cancel()
        agent = r.agent()
        with pytest.raises(RunCancelledError):
            if how == "run":
                agent.run(r.pr._TASK)
            elif how == "stream":
                list(agent.run_stream(r.pr._TASK))
            else:
                rig.run_async(agent.arun(r.pr._TASK))
        assert r.model.log == []

    def test_a_greeting_is_not_spent_on_a_cancelled_context(
        self, tmp_path: Path
    ) -> None:
        from fsm_llm import agents as A
        from fsm_llm.engine.context import RunCancelledError

        r = _Rig(tmp_path)
        r.ctx.cancel.cancel()
        chain = A.PromptChainAgent(r.pr._chain(), config=r.pr._config(), context=r.ctx)
        with pytest.raises(RunCancelledError):
            chain.run(r.pr._TASK)
        assert r.model.log == []

    @pytest.mark.parametrize("how", ["run", "stream", "arun"])
    def test_a_token_set_during_a_step_stops_the_run_at_the_next_check(
        self, how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: the hook not passed; the cancel swallowed by an `except` in either run body;
        the stream unhooked. The conversation is ended and every request has its end."""
        from fsm_llm.agents.base.runs import RunsMixin
        from fsm_llm.engine.context import RunCancelledError

        r = _Rig(tmp_path)
        agent = r.agent(_agent_tools(on_call=lambda: r.ctx.cancel.cancel()))
        ended: list[bool] = []
        original = RunsMixin._end_run_conversation

        def spy(api: Any, conversation_id: str) -> None:
            original(api, conversation_id)
            ended.append(conversation_id not in api.list_active_conversations())

        monkeypatch.setattr(RunsMixin, "_end_run_conversation", staticmethod(spy))
        iterations: list[int] = []
        gate = type(agent)._on_loop_iteration

        def recording(self: Any, api: Any, cid: str, iteration: int) -> None:
            iterations.append(iteration)
            gate(self, api, cid, iteration)

        monkeypatch.setattr(type(agent), "_on_loop_iteration", recording)

        def run() -> Any:
            if how == "run":
                return agent.run(r.pr._TASK)
            if how == "stream":
                return list(agent.run_stream(r.pr._TASK))
            return rig.run_async(agent.arun(r.pr._TASK))

        with rig.HookSpy() as spy_hook, pytest.raises(RunCancelledError):
            run()
        steps = spy_hook.runs[0]["steps"]
        assert steps[-1] > 1 and ended == [True] and r.paired()
        # the context is read BEFORE the HITL driver: a cancelled step never reaches it
        assert iterations == steps[:-1]
        frozen = len(r.model.log)
        # the NEXT call on the same agent ends at its first check: nothing more is asked
        with pytest.raises(RunCancelledError):
            run()
        assert len(r.model.log) == frozen
        # a fresh agent on a fresh context runs
        fresh = _Rig(tmp_path / "fresh")
        assert fresh.agent().run(fresh.pr._TASK).stop_reason == "answered"

    @pytest.mark.parametrize("how", ["run", "stream", "arun"])
    def test_a_ledger_smaller_than_the_ceiling_names_the_shared_budget(
        self, how: str, tmp_path: Path
    ) -> None:
        """Kills: the ledger error mislabelled with the `max_iterations` text; the hook not
        charged (the run would end ok); a per-call charge (the count would exceed the steps)."""
        from fsm_llm.agents.exceptions import BudgetExhaustedError
        from fsm_llm.engine.context import RunBudget

        r = _Rig(tmp_path, budget=RunBudget(max_steps=2))
        agent = r.agent()
        with rig.HookSpy() as spy, pytest.raises(BudgetExhaustedError) as caught:
            if how == "run":
                agent.run(r.pr._TASK)
            elif how == "stream":
                list(agent.run_stream(r.pr._TASK))
            else:
                rig.run_async(agent.arun(r.pr._TASK))
        text = str(caught.value)
        assert "shared run steps" in text and "max_iterations" not in text
        assert "loop turns" not in text
        assert r.ctx.budget.steps_taken == 2
        assert spy.runs[0]["steps"] == [1, 2, 3]  # the third was refused, no step ran
        assert r.paired()

    def test_one_ledger_counts_the_steps_of_every_run_of_one_agent(
        self, tmp_path: Path
    ) -> None:
        """Kills: the hook closure kept on the agent (the last step number leaks across runs and
        the second run charges nothing for steps 1 to 3)."""
        from fsm_llm.engine.context import RunBudget

        r = _Rig(tmp_path, budget=RunBudget(max_steps=50))
        agent = r.agent()
        with rig.HookSpy() as spy:
            agent.run(r.pr._TASK)
            agent.run(r.pr._TASK)
        counts = [len(run["steps"]) for run in spy.runs]
        assert counts == [3, 3] and r.ctx.budget.steps_taken == 6

    def test_a_seconds_ledger_is_the_shared_budget_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
        from fsm_llm.engine.context import RunBudget

        clock = _Clock()
        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", clock)
        r = _Rig(tmp_path, budget=RunBudget(max_seconds=5.0))
        agent = r.agent(_agent_tools(on_call=lambda: setattr(clock, "now", 2000.0)))
        with pytest.raises(BudgetExhaustedError) as caught:
            agent.run(r.pr._TASK)
        assert "shared run seconds" in str(caught.value)
        assert not isinstance(caught.value, AgentTimeoutError)

    def test_the_agents_own_ceiling_keeps_its_text_beside_a_roomy_ledger(
        self, tmp_path: Path
    ) -> None:
        """PARITY: only a SPENT ledger changes the text of the engine's own step ceiling (the
        engine's error is built here: a run that hits the ceiling ships MAX_ITERATIONS instead)."""
        from fsm_llm.agents.definitions import AgentConfig
        from fsm_llm.core.definitions import RunBudgetExceededError
        from fsm_llm.engine.context import RunBudget

        one = AgentConfig(max_iterations=1, model="mock-model")
        engine_error = RunBudgetExceededError("steps", 3, 3)
        roomy = _Rig(tmp_path, budget=RunBudget(max_steps=99)).agent(config=one)
        text = str(roomy._budget_error(engine_error, None))
        assert "loop turns = max_iterations 1 x FSM_BUDGET_MULTIPLIER 3" in text
        assert "shared" not in text
        bare = _Rig(tmp_path / "b").agent(config=one)
        assert str(bare._budget_error(engine_error, None)) == text
        spent = RunBudget(max_steps=1)
        spent.charge()
        full = _Rig(tmp_path / "s", budget=spent).agent(config=one)
        assert "shared run steps" in str(full._budget_error(engine_error, None))

    def test_the_context_is_refused_where_it_cannot_be_honoured(
        self, tmp_path: Path
    ) -> None:
        from fsm_llm import agents as A
        from fsm_llm.agents.agent_graph import AgentGraph
        from fsm_llm.engine.context import RunBudget, RunContext

        r = _Rig(tmp_path)
        with pytest.raises(ValueError, match="max_iterations"):
            A.ReactAgent(
                _agent_tools(), config=r.pr._config(), context=RunContext(max_steps=5)
            )
        with pytest.raises(TypeError, match="RunContext, got dict"):
            A.ReactAgent(_agent_tools(), config=r.pr._config(), context={"a": 1})
        for cls, args in (
            (A.SelfConsistencyAgent, ()),
            (
                A.SwarmAgent,
                (
                    {
                        "a": A.ReactAgent(
                            _agent_tools(), config=r.pr._config(), llm_interface=r.model
                        )
                    },
                    "a",
                ),
            ),
        ):
            with pytest.raises(TypeError, match="context"):
                cls(*args, context=r.ctx)
        with pytest.raises(TypeError, match="context"):
            AgentGraph({}, {}, "a", context=r.ctx)  # type: ignore[call-arg]
        # a ledger alone is fine: it carries no step limit of the agent
        assert r.agent() is not None and RunBudget(max_steps=1)

    def test_an_explicit_keyword_wins_over_the_context(self, tmp_path: Path) -> None:
        """Kills: the explicit keyword overridden by the context."""
        from fsm_llm.runtime import EventLog
        from tests import pattern_runs as pr

        r = _Rig(tmp_path)
        mine, mine_log = pr.ScriptedModel(), EventLog(tmp_path / "mine")
        agent = r.agent(llm_interface=mine, event_log=mine_log)
        assert agent.run(pr._TASK).stop_reason == "answered"
        assert mine.log and r.model.log == []
        assert mine_log.conversation_ids() and r.log.conversation_ids() == []

    def test_a_factory_is_called_once_per_agent_not_per_run(
        self, tmp_path: Path
    ) -> None:
        from fsm_llm import agents as A
        from fsm_llm.engine.context import RunContext
        from tests import pattern_runs as pr

        made: list[Any] = []

        def factory() -> Any:
            made.append(pr.ScriptedModel())
            return made[-1]

        agent = A.ReactAgent(
            _agent_tools(), config=pr._config(), context=RunContext(llm_factory=factory)
        )
        agent.run(pr._TASK)
        agent.run(pr._TASK)
        assert len(made) == 1 and made[0].log

    def test_the_seconds_of_a_context_narrow_the_config_and_never_widen_it(
        self, tmp_path: Path
    ) -> None:
        r = _Rig(tmp_path, max_seconds=7.5)
        assert r.agent().config.timeout_seconds == 7.5
        wide = _Rig(tmp_path / "w", max_seconds=9999.0)
        assert wide.agent().config.timeout_seconds == 300.0

    def test_no_model_anywhere_is_the_usual_refusal(self) -> None:
        """Invariant 5: a context that names no interface invents none."""
        from fsm_llm import agents as A
        from fsm_llm.core.definitions import ModelNotConfiguredError
        from fsm_llm.engine.context import RunContext

        with pytest.raises(ModelNotConfiguredError):
            A.ReactAgent(_agent_tools(), context=RunContext())

    def test_a_standalone_pattern_file_reads_the_interface_of_the_context(
        self, tmp_path: Path
    ) -> None:
        """Kills: the adoption placed after `_config_from_options` (a preset with config options
        would ask `LLM_MODEL`, unset here)."""
        from fsm_llm.agents.pattern_agent import PatternAgent
        from fsm_llm.agents.pattern_files import load_envelope

        r = _Rig(tmp_path)
        agent = PatternAgent(
            load_envelope("summarize"), tools=_agent_tools(), context=r.ctx
        )
        assert agent._context is r.ctx and agent._api_kwargs["llm_interface"] is r.model

    def test_every_concrete_agent_that_takes_a_context_ends_at_a_set_token(
        self, tmp_path: Path
    ) -> None:
        """Kills: one pattern outside the hook, or one whose own `except` eats the cancel (ADaPT's
        `run`, the two run bodies): the double sets the token during its FIRST call and each
        agent raises `RunCancelledError`, never `AgentError`. `native_fc` scripts its replies
        through a litellm stub (`test_native_fc_golden`), so it is cancelled before the run."""
        import fsm_llm.agents.reasoning_react  # noqa: F401  (a subclass of BaseAgent)
        from fsm_llm import agents as A
        from fsm_llm.agents.base.agent import BaseAgent
        from fsm_llm.agents.pattern_files import load_envelope
        from fsm_llm.engine.context import RunCancelledError, RunContext
        from tests import pattern_runs as pr

        class Cancelling(pr.ScriptedModel):
            token: Any = None

            def _note(self, call: str, request: Any) -> None:
                super()._note(call, request)
                self.token.cancel()

        def build(ctx: Any) -> dict[str, Any]:
            kw = {"config": pr._config(), "context": ctx}
            t = pr._tools(many=True)
            return {
                "ReactAgent": A.ReactAgent(t, **kw),
                "ReflexionAgent": A.ReflexionAgent(t, **kw),
                "REWOOAgent": A.REWOOAgent(t, **kw),
                "DebateAgent": A.DebateAgent(**kw),
                "PlanExecuteAgent": A.PlanExecuteAgent(t, **kw),
                "PromptChainAgent": A.PromptChainAgent(pr._chain(), **kw),
                "EvaluatorOptimizerAgent": A.EvaluatorOptimizerAgent(pr._passing, **kw),
                "MakerCheckerAgent": A.MakerCheckerAgent("make it", "check it", **kw),
                "OrchestratorAgent": A.OrchestratorAgent(
                    worker_factory=lambda sub: A.AgentResult(answer="done"), **kw
                ),
                "ParallelReactAgent": A.ParallelReactAgent(t, **kw),
                "ADaPTAgent": A.ADaPTAgent(t, **kw),
                "VerifiedReactAgent": A.VerifiedReactAgent(t, **kw),
                "AutoMemoryReactAgent": A.AutoMemoryReactAgent(
                    t, memory=pr._memory_store(), **kw
                ),
                "NativeFunctionCallingReactAgent": A.NativeFunctionCallingReactAgent(
                    t, **kw
                ),
                "ReasoningReactAgent": A.ReasoningReactAgent(t, **kw),
                "PatternAgent": A.PatternAgent(
                    load_envelope("summarize"), tools=t, **kw
                ),
            }

        def concrete(cls: type) -> set[type]:
            found: set[type] = set()
            for sub in cls.__subclasses__():
                found |= {sub} | concrete(sub)
            return found

        # the shipped classes only: a test module may define subclasses of its own
        taking = {
            c.__name__
            for c in concrete(BaseAgent)
            if c.takes_context and c.__module__.startswith("fsm_llm.")
        }
        names = set(build(RunContext(llm_interface=pr.ScriptedModel())))
        assert names == taking, "a new agent class needs a row here"
        for name in sorted(names):
            model = Cancelling()
            ctx = RunContext(llm_interface=model)
            model.token = ctx.cancel
            if name == "NativeFunctionCallingReactAgent":
                ctx.cancel.cancel()
            with pytest.raises(RunCancelledError):
                build(ctx)[name].run(pr._TASK)
            assert name == "NativeFunctionCallingReactAgent" or model.log, name

    def test_a_cancelled_adapt_subtask_ends_the_run(self, tmp_path: Path) -> None:
        """Kills: `RunCancelledError` missing from `RUN_ENDING_ERRORS` and from ADaPT's `except`."""
        from fsm_llm.agents.handlers import RUN_ENDING_ERRORS, RunEndingErrorHolder
        from fsm_llm.engine.context import RunCancelledError

        assert RunCancelledError in RUN_ENDING_ERRORS
        holder = RunEndingErrorHolder()
        assert holder.capture(RunCancelledError()) and holder.is_set
        with pytest.raises(RunCancelledError):
            holder.raise_if_set()

    def test_the_reasoning_react_inner_engine_gets_the_adopted_interface_and_log(
        self, tmp_path: Path
    ) -> None:
        """The inner engine takes the adopted interface and log (the context itself: step 6, below)."""
        from fsm_llm import agents as A
        from tests import pattern_runs as pr

        r = _Rig(tmp_path)
        agent = A.ReasoningReactAgent(pr._tools(), config=pr._config(), context=r.ctx)
        assert agent._api_kwargs["llm_interface"] is r.model
        engine_kwargs = agent._reasoning_engine.api_kwargs
        assert engine_kwargs["llm_interface"] is r.model
        assert engine_kwargs["event_log"] is r.log and "context" not in engine_kwargs


@pytest.mark.usefixtures("no_model")
class TestAgentContextHoldsNoGrant:
    """Invariant 1: a context carries no approval grant and no approver reaches an agent."""

    def test_the_callables_of_a_context_give_the_agent_no_approver(
        self, tmp_path: Path
    ) -> None:
        """Kills: `callables` consumed (a grant leak path)."""
        from fsm_llm.core.names import NameRegistry
        from fsm_llm.policy import HumanInTheLoop

        hitl = HumanInTheLoop(
            approval_policy=lambda c, x: True, approval_callback=lambda q: True
        )
        callables: NameRegistry[Any] = NameRegistry("test callables")
        callables.register("hitl", hitl)
        r = _Rig(tmp_path, callables=callables)
        agent = r.agent()
        assert agent.hitl is None and agent._hitl_active is False
        assert not any(v is hitl for v in vars(agent).values())

    def test_a_flagged_tool_with_no_hitl_still_refuses_at_construction(
        self, tmp_path: Path
    ) -> None:
        from fsm_llm.core.definitions import AgentError

        r = _Rig(tmp_path)
        with pytest.raises(AgentError, match="approval"):
            r.agent(_agent_tools(flagged=True))
        assert r.model.log == []

    def test_a_grant_in_initial_context_is_still_stripped(self, tmp_path: Path) -> None:
        from fsm_llm.agents.constants import ContextKeys

        r = _Rig(tmp_path)
        seeded = r.agent()._init_context(
            "task", {ContextKeys.DRIVER_APPROVAL: {"tool": "search"}}
        )
        assert ContextKeys.DRIVER_APPROVAL not in seeded


# ---- the reasoning engine (plan 7efd5cb4 step 6, D-009) ----------------------------------------

from fsm_llm.core.definitions import RunBudgetExceededError
from fsm_llm.engine.context import (
    CancelToken,
    RunBudget,
    RunCancelledError,
    RunContext,
)
from fsm_llm.engine.executables.code_backed import cause_chain
from fsm_llm.reasoning import ReasoningEngine, ReasoningExecutionError
from tests.test_fsm_llm_reasoning import test_engine_scripted as sc


class _Stopper(sc._ScriptedLLM):
    """The scripted solve double that runs ``on_field()`` when it is asked for ``field``."""

    def __init__(self, script: dict[str, Any], field: str | None, on_field: Any):
        super().__init__(script)
        self.field, self.on_field = field, on_field

    def extract_field(self, request: Any) -> Any:
        if request.field_name == self.field:
            self.on_field()
        return super().extract_field(request)


class _Solve:
    """One context over a scripted solve double and an ``EventLog`` OBJECT, and its engine."""

    def __init__(
        self,
        tmp_path: Path,
        script: dict[str, Any] | None = None,
        *,
        stop_at: str | None = None,
        **context: Any,
    ) -> None:
        from fsm_llm.runtime import EventLog

        self.token = CancelToken()
        self.llm = _Stopper(script or sc._VALID_SCRIPT, stop_at, self.token.cancel)
        self.log = EventLog(tmp_path / "logs")
        self.ctx = RunContext(
            llm_interface=self.llm, event_log=self.log, cancel=self.token, **context
        )
        self.engine = ReasoningEngine(context=self.ctx)

    @property
    def calls(self) -> int:
        return len(self.llm.requests) + len(self.llm.completions)

    @property
    def taken(self) -> int:
        assert self.ctx.budget is not None
        return self.ctx.budget.steps_taken

    def types(self) -> list[str]:
        return [e["type"] for e in rig.events_of(self.log)]

    def paired(self) -> bool:
        types = self.types()
        ends = types.count("llm/response") + types.count("provider/error")
        return types.count("llm/request") == ends

    def idle(self) -> bool:
        """Nothing active on either API, no strategy frame left, the solve lock free."""
        e = self.engine
        return (
            e.orchestrator.list_active_conversations() == []
            and e.classifier.list_active_conversations() == []
            and dict(e.orchestrator.fsm_manager.instances) == {}
            and not e._solve_lock.locked()
        )

    def solve(self, how: str = "solve", **limits: Any) -> Any:
        if how == "solve":
            return self.engine.solve_problem(sc._PROBLEM, **limits)
        stream = self.engine.solve_problem_stream(sc._PROBLEM, **limits)
        list(stream)
        return stream.result()

    def close(self) -> None:
        self.engine.close()
        self.log.close()


@pytest.fixture
def solve_rig(tmp_path: Path) -> Any:
    made: list[_Solve] = []

    def make(script: dict[str, Any] | None = None, **kw: Any) -> _Solve:
        made.append(_Solve(tmp_path / f"r{len(made)}", script, **kw))
        return made[-1]

    yield make
    for r in made:
        r.close()


def _solve_chain(error: BaseException) -> list[type[BaseException]]:
    return [type(c) for c in cause_chain(error)]


@pytest.mark.usefixtures("no_model")
class TestReasoningUnderContext:
    """ONE log, ONE cancel, ONE budget for ``ReasoningEngine(context=)`` (D-009)."""

    def test_one_log_holds_both_conversations_and_the_double_of_the_context(
        self, solve_rig: Any
    ) -> None:
        """Kills: the context ignored (no model resolves, the log is not shared)."""
        r = solve_rig()
        solution, info = r.solve()
        assert solution == sc._VALID_SCRIPT["final_solution"] and r.llm.requests
        assert len(r.log.conversation_ids()) == 2  # the orchestrator and the classifier
        assert "llm/request" in r.types() and r.paired()
        assert info["reasoning_trace"]["steps"] and r.idle()

    @pytest.mark.parametrize("how", ["solve", "stream"])
    @pytest.mark.parametrize(
        ("script", "orchestrator", "classifier"),
        [(sc._VALID_SCRIPT, 7, 3), (sc._CALC_SCRIPT, 6, 3)],
        ids=["deductive", "calculator"],
    )
    def test_both_run_loops_are_hooked_and_the_ledger_counts_every_step(
        self,
        how: str,
        script: dict[str, Any],
        orchestrator: int,
        classifier: int,
        solve_rig: Any,
    ) -> None:
        """Kills: the classifier loop unhooked (3 steps uncounted), the stream unhooked, a charge
        per call. The total is the engine's own steps_done: orchestrator plus classifier."""
        r = solve_rig(script, budget=RunBudget())
        with rig.HookSpy() as spy:
            r.solve(how)
        assert [(run["hook"], run["steps"]) for run in spy.runs] == [
            (True, list(range(1, orchestrator + 1))),
            (True, list(range(1, classifier + 1))),
        ]
        assert r.taken == orchestrator + classifier

    @pytest.mark.parametrize("how", ["solve", "stream"])
    def test_a_set_token_ends_the_solve_before_any_model_call(
        self, how: str, solve_rig: Any
    ) -> None:
        """Kills: the pre-check dropped (the greeting is spent), the context ignored."""
        r = solve_rig(budget=RunBudget())
        r.token.cancel()
        with pytest.raises(ReasoningExecutionError) as caught:
            r.solve(how)
        assert RunCancelledError in _solve_chain(caught.value)
        assert r.calls == 0 and r.taken == 0 and r.idle()
        r.close()  # still closable

    @pytest.mark.parametrize("how", ["solve", "stream"])
    @pytest.mark.parametrize(
        ("field", "taken", "calls", "through"),
        [
            ("problem_type", 1, 5, "HandlerExecutionError"),
            ("problem_domain", 2, 11, "HandlerExecutionError"),
            ("conclusion", 7, 35, None),
        ],
        ids=["before_the_classifier", "inside_the_classifier", "inside_a_strategy"],
    )
    def test_a_token_set_during_a_step_stops_the_solve_at_the_next_check(
        self,
        how: str,
        field: str,
        taken: int,
        calls: int,
        through: str | None,
        solve_rig: Any,
    ) -> None:
        """Kills: the hook not passed (to the classifier loop, to the stream), the cancel
        swallowed (a tuple returned), `_end_quietly` skipped (a conversation or a strategy frame
        left active). The step in flight ends, then no model call; the NEXT solve on the dead
        context ends at its first check and the engine stays closable."""
        r = solve_rig(budget=RunBudget(), stop_at=field)
        with pytest.raises(ReasoningExecutionError) as caught:
            r.solve(how)
        chain = _solve_chain(caught.value)
        assert RunCancelledError in chain
        assert (through in [c.__name__ for c in chain]) is (through is not None)
        assert (r.taken, r.calls) == (taken, calls)
        assert r.idle() and r.paired()
        with pytest.raises(ReasoningExecutionError, match="cancelled before"):
            r.solve(how)
        assert (r.taken, r.calls) == (taken, calls)
        r.close()

    @pytest.mark.parametrize(
        ("limit", "stops_in"),
        [(3, "classifier"), (6, "orchestrator")],
    )
    def test_a_ledger_smaller_than_the_solve_reads_as_the_shared_budget(
        self, limit: int, stops_in: str, solve_rig: Any
    ) -> None:
        """Kills: the ledger error mislabelled as the 170-step ceiling or as the classifier's own
        10 steps; the classifier loop unhooked. The count is exact and unchanged by the refusal."""
        r = solve_rig(budget=RunBudget(max_steps=limit))
        with pytest.raises(ReasoningExecutionError) as caught:
            r.solve()
        spent = next(
            c
            for c in cause_chain(caught.value)
            if isinstance(c, RunBudgetExceededError)
        )
        assert (spent.budget, spent.limit) == ("steps", limit)
        assert r.taken == limit and r.ctx.budget_spent() == "steps"
        text = " | ".join(str(c) for c in cause_chain(caught.value))
        assert "170" not in text and "did not converge" not in text
        assert ("shared run budget" in text) is (stops_in == "orchestrator")
        assert caught.value.details["partial_context"] is not None
        assert r.idle() and r.paired()

    def test_a_seconds_ledger_is_the_shared_budget_too(
        self, solve_rig: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _Clock()
        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", clock)
        r = solve_rig(budget=RunBudget(max_seconds=5.0), stop_at="conclusion")
        r.llm.on_field = lambda: setattr(clock, "now", clock.now + 100.0)
        with pytest.raises(
            ReasoningExecutionError, match="shared run budget"
        ) as caught:
            r.solve()
        spent = next(
            c
            for c in cause_chain(caught.value)
            if isinstance(c, RunBudgetExceededError)
        )
        assert (spent.budget, spent.limit) == ("seconds", 5.0)
        assert r.ctx.budget_spent() == "seconds" and r.idle()

    def test_the_classifiers_own_ten_steps_keep_their_text(
        self, solve_rig: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parity: an unspent ledger changes nothing about the classifier's own budget error."""
        from fsm_llm.reasoning.constants import ContextKeys
        from fsm_llm.reasoning.exceptions import ReasoningClassificationError

        r = solve_rig(budget=RunBudget(max_steps=1000))

        def spent(conversation_id: str, **kwargs: Any) -> Any:
            raise RunBudgetExceededError("steps", 10, 10)

        monkeypatch.setattr(r.engine.classifier, "run_until_terminal", spent)
        with pytest.raises(ReasoningClassificationError, match="10 steps"):
            r.engine._classify_problem({ContextKeys.PROBLEM_STATEMENT: sc._PROBLEM})
        assert r.ctx.budget_spent() is None

    def test_a_cancel_reaches_the_solve_of_a_reasoning_executable(
        self, solve_rig: Any
    ) -> None:
        """The executable path: a real engine, a token set inside the classifier."""
        from fsm_llm.core.codes import Reason, Status
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.reasoning.executables import SOLVE_ID

        r = solve_rig(budget=RunBudget(), stop_at="problem_domain")
        registry = ExecutableRegistry()
        registry.load_sources()
        result = registry.get(SOLVE_ID).run({"problem": sc._PROBLEM}, context=r.ctx)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert (r.taken, r.calls) == (2, 11) and r.paired()
        again = registry.get(SOLVE_ID).run({"problem": sc._PROBLEM}, context=r.ctx)
        assert (again.status, again.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert r.calls == 11

    def test_a_ledger_reaches_the_solve_of_a_reasoning_executable(
        self, solve_rig: Any
    ) -> None:
        from fsm_llm.core.codes import Reason, Status
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.reasoning.executables import SOLVE_ID

        r = solve_rig(budget=RunBudget(max_steps=6))
        registry = ExecutableRegistry()
        registry.load_sources()
        result = registry.get(SOLVE_ID).run({"problem": sc._PROBLEM}, context=r.ctx)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["shared_budget"] == "steps" and r.taken == 6

    def test_an_explicit_keyword_wins_over_the_context(
        self, solve_rig: Any, tmp_path: Path
    ) -> None:
        """Kills: the explicit `llm_interface` or `event_log` overridden by the context."""
        from fsm_llm.runtime import EventLog

        r = solve_rig()
        mine = sc._ScriptedLLM(sc._VALID_SCRIPT)
        log = EventLog(tmp_path / "mine")
        try:
            with ReasoningEngine(
                llm_interface=mine, event_log=log, context=r.ctx
            ) as engine:
                engine.solve_problem(sc._PROBLEM)
        finally:
            log.close()
        assert mine.requests and r.calls == 0
        assert len(log.conversation_ids()) == 2 and r.log.conversation_ids() == []

    def test_a_factory_is_called_once_per_engine_and_serves_both_apis(
        self, tmp_path: Path
    ) -> None:
        """Kills: the factory resolved per solve or per API (two interfaces, two meters)."""
        made: list[sc._ScriptedLLM] = []

        def factory() -> sc._ScriptedLLM:
            made.append(sc._ScriptedLLM(sc._VALID_SCRIPT))
            return made[-1]

        with ReasoningEngine(context=RunContext(llm_factory=factory)) as engine:
            engine.solve_problem(sc._PROBLEM)
            engine.solve_problem(sc._PROBLEM)
        assert len(made) == 1 and made[0].requests
        assert engine.api_kwargs["llm_interface"] is made[0]

    def test_a_context_that_is_no_run_context_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="RunContext"):
            ReasoningEngine(model="mock", context=object())  # type: ignore[arg-type]

    def test_the_engine_reads_no_callables_and_no_grant_reaches_it(
        self, solve_rig: Any
    ) -> None:
        """Invariant 1. Kills: `callables` consumed by the engine (no approval path exists)."""
        from fsm_llm.core.names import NameRegistry

        marker = object()
        callables: NameRegistry[Any] = NameRegistry("test callables")
        callables.register("hitl", marker)
        r = solve_rig(callables=callables)
        _solution, info = r.solve()
        held = [*vars(r.engine).values(), *r.engine.api_kwargs.values()]
        assert not any(v is marker or v is callables for v in held)
        for key in ("context", "_approval_granted", "approval_granted"):
            assert key not in info["final_context"], key
        assert not any(v is r.ctx for v in info["final_context"].values())


def test_a_cancelled_solve_is_the_next_solve_s_first_check_on_one_engine_only(
    solve_rig: Any,
) -> None:
    """A fresh engine on a FRESH context runs: the death is the context's, not the class's."""
    dead = solve_rig()
    dead.token.cancel()
    alive = solve_rig()
    solution, _ = alive.solve()
    assert solution == sc._VALID_SCRIPT["final_solution"]


class TestReasoningReactUnderContext:
    """The inner engine of ``ReasoningReactAgent`` is under the SAME context as the agent."""

    @staticmethod
    def _models() -> Any:
        from tests import pattern_runs as pr

        class Both(pr.ScriptedModel):
            """One interface for the agent (it picks `reason`) and for the inner solve."""

            def __init__(self, on_field: Any = None, field: str | None = None) -> None:
                super().__init__(tool="reason")
                self.inner = sc._ScriptedLLM(sc._VALID_SCRIPT)
                self.on_field, self.field = on_field, field
                self.inner_calls = 0

            def extract_field(self, request: Any) -> Any:
                if request.field_name in sc._VALID_SCRIPT:
                    self.inner_calls += 1
                    if request.field_name == self.field:
                        self.on_field()
                    return self.inner.extract_field(request)
                return super().extract_field(request)

            def extract_bulk_data(self, request: Any) -> Any:
                names = pr._BULK_NAME.findall(request.system_prompt)
                if any(n in sc._VALID_SCRIPT for n in names):
                    self.inner_calls += 1
                    return self.inner.extract_bulk_data(request)
                return super().extract_bulk_data(request)

            def complete(self, request: Any) -> Any:
                self.inner_calls += 1
                return self.inner.complete(request)

        return Both

    def _agent(self, ctx: Any) -> Any:
        from fsm_llm import agents as A
        from tests import pattern_runs as pr

        return A.ReasoningReactAgent(pr._tools(), config=pr._config(), context=ctx)

    def test_the_context_reaches_the_inner_engine_by_identity(self) -> None:
        """Kills: the context not forwarded (the inner engine runs unhooked)."""
        ctx = RunContext(llm_interface=self._models()())
        agent = self._agent(ctx)
        assert agent._reasoning_engine._context is ctx

    def test_one_ledger_counts_the_agent_and_the_inner_solve_exactly(self) -> None:
        from tests import pattern_runs as pr

        ctx = RunContext(llm_interface=self._models()(), budget=RunBudget())
        with rig.HookSpy() as spy:
            result = self._agent(ctx).run(pr._TASK)
        assert result.stop_reason == "answered"
        runs = [(run["max_steps"], run["steps"]) for run in spy.runs]
        assert [len(steps) for _limit, steps in runs] == [
            3,
            17,
            3,
        ]  # agent, solve, classifier
        assert ctx.budget is not None and ctx.budget.steps_taken == 23

    @pytest.mark.parametrize(
        ("field", "inner_calls", "taken"),
        [("problem_domain", 6, 3), ("conclusion", 19, 8)],
    )
    def test_one_cancel_stops_the_inner_solve_and_the_outer_agent(
        self, field: str, inner_calls: int, taken: int
    ) -> None:
        """Kills: the context not forwarded, the cancel recorded as a failed observation and the
        run going on. Both runs end with their conversations ended and no later model call."""
        from tests import pattern_runs as pr

        token = CancelToken()
        model = self._models()(token.cancel, field)
        ctx = RunContext(llm_interface=model, cancel=token, budget=RunBudget())
        agent = self._agent(ctx)
        with pytest.raises(RunCancelledError):
            agent.run(pr._TASK)
        engine = agent._reasoning_engine
        assert engine.orchestrator.list_active_conversations() == []
        assert engine.classifier.list_active_conversations() == []
        # the inner solve stopped where the token was set (a full solve asks the double 38 times)
        assert (model.inner_calls, ctx.budget.steps_taken) == (inner_calls, taken)  # type: ignore[union-attr]
        frozen = (len(model.log), model.inner_calls)
        with pytest.raises(RunCancelledError):
            agent.run(
                pr._TASK
            )  # the NEXT run of the dead context ends at its first check
        assert (len(model.log), model.inner_calls) == frozen

    def test_a_ledger_shared_by_both_names_the_shared_budget(self) -> None:
        from fsm_llm.agents.exceptions import BudgetExhaustedError
        from tests import pattern_runs as pr

        ctx = RunContext(llm_interface=self._models()(), budget=RunBudget(max_steps=6))
        with pytest.raises(BudgetExhaustedError, match="shared run steps"):
            self._agent(ctx).run(pr._TASK)
        assert ctx.budget is not None and ctx.budget.steps_taken == 6

    @pytest.mark.parametrize("kind", ["cancel", "ledger"])
    def test_a_stop_of_the_context_is_no_failed_reason_observation(
        self, kind: str
    ) -> None:
        """Kills: the `except Exception` of `reason` recording a stop of the context as a failed
        tool. The executor raises it; a real failure is still recorded."""
        from fsm_llm.agents.constants import ContextKeys
        from tests import pattern_runs as pr

        ctx = RunContext(llm_interface=self._models()(), budget=RunBudget(max_steps=1))
        agent = self._agent(ctx)
        engine = agent._reasoning_engine

        def boom(*_a: Any, **_k: Any) -> Any:
            raise ReasoningExecutionError("the inner solve stopped")

        engine.solve_problem = boom  # type: ignore[method-assign]
        handlers = agent._call_handlers(agent.tools)
        execute = agent._make_reasoning_tool_executor(handlers)
        state = {ContextKeys.TOOL_NAME: "reason", ContextKeys.TOOL_INPUT: {"x": "1"}}
        state[ContextKeys.TASK] = pr._TASK
        failed = execute(dict(state))  # a plain failure is recorded, as before
        assert failed[ContextKeys.TOOL_STATUS] == "failed"
        if kind == "cancel":
            ctx.cancel.cancel()
        else:
            ctx.budget.charge()  # type: ignore[union-attr]
        with pytest.raises(ReasoningExecutionError, match="inner solve stopped"):
            execute(dict(state))


# ---- the sweep: every run loop of a runner passes ``before_step`` (invariant 2) ----------------

import ast

_RUN_LOOPS = frozenset(
    {
        "run_until_terminal",
        "arun_until_terminal",
        "run_until_terminal_stream",
        "arun_until_terminal_stream",
    }
)
_SWEPT = ("agents", "reasoning", "workflows")
#: a call site that may run a loop with no hook, as ``"<path under fsm_llm>:<line>": reason``;
#: empty on purpose: a runner that skips the hook is silently outside the cancel and the ledger
_ALLOWED_UNHOOKED: dict[str, str] = {}


def _loop_calls(source: str) -> list[tuple[int, bool]]:
    """Every call that runs a run loop, as ``(line, passes before_step)``.

    Two forms: the direct one (``api.run_until_terminal(...)``, a name or an attribute) and the
    string-named one the agents use (``acall(api, "run_until_terminal", ...)``: the loop name is a
    string constant among the arguments).
    """
    sites: list[tuple[int, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        named = any(
            isinstance(arg, ast.Constant) and arg.value in _RUN_LOOPS
            for arg in node.args
        )
        if name in _RUN_LOOPS or named:
            hooked = any(
                kw.arg == "before_step"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
                for kw in node.keywords
            )  # a literal `before_step=None` is no hook
            sites.append((node.lineno, hooked))
    return sorted(sites)


class TestEveryRunLoopPassesTheHook:
    def test_no_run_loop_of_the_three_packages_runs_unhooked(self) -> None:
        found: dict[str, list[tuple[int, bool]]] = {}
        for package in _SWEPT:
            for path in sorted((REPO / "src" / "fsm_llm" / package).rglob("*.py")):
                sites = _loop_calls(path.read_text(encoding="utf-8"))
                if sites:
                    found[path.relative_to(REPO / "src" / "fsm_llm").as_posix()] = sites
        bare = [
            f"{where}:{line}"
            for where, sites in found.items()
            for line, hooked in sites
            if not hooked and f"{where}:{line}" not in _ALLOWED_UNHOOKED
        ]
        assert bare == [], (
            f"a run loop with no before_step (cancel and ledger skipped): {bare}"
        )
        # the walk must SEE the known sites, or a rename would blind it
        assert sorted(found) == [
            "agents/base/runs.py",
            "reasoning/engine.py",
            "reasoning/engine_handlers.py",
        ]
        assert [len(found[k]) for k in sorted(found)] == [2, 2, 1]

    def test_the_walk_sees_both_forms_and_flags_the_missing_keyword(self) -> None:
        """Negative control: a planted module with each form, once without the hook."""
        source = (
            "def f(api, acall, hook):\n"
            "    api.run_until_terminal(c, max_steps=1)\n"  # line 2: direct, unhooked
            "    api.arun_until_terminal(c, before_step=hook)\n"  # 3: direct, hooked
            "    acall(api, 'run_until_terminal', c)\n"  # 4: string-named, unhooked
            "    acall(api, 'run_until_terminal_stream', c, before_step=hook)\n"  # 5
            "    other(api, 'run_until_terminal_x', c)\n"  # 6: not a loop
            "    api.run_until_terminal(c, before_step=None)\n"  # 7: a literal None is no hook
        )
        assert _loop_calls(source) == [
            (2, False),
            (3, True),
            (4, False),
            (5, True),
            (7, False),
        ]


# ---- the cross-runner acceptance (plan 7efd5cb4 step 8): the done-when, literally -----------------
#
# ONE context, three runners: an agent run, a reasoning solve and a workflow instance (the workflow
# adapter, a step that runs an FSM executable as a child). They write to ONE log (an `EventLog`
# OBJECT), stop on ONE cancellation and share ONE ledger. Every count is exact: a scripted double,
# no clock, no sleep; a thread waits only on an `Event` or a `Barrier` with a timeout. The engine of
# a reasoning solve has no async API, so its async twin runs `solve_problem` on a worker thread.

import asyncio
import threading

#: the steps of each runner under the rig below (agent: react; solve: deductive orchestrator and the
#: classifier loop; workflow: prep and call, then the child FSM `a -> b -> c`)
_AGENT_STEPS, _SOLVE_STEPS, _FLOW_STEPS = 3, 10, 4
_TOTAL = _AGENT_STEPS + _SOLVE_STEPS + _FLOW_STEPS


def _everyone() -> type:
    """The ONE interface of the acceptance rig: the agent's, the solve's and the child FSM's."""
    from tests import pattern_runs as pr

    class Everyone(pr.ScriptedModel):
        """``ScriptedModel`` answers the agent and the child FSM; the scripted solve answers the
        fields of the reasoning script. ``triggers`` maps a field name to a callable that runs when
        that field is asked (a cancel, a clock): the way a double acts INSIDE a runner."""

        def __init__(self, triggers: dict[str, Any] | None = None) -> None:
            super().__init__()
            self.inner = sc._ScriptedLLM(sc._VALID_SCRIPT)
            self.triggers = triggers or {}

        @property
        def calls(self) -> int:
            return (
                len(self.log) + len(self.inner.requests) + len(self.inner.completions)
            )

        def _fire(self, names: Any) -> None:
            for name in names:
                if name in self.triggers:
                    self.triggers[name]()

        def extract_field(self, request: Any) -> Any:
            self._fire([request.field_name])
            if request.field_name in sc._VALID_SCRIPT:
                return self.inner.extract_field(request)
            return super().extract_field(request)

        def extract_bulk_data(self, request: Any) -> Any:
            names = pr._BULK_NAME.findall(request.system_prompt)
            self._fire(names)
            if any(n in sc._VALID_SCRIPT for n in names):
                return self.inner.extract_bulk_data(request)
            return super().extract_bulk_data(request)

        def complete(self, request: Any) -> Any:
            return self.inner.complete(request)

    return Everyone


def _chain_document() -> dict[str, Any]:
    """The child FSM, `a -> b -> c`: two loop steps, `a` extracts ``result``."""
    from tests.test_fsm_llm import kind_corpus as kc

    return {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": "chain",
        "contract": {"inputs": ["topic"], "outputs": ["result"]},
        "body": {
            "fsm": kc._fsm(
                "Chain",
                "a",
                kc._state(
                    "a",
                    extraction_instructions="Extract the result.",
                    required_context_keys=["result"],
                    response_instructions="A.",
                    transitions=[kc._go("b")],
                ),
                kc._state("b", response_instructions="B.", transitions=[kc._go("c")]),
                kc._state("c", response_instructions="C."),
            )
        },
    }


class _Three:
    """One context over three runners: the agent, the reasoning engine and the workflow adapter."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        triggers: dict[str, Any] | None = None,
        on_prep: Any = None,
        on_tool: Any = None,
        hitl: Any = None,
        path_log: bool = False,
        **context: Any,
    ) -> None:
        from fsm_llm import agents as A
        from fsm_llm.core.names import NameRegistry
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.runtime import EventLog
        from tests import pattern_runs as pr
        from tests.test_fsm_llm_workflows.test_workflow_executable import (
            _auto,
            _document,
        )

        self.model = _everyone()(triggers)
        self.path = tmp_path
        self.log = EventLog(tmp_path / "logs")
        shared = tmp_path if path_log else self.log
        self.token = context.pop("cancel", None) or CancelToken()
        names: NameRegistry[Any] = NameRegistry("callables")
        names.register("prep", lambda ctx: on_prep() if on_prep else None)
        if hitl is not None:
            names.register("hitl", hitl)
        registry = ExecutableRegistry()
        registry.add_document(_chain_document())
        call = {
            "type": "ExecutableStep",
            "step_id": "call",
            "name": "call",
            "target": "chain",
            "inputs": {"topic": "subject"},
            "returns": {"final": "result"},
            "success_step_id": "",
        }
        self.workflow = registry.add_document(
            _document(
                {"prep": _auto("prep", "call", "prep"), "call": call},
                ident="compose",
                inputs=("subject",),
                outputs=("final",),
                callables=("prep",),
                executables=("chain",),
            )
        )
        self.registry = registry
        self.ctx = RunContext(
            llm_interface=self.model,
            event_log=shared,
            cancel=self.token,
            callables=names,
            registry=registry,
            **context,
        )
        self.agent = A.ReactAgent(
            _agent_tools(on_call=on_tool), config=pr._config(), context=self.ctx
        )
        self.engine = ReasoningEngine(context=self.ctx)

    @property
    def taken(self) -> int:
        assert self.ctx.budget is not None
        return self.ctx.budget.steps_taken

    async def agent_run(self, how: str) -> Any:
        from tests import pattern_runs as pr

        if how == "sync":
            return await asyncio.to_thread(self.agent.run, pr._TASK)
        return await self.agent.arun(pr._TASK)

    async def solve(self, how: str) -> Any:
        # no async API on the engine: both twins run the sync body on a worker thread
        return await asyncio.to_thread(self.engine.solve_problem, sc._PROBLEM)

    async def flow(self, how: str) -> Any:
        inputs = {"subject": "t"}
        if how == "sync":
            return await asyncio.to_thread(self.workflow.run, inputs, context=self.ctx)
        return await self.workflow.arun(inputs, context=self.ctx)

    async def attempt(self, run: Any, how: str) -> tuple[Any, BaseException | None]:
        try:
            return await run(how), None
        except Exception as exc:
            return None, exc

    def types(self) -> list[str]:
        return [e["type"] for e in rig.events_of(self.log)]

    def paired(self) -> bool:
        types = self.types()
        return types.count("llm/request") == (
            types.count("llm/response") + types.count("provider/error")
        )

    def idle(self) -> bool:
        e = self.engine
        return (
            e.orchestrator.list_active_conversations() == []
            and e.classifier.list_active_conversations() == []
            and dict(e.orchestrator.fsm_manager.instances) == {}
            and not e._solve_lock.locked()
        )

    def close(self) -> None:
        self.engine.close()
        self.log.close()


@pytest.fixture
def three(tmp_path: Path) -> Any:
    made: list[_Three] = []

    def make(**kw: Any) -> _Three:
        made.append(_Three(tmp_path / f"t{len(made)}", **kw))
        return made[-1]

    yield make
    for t in made:
        t.close()


HOW = pytest.mark.parametrize("how", ["sync", "async"])


_ORDERS = [
    ("agent_run", "solve", "flow"),
    ("flow", "agent_run", "solve"),
    ("solve", "flow", "agent_run"),
]


def _cancelled(who: str, result: Any, error: BaseException | None) -> bool:
    """Did runner ``who`` end because the token of the context was set?"""
    from fsm_llm.core.codes import Reason, Status

    if who == "agent_run":
        return isinstance(error, RunCancelledError)
    if who == "solve":
        return isinstance(error, ReasoningExecutionError) and (
            RunCancelledError in _solve_chain(error)
        )
    return error is None and (result.status, result.reason) == (
        Status.ABORTED,
        Reason.CANCELLED,
    )


def _ledger_stop(who: str, result: Any, error: BaseException | None) -> bool:
    """Did runner ``who`` end because the shared ledger refused a step (named as such)?"""
    from fsm_llm.agents.exceptions import BudgetExhaustedError
    from fsm_llm.core.codes import Reason, Status

    if who == "agent_run":
        return isinstance(error, BudgetExhaustedError) and (
            "shared run steps" in str(error)
        )
    if who == "solve":
        return isinstance(error, ReasoningExecutionError) and any(
            isinstance(c, RunBudgetExceededError) for c in cause_chain(error)
        )
    if error is not None:
        return False
    return (result.status, result.reason) == (
        Status.ABORTED,
        Reason.BUDGET_SPENT,
    ) and result.detail.get("shared_budget") == "steps"


def _roots(t: _Three) -> dict[str, list[str]]:
    """The event types of every conversation tree of the log, by the state it started in."""
    roots: dict[str, list[str]] = {}
    for cid in t.log.conversation_ids():
        events = list(t.log.read(cid))
        first = next(e["state"] for e in events if e.get("state"))
        roots[first] = [e["type"] for e in events]
    return roots


class TestOneLog:
    """Row 1: the runners of one context write to the ONE ``EventLog`` object of it."""

    @HOW
    async def test_every_runner_writes_its_trees_to_the_one_log(
        self, how: str, three: Any
    ) -> None:
        t = three(budget=RunBudget())
        agent, e1 = await t.attempt(t.agent_run, how)
        solved, e2 = await t.attempt(t.solve, how)
        flow, e3 = await t.attempt(t.flow, how)
        assert (e1, e2, e3) == (None, None, None)  # no LogLockedError, no refusal
        assert agent.stop_reason == "answered"
        assert solved[0] == sc._VALID_SCRIPT["final_solution"]
        assert flow.outputs == {"final": "text"}
        roots = _roots(t)
        # the agent root, the orchestrator and the classifier roots, the child tree of the workflow
        assert sorted(roots) == ["a", "analyze_domain", "problem_analysis", "think"]
        assert all("llm/request" in types for types in roots.values())
        assert t.paired() and len(t.log.conversation_ids()) == 4
        assert t.idle()

    @HOW
    async def test_a_path_valued_log_gives_each_runner_a_log_of_its_own(
        self, how: str, three: Any
    ) -> None:
        """The runners say it: a PATH is not an object, so each runner opens its own ``EventLog``
        on it (the recommended form is an object, row above). Four trees, four files."""
        t = three(budget=RunBudget(), path_log=True)
        for run in (t.agent_run, t.solve, t.flow):
            _result, error = await t.attempt(run, how)
            assert error is None
        logs = sorted((t.path / "logs").glob("*.jsonl"))
        assert len(logs) == 4


class TestOneCancel:
    """Row 2: ONE token. A set token stops every runner at its next check, none asks the model
    again, the conversations are ended and every request has its end. The step in flight when
    the token is set ends first (the cancel is cooperative and between steps)."""

    @HOW
    async def test_a_token_set_before_the_runs_stops_all_three_before_any_model_call(
        self, how: str, three: Any
    ) -> None:
        """Kills: any runner outside the cancel (it would ask the double); the pre-run check
        dropped from the agent, the engine or the workflow adapter."""
        t = three(budget=RunBudget())
        t.token.cancel()
        for who in ("agent_run", "solve", "flow"):
            result, error = await t.attempt(getattr(t, who), how)
            assert _cancelled(who, result, error), who
        assert t.model.calls == 0 and t.taken == 0
        assert t.types().count("llm/request") == 0 and t.idle()
        t.engine.close()  # still closable after a cancelled solve

    @HOW
    async def test_a_token_set_inside_the_agent_run_stops_the_two_runners_that_follow(
        self, how: str, three: Any
    ) -> None:
        """The double (the tool of the agent) sets the token during step 1: the agent ends at
        step 2's check, BEFORE the charge; then the solve and the workflow end at their first
        check with the double's count unchanged."""
        t = three(on_tool=lambda: t.token.cancel(), budget=RunBudget())
        result, error = await t.attempt(t.agent_run, how)
        assert _cancelled("agent_run", result, error)
        frozen = t.model.calls
        assert frozen > 0 and t.taken == 1  # the check goes before the charge of step 2
        for who in ("solve", "flow"):
            result, error = await t.attempt(getattr(t, who), how)
            assert _cancelled(who, result, error), who
        assert (t.model.calls, t.taken) == (frozen, 1)
        assert t.paired() and t.idle()

    @HOW
    @pytest.mark.parametrize(
        ("who", "trigger", "taken", "calls"),
        [
            ("agent_run", None, 1, None),
            ("solve", "problem_domain", 2, 11),
            ("solve", "conclusion", 7, 35),
            ("flow", "result", 3, None),
        ],
        ids=["agent_tool", "solve_classifier", "solve_strategy", "workflow_child"],
    )
    async def test_a_token_set_during_each_runner_stops_that_runner_at_its_next_check(
        self,
        how: str,
        who: str,
        trigger: str | None,
        taken: int,
        calls: int | None,
        three: Any,
    ) -> None:
        """Kills: the hook not passed to that runner (M25), the cancel swallowed (M13), the
        token replaced by a new one in a nested run (M27: the child FSM of the workflow would run
        to its end), the charge before the check (M2). The conversation is ended, every request
        has its end, and the NEXT call of the runner ends at its first check."""
        holder: list[_Three] = []
        cancel = lambda: holder[0].token.cancel()  # noqa: E731
        t = three(
            triggers={trigger: cancel} if trigger else None,
            on_tool=cancel if who == "agent_run" else None,
            budget=RunBudget(),
        )
        holder.append(t)
        result, error = await t.attempt(getattr(t, who), how)
        assert _cancelled(who, result, error), (result, error)
        assert t.taken == taken
        if calls is not None:
            assert t.model.calls == calls
        assert t.idle() and t.paired()
        frozen = (t.model.calls, t.taken)
        result, error = await t.attempt(getattr(t, who), how)
        assert _cancelled(who, result, error)
        assert (t.model.calls, t.taken) == frozen


class TestOneBudget:
    """Row 3: ONE ledger. ``budget.steps_taken`` is the sum of the exact counts of the runners
    (the engine loop steps of the agent, of the solve and its classifier, of the child FSM, and
    the steps of the workflow itself); a spent ledger refuses the next run at its first step."""

    @HOW
    async def test_the_ledger_equals_the_sum_of_the_three_exact_counts(
        self, how: str, three: Any
    ) -> None:
        """Kills: a runner outside the ledger (M25, M20: the classifier's 3 steps), a ledger
        copied by ``derive`` (M4, M26: the child FSM's 2 steps are lost) or lost by the second
        ``replace`` (M3), a charge per call (M1)."""
        t = three(budget=RunBudget())
        counts = []
        with rig.HookSpy() as spy:
            for run in (t.agent_run, t.solve, t.flow):
                result, error = await t.attempt(run, how)
                assert error is None
                counts.append(t.taken)
        flow = result
        assert counts == [_AGENT_STEPS, _AGENT_STEPS + _SOLVE_STEPS, _TOTAL]
        loops = [run["steps"] for run in spy.runs]
        # agent, orchestrator (7), classifier (3), the child FSM (2): every loop hooked, once a step
        assert [len(s) for s in loops] == [3, 7, 3, 2]
        assert all(s == list(range(1, len(s) + 1)) for s in loops)
        assert flow.detail["steps"] == 2  # the workflow's own steps, no loop under them
        assert t.taken == sum(len(s) for s in loops) + flow.detail["steps"] == 17

    def test_a_derived_context_shares_the_same_ledger_by_identity(
        self, three: Any
    ) -> None:
        t = three(budget=RunBudget(max_steps=5))
        child = t.ctx.derive(parent="w", max_steps=2)
        grandchild = child.derive(parent="x")
        assert child.budget is t.ctx.budget is grandchild.budget
        assert dataclasses.replace(child, max_seconds=3.0).budget is t.ctx.budget

    @HOW
    async def test_a_ledger_as_large_as_the_three_runs_lets_all_three_finish(
        self, how: str, three: Any
    ) -> None:
        """Kills: ``spent()`` off by one (M7) on the full side: the last charge is allowed."""
        t = three(budget=RunBudget(max_steps=_TOTAL))
        for run in (t.agent_run, t.solve, t.flow):
            _result, error = await t.attempt(run, how)
            assert error is None
        assert t.taken == _TOTAL and t.ctx.budget_spent() == "steps"
        assert t.paired() and t.idle()

    @HOW
    @pytest.mark.parametrize("order", _ORDERS, ids=lambda o: "-".join(o))
    async def test_a_ledger_of_total_minus_one_stops_exactly_the_last_runner_mid_run(
        self, how: str, order: tuple[str, ...], three: Any
    ) -> None:
        """Kills: ``spent()`` off by one (M7) on the empty side; a runner that ignores the
        ledger. The count is exactly total - 1 whichever runner comes last; the refused step
        asked the double nothing; the stopped runner's conversations are ended."""
        t = three(budget=RunBudget(max_steps=_TOTAL - 1))
        stopped = []
        for who in order:
            result, error = await t.attempt(getattr(t, who), how)
            if error is not None or (
                getattr(result, "status", None) is not None
                and result.status.value != "ok"
            ):
                stopped.append(who)
                assert _ledger_stop(who, result, error), (who, result, error)
        assert stopped == [order[-1]]
        assert t.taken == _TOTAL - 1 and t.ctx.budget_spent() == "steps"
        assert t.paired() and t.idle()

    @HOW
    async def test_a_further_run_is_refused_at_its_first_step_and_asks_the_double_nothing(
        self, how: str, three: Any
    ) -> None:
        """Kills: no pre-run check (a greeting or a first step is spent), a refusal that
        charges, a runner that reads no ledger. The reasoning run goes through the executable
        (the engine alone spends its greeting on a spent ledger: a known limit, recorded)."""
        from fsm_llm.core.codes import Reason, Status
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.reasoning.executables import SOLVE_ID

        t = three(budget=RunBudget(max_steps=_TOTAL))
        for run in (t.agent_run, t.solve, t.flow):
            _result, error = await t.attempt(run, how)
            assert error is None
        frozen = (t.model.calls, t.taken, t.types().count("llm/request"))
        assert frozen[1] == _TOTAL

        result, error = await t.attempt(t.agent_run, how)
        assert _ledger_stop("agent_run", result, error)
        result, error = await t.attempt(t.flow, how)
        assert _ledger_stop("flow", result, error) and result.handle is None
        solve_registry = ExecutableRegistry()
        solve_registry.load_sources()
        solve = solve_registry.get(SOLVE_ID)
        if how == "sync":
            done = await asyncio.to_thread(
                solve.run, {"problem": sc._PROBLEM}, context=t.ctx
            )
        else:
            done = await solve.arun({"problem": sc._PROBLEM}, context=t.ctx)
        assert (done.status, done.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert done.detail["shared_budget"] == "steps"
        assert (t.model.calls, t.taken, t.types().count("llm/request")) == frozen

    @HOW
    async def test_the_engine_alone_on_a_spent_ledger_names_the_shared_budget(
        self, how: str, three: Any
    ) -> None:
        spent = RunBudget(max_steps=1)
        spent.charge()
        t = three(budget=spent)
        result, error = await t.attempt(t.solve, how)
        assert _ledger_stop("solve", result, error)
        assert "170" not in str(error) and t.taken == 1 and t.idle()

    @HOW
    async def test_a_workflow_step_refused_by_the_ledger_is_aborted_and_asks_the_double_nothing(
        self, how: str, three: Any
    ) -> None:
        """The workflow's OWN hook: step 1 (`prep`) is charged, step 2 (`call`) is refused and the
        stop is `aborted/budget_spent` with the fact of the ledger; the child it starts is refused
        before any model call (the pre-run check), so the double was never asked."""
        t = three(budget=RunBudget(max_steps=1))
        result, error = await t.attempt(t.flow, how)
        assert error is None and _ledger_stop("flow", result, error)
        assert result.detail["shared_budget"] == "steps"
        assert (t.taken, t.model.calls) == (1, 0) and t.paired()

    @HOW
    async def test_a_ledger_stop_inside_the_child_run_is_aborted_at_the_workflow_level(
        self, how: str, three: Any
    ) -> None:
        """Plan 7efd5cb4 step 8.1 (D-011; step 8 had pinned this as a known limit). The ledger
        refuses a step of the CHILD FSM of an `ExecutableStep`: the child ends
        `aborted/budget_spent`, the instance is `failed` for the engine, and the adapter maps it
        to `aborted/budget_spent` with the fact of the ledger and the refusal's budget and limit.
        The count is exact, the child's conversation is ended, the double asked nothing more."""
        from fsm_llm.core.codes import Reason, Status

        t = three(
            budget=RunBudget(max_steps=3)
        )  # prep, call, the child's step 1; step 2 refused
        result, error = await t.attempt(t.flow, how)
        assert error is None and _ledger_stop("flow", result, error)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (result.detail["budget"], result.detail["limit"]) == ("steps", 3)
        assert result.detail["workflow_status"] == "failed" and result.handle
        assert (t.taken, t.ctx.budget_spent()) == (3, "steps") and t.paired()

    @HOW
    async def test_a_seconds_ledger_starts_at_its_first_charge_and_stops_every_runner(
        self, how: str, three: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: the clock started at construction (M6: the first charge would find 8000 s
        spent). The clock is a stand-in that moves only when the double moves it: no sleep."""
        from fsm_llm.agents.exceptions import BudgetExhaustedError
        from fsm_llm.core.codes import Reason, Status

        clock = _Clock()
        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", clock)
        ledger = RunBudget(max_seconds=5.0)  # built at 1000
        clock.now = 9000.0  # a long wait before the first run
        jump = lambda: setattr(clock, "now", clock.now + 100.0)  # noqa: E731
        t = three(budget=ledger, triggers={"conclusion": jump})
        result, error = await t.attempt(t.agent_run, how)
        assert error is None and result.stop_reason == "answered"
        # the solve: the double moves the clock inside the strategy; the next check refuses it
        _solved, error = await t.attempt(t.solve, how)
        assert isinstance(error, ReasoningExecutionError)
        spent = next(
            c for c in cause_chain(error) if isinstance(c, RunBudgetExceededError)
        )
        assert (spent.budget, spent.limit) == ("seconds", 5.0)
        assert t.ctx.budget_spent() == "seconds" and t.idle()
        frozen = (t.model.calls, t.taken)
        # then both others are refused at their first check, the double is not asked
        result, error = await t.attempt(t.agent_run, how)
        assert isinstance(error, BudgetExhaustedError)
        assert "shared run seconds" in str(error)
        result, error = await t.attempt(t.flow, how)
        assert error is None
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["shared_budget"] == "seconds"
        assert (t.model.calls, t.taken) == frozen and t.paired()


class TestNoGrant:
    """Invariant 1 across the three runners: a context carries no grant and no approver."""

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
        """Every key set any ``API.update_context`` of the process was asked to write."""
        from fsm_llm import API

        writes: list[set[str]] = []
        original = API.update_context

        def spy(self: Any, conversation_id: str, update: dict[str, Any]) -> Any:
            writes.append(set(update))
            return original(self, conversation_id, update)

        monkeypatch.setattr(API, "update_context", spy)
        return writes

    def test_the_spy_sees_a_grant_when_an_agent_with_its_own_approver_writes_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative control: the spy is able to FAIL (a HITL agent writes the driver grant)."""
        from fsm_llm import agents as A
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.policy import HumanInTheLoop
        from tests import pattern_runs as pr

        writes = self._spy(monkeypatch)
        agent = A.ReactAgent(
            _agent_tools(flagged=True),
            config=pr._config(),
            llm_interface=pr.ScriptedModel(),
            hitl=HumanInTheLoop(
                approval_policy=lambda c, x: True, approval_callback=lambda q: True
            ),
        )
        agent.run(pr._TASK)
        assert any(ContextKeys.DRIVER_APPROVAL in keys for keys in writes)

    @HOW
    async def test_no_runner_of_a_context_consults_an_approver_or_writes_a_grant(
        self, how: str, three: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: ``callables`` consumed by a runner (M12: the agent would hold the approver and
        ask it), a grant written for a context run, a grant in a result. The context holds an
        approver under ``hitl`` in ITS ``callables`` and a ledger: neither is a grant."""
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.policy import HumanInTheLoop

        asked: list[str] = []
        hitl = HumanInTheLoop(
            approval_policy=lambda c, x: asked.append("policy") or True,
            approval_callback=lambda q: asked.append("callback") or True,
        )
        writes = self._spy(monkeypatch)
        t = three(budget=RunBudget(), hitl=hitl)
        _agent, e1 = await t.attempt(t.agent_run, how)
        solved, e2 = await t.attempt(t.solve, how)
        flow, e3 = await t.attempt(t.flow, how)
        assert (e1, e2, e3) == (None, None, None)
        assert asked == []
        assert not any(ContextKeys.DRIVER_APPROVAL in keys for keys in writes)
        held = [
            *vars(t.agent).values(),
            *t.agent._api_kwargs.values(),
            *vars(t.engine).values(),
            *t.engine.api_kwargs.values(),
        ]
        assert not any(v is hitl for v in held)
        assert t.agent.hitl is None and t.agent._hitl_active is False
        for data in (solved[1]["final_context"], flow.outputs):
            assert ContextKeys.DRIVER_APPROVAL not in data
            assert not any(v is t.ctx or v is hitl for v in data.values())
        # nothing in the context or its children is a data member: the declared fields only
        fields = {f.name for f in dataclasses.fields(RunContext)}
        assert fields == {
            "llm_interface",
            "llm_factory",
            "event_log",
            "max_steps",
            "max_seconds",
            "cancel",
            "callables",
            "parent",
            "registry",
            "budget",
        }
        assert {f.name for f in dataclasses.fields(t.ctx.derive(parent="w"))} == fields


def _on_three_threads(t: _Three) -> dict[str, tuple[Any, BaseException | None]]:
    """The three runners, one thread each, started together; ``{runner: (result, error)}``."""
    from tests import pattern_runs as pr

    outcomes: dict[str, tuple[Any, BaseException | None]] = {}
    guard = threading.Lock()
    bodies = {
        "agent_run": lambda: t.agent.run(pr._TASK),
        "solve": lambda: t.engine.solve_problem(sc._PROBLEM),
        "flow": lambda: t.workflow.run({"subject": "t"}, context=t.ctx),
    }

    def work(name: str) -> None:
        try:
            result, error = bodies[name](), None
        except Exception as exc:
            result, error = None, exc
        with guard:
            outcomes[name] = (result, error)

    threads = [threading.Thread(target=work, args=(n,), daemon=True) for n in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()  # every wait inside a runner is bounded (a barrier with a timeout)
    return outcomes


#: the steps the ledger holds when each runner first asks for its step 2: the agent's step 1, the
#: solve's (orchestrator step 1 and the classifier's first step), the workflow's `prep`
_GATE_PREFIX = 4


class TestThreeThreadsOneLedger:
    """Row 4: the three runners on three threads, one ledger, 8 of 8 repeats.

    The lock is proven here the way the WINDOW row of ``test_run_context.py`` proves it: a
    stand-in clock that waits for the other threads INSIDE the budget check opens the window. The
    three runners are held at their first step 2 (a ``Barrier`` with a timeout), then released
    together onto a ledger with room for exactly one more step: with the lock one is accepted and
    two are refused; without it all three pass the check and the ledger overshoots."""

    @pytest.mark.parametrize("repeat", range(8))
    def test_three_runners_on_three_threads_count_exactly_the_sum(
        self, repeat: int, three: Any
    ) -> None:
        t = three(budget=RunBudget(max_steps=_TOTAL))
        outcomes = _on_three_threads(t)
        assert {n: e for n, (_r, e) in outcomes.items()} == {
            "agent_run": None,
            "solve": None,
            "flow": None,
        }
        assert t.taken == _TOTAL and t.paired() and t.idle()
        assert len(t.log.conversation_ids()) == 4

    @pytest.mark.parametrize("repeat", range(8))
    def test_released_together_onto_one_free_step_exactly_one_is_accepted(
        self, repeat: int, three: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The WINDOW row of ``test_run_context.py``, with the real runners. The three are held
        at their first step 2 (a ``Barrier`` with a timeout) and released together onto a ledger
        with room for ONE more step. The ledger's record waits for the other two threads right
        AFTER its check and before it counts: with the lock one thread is inside at a time (the
        wait times out harmlessly, once), without it all three pass the check. Kills: the lock
        removed (M5): three charges accepted, the ledger over its limit."""
        from fsm_llm.engine.fsm.budget import TurnBudget

        seen: set[int] = set()
        arrivals: list[int] = []
        accepted: list[int] = []
        armed = threading.Event()
        window = threading.Barrier(3, timeout=0.3)

        class Racy(TurnBudget):
            def spent(self) -> Any:
                verdict = super().spent()  # the check is done ...
                if armed.is_set():
                    try:
                        window.wait()  # ... and the others have done theirs
                    except threading.BrokenBarrierError:
                        pass
                return verdict

        ledger = RunBudget(max_steps=_GATE_PREFIX + 1)
        ledger._record = Racy(_GATE_PREFIX + 1, None)
        charge = ledger.charge

        def counting_charge() -> None:
            charge()
            if armed.is_set():
                accepted.append(1)

        monkeypatch.setattr(ledger, "charge", counting_charge)

        def at_the_gate() -> (
            None
        ):  # once, in the last thread to arrive, before the release
            arrivals.append(ledger.steps_taken)
            armed.set()

        gate = threading.Barrier(3, timeout=20, action=at_the_gate)
        original = RunContext.step_hook

        def gated(self: RunContext) -> Any:
            hook = original(self)
            owner = threading.get_ident()

            def wrapped(step: int) -> None:
                if step == 2 and owner not in seen:
                    seen.add(owner)
                    gate.wait()
                hook(step)

            return wrapped

        monkeypatch.setattr(RunContext, "step_hook", gated)
        t = three(budget=ledger)
        outcomes = _on_three_threads(t)
        assert arrivals == [_GATE_PREFIX]
        assert (
            len(accepted) == 1
        )  # exactly one of the three released charges was accepted
        assert ledger.steps_taken == _GATE_PREFIX + 1
        for who, (result, error) in outcomes.items():
            assert _ledger_stop(who, result, error), (who, result, error)
        assert t.paired() and t.idle()


class TestOneBudgetOverARunLoop:
    """The ledger of the context counts STEPS, not hook calls: a run loop that offers one step
    number twice (the ended pushed frame it asks the hook to pop) is charged once."""

    @pytest.mark.parametrize("how", ["sync", "async"])
    async def test_a_loop_that_offers_a_number_twice_is_charged_once_on_the_shared_ledger(
        self, how: str, three: Any
    ) -> None:
        """Kills: a charge per hook call (M1). The same ledger the three runners share counts
        this loop's engine steps (3) and not its four hook calls."""
        from tests.test_fsm_llm.test_advance import _ScriptedLLM, _start, _trip_fsm
        from tests.test_fsm_llm.test_run_until_terminal import _push_sub_on_entry

        t = three(budget=RunBudget())
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        _push_sub_on_entry(api, conv_id, "plan")
        hook = t.ctx.step_hook()
        seen: list[int] = []

        def before_step(step: int) -> None:
            seen.append(step)
            if api.get_stack_depth(conv_id) > 1 and api.has_conversation_ended(conv_id):
                api.pop_fsm(conv_id)
            hook(step)

        if how == "sync":
            results = await asyncio.to_thread(
                api.run_until_terminal, conv_id, max_steps=10, before_step=before_step
            )
        else:
            results = await api.arun_until_terminal(
                conv_id, max_steps=10, before_step=before_step
            )
        assert seen == [1, 2, 3, 3] and len(results) == 3
        assert t.taken == 3


class TestDirectWorkflowEngineOnTheContext:
    """Plan 7efd5cb4 step 9 (D-012): a DIRECT ``WorkflowEngine(context=ctx)`` (no adapter) runs the
    same workflow as the third runner of the acceptance under the same context: its own steps and
    the child FSM of its ``ExecutableStep`` are charged to ONE ledger, the child writes to the
    shared log, and the token and the ledger stop it before a step starts. The engine has an async
    API only."""

    @staticmethod
    async def _run(t: Any) -> tuple[Any, str, Any]:
        from fsm_llm.workflows import WorkflowEngine

        engine = WorkflowEngine(context=t.ctx)
        definition = t.workflow.build_definition(t.ctx.callables, t.ctx.registry)
        engine.register_workflow(definition)
        iid = await asyncio.wait_for(
            engine.start(definition.workflow_id, initial_context={"subject": "t"}), 120
        )
        instance = engine.get_instance(iid)
        await engine.close()
        return engine, iid, instance

    async def test_its_steps_and_its_childs_steps_share_one_ledger_and_one_log(
        self, three: Any
    ) -> None:
        """Kills: the engine not charging (taken 2), the scope not set per step (the child would
        build a bare context: no double, no ledger), the ledger copied into the child."""
        t = three(budget=RunBudget())
        _, _, instance = await self._run(t)
        assert instance.status.value == "completed"
        assert (
            t.taken == _FLOW_STEPS
        )  # prep, call, then the two loop steps of the child
        assert t.types().count("llm/request") > 0 and t.model.calls > 0
        assert t.paired()

    async def test_a_ledger_that_ends_in_the_child_fails_the_instance(
        self, three: Any
    ) -> None:
        t = three(budget=RunBudget(max_steps=3))
        _, _, instance = await self._run(t)
        assert instance.status.value == "failed" and t.taken == 3
        assert t.ctx.last_refusal() is not None and t.paired()

    async def test_a_ledger_refusal_by_the_engine_itself_starts_no_step(
        self, three: Any
    ) -> None:
        """The ledger allows only ``prep``: the refusal of step ``call`` is the engine's, and the
        double is never asked (the child never started)."""
        t = three(budget=RunBudget(max_steps=1))
        _, _, instance = await self._run(t)
        assert instance.status.value == "failed" and t.taken == 1
        assert t.model.calls == 0 and t.types().count("llm/request") == 0

    async def test_a_token_set_by_a_step_cancels_before_the_next_step(
        self, three: Any
    ) -> None:
        """Kills: the engine not reading the token (the child would run and ask the double)."""
        holder: list[Any] = []
        t = three(on_prep=lambda: holder[0].token.cancel(), budget=RunBudget())
        holder.append(t)
        _, _, instance = await self._run(t)
        assert instance.status.value == "cancelled"
        assert t.taken == 1 and t.model.calls == 0
