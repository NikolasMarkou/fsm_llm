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
is reported. Later steps of the plan add the acceptance class to this file.
"""

from __future__ import annotations

import copy
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
        found = first_difference(sections[section][case], GOLDEN[section][case])
        assert found is None, f"{section}/{case}{found}"


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
        """The context itself reaches the engine with step 6; the interface and log already do."""
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
