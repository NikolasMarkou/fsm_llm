"""The fix round of the phase review of plan 7efd5cb4 (step 17, D-015): one class per finding.

Every row runs on doubles only. A row names the mutant it kills; each mutant was run on a full copy
of the tree with this file copied in, and every row of 17.1 to 17.4 was red on a ``git archive`` of
the parent commit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions import RunBudgetExceededError
from fsm_llm.engine.context import RunBudget, RunContext
from tests.test_fsm_llm.test_run_context_adoption import _Rig, _Solve


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)


@pytest.fixture
def solve_rig(tmp_path: Path) -> Any:
    made: list[_Solve] = []

    def make(script: dict[str, Any] | None = None, **kw: Any) -> _Solve:
        made.append(_Solve(tmp_path / f"r{len(made)}", script, **kw))
        return made[-1]

    yield make
    for r in made:
        r.close()


def refusal_of(budget: RunBudget) -> RunBudgetExceededError:
    """The error a REAL refused charge of *budget* raises (spends nothing when it is not full)."""
    with pytest.raises(RunBudgetExceededError) as caught:
        budget.charge()
    return caught.value


def full(max_steps: int) -> RunBudget:
    """A ledger charged exactly to its limit: spent, and nothing was ever refused."""
    budget = RunBudget(max_steps=max_steps)
    for _ in range(max_steps):
        budget.charge()
    assert budget.last_refusal is None
    return budget


@pytest.mark.usefixtures("no_model")
class TestALedgerStopIsDecidedByIdentity:
    """17.1: a run's OWN limit is never read as the shared ledger because the ledger happens to be
    spent too; the stop is the ledger's only when the ledger itself refused (D-015)."""

    def test_the_agents_own_timeout_beside_an_exactly_spent_ledger_stays_its_own(
        self, tmp_path: Path
    ) -> None:
        """Kills M36: `_budget_error` asks `budget_spent()` (a fact read afterwards)."""
        from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError

        agent = _Rig(tmp_path, budget=full(2)).agent()
        own = agent._budget_error(RunBudgetExceededError("seconds", 30.0, 2), None)
        assert isinstance(own, AgentTimeoutError)
        steps = agent._budget_error(RunBudgetExceededError("steps", 3, 3), None)
        assert isinstance(steps, BudgetExhaustedError)
        assert "loop turns" in str(steps) and "shared" not in str(steps)

    def test_a_real_refusal_of_the_ledger_is_the_shared_budget_with_its_own_limit(
        self, tmp_path: Path
    ) -> None:
        from fsm_llm.agents.exceptions import BudgetExhaustedError

        budget = full(2)
        agent = _Rig(tmp_path, budget=budget).agent()
        mine = refusal_of(budget)
        refused = agent._budget_error(mine, None)
        assert isinstance(refused, BudgetExhaustedError)
        assert "shared run steps" in str(refused)
        # a concurrent runner is refused next: `last_refusal` moves on, this stop stays the ledger's
        refusal_of(budget)
        assert budget.last_refusal is not mine
        assert "shared run steps" in str(agent._budget_error(mine, None))

    def test_the_refusal_of_another_ledger_is_not_this_contexts(
        self, tmp_path: Path
    ) -> None:
        """Kills M43: any refusal of any ledger counts (a nested context with its own ledger)."""
        elsewhere = refusal_of(full(1))
        agent = _Rig(tmp_path, budget=RunBudget(max_steps=1)).agent()
        assert "shared" not in str(agent._budget_error(elsewhere, None))
        assert RunContext(budget=RunBudget()).stopped_by_ledger(elsewhere) is None
        assert RunContext().stopped_by_ledger(elsewhere) is None

    def test_a_cause_chain_is_read_and_a_plain_error_never_is_the_ledger(self) -> None:
        budget = full(1)
        ctx = RunContext(budget=budget)
        mine = refusal_of(budget)
        try:
            raise ValueError("wrapped") from mine
        except ValueError as wrapped:
            assert ctx.stopped_by_ledger(wrapped) is mine
        assert ctx.stopped_by_ledger(RunBudgetExceededError("steps", 1, 1)) is None
        assert ctx.stopped_by_ledger(ValueError("x")) is None
        assert ctx.ledger_detail(mine) == {"shared_budget": "steps"}
        assert ctx.ledger_detail(RunBudgetExceededError("steps", 1, 1)) == {}
        assert ctx.ledger_detail() == {"shared_budget": "steps"}  # the pre-run fact

    def test_the_engines_own_steps_tie_with_the_ledger_is_the_runs_own_stop(
        self,
    ) -> None:
        """Kills M40: the adapter of an FSM body adds `shared_budget` to ITS OWN step ceiling
        because the ledger reached its limit on the same step."""
        from fsm_llm.core.codes import Reason, Status
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.testing import ScriptedLLM
        from tests import run_context_rig as rig

        long = rig._fsm_documents()["long"]

        def run(ledger: int, own: int) -> Any:
            ctx = RunContext(
                llm_interface=ScriptedLLM(default_response="r"),
                max_steps=own,
                budget=RunBudget(max_steps=ledger),
            )
            return (
                ExecutableRegistry().add_document(long).run({"topic": "t"}, context=ctx)
            )

        tie = run(ledger=2, own=2)
        assert (tie.status, tie.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (tie.detail["budget"], tie.detail["limit"]) == ("steps", 2)
        assert "shared_budget" not in tie.detail
        refused = run(ledger=2, own=3)
        assert refused.detail["shared_budget"] == "steps"
        assert refused.detail["limit"] == 2

    def test_an_agent_executable_reads_the_ledger_only_from_its_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills M41: `PatternExecutable.execute` adds the ledger fact to its own timeout."""
        from fsm_llm import agents as A
        from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
        from fsm_llm.engine.executables import ExecutableRegistry
        from tests import pattern_runs as pr

        registry = ExecutableRegistry()
        registry.load_sources()
        ledgers: list[RunBudget] = []

        def context() -> RunContext:
            ledgers.append(RunBudget(max_steps=2))
            return RunContext(llm_interface=pr.ScriptedModel(), budget=ledgers[-1])

        def own(self: Any, *a: Any, **k: Any) -> Any:
            ledgers[-1].charge()
            ledgers[-1].charge()  # exactly spent, nothing refused
            raise AgentTimeoutError(30.0)

        monkeypatch.setattr(A.DebateAgent, "run", own)
        result = registry.get("debate").run({"task": "t"}, context=context())
        assert result.detail["error"] == "AgentTimeoutError"
        assert "shared_budget" not in result.detail

        def refused(self: Any, *a: Any, **k: Any) -> Any:
            ledgers[-1].charge()
            ledgers[-1].charge()
            raise BudgetExhaustedError("shared run steps", 2) from refusal_of(
                ledgers[-1]
            )

        monkeypatch.setattr(A.DebateAgent, "run", refused)
        result = registry.get("debate").run({"task": "t"}, context=context())
        assert result.detail["shared_budget"] == "steps"


@pytest.mark.usefixtures("no_model")
class TestTheReasoningRunnersReadTheLedgerByIdentity:
    """17.1 for the reasoning engine, its classifier, its executable and the reason tool."""

    def test_the_orchestrators_own_170_steps_beside_a_spent_ledger_keep_their_text(
        self, solve_rig: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills M37: the message of an unfinished solve says `shared run budget` for the
        orchestrator's OWN error because the ledger is spent."""
        from fsm_llm.reasoning import ReasoningExecutionError

        r = solve_rig(budget=RunBudget(max_steps=50))

        def own(conversation_id: str, **kwargs: Any) -> Any:
            while r.ctx.budget_spent() is None:
                r.ctx.budget.charge()  # type: ignore[union-attr]
            raise RunBudgetExceededError("steps", 170, 170)

        monkeypatch.setattr(r.engine.orchestrator, "run_until_terminal", own)
        with pytest.raises(ReasoningExecutionError) as caught:
            r.solve()
        assert "within 170 steps" in str(caught.value)
        assert "shared run budget" not in str(caught.value)

    def test_the_classifiers_own_ten_steps_beside_a_spent_ledger_keep_their_text(
        self, solve_rig: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills M38: the classifier re-raises its OWN budget error as the ledger's."""
        from fsm_llm.reasoning.constants import ContextKeys
        from fsm_llm.reasoning.exceptions import ReasoningClassificationError
        from tests.test_fsm_llm_reasoning import test_engine_scripted as sc

        r = solve_rig(budget=RunBudget(max_steps=3))

        def own(conversation_id: str, **kwargs: Any) -> Any:
            while r.ctx.budget_spent() is None:
                r.ctx.budget.charge()  # type: ignore[union-attr]
            raise RunBudgetExceededError("steps", 10, 10)

        monkeypatch.setattr(r.engine.classifier, "run_until_terminal", own)
        with pytest.raises(ReasoningClassificationError, match="10 steps"):
            r.engine._classify_problem({ContextKeys.PROBLEM_STATEMENT: sc._PROBLEM})

    def test_the_solve_executable_adds_the_ledger_only_for_its_refusal(
        self, solve_rig: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills M42: `SolveExecutable._mapped` adds the fact to the engine's own budget error."""
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.reasoning import ReasoningEngine, ReasoningExecutionError
        from fsm_llm.reasoning.executables import SOLVE_ID
        from tests.test_fsm_llm_reasoning import test_engine_scripted as sc

        r = solve_rig(budget=RunBudget(max_steps=50))
        registry = ExecutableRegistry()
        registry.load_sources()

        def stop(cause: Any) -> Any:
            def solve(self: Any, *a: Any, **k: Any) -> Any:
                while r.ctx.budget_spent() is None:
                    r.ctx.budget.charge()  # type: ignore[union-attr]
                raise ReasoningExecutionError("did not finish") from cause()

            return solve

        fresh = lambda: RunContext(  # noqa: E731
            llm_interface=r.llm, budget=RunBudget(max_steps=2)
        )
        own = fresh()
        r.ctx = own
        monkeypatch.setattr(
            ReasoningEngine,
            "solve_problem",
            stop(lambda: RunBudgetExceededError("steps", 170, 170)),
        )
        result = registry.get(SOLVE_ID).run({"problem": sc._PROBLEM}, context=own)
        assert (result.detail["budget"], result.detail["limit"]) == ("steps", 170)
        assert "shared_budget" not in result.detail
        real = fresh()
        r.ctx = real
        monkeypatch.setattr(
            ReasoningEngine,
            "solve_problem",
            stop(lambda: refusal_of(real.budget)),  # type: ignore[arg-type]
        )
        result = registry.get(SOLVE_ID).run({"problem": sc._PROBLEM}, context=real)
        assert result.detail["shared_budget"] == "steps"

    def test_a_spent_ledger_beside_a_plain_failure_is_a_failed_reason_observation(
        self,
    ) -> None:
        """Kills M39: the reason tool re-raises ANY failure once the ledger is spent."""
        from fsm_llm import agents as A
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.reasoning import ReasoningExecutionError
        from tests import pattern_runs as pr
        from tests.test_fsm_llm.test_run_context_adoption import (
            TestReasoningReactUnderContext as T,
        )

        budget = RunBudget(max_steps=1)
        ctx = RunContext(llm_interface=T._models()(), budget=budget)
        agent = A.ReasoningReactAgent(pr._tools(), config=pr._config(), context=ctx)
        budget.charge()  # exactly spent, nothing refused

        def boom(*_a: Any, **_k: Any) -> Any:
            raise ReasoningExecutionError("the inner solve failed on its own")

        agent._reasoning_engine.solve_problem = boom  # type: ignore[method-assign]
        execute = agent._make_reasoning_tool_executor(agent._call_handlers(agent.tools))
        state = {ContextKeys.TOOL_NAME: "reason", ContextKeys.TOOL_INPUT: {"x": "1"}}
        state[ContextKeys.TASK] = pr._TASK
        assert execute(dict(state))[ContextKeys.TOOL_STATUS] == "failed"


class TestTheOfferOfAnEndedFrameIsNoStep:
    """17.2: `step_hook` used as a caller's `before_step` charges the step a run takes, never the
    offer of an ended pushed frame nobody pops (D-015). One rule for every use of the hook."""

    @pytest.mark.parametrize("how", ["run", "arun"])
    @pytest.mark.parametrize("limit", [None, 2])
    def test_a_pushed_frame_nobody_pops_costs_two_steps_and_two_charges(
        self, how: str, limit: int | None
    ) -> None:
        """Kills M45: the hook charges the offer (ledger 3 for 2 steps, and a ledger of 2 raises
        in the state of the frame instead of ending normally)."""
        from tests.test_fsm_llm.test_run_until_terminal import (
            _push_sub_on_entry,
            _ScriptedLLM,
            _start,
            _trip_fsm,
        )

        api, cid = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        _push_sub_on_entry(api, cid, "plan")
        budget = RunBudget(max_steps=limit)
        hook = RunContext(budget=budget).step_hook()
        if how == "run":
            results = api.run_until_terminal(cid, max_steps=10, before_step=hook)
        else:
            results = asyncio.run(
                api.arun_until_terminal(cid, max_steps=10, before_step=hook)
            )
        assert len(results) == 2 and budget.steps_taken == 2
        assert budget.last_refusal is None

    def test_the_offer_is_checked_for_the_token_and_charged_nothing(self) -> None:
        from fsm_llm.engine.context import RunCancelledError
        from fsm_llm.engine.fsm.budget import EndedFrameOffer

        ctx = RunContext(budget=RunBudget())
        hook = ctx.step_hook()
        hook(1)
        hook(EndedFrameOffer(2))
        assert ctx.budget.steps_taken == 1  # type: ignore[union-attr]
        hook(2)  # the step the pop made possible: charged now, once
        hook(2)
        assert ctx.budget.steps_taken == 2  # type: ignore[union-attr]
        ctx.cancel.cancel()
        with pytest.raises(RunCancelledError):
            hook(EndedFrameOffer(3))


class TestTheMetaBuilderRefusesAContext:
    """17.3: `MetaBuilderAgent` is not a `BaseAgent` and a run of it is not under a context in this
    release, so `context=` is refused by name at construction, not swallowed into the `API`s it
    builds later (partial adoption: a set token refused `start()`, the ledger was never charged)."""

    @pytest.mark.parametrize("value", ["anything", RunContext()])
    def test_the_keyword_is_a_type_error_naming_it(self, value: Any) -> None:
        """Kills M48: `context` not in the refusal (it is forwarded to every API built later)."""
        from fsm_llm.agents import MetaBuilderAgent
        from tests import pattern_runs as pr

        with pytest.raises(TypeError, match="context") as caught:
            MetaBuilderAgent(llm_interface=pr.ScriptedModel(), context=value)
        assert "RunContext" in str(caught.value)

    def test_no_context_builds_as_before(self) -> None:
        from fsm_llm.agents import MetaBuilderAgent
        from tests import pattern_runs as pr

        model = pr.ScriptedModel()
        assert MetaBuilderAgent(llm_interface=model) is not None
        assert MetaBuilderAgent(llm_interface=model, context=None) is not None


class TestACompletedInstanceStaysOk:
    """17.4: the commit wins. An instance that COMPLETED is `ok` even when its token was set by
    the last step (nothing is left to cancel, and the outputs are real); only an instance that did
    not complete reads the token (the row step 9 deleted with its old-design neighbours)."""

    @pytest.mark.parametrize("how", ["run", "arun"])
    def test_a_token_set_by_the_last_step_leaves_the_run_ok(self, how: str) -> None:
        """Kills M49: `_ended` reads the token BEFORE the COMPLETED branch."""
        from fsm_llm.core.codes import Status
        from fsm_llm.engine.context import CancelToken
        from tests.test_fsm_llm_workflows.test_workflow_executable import (
            _auto,
            _document,
            _executable,
            _names,
        )

        token = CancelToken()
        executable = _executable(
            _document({"a": _auto("a", "", "stop")}, callables=("stop",))
        )
        context = RunContext(
            cancel=token, callables=_names(stop=lambda _ctx: token.cancel())
        )
        if how == "run":
            result = executable.run({}, context=context)
        else:
            result = asyncio.run(executable.arun({}, context=context))
        assert token.cancelled and result.status is Status.OK
        assert result.detail["workflow_status"] == "completed"


@pytest.mark.usefixtures("no_model")
class TestCreateAgentReadsTheInterfaceOfTheContext:
    """17.5b: `create_agent(system_prompt=..., context=ctx)` builds the default config it needs for
    the instructions from the interface of the context, as `BaseAgent` does; with no `LLM_MODEL` it
    no longer raises `ModelNotConfiguredError` for a context that holds an interface."""

    def test_a_system_prompt_with_a_context_interface_and_no_model_builds(self) -> None:
        """Kills M50: the default config is built from the `llm_interface` keyword only."""
        from fsm_llm.agents import create_agent
        from tests import pattern_runs as pr
        from tests.test_fsm_llm.test_run_context_adoption import _agent_tools

        context = RunContext(llm_interface=pr.ScriptedModel())
        agent = create_agent(
            "react", _agent_tools(), system_prompt="Cite sources.", context=context
        )
        assert agent.config.instructions == "Cite sources."

    def test_no_interface_anywhere_is_still_the_usual_refusal(self) -> None:
        from fsm_llm.agents import create_agent
        from fsm_llm.core.definitions import ModelNotConfiguredError
        from tests.test_fsm_llm.test_run_context_adoption import _agent_tools

        with pytest.raises(ModelNotConfiguredError):
            create_agent(
                "react", _agent_tools(), system_prompt="x", context=RunContext()
            )


class TestTheApiDuckCheck:
    """17.5c: `API(context=)` needs callables where it calls and says so; the refusals read right."""

    @pytest.mark.parametrize("name", ["step_hook", "resolve_llm"])
    def test_a_name_that_is_called_must_be_callable(self, name: str) -> None:
        """Kills M51: `hasattr` only (a `step_hook = None` passed and failed inside the run loop)."""
        from fsm_llm import API
        from tests.test_fsm_llm.test_run_context_adoption import _Duck
        from tests.test_fsm_llm.test_run_until_terminal import (
            _ScriptedLLM,
            _trip_fsm,
        )

        duck = _Duck(_ScriptedLLM())
        setattr(duck, name, None)
        with pytest.raises(TypeError, match=name):
            API.from_definition(_trip_fsm(), context=duck)

    def test_the_refusal_names_the_type_and_never_echoes_the_object(self) -> None:
        from fsm_llm import API
        from tests.test_fsm_llm.test_run_until_terminal import _trip_fsm

        class Odd:
            def __repr__(self) -> str:
                return "SECRET-REPR"

        with pytest.raises(TypeError, match="Odd") as caught:
            API.from_definition(_trip_fsm(), context=Odd())
        assert "SECRET-REPR" not in str(caught.value)

    def test_a_start_refused_for_a_spent_ledger_says_nothing_was_started(self) -> None:
        """Kills M52: the start refusal reads "Run stopped before the conversation ended"."""
        from fsm_llm import API
        from tests.test_fsm_llm.test_run_until_terminal import (
            _ScriptedLLM,
            _trip_fsm,
        )

        api = API.from_definition(
            _trip_fsm(plan_speaks=True),
            context=RunContext(llm_interface=_ScriptedLLM(), budget=full(1)),
        )
        with pytest.raises(RunBudgetExceededError, match="not started") as caught:
            api.start_conversation({})
        assert "before the conversation ended" not in str(caught.value)
        assert (caught.value.budget, caught.value.limit) == ("steps", 1)
