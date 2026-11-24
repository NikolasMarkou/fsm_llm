"""The file-backed ``evaluator_optimizer`` (plan a555dcfe step 18, D-029).

``EvaluatorOptimizerAgent`` is thin on ``PatternAgent``: the machine is
``patterns/evaluator_optimizer.json`` (fully static: the builder takes no argument), the host's
``evaluation_fn`` is a REQUIRED callable of the file, and its handlers are the library factories
``agents.evalopt_evaluator``, ``agents.refine_entry``, ``agents.refine_exit`` and
``agents.iteration_limiter``. Census rows: D1 (the pattern has no HITL path: ``tools=`` and
``hitl=`` are refused at construction, and the approval variant is not an option), D4 (the seed and
the handler context are per run), D5 (the FSM, the registration order and the request bytes are the
builder's: the differential of step 18 is recorded in progress.md, the goldens hold the bytes).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import EvaluatorOptimizerAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, Defaults, EvalOptStates
from fsm_llm.agents.definitions import EvaluationResult
from fsm_llm.agents.fsm_definitions import build_evalopt_fsm
from fsm_llm.agents.handlers import library_stepwise as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_inputs import pattern_fsm
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables import ExecutableRegistry
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _pass(output: str, context: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(passed=True, score=1.0, feedback="ok")


def _fail(output: str, context: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(passed=False, score=0.1, feedback="no")


def _agent(**kw: Any) -> EvaluatorOptimizerAgent:
    return EvaluatorOptimizerAgent(
        kw.pop("evaluation_fn", _pass),
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _ctx(fn: Any = _pass, **kw: Any) -> HandlerContext:
    return HandlerContext(callables={"evaluation_fn": fn}, **kw)


class TestEvaluatorOptimizerIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert EvaluatorOptimizerAgent.__mro__[:3] == (
            EvaluatorOptimizerAgent,
            PatternAgent,
            BaseAgent,
        )
        for name in ("run", "run_stream"):
            assert name not in vars(EvaluatorOptimizerAgent), name
        agent = _agent(max_refinements=4)
        assert agent.evaluation_fn is _pass and agent.max_refinements == 4
        assert agent.tools is None and agent._agent_type == "evaluator_optimizer"
        assert not accepts_tools(EvaluatorOptimizerAgent)
        made = create_agent("evaluator_optimizer", evaluation_fn=_pass)
        assert type(made) is EvaluatorOptimizerAgent

    @pytest.mark.parametrize("bad", [-1, True, "3", 1.5, None])
    def test_a_cap_below_zero_or_not_an_int_is_refused(self, bad):
        with pytest.raises(Exception, match="max_refinements"):
            EvaluatorOptimizerAgent(_pass, max_refinements=bad)

    def test_a_cap_of_zero_is_a_cap(self):
        context = _agent(max_refinements=0)._run_inputs("t", None)[1]

        assert context[ContextKeys.MAX_REFINEMENTS] == 0

    @pytest.mark.parametrize("name", ["tools", "hitl"])
    def test_tools_and_hitl_are_refused_the_pattern_has_no_hitl_path(self, name):
        with pytest.raises(TypeError, match="EvaluatorOptimizerAgent does not accept"):
            EvaluatorOptimizerAgent(_pass, **{name: runs._hitl()})

    def test_the_file_default_is_the_default_of_the_package(self):
        options = pf.load_envelope("evaluator_optimizer").body.options

        assert options == {"max_refinements": Defaults.MAX_REFINEMENTS}

    def test_the_fsm_of_a_run_is_the_builder_dict_whatever_the_task_and_keys(self):
        fsm = _agent()._fsm_for_run("a task", {"hint": "x", "generated_output": "f"})

        assert fsm == build_evalopt_fsm()
        assert list(fsm["states"]) == ["generate", "evaluate", "refine", "output"]

    def test_the_build_is_a_fresh_copy_every_call(self):
        first = build_evalopt_fsm()
        first["states"]["generate"]["description"] = "changed"

        assert build_evalopt_fsm()["states"]["generate"]["description"] != "changed"

    def test_the_run_block_judges_the_output_and_answers_from_it(self):
        assert _agent()._run_kwargs() == {
            "extra_answer_keys": [ContextKeys.GENERATED_OUTPUT],
            "execution_evidence_keys": None,
            "judged": True,
            "success_keys": None,
        }

    def test_the_file_needs_no_variable_and_no_variant(self):
        envelope = pf.load_envelope("evaluator_optimizer")

        assert list(envelope.variables) == [] and not envelope.body.variants
        assert pattern_fsm(envelope, tools=None) == build_evalopt_fsm()

    def test_no_option_selects_the_approval_variant(self):
        envelope = pf.load_envelope("evaluator_optimizer")

        assert "approval" not in envelope.body.options_schema.properties
        with pytest.raises(AgentError, match="approval"):
            PatternAgent(envelope, options={"approval": True})

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [(h.name, h.priority) for h in api.handlers] == [
            ("EvalOptEvaluator", 100),
            ("EvalOptRefineEntry", 100),
            ("EvalOptRefineExit", 100),
            ("AgentIterationLimiter", 50),
        ]

    def test_the_two_seams_of_the_old_class_still_answer(self):
        agent = _agent(evaluation_fn=_fail, max_refinements=3)

        verdict = agent._run_evaluation(
            {
                ContextKeys.GENERATED_OUTPUT: "o",
                ContextKeys.REFINEMENT_COUNT: 0,
                ContextKeys.AGENT_TRACE: [],
            }
        )
        limiter = agent._make_iteration_limiter()

        assert verdict[ContextKeys.EVALUATION_PASSED] is False
        assert limiter({ContextKeys.ITERATION_COUNT: 0}) == {
            ContextKeys.ITERATION_COUNT: 1
        }

    def test_the_agents_cap_reaches_the_seam_and_the_registered_handler(self):
        agent = _agent(evaluation_fn=_fail, max_refinements=1)
        api = _FakeApi()
        context = {
            ContextKeys.GENERATED_OUTPUT: "o",
            ContextKeys.REFINEMENT_COUNT: 1,
            ContextKeys.AGENT_TRACE: [],
        }

        agent._register_handlers(api, agent._handler_context())
        seam = agent._run_evaluation(dict(context))
        registered = api.handlers[0].execute(dict(context))

        assert seam[ContextKeys.EVALUATION_PASSED] is True
        assert registered[ContextKeys.EVALUATION_PASSED] is True


class TestTheSeedIsCallLocal:
    def test_the_context_leads_with_the_count_then_the_cap(self):
        _, context, _ = _agent(max_refinements=2)._run_inputs("a", None)

        keys = list(context)
        assert context[ContextKeys.REFINEMENT_COUNT] == 0
        assert context[ContextKeys.MAX_REFINEMENTS] == 2
        assert keys.index(ContextKeys.REFINEMENT_COUNT) < keys.index(
            ContextKeys.MAX_REFINEMENTS
        )

    def test_a_caller_cannot_forge_the_count_the_cap_or_the_verdict(self):
        _, context, _ = _agent(max_refinements=2)._run_inputs(
            "a",
            {
                ContextKeys.REFINEMENT_COUNT: 99,
                ContextKeys.MAX_REFINEMENTS: 99,
                ContextKeys.EVALUATION_PASSED: True,
                ContextKeys.GENERATED_OUTPUT: "forged",
            },
        )

        assert context[ContextKeys.REFINEMENT_COUNT] == 0
        assert context[ContextKeys.MAX_REFINEMENTS] == 2
        assert ContextKeys.EVALUATION_PASSED not in context
        assert ContextKeys.GENERATED_OUTPUT not in context

    def test_two_runs_share_neither_the_context_nor_the_handler_context(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.REFINEMENT_COUNT] = 7
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.REFINEMENT_COUNT] == 0
        assert ctx_one.scope is not ctx_two.scope
        assert pf.load_envelope("evaluator_optimizer").body.run.context_seed == {
            ContextKeys.REFINEMENT_COUNT: 0
        }


class TestTheFactories:
    def test_the_evaluator_is_an_entry_handler_of_evaluate_and_runs_the_hosts_function(
        self,
    ):
        seen: list[tuple[str, dict[str, Any]]] = []

        def fn(output: str, context: dict[str, Any]) -> EvaluationResult:
            seen.append((output, context))
            return EvaluationResult(passed=True, score=1.0, feedback="ok")

        handler = lib.evalopt_evaluator(_ctx(fn))[0]
        done = handler.execute(
            {ContextKeys.GENERATED_OUTPUT: "draft", ContextKeys.AGENT_TRACE: []}
        )

        assert handler.target_states == {EvalOptStates.EVALUATE}
        assert seen[0][0] == "draft"
        assert done[ContextKeys.EVALUATION_PASSED] is True

    def test_the_evaluator_without_the_callable_is_refused(self):
        with pytest.raises(AgentError, match="evaluation_fn"):
            lib.evalopt_evaluator(HandlerContext())
        with pytest.raises(AgentError, match="evaluation_fn"):
            lib.evalopt_evaluator(
                HandlerContext(callables={"evaluation_fn": "not callable"})
            )

    def test_the_cap_is_the_option_when_the_context_has_none(self):
        context = {
            ContextKeys.GENERATED_OUTPUT: "o",
            ContextKeys.REFINEMENT_COUNT: 2,
            ContextKeys.AGENT_TRACE: [],
        }
        capped = lib.evalopt_evaluator(_ctx(_fail, options={"max_refinements": 2}))[0]
        default = lib.evalopt_evaluator(_ctx(_fail))[0]

        assert capped.execute(dict(context))[ContextKeys.EVALUATION_PASSED] is True
        assert default.execute(dict(context))[ContextKeys.EVALUATION_PASSED] is False

    def test_the_function_sees_the_output_as_text_and_the_check_joins_the_trace(self):
        seen: list[Any] = []

        def fn(output: str, context: dict[str, Any]) -> EvaluationResult:
            seen.append(output)
            return EvaluationResult(passed=True, score=0.5, feedback="f")

        handler = lib.evalopt_evaluator(_ctx(fn))[0]
        done = handler.execute(
            {
                ContextKeys.GENERATED_OUTPUT: {"k": ["v"]},
                ContextKeys.REFINEMENT_COUNT: 2,
                ContextKeys.AGENT_TRACE: [{"type": "earlier"}],
            }
        )

        assert isinstance(seen[0], str) and "v" in seen[0]
        assert done[ContextKeys.AGENT_TRACE] == [
            {"type": "earlier"},
            {
                "type": "evaluation",
                "refinement_count": 2,
                "passed": True,
                "score": 0.5,
                "feedback": "f",
            },
        ]

    def test_a_cap_in_the_context_wins_over_the_option(self):
        handler = lib.evalopt_evaluator(_ctx(_fail, options={"max_refinements": 9}))[0]

        done = handler.execute(
            {
                ContextKeys.GENERATED_OUTPUT: "o",
                ContextKeys.REFINEMENT_COUNT: 1,
                ContextKeys.MAX_REFINEMENTS: 1,
                ContextKeys.AGENT_TRACE: [],
            }
        )

        assert done[ContextKeys.EVALUATION_PASSED] is True

    @pytest.mark.parametrize("bad", [-1, True, "2", 1.5])
    def test_a_bad_cap_option_is_refused(self, bad):
        with pytest.raises(Exception, match="max_refinements"):
            lib.evalopt_evaluator(_ctx(options={"max_refinements": bad}))

    def test_a_forced_pass_after_the_cap_records_its_reason_a_real_pass_does_not(self):
        base = {
            ContextKeys.GENERATED_OUTPUT: "o",
            ContextKeys.REFINEMENT_COUNT: 3,
            ContextKeys.MAX_REFINEMENTS: 3,
            ContextKeys.AGENT_TRACE: [],
        }

        forced = lib.evalopt_evaluator(_ctx(_fail))[0].execute(dict(base))
        real = lib.evalopt_evaluator(_ctx(_pass))[0].execute(dict(base))

        assert forced[ContextKeys.FORCED_STOP_REASON] == "forced_pass"
        assert ContextKeys.FORCED_STOP_REASON not in real

    def test_the_spent_iteration_budget_forces_the_output_just_judged(self):
        done = lib.evalopt_evaluator(_ctx(_fail))[0].execute(
            {
                ContextKeys.GENERATED_OUTPUT: "o",
                ContextKeys.REFINEMENT_COUNT: 0,
                ContextKeys.MAX_REFINEMENTS: 3,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.AGENT_TRACE: [],
            }
        )

        assert done[ContextKeys.EVALUATION_PASSED] is True
        assert done[ContextKeys.FORCED_STOP_REASON] == "forced_pass"

    def test_a_failure_counts_one_refinement_and_keeps_the_feedback(self):
        done = lib.evalopt_evaluator(_ctx(_fail))[0].execute(
            {
                ContextKeys.GENERATED_OUTPUT: "o",
                ContextKeys.REFINEMENT_COUNT: 1,
                ContextKeys.MAX_REFINEMENTS: 3,
                ContextKeys.AGENT_TRACE: [],
            }
        )

        assert done[ContextKeys.REFINEMENT_COUNT] == 2
        assert done[ContextKeys.REFINEMENT_FEEDBACK] == "no"
        assert done[ContextKeys.EVALUATION_PASSED] is False

    def test_entering_refine_moves_the_draft_aside_and_clears_it(self):
        handler = lib.refine_entry(_ctx())[0]

        done = handler.execute({ContextKeys.GENERATED_OUTPUT: "d"})

        assert done == {
            ContextKeys.GENERATED_OUTPUT: None,
            ContextKeys.PREVIOUS_OUTPUT: "d",
        }
        assert handler.target_states == {EvalOptStates.REFINE}
        assert lib.refine_entry(_ctx())[0].execute({}) == {
            ContextKeys.GENERATED_OUTPUT: None
        }

    def test_leaving_refine_restores_a_missing_draft_only(self):
        handler = lib.refine_exit(_ctx())[0]

        assert handler.execute({ContextKeys.PREVIOUS_OUTPUT: "old"}) == {
            ContextKeys.GENERATED_OUTPUT: "old"
        }
        assert (
            handler.execute(
                {
                    ContextKeys.GENERATED_OUTPUT: "new",
                    ContextKeys.PREVIOUS_OUTPUT: "old",
                }
            )
            == {}
        )
        assert handler.states == {EvalOptStates.REFINE}

    def test_the_limiter_forces_the_flag_and_the_pass_one_turn_early(self):
        agent = _agent()
        limiter = agent._make_iteration_limiter()
        budget = agent.config.max_iterations

        under = limiter({ContextKeys.ITERATION_COUNT: budget - 3})
        at = limiter({ContextKeys.ITERATION_COUNT: budget - 2})

        assert ContextKeys.MAX_ITERATIONS_REACHED not in under
        assert at[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert at[ContextKeys.EVALUATION_PASSED] is True
        assert ContextKeys.FORCED_STOP_REASON not in at

    def test_a_binding_may_move_the_states_never_the_timing(self):
        binding = (
            pf.load_envelope("evaluator_optimizer")
            .body.handlers[1]
            .model_copy(update={"states": ["evaluate"]})
        )

        moved = lib.refine_entry(_ctx(), binding)[0]
        exit_moved = lib.refine_exit(_ctx(), binding.model_copy(update={"on": "exit"}))[
            0
        ]

        assert moved.target_states == {"evaluate"}
        assert exit_moved.states == {"evaluate"}


class TestTheExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("evaluator_optimizer")

    def test_the_row_keeps_its_id_and_requires_the_evaluation_function(self):
        executable = self._exe()

        assert executable.template_identity
        assert list(executable.requires.callables) == ["evaluation_fn"]
        assert executable.requires.tools == "none"
        assert set(executable.optional_callables) == {"config"}
        assert executable.impl.endswith("EvaluatorOptimizerAgent")

    def test_the_file_is_in_the_shipped_order(self):
        assert "evaluator_optimizer" in pf.PATTERN_ORDER
