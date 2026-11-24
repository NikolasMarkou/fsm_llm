"""The file-backed ``orchestrator`` and the ``after_run`` hook seam (plan a555dcfe step 13, D-022).

``OrchestratorAgent`` is thin on ``PatternAgent``: the machine is ``patterns/orchestrator.json``, the
delegator and the decision reset are the library factories ``agents.orchestrator_*`` and the error a
worker ends the run with is re-raised by the hook ``agents.raise_run_ending_error`` (``run.hooks``,
``at: after_run``). Census rows: D1 (a hook is operator code: a model-origin document cannot bind
one, and a file cannot bind a hook the agent has no seam for), D4 (the holder and the seed are per
run, nothing is kept on the agent), D5 (the FSM, the registration order and the request bytes are the
builder's: the differential of step 13 is recorded in progress.md, the goldens hold the bytes).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import OrchestratorAgent, ReflexionAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, Defaults, OrchestratorStates
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.agents.fsm_definitions import build_orchestrator_fsm
from fsm_llm.agents.handlers import library_workers as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope, parse_executable
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.namespaces import HANDLERS, namespace_resolver
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _worker(subtask: str) -> AgentResult:
    return AgentResult(answer=f"did {subtask}", status=Status.OK)


def _agent(**kw: Any) -> OrchestratorAgent:
    return OrchestratorAgent(
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _ctx(**kw: Any) -> HandlerContext:
    return HandlerContext(**kw)


class TestOrchestratorIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert OrchestratorAgent.__mro__[:3] == (
            OrchestratorAgent,
            PatternAgent,
            BaseAgent,
        )
        for name in ("run", "run_stream"):
            assert name not in vars(OrchestratorAgent), name
        agent = _agent(worker_factory=_worker, max_workers=2, tools=runs._tools())
        assert agent.worker_factory is _worker and agent.max_workers == 2
        assert agent.tools is not None and agent._agent_type == "orchestrator"
        assert accepts_tools(OrchestratorAgent)
        assert type(create_agent("orchestrator", runs._tools())) is OrchestratorAgent

    @pytest.mark.parametrize("bad", [0, -1, True, "3", 1.5])
    def test_a_cap_below_one_or_not_an_int_is_refused(self, bad):
        with pytest.raises(Exception, match="max_workers"):
            OrchestratorAgent(max_workers=bad)

    def test_hitl_stays_a_type_error_as_for_every_tool_less_pattern(self):
        with pytest.raises(TypeError, match="OrchestratorAgent does not accept"):
            OrchestratorAgent(hitl=runs._hitl())

    def test_a_flagged_tool_is_not_refused_the_orchestrator_never_runs_one(self):
        # the tools of the constructor are reserved for worker_factory implementations
        agent = _agent(tools=runs._tools(flagged=True))

        assert agent.run("go").answer is not None

    def test_the_default_cap_of_the_file_is_the_default_of_the_package(self):
        envelope = pf.load_envelope("orchestrator")

        assert envelope.body.options["max_workers"] == Defaults.MAX_WORKERS
        assert envelope.requires.callables == [] and envelope.requires.tools == "none"

    def test_the_fsm_of_a_run_is_the_file_and_leaves_out_its_own_outputs(self):
        agent = _agent()

        fsm = agent._fsm_for_run(
            "a task", {"hint": "x", "subtasks": ["a"], "worker_results": [1]}
        )

        assert fsm == build_orchestrator_fsm(("hint",))
        keys = fsm["states"]["orchestrate"]["field_extractions"][0]["context_keys"]
        assert keys == ["task", "observations", "worker_results", "hint"]

    def test_the_react_loop_lists_every_caller_key_as_the_builders_did(self):
        reflexion = ReflexionAgent(
            runs._tools(), config=runs._config(), llm_interface=runs.ScriptedModel()
        )

        fsm = reflexion._fsm_for_run("t", {"reflection": "r", "hint": "x"})

        keys = fsm["states"]["think"]["field_extractions"][0]["context_keys"]
        assert keys[-2:] == ["reflection", "hint"]

    def test_only_synthesize_speaks(self):
        # D-013 of plan 0068b055: an intermediate state makes no model call
        states = _agent()._fsm_for_run("t", None)["states"]

        assert [
            sid for sid, state in states.items() if state.get("response_instructions")
        ] == [OrchestratorStates.SYNTHESIZE]
        assert not states[OrchestratorStates.COLLECT].get("extraction_instructions")

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "OrchestratorDelegator",
            "OrchestratorDecisionReset",
            "AgentIterationLimiter",
        ]

    def test_the_two_seams_of_the_old_class_still_answer(self):
        agent = _agent(worker_factory=_worker, max_workers=1)

        done = agent._delegate_to_workers({ContextKeys.SUBTASKS: ["a", "b"]})
        limiter = agent._make_iteration_limiter()

        assert [r["subtask"] for r in done[ContextKeys.WORKER_RESULTS]] == ["a"]
        assert done[ContextKeys.SKIPPED_SUBTASKS] == ["b"]
        assert limiter({ContextKeys.ITERATION_COUNT: 0}) == {
            ContextKeys.ITERATION_COUNT: 1
        }


class TestTheSeedAndTheRunAreCallLocal:
    def test_the_context_leads_with_the_file_seed_then_the_config_budget(self):
        agent = _agent(config=runs._config())
        _, context, _ = agent._run_inputs("a", None)
        _, forged, _ = agent._run_inputs("a", {"_max_iterations": 99})

        keys = list(context)
        assert forged[ContextKeys.ITERATION_BUDGET] == agent.config.max_iterations
        assert context[ContextKeys.WORKER_RESULTS] == []
        assert keys.index(ContextKeys.WORKER_RESULTS) < keys.index(
            ContextKeys.ITERATION_BUDGET
        )
        assert context[ContextKeys.ITERATION_BUDGET] == agent.config.max_iterations

    def test_two_runs_share_neither_the_seed_the_context_nor_the_holder(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.WORKER_RESULTS].append("kept?")
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.WORKER_RESULTS] == []
        assert ctx_one.scope is not ctx_two.scope
        assert "_holder" not in vars(agent) and "holder" not in vars(agent)
        assert pf.load_envelope("orchestrator").body.run.context_seed == {
            ContextKeys.WORKER_RESULTS: []
        }


class TestTheFactories:
    def test_the_delegator_reads_its_cap_and_its_callable_from_the_run(self):
        handler = lib.orchestrator_delegator(
            _ctx(options={"max_workers": 1}, callables={"worker_factory": _worker})
        )[0]

        done = handler.execute({ContextKeys.SUBTASKS: ["a", "b"]})

        assert [r["subtask"] for r in done[ContextKeys.WORKER_RESULTS]] == ["a"]
        assert done[ContextKeys.SKIPPED_SUBTASKS] == ["b"]
        assert done[ContextKeys.SUBTASKS] is None

    def test_without_a_callable_the_model_fills_each_subtask(self):
        handler = lib.orchestrator_delegator(_ctx())[0]

        done = handler.execute({ContextKeys.SUBTASKS: ["a"]})

        assert done[ContextKeys.WORKER_RESULTS][0]["success"] is False
        assert "Pending LLM processing" in done[ContextKeys.WORKER_RESULTS][0]["answer"]

    @pytest.mark.parametrize("bad", [0, -1, True, "3", 1.5])
    def test_a_cap_below_one_or_not_an_int_is_refused(self, bad):
        with pytest.raises(AgentError, match="max_workers"):
            lib.orchestrator_delegator(_ctx(options={"max_workers": bad}))

    def test_the_decision_reset_clears_all_collected_on_entry_to_orchestrate(self):
        handler = lib.orchestrator_decision_reset(_ctx())[0]

        assert handler.execute({ContextKeys.ALL_COLLECTED: False}) == {
            ContextKeys.ALL_COLLECTED: None
        }

    def test_a_worker_that_ends_the_run_stops_the_next_worker_and_the_hook_raises_it(
        self,
    ):
        ctx = _ctx(
            callables={
                "worker_factory": lambda subtask: (_ for _ in ()).throw(
                    AgentTimeoutError(1.0)
                )
            }
        )
        handler = lib.orchestrator_delegator(ctx)[0]
        hook = lib.raise_run_ending_error(ctx)

        done = handler.execute({ContextKeys.SUBTASKS: ["a", "b"]})

        assert done[ContextKeys.WORKER_RESULTS] == []
        with pytest.raises(AgentTimeoutError):
            hook()

    def test_the_holder_is_the_runs_and_a_new_run_has_none(self):
        ctx = _ctx()

        assert lib.raise_run_ending_error(ctx).__self__ is lib._run_holder(ctx)
        assert lib._run_holder(ctx.next_run()) is not lib._run_holder(ctx)

    def test_a_hook_with_nothing_held_does_nothing(self):
        assert lib.raise_run_ending_error(_ctx())() is None


def _timeout_worker(subtask: str) -> AgentResult:
    raise AgentTimeoutError(1.0)


class TestTheRunReRaisesAWorkersEndOfRun:
    @pytest.mark.parametrize(
        "error", [AgentTimeoutError(1.0), BudgetExhaustedError("tokens", 5)]
    )
    def test_a_run_ending_error_of_a_worker_propagates_out_of_run(self, error):
        def worker(subtask: str) -> AgentResult:
            raise error

        agent = _agent(worker_factory=worker)

        with pytest.raises(type(error)) as info:
            agent.run("go")

        assert info.value is error

    def test_the_same_error_propagates_out_of_the_stream(self):
        agent = _agent(worker_factory=_timeout_worker)

        with pytest.raises(AgentTimeoutError):
            list(agent.run_stream("go"))

    def test_any_other_worker_error_is_a_failed_result_and_the_run_ends(self):
        def worker(subtask: str) -> AgentResult:
            raise ValueError("boom")

        result = _agent(worker_factory=worker).run("go")

        entries = result.final_context[ContextKeys.WORKER_RESULTS]
        assert entries and all(e["success"] is False for e in entries)
        assert "Worker error: boom" in entries[0]["answer"]
        assert result.success is False

    def test_a_run_that_succeeds_with_a_worker_reports_it(self):
        result = _agent(worker_factory=_worker).run("go")

        assert result.final_context[ContextKeys.WORKER_RESULTS][0]["success"] is True
        assert result.success is True

    def test_a_second_run_after_a_failed_one_starts_with_no_error_held(self):
        calls = iter([_timeout_worker, _worker])
        agent = _agent(worker_factory=lambda s: next(calls)(s))

        with pytest.raises(AgentTimeoutError):
            agent.run("go")
        result = agent.run("go")

        assert result.final_context[ContextKeys.WORKER_RESULTS][0]["success"] is True


def _user_doc(hooks: list[dict[str, Any]], doc_id: str = "user.orch") -> dict[str, Any]:
    doc = pf.load_document("orchestrator")
    doc["id"] = doc_id
    doc["body"]["run"]["hooks"] = hooks
    return doc


def _envelope(doc: dict[str, Any]) -> PatternEnvelope:
    envelope = parse_executable(doc)
    assert isinstance(envelope, PatternEnvelope)
    return envelope


def _hooked(monkeypatch, events: list[str], hooks: dict[str, Any]) -> PatternAgent:
    """A PatternAgent over the orchestrator file with *hooks* (name -> callable) bound, in order."""
    for name, call in hooks.items():
        monkeypatch.setitem(HANDLERS._objects, name, lambda ctx, binding, c=call: c)

    def loop(self: Any, *args: Any, **kwargs: Any) -> AgentResult:
        events.append("loop")
        return AgentResult(answer="ok", status=Status.OK)

    monkeypatch.setattr(PatternAgent, "_standard_run", loop)
    doc = _user_doc([{"hook": name, "at": "after_run"} for name in hooks])
    return PatternAgent(
        _envelope(doc), config=runs._config(), llm_interface=runs.ScriptedModel()
    )


class TestAfterRunHooks:
    def test_a_hook_runs_after_the_loop_and_in_declared_order(self, monkeypatch):
        events: list[str] = []
        agent = _hooked(
            monkeypatch,
            events,
            {
                "test.first": lambda: events.append("first"),
                "test.second": lambda: events.append("second"),
                "test.third": lambda: events.append("third"),
            },
        )

        agent.run("go")

        assert events == ["loop", "first", "second", "third"]

    def test_the_hooks_of_a_run_get_that_runs_own_handler_context(self, monkeypatch):
        seen: list[HandlerContext] = []
        for name in ("test.a", "test.b"):
            monkeypatch.setitem(
                HANDLERS._objects, name, lambda ctx, binding: lambda: seen.append(ctx)
            )
        monkeypatch.setattr(
            PatternAgent,
            "_standard_run",
            lambda self, *a, **k: AgentResult(answer="", status=Status.OK),
        )
        agent = PatternAgent(
            _envelope(
                _user_doc(
                    [
                        {"hook": "test.a", "at": "after_run"},
                        {"hook": "test.b", "at": "after_run"},
                    ]
                )
            ),
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
        )

        agent.run("one")
        agent.run("two")

        assert seen[0] is seen[1] and seen[2] is seen[3] and seen[0] is not seen[2]

    def test_a_hook_error_propagates_and_the_hooks_after_it_do_not_run(
        self, monkeypatch
    ):
        events: list[str] = []

        def boom() -> None:
            raise RuntimeError("hook failed")

        agent = _hooked(
            monkeypatch,
            events,
            {"test.boom": boom, "test.never": lambda: events.append("never")},
        )

        with pytest.raises(RuntimeError, match="hook failed"):
            agent.run("go")

        assert events == ["loop"]

    def test_a_hook_runs_when_the_loop_raised_and_the_loop_error_wins(
        self, monkeypatch
    ):
        events: list[str] = []
        agent = _hooked(
            monkeypatch, events, {"test.after": lambda: events.append("hook")}
        )

        def failing(self: Any, *args: Any, **kwargs: Any) -> AgentResult:
            raise AgentError("loop failed")

        monkeypatch.setattr(PatternAgent, "_standard_run", failing)

        with pytest.raises(AgentError, match="loop failed"):
            agent.run("go")

        assert events == ["hook"]

    def test_the_stream_runs_the_hooks_after_its_last_chunk(self, monkeypatch):
        events: list[str] = []
        agent = _hooked(
            monkeypatch, events, {"test.after": lambda: events.append("hook")}
        )

        def stream(self: Any, *args: Any, **kwargs: Any):
            events.append("start")
            yield "a"
            events.append("end")
            yield "b"

        monkeypatch.setattr(PatternAgent, "_standard_run_stream", stream)

        assert list(agent.run_stream("go")) == ["a", "b"]
        assert events == ["start", "end", "hook"]

    def test_a_file_with_no_hook_runs_nothing_after_the_loop(self, monkeypatch):
        events: list[str] = []
        agent = _hooked(monkeypatch, events, {})

        assert agent.run("go").answer == "ok"
        assert events == ["loop"]

    def test_a_handler_bound_as_a_hook_and_a_hook_bound_as_a_handler_are_refused(self):
        as_hook = PatternAgent(
            _envelope(
                _user_doc(
                    [{"hook": "agents.orchestrator_decision_reset", "at": "after_run"}]
                )
            ),
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
        )
        doc = _user_doc([])
        doc["body"]["handlers"][0] = {
            "handler": "agents.raise_run_ending_error",
            "on": "entry",
            "states": ["delegate"],
        }
        as_handler = PatternAgent(
            _envelope(doc), config=runs._config(), llm_interface=runs.ScriptedModel()
        )

        with pytest.raises(AgentError, match="is a handler, not a hook"):
            as_hook._after_run(as_hook._handler_context())
        with pytest.raises(AgentError, match="is a hook, not a handler"):
            as_handler._register_handlers(_FakeApi(), as_handler._handler_context())

    def test_an_unknown_hook_name_fails_the_run_closed(self):
        agent = PatternAgent(
            _envelope(_user_doc([{"hook": "agents.nothing", "at": "after_run"}])),
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
        )

        with pytest.raises(Exception, match=r"agents\.nothing"):
            agent._after_run(agent._handler_context())


class TestAModelDocumentCannotBindAHook:
    def test_the_after_run_hook_of_the_shipped_file_is_refused_for_a_model_origin(self):
        doc = pf.load_document("orchestrator")

        envelope, result = load_executable(
            doc, origin="model", resolver=namespace_resolver()
        )
        lines = [e for e in result.errors if "body.run.hooks" in e]
        operator, report = load_executable(
            doc, origin="operator", resolver=namespace_resolver()
        )

        assert envelope is None and len(lines) == 1, result.errors
        assert "agents.raise_run_ending_error" in lines[0]
        assert operator is not None and report.is_valid, report.errors

    def test_the_hook_name_is_resolved_like_a_handler_name_at_load(self):
        doc = _user_doc([{"hook": "agents.nothing", "at": "after_run"}])
        doc["requires"]["handlers"] = list(doc["requires"]["handlers"])

        envelope, result = load_executable(
            doc, origin="operator", resolver=namespace_resolver()
        )

        assert envelope is None and any("agents.nothing" in e for e in result.errors)


class TestTheOrchestratorExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("orchestrator")

    def test_the_row_keeps_its_id_and_reads_its_objects_when_present(self):
        executable = self._exe()

        assert executable.template_identity
        assert set(executable.optional_callables) == {
            "tools",
            "worker_factory",
            "config",
        }
        assert list(executable.requires.callables) == []
        assert executable.requires.tools == "none"

    def test_the_hosts_worker_factory_reaches_the_packaged_class(self, monkeypatch):
        seen: list[Any] = []

        def fake_run(self: Any, task: str, initial_context: Any = None) -> AgentResult:
            seen.append((type(self), self.worker_factory))
            return AgentResult(answer="patched", status=Status.OK)

        from fsm_llm.core.names import NameRegistry
        from fsm_llm.engine.context import RunContext

        monkeypatch.setattr(OrchestratorAgent, "run", fake_run)
        registry: NameRegistry[Any] = NameRegistry("callables")
        registry.register("worker_factory", _worker)

        self._exe().run(
            {"task": "go"},
            context=RunContext(llm_interface=runs.ScriptedModel(), callables=registry),
        )

        assert seen == [(OrchestratorAgent, _worker)]
