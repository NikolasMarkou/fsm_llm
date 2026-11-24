"""The run hooks of a pattern file and the react presets (plan a555dcfe step 21, D-032).

``PatternAgent.run`` is the one seam: ``before_run`` once, then attempts (each with its own
``HandlerContext``) whose ``after_run`` hooks may reject with a ``Retry``; ``on_iteration`` hooks run
at every step AFTER the HITL driver and write through a writer that refuses an internal-prefixed key.
``VerifiedReactAgent`` and ``AutoMemoryReactAgent`` are thin classes over the presets
``patterns/verified_react.json`` and ``patterns/auto_memory.json``. Census rows: D1 (order against the
HITL driver, no grant path, V7 on hooks), D4 (per-attempt context, nothing kept on the agent).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from fsm_llm.agents import (
    AgentConfig,
    AutoMemoryReactAgent,
    ReactAgent,
    SemanticMemoryStore,
    VerifiedReactAgent,
)
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.agents.definitions import AgentResult, AgentTrace
from fsm_llm.agents.handlers import hooks as hook_lib
from fsm_llm.agents.handlers.context import (
    RUN_ATTEMPT,
    RUN_RESULT,
    RUN_TASK,
    HandlerContext,
    Retry,
)
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope, parse_executable
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables.namespaces import HANDLERS, namespace_resolver
from tests import pattern_runs as runs


def _result(answer: str = "ok", ok: bool = True) -> AgentResult:
    return AgentResult(
        answer=answer,
        status=Status.OK if ok else Status.FAILED,
        trace=AgentTrace(tool_calls=[], total_iterations=1),
    )


def _verified(verify: Any, retries: int = 1, **kw: Any) -> VerifiedReactAgent:
    return VerifiedReactAgent(
        runs._tools(),
        config=AgentConfig(model="mock/model", verification_fn=verify, **kw),
        max_verify_retries=retries,
        llm_interface=runs.ScriptedModel(),
    )


def _embed(text: str) -> list[float]:
    t = text.lower()
    return [float(t.count("python") + t.count("language") + t.count("zephyr")), 0.0]


def _store() -> SemanticMemoryStore:
    return SemanticMemoryStore(embed_fn=_embed)


def _memory_agent(store: Any = None, **kw: Any) -> AutoMemoryReactAgent:
    return AutoMemoryReactAgent(
        runs._tools(),
        config=runs._config(),
        memory=store if store is not None else _store(),
        llm_interface=runs.ScriptedModel(),
        **kw,
    )


def _loop(monkeypatch, answers: Any, tasks: list[str] | None = None) -> None:
    """Replace the inner attempt: each call returns the next answer (an exception is raised)."""
    queue = iter(answers)

    def attempt(
        self: Any, task: str, initial_context: Any = None, handlers: Any = None
    ):
        if tasks is not None:
            tasks.append(task)
        value = next(queue)
        if isinstance(value, BaseException):
            raise value
        return _result(value)

    monkeypatch.setattr(PatternAgent, "_run_loop", attempt)


def _bound(monkeypatch, name: str, factory: Any) -> None:
    monkeypatch.setitem(HANDLERS._objects, name, factory)


def _hooked(hooks: list[tuple[str, str]], **kw: Any) -> PatternAgent:
    """A PatternAgent over a copy of the react document that binds *hooks* ``(name, at)``."""
    doc = pf.load_document("react")
    doc["id"] = "user.hooked"
    doc["body"]["run"]["hooks"] = [{"hook": n, "at": at} for n, at in hooks]
    envelope = parse_executable(doc)
    assert isinstance(envelope, PatternEnvelope)
    return PatternAgent(
        envelope,
        tools=runs._tools(),
        config=runs._config(),
        llm_interface=runs.ScriptedModel(),
        **kw,
    )


class TestVerifyRetry:
    def test_a_rejected_answer_reruns_with_the_feedback_up_to_the_option(
        self, monkeypatch
    ):
        tasks: list[str] = []
        _loop(monkeypatch, ["bad", "bad", "good"], tasks)
        agent = _verified(lambda a, c: {"ok": a == "good", "feedback": "say good"}, 2)

        result = agent.run("q")

        assert result.answer == "good" and result.success
        assert tasks[0] == "q" and len(tasks) == 3
        # the note is folded into the BASE task each time, never into the previous attempt
        assert tasks[1] == tasks[2]
        assert tasks[1].startswith("q\n\n[A previous answer was rejected")
        assert (
            tasks[1].count("rejected by verification") == 1 and "say good" in tasks[1]
        )

    def test_the_last_rejected_answer_ships_as_verification_failed(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, ["a", "b", "c"], tasks)

        result = _verified(lambda a, c: False, 1).run("q")

        assert len(tasks) == 2  # one extra attempt, as max_verify_retries says
        assert result.answer == "b" and result.success is False
        assert result.stop_reason == StopReason.VERIFICATION_FAILED

    def test_a_raising_verifier_is_a_rejection_never_a_pass(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, ["x", "x"], tasks)

        def boom(answer, context):
            raise RuntimeError("verifier crashed")

        result = _verified(boom, 1).run("q")

        assert len(tasks) == 2 and "verifier crashed" in tasks[1]
        assert result.success is False
        assert result.stop_reason == StopReason.VERIFICATION_FAILED

    def test_zero_retries_means_one_attempt(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, ["a", "b"], tasks)

        result = _verified(lambda a, c: False, 0).run("q")

        assert len(tasks) == 1 and result.stop_reason == StopReason.VERIFICATION_FAILED

    def test_no_verifier_runs_once_and_a_raising_loop_is_not_retried(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, [RuntimeError("provider down")], tasks)

        with pytest.raises(RuntimeError, match="provider down"):
            _verified(lambda a, c: False, 3).run("q")

        assert tasks == ["q"]  # nothing to verify, so nothing to retry
        tasks.clear()
        _loop(monkeypatch, ["a", "b"], tasks)
        assert _verified(None, 3).run("q").answer == "a" and tasks == ["q"]

    def test_each_attempt_has_its_own_context_with_the_attempt_number(
        self, monkeypatch
    ):
        scopes: list[dict[str, Any]] = []

        def attempt(self, task, initial_context=None, handlers=None):
            scopes.append(handlers.scope)
            return _result("x")

        monkeypatch.setattr(PatternAgent, "_run_loop", attempt)
        _verified(lambda a, c: False, 2).run("caller task")

        assert len({id(s) for s in scopes}) == 3
        assert [s[RUN_ATTEMPT] for s in scopes] == [0, 1, 2]
        assert {s[RUN_TASK] for s in scopes} == {"caller task"}
        assert all(RUN_RESULT in s for s in scopes)


class TestRecallAndRemember:
    def test_recall_rewrites_the_task_before_and_remember_stores_the_original(
        self, monkeypatch
    ):
        store = _store()
        store.add("My favorite language is Python")
        tasks: list[str] = []
        _loop(monkeypatch, ["Python"], tasks)

        _memory_agent(store).run("what language do I like?")

        assert tasks[0].startswith("[Relevant things you remember]")
        assert "Python" in tasks[0]
        added = [text for text, _, _ in store.search("what language do I like?", k=5)]
        assert any(
            t.startswith("Q: what language do I like?\nA: Python") for t in added
        )
        assert not any("[Relevant things" in t for t in added)

    def test_remember_runs_in_a_finally_when_the_run_raises(self, monkeypatch):
        store = _store()
        _loop(monkeypatch, [RuntimeError("budget exhausted")])

        with pytest.raises(RuntimeError, match="budget exhausted"):
            _memory_agent(store).run("remember: my name is Nikolas")

        assert len(store) == 1  # the raw task, as the class did

    def test_the_options_switch_remember_and_recall_off(self, monkeypatch):
        store = _store()
        store.add("zephyr fact")
        tasks: list[str] = []
        _loop(monkeypatch, ["a", "b"], tasks)

        _memory_agent(store, auto_remember=False, recall_k=0).run("zephyr?")
        assert len(store) == 1 and tasks == ["zephyr?"]
        agent = _memory_agent(store, remember_only_on_success=True)
        monkeypatch.setattr(
            PatternAgent,
            "_run_loop",
            lambda self, t, c=None, h=None: _result("n", False),
        )
        agent.run("q")
        assert len(store) == 1  # a failed run is not stored when opted in

    def test_a_retry_chain_remembers_once_for_the_final_attempt(self, monkeypatch):
        store = _store()
        _loop(monkeypatch, ["bad", "good"])
        doc = pf.load_document("auto_memory")
        doc["id"] = "user.both"
        doc["body"]["run"]["hooks"].insert(
            0, {"hook": "agents.verify_retry", "at": "after_run"}
        )
        doc["body"]["options_schema"]["properties"]["max_verify_retries"] = {
            "type": "integer",
            "minimum": 0,
        }
        doc["body"]["options"]["max_verify_retries"] = 2
        envelope = parse_executable(doc)
        assert isinstance(envelope, PatternEnvelope)
        agent = PatternAgent(
            envelope,
            tools=runs._tools(),
            config=AgentConfig(
                model="mock/model",
                verification_fn=lambda a, c: a == "good",
            ),
            options={"max_verify_retries": 2},
            callables={"memory": store},
            llm_interface=runs.ScriptedModel(),
        )

        assert agent.run("q").answer == "good"
        assert len(store) == 1  # the rejected attempt stopped the chain before remember

    def test_a_missing_memory_object_is_a_named_error(self):
        ctx = HandlerContext()
        for factory in (hook_lib.memory_recall, hook_lib.memory_remember):
            with pytest.raises(AgentError, match="'memory'"):
                factory(ctx, None)


class TestStreamsGoThroughTheSameChain:
    def test_a_class_stream_runs_the_chain_and_yields_the_answer_once(
        self, monkeypatch
    ):
        store = _store()
        store.add("zephyr fact")
        tasks: list[str] = []
        _loop(monkeypatch, ["answer"], tasks)

        out = list(_memory_agent(store).run_stream("zephyr?"))

        assert out == ["answer"] and "[Relevant things" in tasks[0]
        assert len(store) == 2

    def test_a_verified_stream_verifies_and_a_plain_one_streams(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, ["bad", "bad"], tasks)

        assert list(_verified(lambda a, c: False, 1).run_stream("q")) == ["bad"]
        assert len(tasks) == 2

    def test_a_file_stream_applies_before_run_and_runs_after_run_with_no_result(
        self, monkeypatch
    ):
        seen: dict[str, Any] = {}

        def stream(self, task, fsm_def, context, agent_type, **kw):
            seen["task"] = task
            yield "tok"

        monkeypatch.setattr(PatternAgent, "_standard_run_stream", stream)
        _bound(monkeypatch, "test.before", lambda ctx, b: lambda task: task + "+")
        _bound(
            monkeypatch,
            "test.after",
            lambda ctx, b: lambda: seen.update(slot=ctx.scope.get(RUN_RESULT)),
        )
        agent = _hooked([("test.before", "before_run"), ("test.after", "after_run")])

        assert list(agent.run_stream("go")) == ["tok"]

        assert seen == {"task": "go+", "slot": None}


class TestOnIterationRunsAfterTheHitlDriver:
    def _agent(self, monkeypatch, events: list[str], hook: Any):
        _bound(monkeypatch, "test.step", lambda ctx, b: hook)
        agent = _hooked([("test.step", "on_iteration")])
        monkeypatch.setattr(
            agent, "_handle_hitl_approval", lambda api, cid: events.append("driver")
        )
        return agent

    def test_the_driver_runs_first_and_the_hook_gets_the_step_a_reader_and_a_writer(
        self, monkeypatch
    ):
        events: list[str] = []
        api = MagicMock()
        api.get_data.return_value = {"k": 1}

        def hook(iteration, read, write):
            events.append(f"hook{iteration}")
            write({"note": read()["k"]})

        self._agent(monkeypatch, events, hook)._on_loop_iteration(api, "cid", 3)

        assert events == ["driver", "hook3"]
        api.update_context.assert_called_once_with("cid", {"note": 1})

    def test_a_hook_that_raises_does_not_skip_the_driver_and_the_error_propagates(
        self, monkeypatch
    ):
        events: list[str] = []

        def hook(iteration, read, write):
            raise RuntimeError("hook failed")

        agent = self._agent(monkeypatch, events, hook)
        with pytest.raises(RuntimeError, match="hook failed"):
            agent._on_loop_iteration(MagicMock(), "cid", 1)

        assert events == ["driver"]

    def test_a_driver_that_raises_runs_no_hook(self, monkeypatch):
        events: list[str] = []
        agent = self._agent(monkeypatch, events, lambda i, r, w: events.append("hook"))

        def refuse(api, cid):
            raise AgentError("denied")

        monkeypatch.setattr(agent, "_handle_hitl_approval", refuse)
        with pytest.raises(AgentError, match="denied"):
            agent._on_loop_iteration(MagicMock(), "cid", 1)

        assert events == []

    def test_a_hook_has_no_path_to_the_grant(self, monkeypatch):
        api = MagicMock()
        for key in ("_approval_granted", "_run_output_keys", "__x", "_anything"):
            agent = self._agent(
                monkeypatch, [], lambda i, r, w, key=key: w({"fine": 1, key: True})
            )
            with pytest.raises(AgentError, match="internal keys"):
                agent._on_loop_iteration(api, "cid", 1)

        api.update_context.assert_not_called()  # nothing was written, not even "fine"

    def test_the_reflection_hook_writes_only_agent_feedback_on_its_cadence(self):
        written: list[dict[str, Any]] = []
        ctx = HandlerContext(config=AgentConfig(model="m", reflect_every_n=2))
        reflect = hook_lib.periodic_reflection(ctx, None)

        for step in (1, 2, 3, 4):
            reflect(step, lambda: {}, written.append)

        assert [list(w) for w in written] == [[ContextKeys.AGENT_FEEDBACK]] * 2

    def test_no_hooks_means_no_context_is_built_per_step(self, monkeypatch):
        agent = ReactAgent(
            runs._tools(), config=runs._config(), llm_interface=runs.ScriptedModel()
        )
        monkeypatch.setattr(agent, "_handle_hitl_approval", lambda api, cid: None)
        monkeypatch.setattr(
            agent,
            "_handler_context",
            lambda: pytest.fail("a context was built with no on_iteration hook"),
        )

        agent._on_loop_iteration(MagicMock(), "cid", 1)


class TestTheSeamShapes:
    def test_an_after_run_hook_returning_something_else_is_refused(self, monkeypatch):
        _loop(monkeypatch, ["a"])
        _bound(monkeypatch, "test.odd", lambda ctx, b: lambda: "again")

        with pytest.raises(AgentError, match="not a Retry"):
            _hooked([("test.odd", "after_run")]).run("go")

    def test_a_handler_factory_is_no_hook_at_any_position(self, monkeypatch):
        _loop(monkeypatch, ["a"])
        _bound(monkeypatch, "test.list", lambda ctx, b: [])
        for at in ("before_run", "after_run", "on_iteration"):
            agent = _hooked([("test.list", at)])
            with pytest.raises(AgentError, match="is a handler, not a hook"):
                if at == "on_iteration":
                    agent._on_loop_iteration(MagicMock(), "cid", 1)
                else:
                    agent.run("go")

    def test_the_first_retry_ends_the_chain_of_its_attempt(self, monkeypatch):
        events: list[str] = []
        _loop(monkeypatch, ["a", "b"])
        counter = iter([Retry("again"), None])
        _bound(monkeypatch, "test.judge", lambda ctx, b: lambda: next(counter))
        _bound(monkeypatch, "test.last", lambda ctx, b: lambda: events.append("last"))

        agent = _hooked([("test.judge", "after_run"), ("test.last", "after_run")])

        assert agent.run("go").answer == "b"
        assert events == ["last"]  # only the accepted attempt reached the second hook

    def test_before_run_hooks_apply_in_declared_order_once_per_run(self, monkeypatch):
        tasks: list[str] = []
        _loop(monkeypatch, ["x", "y"], tasks)
        built: list[str] = []

        def one(ctx, binding):
            built.append("one")
            return lambda t: t + "1"

        _bound(monkeypatch, "test.one", one)
        _bound(monkeypatch, "test.two", lambda ctx, b: lambda t: t + "2")
        counter = iter([Retry("n"), None])
        _bound(monkeypatch, "test.judge", lambda ctx, b: lambda: next(counter))
        agent = _hooked(
            [
                ("test.one", "before_run"),
                ("test.two", "before_run"),
                ("test.judge", "after_run"),
            ]
        )

        agent.run("t")

        assert built == ["one"]  # built and applied once, not once per attempt
        assert tasks == ["t12", "t12\n\nn"]  # retried on the rewritten base


class TestPresetsAndClasses:
    @pytest.mark.parametrize("name", ["verified_react", "auto_memory"])
    def test_a_preset_extends_react_and_holds_no_copy_of_its_template(self, name):
        body = pf.load_document(name)["body"]
        react, flat = pf.load_envelope("react"), pf.load_envelope(name)

        assert body["extends"] == "react"
        assert not {"fsm", "handlers", "variants"} & set(body)
        assert (
            flat.body.fsm == react.body.fsm
            and flat.body.handlers == react.body.handlers
        )
        assert flat.body.run.agent_type == "react" and flat.requires.executables == [
            "react"
        ]
        added = set(flat.body.options) - set(react.body.options)
        assert added == set(body["options_schema"]["properties"])
        assert all(h["hook"] in HANDLERS for h in body["run"]["hooks"])

    def test_the_option_defaults_of_the_files_are_those_of_the_hooks_and_classes(self):
        verified = pf.load_envelope("verified_react").body.options
        memory = pf.load_envelope("auto_memory").body.options

        assert verified["max_verify_retries"] == hook_lib.DEFAULT_MAX_VERIFY_RETRIES
        assert memory["recall_k"] == hook_lib.DEFAULT_RECALL_K
        assert memory["recall_min_score"] == hook_lib.DEFAULT_RECALL_MIN_SCORE
        assert memory["auto_remember"] is True
        assert memory["remember_only_on_success"] is False

    @pytest.mark.parametrize("name", ["verified_react", "auto_memory"])
    def test_a_model_origin_copy_with_hooks_is_refused_by_v7(self, name):
        document = pf.load_document(name)

        _, report = load_executable(
            document,
            resolver=namespace_resolver(),
            origin="model",
            allow={"tools": [], "handlers": [], "hooks": [], "executables": ["react"]},
            bases=lambda n: parse_executable(pf.load_document(n)),
        )

        assert not report.is_valid
        assert any("body.run.hooks" in e for e in report.errors)

    def test_the_classes_define_no_run_of_their_own_and_react_keeps_its_names(self):
        for cls in (VerifiedReactAgent, AutoMemoryReactAgent):
            assert "run" not in vars(cls) and "_on_loop_iteration" not in vars(cls)
            assert issubclass(cls, ReactAgent)
        for name in ("run", "run_stream", "_stream_via_run"):
            assert name in vars(ReactAgent)

    def test_a_negative_retry_count_is_refused_and_the_attributes_stay(self):
        with pytest.raises(Exception, match="max_verify_retries"):
            VerifiedReactAgent(runs._tools(), max_verify_retries=-1)
        agent = _memory_agent(recall_k=5, recall_min_score=0.5, auto_remember=False)
        assert (agent.recall_k, agent.recall_min_score, agent.auto_remember) == (
            5,
            0.5,
            False,
        )
        assert _verified(None, 3).max_verify_retries == 3
