"""The shared rig of the RunContext adoption plan (plan 7efd5cb4, roadmap row 1.14).

Three packages need the same doubles (agents, the reasoning engine, the workflow engine), so they
live here once: a capturing model double, an ``EventLog`` object over a temp directory, a spy on the
one run loop body (``fsm_llm.engine.api.run._run_rounds``, so every ``run_until_terminal*`` of every
runner is seen once), and a stand-in clock. ``collect_sections(work_dir)`` runs the parity matrix
(PG-agent, PG-reasoning, PG-workflow, PG-adapters) with NO ``context=`` anywhere and returns plain
JSON data: ``tests/test_fsm_llm/test_run_context_adoption.py`` compares the tree to
``tests/fixtures/run_context_parity_golden.json``, which an out-of-repo script wrote by running
this very function on a ``git archive`` of the plan base 51366ed3 (the golden is the parent's, never
the tree's).

Interface contract:

- ``collect_sections(work_dir) -> dict[str, dict[str, Any]]``: ``{section: {case: record}}``.
  A record is ``{"result": ..., "error": ..., "requests": [...], "events": [...], "runs": [...]}``
  with the keys the case has; every value is JSON data. ``work_dir`` must exist (one sub directory
  per case holds its event log). A case never raises: an error is recorded as ``{"type", "message",
  "cause"}`` data, so a refusal is part of the golden.
- ``events_of(log)``: the stable fields of every event, per conversation in first-seen order;
  ``ts``, ids, ``*_ref`` hashes and timings are dropped, conversation ids become ``c0``, ``c1``...
- ``HookSpy``: ``with HookSpy() as spy`` records one entry per run loop call (``max_steps``,
  whether a time budget was set, whether a ``before_step`` hook was given) and the step numbers that
  hook received; the hook is wrapped, never replaced, so the run is unchanged. ``spy.runs``.
- ``StandInClock``: a ``time`` stand-in whose ``monotonic()`` is one second later on every read.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

#: keys that carry a wall clock, an id or a content hash: never part of a parity record
_UNSTABLE_KEYS = frozenset(
    {"ts", "conversation_id", "call_id", "elapsed_ms", "handle", "timestamp", "id"}
)
_TS = re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_FLOAT_SECONDS = re.compile(r"(\d+\.\d{3,})")


def _unstable(key: str) -> bool:
    return (
        key in _UNSTABLE_KEYS
        or key.endswith("_ref")
        or key.endswith(("_seconds", "_time", "_ms"))
        or key.startswith("elapsed")
        or key.startswith("execution_time")
    )


def plain(value: Any) -> Any:
    """JSON data of ``value``: unstable keys dropped, text normalised, containers sorted by key."""
    if isinstance(value, dict):
        return {
            str(k): plain(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if not _unstable(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, str):
        return _UUID.sub("<uuid>", _TS.sub("<ts>", value))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (str, int)):
        return plain(enum_value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return plain(dump())
    return plain(repr(value))


def error_of(exc: BaseException) -> dict[str, Any]:
    """The type, the message and the type chain of the causes of an error, as data."""
    chain: list[str] = []
    cause = exc.__cause__
    while cause is not None and len(chain) < 8:
        chain.append(f"{type(cause).__name__}: {plain(str(cause))}")
        cause = cause.__cause__
    details = getattr(exc, "details", None)
    return {
        "type": type(exc).__name__,
        "message": plain(str(exc)),
        "cause": chain,
        "details": sorted(details) if isinstance(details, dict) else None,
    }


def request_row(call: str, request: dict[str, Any]) -> dict[str, str]:
    """One request of a capturing double: the call, and the sha256 of its normalised JSON."""
    text = plain(json.loads(json.dumps(request, sort_keys=True, default=str)))
    blob = json.dumps(text, sort_keys=True, ensure_ascii=True)
    label = request.get("field_name") if call == "extract_field" else call
    return {
        "call": call,
        "label": str(label),
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }


def events_of(log: Any) -> list[dict[str, Any]]:
    """Every event of ``log``, conversation by conversation, as stable data."""
    ids: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for cid in sorted(log.conversation_ids(), key=lambda c: _first_ts(log, c)):
        ids[cid] = f"c{len(ids)}"
    for cid, name in ids.items():
        for event in log.read(cid):
            row = plain({k: v for k, v in event.items() if k != "request_seq"})
            row["conversation"] = name
            rows.append(row)
    return rows


def _first_ts(log: Any, cid: str) -> tuple[str, str]:
    first = next(iter(log.read(cid)), {})
    return (str(first.get("ts", "")), cid)


class StandInClock:
    """A ``time`` stand-in: ``monotonic`` is one second later on every read, the rest is real."""

    def __init__(self, start: float = 1000.0) -> None:
        self._ticks = itertools.count(start, 1.0)

    def monotonic(self) -> float:
        return next(self._ticks)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class HookSpy:
    """Records every run loop call and the step numbers its ``before_step`` hook received."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self._patch: Any = None

    def __enter__(self) -> HookSpy:
        from fsm_llm.engine.api import run as run_module

        original = run_module._run_rounds
        spy = self

        def spied(
            api: Any,
            conversation_id: str,
            max_steps: int,
            max_seconds: float | None,
            before_step: Callable[[int], Any] | None,
            *rest: Any,
        ) -> Any:
            entry: dict[str, Any] = {
                "max_steps": max_steps,
                "seconds": None if max_seconds is None else "set",
                "hook": before_step is not None,
                "steps": [],
            }
            spy.runs.append(entry)
            hook = before_step
            if before_step is not None:

                def recording(step: int) -> Any:
                    entry["steps"].append(step)
                    return before_step(step)

                hook = recording
            return original(api, conversation_id, max_steps, max_seconds, hook, *rest)

        self._patch = mock.patch.object(run_module, "_run_rounds", spied)
        self._patch.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._patch.stop()


@contextmanager
def frozen_today() -> Iterator[None]:
    """``datetime.date.today()`` answers one recorded day (a prompt may name it)."""
    from tests.test_fsm_llm.kind_corpus import _frozen_today

    with _frozen_today():
        yield


def record_case(
    work_dir: Path,
    name: str,
    body: Callable[[Any], dict[str, Any]],
    *,
    log: bool = True,
) -> dict[str, Any]:
    """Run ``body(event_log_or_None)`` under the hook spy and a fresh event log; never raises.

    ``body`` returns the ``result`` data (and may put ``requests`` beside it); the record adds the
    ``events`` and the ``runs`` of the spy. An exception of ``body`` is recorded as ``error``.
    """
    from fsm_llm.runtime import EventLog

    case_dir = work_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {}
    with frozen_today(), HookSpy() as spy:
        if log:
            with EventLog(case_dir / "log") as event_log:
                _fill(record, body, event_log)
                record["events"] = events_of(event_log)
        else:
            _fill(record, body, None)
    record["runs"] = plain(spy.runs)
    return record


def _fill(record: dict[str, Any], body: Callable[[Any], Any], event_log: Any) -> None:
    try:
        record.update(plain(body(event_log)))
    except Exception as exc:  # recorded, never propagated
        record["error"] = error_of(exc)


def run_async(coro: Any) -> Any:
    """Run one coroutine to its end on a fresh loop."""
    return asyncio.run(coro)


# ---- PG-agent ---------------------------------------------------------------------------------


def _agent_outcome(result: Any) -> dict[str, Any]:
    """What an ``AgentResult`` says: answer, status, reasons, the tool trace and the iterations."""
    return {
        "answer": result.answer,
        "status": result.status,
        "reason": result.reason,
        "stop_reason": result.stop_reason,
        "success": result.success,
        "tools": [[c.tool_name, c.parameters] for c in result.trace.tool_calls],
        "iterations": result.trace.total_iterations,
        "context_keys": sorted(result.final_context),
    }


def _models() -> dict[str, type]:
    from fsm_llm.core.definitions import ResponseGenerationResponse
    from tests.pattern_runs import ScriptedModel

    class NeverDone(ScriptedModel):
        """``should_terminate`` is never true: only the iteration ceiling ends the run."""

        def extract_field(self, request: Any) -> Any:
            response = super().extract_field(request)
            if request.field_name == "should_terminate":
                return response.model_copy(update={"value": False})
            return response

    class Cut(ScriptedModel):
        """Every reply is cut at ``max_tokens``."""

        def generate_response(self, request: Any) -> Any:
            super().generate_response(request)
            return ResponseGenerationResponse(
                message="Answer: partial",
                message_type="response",
                reasoning="m",
                truncated=True,
            )

    return {"scripted": ScriptedModel, "never_done": NeverDone, "cut": Cut}


def _failing_tools() -> Any:
    from fsm_llm.tools import ToolRegistry

    def boom(params: dict[str, Any]) -> str:
        raise ValueError("the tool failed")

    registry = ToolRegistry()
    registry.register_function(
        boom,
        name="search",
        description="Tool search",
        parameter_schema={"properties": {"x": {"type": "string"}}},
    )
    return registry


def _agent_case(
    work_dir: Path,
    name: str,
    build: Callable[[Any, Any], Any],
    *,
    model: str = "scripted",
    mode: str = "run",
    clock: bool = False,
) -> dict[str, Any]:
    """One agent case: ``build(model, event_log)`` makes the agent, ``mode`` is run, stream, arun."""
    models = _models()

    def body(log: Any) -> dict[str, Any]:
        from tests.pattern_runs import _TASK

        double = models[model]()
        out: dict[str, Any] = {}
        stand_in = (
            mock.patch("fsm_llm.agents.base.runs.time", StandInClock())
            if clock
            else nullcontext()
        )
        try:
            with stand_in:
                agent = build(double, log)
                if mode == "stream":
                    out["result"] = {"chunks": list(agent.run_stream(_TASK))}
                elif mode == "arun":
                    out["result"] = _agent_outcome(run_async(agent.arun(_TASK)))
                else:
                    out["result"] = _agent_outcome(agent.run(_TASK))
        except Exception as exc:
            out["error"] = error_of(exc)
        out["requests"] = [request_row(c, r) for c, r in double.log]
        return out

    return record_case(work_dir, name, body)


def agent_section(work_dir: Path) -> dict[str, Any]:
    """PG-agent: react, reflexion, a pattern file, parallel react, every stop path."""
    from fsm_llm import agents as A
    from fsm_llm.agents.definitions import AgentConfig
    from tests import pattern_runs as pr

    def kw(model: Any, log: Any, **over: Any) -> dict[str, Any]:
        return {
            "config": pr._config(),
            "llm_interface": model,
            "event_log": log,
            **over,
        }

    def react(model: Any, log: Any) -> Any:
        return A.ReactAgent(pr._tools(many=True), **kw(model, log))

    work_dir.mkdir(parents=True, exist_ok=True)
    one = AgentConfig(max_iterations=1, model="mock-model")
    timed = AgentConfig(max_iterations=5, model="mock-model", timeout_seconds=2.5)
    cases: dict[str, dict[str, Any]] = {
        "react_ok": _agent_case(work_dir, "react_ok", react),
        "react_arun": _agent_case(work_dir, "react_arun", react, mode="arun"),
        "react_stream": _agent_case(work_dir, "react_stream", react, mode="stream"),
        "reflexion_ok": _agent_case(
            work_dir,
            "reflexion_ok",
            lambda m, log: A.ReflexionAgent(pr._tools(many=True), **kw(m, log)),
        ),
        "prompt_chain_ok": _agent_case(
            work_dir,
            "prompt_chain_ok",
            lambda m, log: A.PromptChainAgent(pr._chain(), **kw(m, log)),
        ),
        "parallel_react_ok": _agent_case(
            work_dir,
            "parallel_react_ok",
            lambda m, log: A.ParallelReactAgent(pr._tools(many=True), **kw(m, log)),
        ),
        "react_max_iterations": _agent_case(
            work_dir,
            "react_max_iterations",
            lambda m, log: A.ReactAgent(pr._tools(), **kw(m, log, config=one)),
            model="never_done",
        ),
        "react_timeout": _agent_case(
            work_dir,
            "react_timeout",
            lambda m, log: A.ReactAgent(pr._tools(), **kw(m, log, config=timed)),
            model="never_done",
            clock=True,
        ),
        "react_tool_error": _agent_case(
            work_dir,
            "react_tool_error",
            lambda m, log: A.ReactAgent(_failing_tools(), **kw(m, log)),
        ),
        "react_flagged_no_hitl": _agent_case(
            work_dir,
            "react_flagged_no_hitl",
            lambda m, log: A.ReactAgent(pr._tools(flagged=True), **kw(m, log)),
        ),
        "react_cut_reply": _agent_case(work_dir, "react_cut_reply", react, model="cut"),
    }
    return cases


# ---- PG-reasoning -----------------------------------------------------------------------------


def _solve_record(
    work_dir: Path,
    name: str,
    script: dict[str, Any],
    *,
    stream: bool = False,
    max_steps: int | None = None,
    classifier_budget: bool = False,
) -> dict[str, Any]:
    from fsm_llm import RunBudgetExceededError
    from fsm_llm.reasoning import ReasoningEngine
    from fsm_llm.reasoning.constants import ContextKeys
    from tests.test_fsm_llm_reasoning.test_engine_scripted import _PROBLEM, _ScriptedLLM

    def body(log: Any) -> dict[str, Any]:
        llm = _ScriptedLLM(script)
        out: dict[str, Any] = {}
        try:
            with ReasoningEngine(
                model="mock", llm_interface=llm, event_log=log
            ) as engine:
                out["default_steps"] = engine._solve_steps
                if classifier_budget:
                    engine.classifier.run_until_terminal = _spent_classifier(
                        RunBudgetExceededError
                    )
                    engine._classify_problem(
                        {
                            ContextKeys.PROBLEM_STATEMENT: _PROBLEM,
                            ContextKeys.PROBLEM_TYPE: "logic",
                            ContextKeys.PROBLEM_COMPONENTS: ["a"],
                        }
                    )
                elif stream:
                    chunks = engine.solve_problem_stream(_PROBLEM, max_steps=max_steps)
                    out["chunks"] = list(chunks)
                    solution, info = chunks.result()
                    out["result"] = _solve_outcome(solution, info)
                else:
                    solution, info = engine.solve_problem(_PROBLEM, max_steps=max_steps)
                    out["result"] = _solve_outcome(solution, info)
                out["active_after"] = len(
                    engine.orchestrator.list_active_conversations()
                )
        except Exception as exc:
            out["error"] = error_of(exc)
        out["requests"] = [
            request_row(type(r).__name__, json.loads(r.model_dump_json()))
            for r in llm.requests
        ]
        out["completions"] = [
            request_row("complete", json.loads(r.model_dump_json()))
            for r in llm.completions
        ]
        return out

    return record_case(work_dir, name, body)


def _spent_classifier(error_type: Any) -> Callable[..., Any]:
    def spent(conversation_id: str, **kwargs: Any) -> Any:
        raise error_type("steps", 10, 10)

    return spent


def _solve_outcome(solution: str, info: dict[str, Any]) -> dict[str, Any]:
    trace = info["reasoning_trace"]
    return {
        "solution": solution,
        "types": trace["reasoning_types_used"],
        "visited": [step["to"] for step in trace["steps"]],
        "responses": info["all_responses"],
        "summary": info["summary"],
        "final_keys": sorted(info["final_context"]),
        "confidence": trace["final_confidence"],
    }


def reasoning_section(work_dir: Path) -> dict[str, Any]:
    """PG-reasoning: scripted solves (analytical, calculator), the stream, the two budget errors."""
    from tests.test_fsm_llm_reasoning import test_engine_scripted as sc

    work_dir.mkdir(parents=True, exist_ok=True)
    analytical = {
        **sc._VALID_SCRIPT,
        "recommended_reasoning_type": "analytical",
        "reasoning_strategy": "analytical",
        "integrated_analysis": "The argument is valid: Socrates is mortal.",
    }
    return {
        "analytical_ok": _solve_record(work_dir, "analytical_ok", analytical),
        "deductive_ok": _solve_record(work_dir, "deductive_ok", sc._VALID_SCRIPT),
        "calculator_ok": _solve_record(work_dir, "calculator_ok", sc._CALC_SCRIPT),
        "retry_loop": _solve_record(work_dir, "retry_loop", sc._FAILING_SCRIPT),
        "stream_ok": _solve_record(
            work_dir, "stream_ok", sc._VALID_SCRIPT, stream=True
        ),
        "never_ending_7": _solve_record(
            work_dir, "never_ending_7", sc._NEVER_ENDING_SCRIPT, max_steps=7
        ),
        "classifier_budget": _solve_record(
            work_dir,
            "classifier_budget",
            sc._VALID_SCRIPT,
            classifier_budget=True,
        ),
    }


# ---- PG-workflow ------------------------------------------------------------------------------


def _workflows() -> dict[str, Any]:
    """The three definitions of the matrix: linear, branching, retry (and a longer chain)."""
    from fsm_llm.workflows import (
        auto_step,
        condition_step,
        create_workflow,
        retry_step,
    )

    def chain(workflow_id: str, count: int) -> Any:
        wf = create_workflow(workflow_id, workflow_id)
        for i in range(1, count + 1):
            step = auto_step(
                f"s{i}",
                f"s{i}",
                f"s{i + 1}" if i < count else "",
                action=lambda ctx, i=i: {f"k{i}": i},
            )
            wf.with_initial_step(step) if i == 1 else wf.with_step(step)
        return wf

    branch = create_workflow("branch", "branch")
    branch.with_initial_step(
        condition_step("c", "c", lambda ctx: bool(ctx.get("flag")), "yes", "no")
    )
    branch.with_step(auto_step("yes", "yes", "", action=lambda ctx: {"took": "yes"}))
    branch.with_step(auto_step("no", "no", "", action=lambda ctx: {"took": "no"}))

    attempts: list[int] = []

    def flaky(ctx: dict[str, Any]) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("flaky step failed")
        return {"flaky": len(attempts)}

    retry = create_workflow("retry", "retry")
    retry.with_initial_step(
        retry_step(
            "r",
            "r",
            auto_step("inner", "inner", "", action=flaky),
            max_retries=3,
            backoff_factor=0.0,
        )
    )
    return {
        "linear": chain("linear", 3),
        "branch": branch,
        "retry": retry,
        "long": chain("long", 5),
    }


async def _run_workflow(
    definition: Any, context: dict[str, Any], max_steps: int
) -> Any:
    from fsm_llm.workflows import WorkflowEngine

    hooks: list[list[Any]] = []

    def hook(event: str, instance: Any, data: dict[str, Any]) -> None:
        hooks.append([event, data.get("step_id") or str(data.get("status"))])

    engine = WorkflowEngine(max_steps=max_steps)
    engine.add_hook(hook)
    engine.register_workflow(definition)
    instance_id = await engine.start(definition.workflow_id, context)
    instance = engine.get_instance(instance_id)
    return {
        "status": instance.status,
        "error": instance.error,
        "history": [
            [h.step_id, h.status, h.message, h.error, sorted(h.data)]
            for h in instance.history
        ],
        "hooks": hooks,
        "context": {k: v for k, v in instance.context.items() if not k.startswith("_")},
    }


def workflow_section(work_dir: Path) -> dict[str, Any]:
    """PG-workflow: ``WorkflowEngine`` with no context on linear, branching and retry workflows."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("linear", "linear", {}, 100),
        ("branch_yes", "branch", {"flag": True}, 100),
        ("branch_no", "branch", {"flag": False}, 100),
        ("retry", "retry", {}, 100),
        ("step_budget", "long", {}, 3),
    )
    out: dict[str, Any] = {}
    for name, workflow_id, context, max_steps in cases:
        definition = _workflows()[workflow_id]
        out[name] = record_case(
            work_dir,
            name,
            lambda _log, d=definition, c=context, m=max_steps: {
                "result": run_async(_run_workflow(d, dict(c), m))
            },
            log=False,
        )
    return out


# ---- PG-adapters ------------------------------------------------------------------------------


def _fsm_documents() -> dict[str, dict[str, Any]]:
    from tests.test_fsm_llm import kind_corpus as kc

    short = kc._fsm(
        "Demo",
        "a",
        kc._state(
            "a",
            extraction_instructions="Extract the result.",
            required_context_keys=["result"],
            response_instructions="A.",
            transitions=[kc._go("b")],
        ),
        kc._state("b", response_instructions="B."),
    )
    long = kc._fsm(
        "Long",
        "a",
        kc._state("a", response_instructions="A.", transitions=[kc._go("b")]),
        kc._state("b", response_instructions="B.", transitions=[kc._go("c")]),
        kc._state("c", response_instructions="C.", transitions=[kc._go("d")]),
        kc._state("d", response_instructions="D."),
    )

    def doc(ident: str, fsm: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": "fsm-llm/executable",
            "kind": "fsm",
            "id": ident,
            "contract": {"inputs": ["topic"], "outputs": ["result"]},
            "body": {"fsm": fsm},
        }

    return {"short": doc("demo", short), "long": doc("long", long)}


def _workflow_document(count: int) -> dict[str, Any]:
    steps = {
        f"s{i}": {
            "type": "AutoStep",
            "step_id": f"s{i}",
            "name": f"s{i}",
            "next_step_id": f"s{i + 1}" if i < count else "",
        }
        for i in range(1, count + 1)
    }
    return {
        "schema": "fsm-llm/executable",
        "kind": "workflow",
        "id": "wf",
        "name": "wf",
        "contract": {"inputs": [], "outputs": []},
        "requires": {"tools": "none", "callables": [], "executables": []},
        "lifecycle": "draft",
        "body": {"initial_step_id": "s1", "steps": steps},
    }


def _adapter_case(
    work_dir: Path,
    name: str,
    make: Callable[[Any], tuple[Any, dict[str, Any], Any]],
) -> dict[str, Any]:
    """``make(log)`` returns ``(executable, inputs, context)``; the whole ``RunResult`` is the data."""

    def body(log: Any) -> dict[str, Any]:
        executable, inputs, context = make(log)
        return {"result": executable.run(inputs, context=context).model_dump()}

    return record_case(work_dir, name, body)


def adapter_section(work_dir: Path) -> dict[str, Any]:
    """PG-adapters: the four adapters under a ``RunContext`` that holds no ledger."""
    from fsm_llm.core.names import NameRegistry
    from fsm_llm.engine.context import RunContext
    from fsm_llm.engine.executables import ExecutableRegistry
    from fsm_llm.reasoning.executables import SOLVE_ID
    from fsm_llm.testing import ScriptedLLM
    from tests import pattern_runs as pr
    from tests.test_fsm_llm_reasoning import test_engine_scripted as sc

    work_dir.mkdir(parents=True, exist_ok=True)
    docs = _fsm_documents()

    def fsm_llm_() -> Any:
        return ScriptedLLM(extraction_data={"result": "42"}, default_response="r")

    def fsm(
        key: str, **limits: Any
    ) -> Callable[[Any], tuple[Any, dict[str, Any], Any]]:
        def make(log: Any) -> tuple[Any, dict[str, Any], Any]:
            exe = ExecutableRegistry().add_document(docs[key])
            context = RunContext(llm_interface=fsm_llm_(), event_log=log, **limits)
            return exe, {"topic": "t"}, context

        return make

    def cancelled(make: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapped(log: Any) -> Any:
            exe, inputs, context = make(log)
            context.cancel.cancel()
            return exe, inputs, context

        return wrapped

    def react(**limits: Any) -> Callable[[Any], tuple[Any, dict[str, Any], Any]]:
        def make(log: Any) -> tuple[Any, dict[str, Any], Any]:
            names: NameRegistry[Any] = NameRegistry("callables")
            names.register("tools", pr._tools(many=True))
            names.register("config", pr._config())
            registry = ExecutableRegistry()
            registry.load_sources()
            context = RunContext(
                llm_interface=pr.ScriptedModel(),
                event_log=log,
                callables=names,
                **limits,
            )
            return registry.get("react"), {"task": pr._TASK}, context

        return make

    def solve(**limits: Any) -> Callable[[Any], tuple[Any, dict[str, Any], Any]]:
        def make(log: Any) -> tuple[Any, dict[str, Any], Any]:
            registry = ExecutableRegistry()
            registry.load_sources()
            context = RunContext(
                llm_interface=sc._ScriptedLLM(
                    sc._NEVER_ENDING_SCRIPT if limits else sc._VALID_SCRIPT
                ),
                event_log=log,
                **limits,
            )
            return registry.get(SOLVE_ID), {"problem": sc._PROBLEM}, context

        return make

    def workflow(count: int, **limits: Any) -> Callable[[Any], Any]:
        def make(log: Any) -> tuple[Any, dict[str, Any], Any]:
            from fsm_llm.core.definitions.executable import parse_executable
            from fsm_llm.workflows.executable import workflow_adapter

            exe = workflow_adapter(parse_executable(_workflow_document(count)))
            return exe, {}, RunContext(event_log=log, **limits)

        return make

    cases: dict[str, Callable[[Any], Any]] = {
        "fsm_ok": fsm("short"),
        "fsm_budget": fsm("long", max_steps=2),
        "fsm_cancelled": cancelled(fsm("short")),
        "pattern_react_ok": react(),
        "pattern_react_seconds": react(max_seconds=100.0),
        "pattern_react_max_steps": react(max_steps=3),
        "pattern_react_cancelled": cancelled(react()),
        "solve_ok": solve(),
        "solve_budget": solve(max_steps=6),
        "solve_cancelled": cancelled(solve()),
        "workflow_ok": workflow(3),
        "workflow_budget": workflow(5, max_steps=2),
        "workflow_cancelled": cancelled(workflow(3)),
    }
    return {name: _adapter_case(work_dir, name, make) for name, make in cases.items()}


# ---- the matrix -------------------------------------------------------------------------------


def collect_sections(work_dir: Path) -> dict[str, dict[str, Any]]:
    """The whole parity matrix with no ``context=`` anywhere; see the module docstring."""
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "PG-agent": agent_section(work_dir / "agent"),
        "PG-reasoning": reasoning_section(work_dir / "reasoning"),
        "PG-workflow": workflow_section(work_dir / "workflow"),
        "PG-adapters": adapter_section(work_dir / "adapters"),
    }
