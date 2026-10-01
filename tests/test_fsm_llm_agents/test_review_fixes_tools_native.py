"""Review round 1 fixes for tools and native function calling (plan 944e2692
step 18.2, D-029 item list "18.2", decisions D-032 and D-033).

Each class pins one accepted finding and fails on the parent commit 07900ff:
the tool timeout bound and its unknown outcome, the cache never serving a
granted call, one schema source for prompt-mode tool descriptions, the
``gated`` keyword checked at construction, the native run finishing its
post-loop turns past the wall clock, the forced turn's raw arguments kept
out of ``final_context``, the ``system_policy`` type, and the native
handler rules whose mutants survived review (M01, M16, M17, M18, M19, M23).
"""

from __future__ import annotations

import contextvars
import json
import math
import threading
import time
from typing import Any, Optional

import pydantic
import pytest
from pydantic import BaseModel

import fsm_llm.api as api_module
from fsm_llm.agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    PlanExecuteAgent,
    ReactAgent,
    ReflexionAgent,
    REWOOAgent,
    ToolAnnotations,
)
from fsm_llm.agents.constants import ContextKeys, NativeFCContextKeys
from fsm_llm.agents.definitions import ToolCall, ToolDefinition, ToolResult
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.native_fc import NativeFCHandlers
from fsm_llm.agents.prompts import build_think_extraction_instructions
from fsm_llm.agents.tool_registries import CachingToolRegistry, RetryingToolRegistry
from fsm_llm.agents.tools import ToolRegistry
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents import test_native_fc_golden as golden
from tests.test_fsm_llm_agents.test_toolspec import (
    _approved_context,
    _policy_gates_everything,
)

_K = NativeFCContextKeys
_SECRET = "sk-test-0123456789abcdef0123456789abcdef"

# ---------------------------------------------------------------------------
# Tool timeout: finite, bounded, and nothing raises after the worker starts
# ---------------------------------------------------------------------------


def _counting_tool(ran: list[int], seconds: float = 0.0) -> Any:
    def slow(x: int) -> str:
        time.sleep(seconds)
        ran.append(x)
        return "ok"

    return slow


class TestTimeoutBound:
    @pytest.mark.parametrize("bad", [math.inf, 1e12, math.nan])
    def test_definition_refuses_an_unwaitable_timeout(self, bad):
        with pytest.raises(pydantic.ValidationError):
            ToolDefinition(
                name="t", description="d", execute_fn=lambda: 1, timeout_s=bad
            )

    @pytest.mark.parametrize("bad", [math.inf, 1e12])
    def test_register_function_refuses_it(self, bad):
        with pytest.raises(ValueError):
            ToolRegistry().register_function(
                _counting_tool([]), name="slow", description="d", timeout_s=bad
            )

    def test_the_largest_waitable_timeout_is_accepted(self):
        definition = ToolDefinition(
            name="t",
            description="d",
            execute_fn=lambda: 1,
            timeout_s=threading.TIMEOUT_MAX,
        )
        assert definition.timeout_s == threading.TIMEOUT_MAX

    def test_a_timeout_set_after_registration_never_runs_the_tool(self):
        """Bypassing the field check is refused before the worker starts.

        On the parent the worker ran the tool, then the wait raised
        ``OverflowError``: a tool that ran was reported failed and a
        retry-safe one ran three times.
        """
        ran: list[int] = []
        registry = RetryingToolRegistry(max_retries=2)
        registry.register_function(
            _counting_tool(ran),
            name="slow",
            description="d",
            timeout_s=1.0,
            annotations=ToolAnnotations(idempotent=True),
        )
        registry.get("slow").timeout_s = 1e12
        result = registry.execute(ToolCall(tool_name="slow", parameters={"x": 1}))
        time.sleep(0.2)
        assert ran == []
        assert result.success is False
        assert "invalid timeout_s" in (result.error or "")
        assert result.timed_out is False


class TestTimedOutOutcomeIsUnknown:
    def _registry(self, release: threading.Event, ran: list[str]) -> ToolRegistry:
        def pay(amount: str) -> str:
            release.wait(5.0)
            ran.append(amount)
            return "paid"

        registry = ToolRegistry()
        registry.register_function(
            pay, name="pay", description="d", requires_approval=True, timeout_s=0.1
        )
        return registry

    def test_result_is_marked_timed_out_and_says_so(self):
        release, ran = threading.Event(), []
        result = self._registry(release, ran).execute(
            ToolCall(tool_name="pay", parameters={"amount": "9"})
        )
        release.set()
        assert result.success is False
        assert result.timed_out is True
        assert "outcome is unknown" in (result.error or "")
        assert "may still happen" in (result.error or "")
        assert result.observation.startswith("[TOOL OUTCOME UNKNOWN] ")

    def test_hitl_path_tells_the_model_the_outcome_is_unknown(self):
        """An approved (gated) call that times out is not shown as failed."""
        release, ran = threading.Event(), []
        registry = self._registry(release, ran)
        handlers = AgentHandlers(registry, requires_approval=_policy_gates_everything)
        delta = handlers.execute_tool(_approved_context())
        release.set()
        observation = delta[ContextKeys.OBSERVATIONS][-1]
        assert "[TOOL OUTCOME UNKNOWN]" in observation
        assert "[TOOL FAILED]" not in observation
        assert "outcome is unknown" in observation

    def test_failure_and_success_observations_are_unchanged(self):
        failed = ToolResult(tool_name="t", success=False, error="boom")
        assert failed.observation == "[TOOL FAILED] Error: boom"
        ok = ToolResult(tool_name="t", success=True, result="fine")
        assert ok.observation == "fine"

    def test_worker_sees_the_callers_context_variables(self):
        """Review NOTE 5 (mutant M6): the worker runs in a copy of the
        caller's context variables."""
        var: contextvars.ContextVar[str] = contextvars.ContextVar("v", default="-")
        var.set("caller")

        def read() -> str:
            return var.get()

        registry = ToolRegistry()
        registry.register_function(read, name="read", description="d", timeout_s=5.0)
        assert registry.execute(ToolCall(tool_name="read")).result == "caller"


# ---------------------------------------------------------------------------
# CachingToolRegistry never serves or stores a granted call
# ---------------------------------------------------------------------------


class TestCacheSkipsGatedCalls:
    def _registry(self, paid: list[int]) -> CachingToolRegistry:
        def pay(amount: int) -> str:
            paid.append(amount)
            return f"paid {amount}"

        registry = CachingToolRegistry()
        registry.register_function(pay, requires_approval=True)
        return registry

    def test_each_granted_call_runs(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        for _ in range(2):
            assert registry.execute(call, gated=True).success
        assert paid == [10, 10]
        assert registry.cache_hits == 0

    def test_a_granted_result_is_never_served_to_a_later_call(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        registry.execute(call, gated=True)
        registry.execute(call)
        assert paid == [10, 10]

    def test_ungated_calls_are_still_cached(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        registry.execute(call)
        registry.execute(call)
        assert paid == [10]
        assert registry.cache_hits == 1


# ---------------------------------------------------------------------------
# One schema source: prompt mode reads the args_model schema
# ---------------------------------------------------------------------------


class _Inner(BaseModel):
    title: str


def _typed_search(
    query: str,
    limit: Optional[int] = None,  # noqa: UP045 (the review case, verbatim)
    inner: _Inner | None = None,
) -> str:
    """Search the catalogue."""
    return query


class TestPromptSchemaSource:
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_function(_typed_search, name="search")
        return registry

    def test_description_shows_optional_int_as_integer_or_null(self):
        text = self._registry().to_prompt_description()
        assert "limit [optional] (integer or null)" in text
        assert "query [REQUIRED] (string)" in text
        assert "inner [optional] (object or null)" in text

    def test_description_and_native_schema_share_one_source(self):
        registry = self._registry()
        native = registry.get_json_schemas()[0]["function"]["parameters"]
        assert native["properties"]["limit"]["anyOf"] == [
            {"type": "integer"},
            {"type": "null"},
        ]
        assert "(integer or null)" in registry.to_prompt_description()

    def test_think_example_uses_the_typed_schema(self):
        text = build_think_extraction_instructions(self._registry())
        example = next(
            json.loads(line) for line in text.splitlines() if '"search"' in line
        )
        assert example["tool_input"]["limit"] == 0

    def test_prompt_mode_agent_is_shown_integer_or_null(self):
        """The text a ReAct agent's model actually receives."""
        llm = PromptGroundedLLM()
        agent = ReactAgent(
            self._registry(),
            config=AgentConfig(model="m", max_iterations=1),
            llm_interface=llm,
        )
        try:
            agent.run("find shoes")
        except AgentError:
            pass
        prompts = [request.system_prompt for request in llm.calls("extract_field")]
        assert prompts
        assert any("limit [optional] (integer or null)" in p for p in prompts)
        assert not any("limit [optional] (string)" in p for p in prompts)

    def test_explicit_schema_tools_are_unchanged(self):
        text = golden._registry().to_prompt_description()
        assert "city [REQUIRED] (string): City" in text


# ---------------------------------------------------------------------------
# A registry whose execute lacks `gated` is refused at construction
# ---------------------------------------------------------------------------


class _OldRegistry(ToolRegistry):
    def execute(self, tool_call: ToolCall) -> ToolResult:  # type: ignore[override]
        return super().execute(tool_call)


class _KwargsRegistry(ToolRegistry):
    def execute(self, tool_call: ToolCall, **kwargs: Any) -> ToolResult:
        return super().execute(tool_call, **kwargs)


def _with_tool(registry: ToolRegistry) -> ToolRegistry:
    registry.register_function(lambda q: q, name="echo", description="Echo q")
    return registry


class TestGatedKeywordRequired:
    """Refused where ``gated=`` is passed (``AgentHandlers``, built at the
    start of every run of the agents it executes for), before any model
    call. On the parent each tool call raised a bare ``TypeError`` inside
    the handler, which the run turned into a failed tool."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda r, llm: ReactAgent(
                r, config=AgentConfig(model="m"), llm_interface=llm
            ),
            lambda r, llm: ReflexionAgent(
                r, config=AgentConfig(model="m"), llm_interface=llm
            ),
            lambda r, llm: PlanExecuteAgent(
                r, config=AgentConfig(model="m"), llm_interface=llm
            ),
        ],
        ids=["react", "reflexion", "plan_execute"],
    )
    def test_old_signature_is_refused_before_any_model_call(self, build):
        llm = PromptGroundedLLM()
        agent = build(_with_tool(_OldRegistry()), llm)
        with pytest.raises(AgentError, match=r"_OldRegistry\.execute .*'gated'"):
            agent.run("echo hi")
        assert llm.requests == []

    def test_executor_construction_refuses_it(self):
        with pytest.raises(AgentError, match="gated"):
            AgentHandlers(_with_tool(_OldRegistry()))

    def test_reasoning_react_checks_the_callers_registry(self):
        pytest.importorskip("fsm_llm.reasoning")
        from fsm_llm.agents import ReasoningReactAgent

        with pytest.raises(AgentError, match="_OldRegistry"):
            ReasoningReactAgent(
                _with_tool(_OldRegistry()), config=AgentConfig(model="m")
            )

    @pytest.mark.parametrize(
        "build",
        [
            lambda r: REWOOAgent(r, config=AgentConfig(model="m")),
            lambda r: NativeFunctionCallingReactAgent(r, config=AgentConfig(model="m")),
        ],
        ids=["rewoo", "native_fc"],
    )
    def test_agents_that_never_pass_gated_accept_it(self, build):
        """They call ``execute(call)``: an older override still runs there
        (the harness live suite spies on ``execute`` this way)."""
        build(_with_tool(_OldRegistry()))

    def test_kwargs_and_the_shipped_registries_are_accepted(self):
        for registry in (
            _KwargsRegistry(),
            ToolRegistry(),
            CachingToolRegistry(),
            RetryingToolRegistry(),
        ):
            AgentHandlers(_with_tool(registry))


# ---------------------------------------------------------------------------
# native_fc: a final answer is never lost to the wall clock (D-032)
# ---------------------------------------------------------------------------


class _FakeClock:
    """Stands in for ``fsm_llm.api``'s ``time``; each model call costs time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _timed_recorder(
    monkeypatch: pytest.MonkeyPatch, script: list[Any], seconds: float
) -> golden._Recorder:
    clock = _FakeClock()
    monkeypatch.setattr(api_module, "time", clock)
    recorder = golden._Recorder(script)

    def call(**kwargs: Any) -> Any:
        clock.now += seconds
        return recorder(**kwargs)

    for binding in golden._BINDINGS:
        monkeypatch.setattr(binding, call)
    return recorder


def _native(**config: Any) -> NativeFunctionCallingReactAgent:
    settings: dict[str, Any] = {"model": "gpt-4o-mini", "max_iterations": 5, **config}
    return NativeFunctionCallingReactAgent(
        golden._registry(), config=AgentConfig(**settings)
    )


class TestPostLoopTurnsFinishPastTheDeadline:
    """``budget.py clock`` of review pass 5: each call costs 0.6 s of a 1.0 s
    budget, the loop ends with a final answer at 1.2 s. e1f63a9 ran the
    forced write and the repair; the parent raised AgentTimeoutError."""

    def test_forced_write_runs_and_the_answer_ships(self, monkeypatch):
        recorder = _timed_recorder(
            monkeypatch,
            [
                golden._calls(golden._PARIS),
                golden._final("plain answer"),
                golden._calls(golden._NOTE),
            ],
            0.6,
        )
        result = _native(timeout_seconds=1.0, force_final_tool="write_note").run("t")
        assert len(recorder.requests) == 3
        assert result.success is True
        assert result.answer == "plain answer"
        assert [c.tool_name for c in result.trace.tool_calls] == [
            "weather",
            "write_note",
        ]

    def test_repair_runs_and_its_answer_ships(self, monkeypatch):
        recorder = _timed_recorder(
            monkeypatch,
            [
                golden._calls(golden._PARIS),
                golden._final("plain answer"),
                golden._final(golden._REPORT),
            ],
            0.6,
        )
        result = _native(timeout_seconds=1.0, output_schema=golden._WeatherReport).run(
            "t"
        )
        assert len(recorder.requests) == 3
        assert result.success is True
        assert result.structured_output == golden._WeatherReport(
            city="Paris", conditions="sunny"
        )

    def test_a_loop_turn_past_the_deadline_still_times_out(self, monkeypatch):
        from fsm_llm.agents.exceptions import AgentTimeoutError

        recorder = _timed_recorder(
            monkeypatch,
            [golden._calls(golden._PARIS), golden._calls(golden._ROME)],
            0.6,
        )
        with pytest.raises(AgentTimeoutError):
            _native(timeout_seconds=1.0).run("t")
        assert len(recorder.requests) == 2


# ---------------------------------------------------------------------------
# native_fc: raw call arguments never reach final_context
# ---------------------------------------------------------------------------


class TestRawArgumentsStayOut:
    def test_forced_turn_arguments_are_not_in_final_context(self, monkeypatch):
        secret_note = ("call_9", "write_note", {"text": "done", "api_key": _SECRET})
        recorder = golden._Recorder(
            [
                golden._calls(golden._PARIS),
                golden._final("Paris is sunny."),
                golden._calls(secret_note),
            ]
        )
        for binding in golden._BINDINGS:
            monkeypatch.setattr(binding, recorder)
        result = _native(force_final_tool="write_note").run("t")
        assert result.success is True
        assert _SECRET not in json.dumps(result.final_context, default=str)
        assert _K.FORCED_REPLY not in result.final_context
        shown = result.trace.tool_calls[-1].parameters
        assert shown["api_key"] == "<redacted>"

    def test_a_refused_turn_clears_its_reply(self):
        handlers = NativeFCHandlers(golden._registry(), max_tool_turns=5)
        context = {
            ContextKeys.AGENT_TRACE: [],
            _K.MODEL_REPLY: {
                "kind": "calls",
                "text": None,
                "calls": [{"id": "c", "name": "", "arguments": {"api_key": _SECRET}}],
            },
        }
        delta = handlers.run_tools(context)
        assert delta[_K.MODEL_REPLY] is None
        assert delta[_K.LOOP_END] == "malformed"


# ---------------------------------------------------------------------------
# native_fc: system_policy type, empty tool name
# ---------------------------------------------------------------------------


class TestSystemPolicyType:
    def test_a_callable_in_the_old_complete_fn_slot_is_refused(self):
        def legacy_complete(*args: Any) -> dict[str, Any]:
            return {}

        with pytest.raises(TypeError, match="system_policy must be a str or None"):
            NativeFunctionCallingReactAgent(
                golden._registry(), AgentConfig(model="m"), legacy_complete
            )

    def test_setting_a_non_str_later_is_refused(self):
        agent = _native()
        with pytest.raises(TypeError):
            agent.system_policy = 42  # type: ignore[assignment]

    def test_str_and_none_are_accepted(self):
        agent = _native()
        agent.system_policy = "Be brief."
        assert agent._system_message().endswith("\n\nBe brief.")
        agent.system_policy = None
        assert agent.system_policy is None


class TestEmptyToolNameIsMalformed:
    def test_nameless_call_ends_the_loop_and_runs_nothing(self, monkeypatch):
        recorder = golden._Recorder(
            [golden._raw_calls(("call_1", "", '{"city": "Paris"}'))]
        )
        for binding in golden._BINDINGS:
            monkeypatch.setattr(binding, recorder)
        result = _native().run("t")
        assert len(recorder.requests) == 1
        assert result.success is False
        assert result.stop_reason == "no_result"
        assert result.trace.tool_calls == []


# ---------------------------------------------------------------------------
# native_fc handler rules whose mutants survived review pass 5
# ---------------------------------------------------------------------------


def _scripted(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> golden._Recorder:
    recorder = golden._Recorder(script)
    for binding in golden._BINDINGS:
        monkeypatch.setattr(binding, recorder)
    return recorder


class TestSurvivingMutants:
    def test_m01_whitespace_answer_after_a_tool_is_no_result(self, monkeypatch):
        _scripted(monkeypatch, [golden._calls(golden._PARIS), golden._final("   ")])
        result = _native().run("t")
        assert result.success is False
        assert result.stop_reason == "no_result"

    def test_m16_answer_is_the_model_text_verbatim(self, monkeypatch):
        _scripted(monkeypatch, [golden._final("  Paris is sunny.\n")])
        result = _native().run("t")
        assert result.answer == "  Paris is sunny.\n"

    def test_m17_caller_context_never_stands_in_for_the_answer(self, monkeypatch):
        _scripted(
            monkeypatch,
            [golden._final("not json"), golden._final("still not json")],
        )
        result = _native(output_schema=golden._WeatherReport).run(
            "t", initial_context={"city": "Forged", "conditions": "forged"}
        )
        assert result.final_context["city"] == "Forged"
        assert result.structured_output is None

    @pytest.mark.parametrize(
        ("script", "max_iterations", "expected"),
        [
            ([golden._final("x")], 5, 1),
            ([golden._calls(golden._PARIS), golden._final("x")], 5, 2),
            ([golden._calls(golden._PARIS), golden._calls(golden._ROME)], 2, 2),
        ],
        ids=["final", "tool_then_final", "exhausted"],
    )
    def test_m18_iterations_used_counts_loop_model_turns(
        self, monkeypatch, script, max_iterations, expected
    ):
        _scripted(monkeypatch, script)
        result = _native(max_iterations=max_iterations).run("t")
        assert result.iterations_used == expected

    def test_m19_a_raising_registry_fails_the_run(self, monkeypatch):
        class Broken(ToolRegistry):
            def execute(self, tool_call: ToolCall, *, gated: bool = False) -> Any:
                raise RuntimeError("registry broke its never-raise contract")

        registry = Broken()
        for tool_def in golden._registry().list_tools():
            registry.register(tool_def)
        recorder = _scripted(
            monkeypatch, [golden._calls(golden._PARIS), golden._final("x")]
        )
        agent = NativeFunctionCallingReactAgent(
            registry, config=AgentConfig(model="gpt-4o-mini")
        )
        with pytest.raises(AgentError, match="never-raise"):
            agent.run("t")
        assert len(recorder.requests) == 1

    def test_m23_repair_turn_never_sees_the_forced_exchange(self, monkeypatch):
        recorder = _scripted(
            monkeypatch,
            [
                golden._calls(golden._PARIS),
                golden._final("Paris is sunny."),
                golden._calls(golden._NOTE),
                golden._final(golden._REPORT),
            ],
        )
        _native(force_final_tool="write_note", output_schema=golden._WeatherReport).run(
            "t"
        )
        repair = recorder.requests[-1]["messages"]
        assert "response_format" in recorder.requests[-1]
        assert not any(m.get("tool_calls") for m in repair[3:])
        assert not any(m.get("tool_call_id") == "call_9" for m in repair)
