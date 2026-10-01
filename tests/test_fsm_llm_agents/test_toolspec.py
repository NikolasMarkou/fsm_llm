"""ToolSpec layer (ported from 675b3cf, plan 944e2692 step 14, D-008).

``ToolAnnotations``, ``ToolDefinition.timeout_s``/``args_model``, the exact
argument schema that ``get_json_schemas`` emits (TOOL-04), the annotation
kwargs of ``@tool``/``register_function``, MCP annotation mapping, the
``RetryingToolRegistry`` rule (retry only idempotent/read-only tools, never a
call executed with ``gated=True``, TOOL-07), the ``gated`` keyword on every
``execute`` and the ``timeout_s`` enforced by ``ToolRegistry.execute``.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from typing import Annotated, Any, Literal, Optional, Union
from unittest.mock import Mock

import pydantic
import pytest

from fsm_llm.agents import ToolAnnotations, tool
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import ToolCall, ToolDefinition
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.tool_registries import CachingToolRegistry, RetryingToolRegistry
from fsm_llm.agents.tools import ToolRegistry

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _params(fn: Any) -> dict[str, Any]:
    """The ``parameters`` object ``get_json_schemas`` emits for ``@tool`` *fn*."""
    registry = ToolRegistry()
    registry.register(fn._tool_definition)
    return registry.get_json_schemas()[0]["function"]["parameters"]


def _registered_params(fn: Any, **kwargs: Any) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register_function(fn, name="t", description="d", **kwargs)
    return registry.get_json_schemas()[0]["function"]["parameters"]


# ---------------------------------------------------------------------------
# TOOL-04: exact argument schemas (RED on the parent: every one was "string")
# ---------------------------------------------------------------------------


@tool
def typed(
    opt: Optional[int],  # noqa: UP045 - the typing spelling is the case under test
    either: Union[int, str],  # noqa: UP007
    choice: Literal["x", "y"],
    pipe: int | None = None,
) -> str:
    """A tool with non-trivial hints."""
    return "ok"


class TestExactArgumentSchema:
    def test_optional_int_is_integer_or_null(self):
        prop = _params(typed)["properties"]["opt"]
        assert prop.get("type") != "string"
        assert prop["anyOf"] == [{"type": "integer"}, {"type": "null"}]

    def test_union_int_str_lists_both(self):
        prop = _params(typed)["properties"]["either"]
        assert prop["anyOf"] == [{"type": "integer"}, {"type": "string"}]

    def test_literal_is_an_enum(self):
        prop = _params(typed)["properties"]["choice"]
        assert prop["enum"] == ["x", "y"]

    def test_pipe_union_with_none_is_integer_or_null(self):
        prop = _params(typed)["properties"]["pipe"]
        assert prop["anyOf"] == [{"type": "integer"}, {"type": "null"}]
        assert prop["default"] is None

    def test_required_lists_only_parameters_without_defaults(self):
        assert _params(typed)["required"] == ["opt", "either", "choice"]

    def test_register_function_emits_the_same_schema(self):
        def fn(n: Optional[int], mode: Literal["a", "b"] = "a") -> str:  # noqa: UP045
            return "ok"

        params = _registered_params(fn)
        assert params["properties"]["n"]["anyOf"] == [
            {"type": "integer"},
            {"type": "null"},
        ]
        assert params["properties"]["mode"]["enum"] == ["a", "b"]
        assert params["required"] == ["n"]

    def test_parameter_schema_stays_coarse_for_the_prompt(self):
        # The prompt description and the dict-style binder read the coarse
        # schema; only the provider schema changed.
        schema = typed._tool_definition.parameter_schema
        assert schema["properties"]["opt"] == {"type": "string"}

    def test_no_pydantic_titles_leak(self):
        params = _params(typed)
        assert "title" not in params
        assert all("title" not in p for p in params["properties"].values())

    def test_a_parameter_named_title_is_kept(self):
        @tool
        def post(title: str, body: str = "") -> str:
            """Post."""
            return title

        params = _params(post)
        assert params["properties"]["title"] == {"type": "string"}
        assert params["required"] == ["title"]

    def test_annotated_description_reaches_the_schema(self):
        @tool
        def search(query: Annotated[str, "what to look for"]) -> str:
            """Search."""
            return query

        assert _params(search)["properties"]["query"] == {
            "type": "string",
            "description": "what to look for",
        }

    def test_list_param_keeps_items(self):
        @tool
        def tag(names: list[str]) -> str:
            """Tag."""
            return ",".join(names)

        assert _params(tag)["properties"]["names"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_simple_tool_shape_matches_the_coarse_schema(self):
        # The harness workspace tools are all ``str`` params: the emitted
        # schema is unchanged for them.
        @tool
        def read_file(path: str) -> str:
            """Read."""
            return path

        assert _params(read_file) == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }


# ---------------------------------------------------------------------------
# args_model: when it exists, what it validates, and that it is not dumped
# ---------------------------------------------------------------------------


class TestArgsModel:
    def test_typed_tool_has_a_model(self):
        model = typed._tool_definition.args_model
        assert model is not None
        assert issubclass(model, pydantic.BaseModel)

    def test_model_validates_and_coerces(self):
        model = typed._tool_definition.args_model
        args = model.model_validate({"opt": "5", "either": "a", "choice": "x"})
        assert args.opt == 5
        assert args.pipe is None
        with pytest.raises(pydantic.ValidationError):
            model.model_validate({"opt": 1, "either": 1, "choice": "z"})

    def test_excluded_from_dumps(self):
        dump = typed._tool_definition.model_dump()
        assert "args_model" not in dump
        assert "execute_fn" not in dump
        assert dump["annotations"] == {
            "read_only": None,
            "destructive": None,
            "idempotent": None,
            "open_world": None,
        }
        assert dump["timeout_s"] is None

    def test_dict_style_tool_has_no_model_and_keeps_working(self):
        seen: list[dict[str, Any]] = []

        def legacy(params: dict) -> str:
            seen.append(params)
            return "ok"

        registry = ToolRegistry()
        registry.register_function(legacy, name="d", description="d")
        definition = registry.get("d")
        assert definition.args_model is None
        assert definition.parameter_schema == {}
        result = registry.execute(ToolCall(tool_name="d", parameters={"a": 1}))
        assert result.success
        assert seen == [{"a": 1}]
        assert registry.get_json_schemas()[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_unannotated_param_has_no_model(self):
        def fn(a: int, b) -> str:  # type: ignore[no-untyped-def]
            return "ok"

        registry = ToolRegistry()
        registry.register_function(fn, name="u", description="d")
        assert registry.get("u").args_model is None
        # The coarse schema is emitted instead.
        params = registry.get_json_schemas()[0]["function"]["parameters"]
        assert params["properties"]["b"] == {"type": "string"}

    def test_var_keyword_tool_has_no_model(self):
        @tool
        def fn(a: int, **extra: Any) -> str:
            """Kw."""
            return "ok"

        assert fn._tool_definition.args_model is None

    def test_explicit_schema_has_no_model(self):
        schema = {"properties": {"q": {"type": "string"}}, "required": ["q"]}

        @tool(parameter_schema=schema)
        def fn(q: Optional[int]) -> str:  # noqa: UP045
            """Explicit."""
            return "ok"

        assert fn._tool_definition.args_model is None
        assert _params(fn)["properties"] == {"q": {"type": "string"}}

    def test_unschemable_type_falls_back(self):
        class Opaque:
            pass

        def fn(thing: Opaque) -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register_function(fn, name="o", description="d")
        assert registry.get("o").args_model is None

    def test_partial_gets_a_model_from_the_wrapped_function(self):
        import functools

        def fn(a: int, b: Optional[int] = None) -> str:  # noqa: UP045
            return "ok"

        registry = ToolRegistry()
        registry.register_function(functools.partial(fn), name="p", description="d")
        params = registry.get_json_schemas()[0]["function"]["parameters"]
        assert params["properties"]["b"]["anyOf"][1] == {"type": "null"}


# ---------------------------------------------------------------------------
# ToolAnnotations, timeout_s and the decorator/registration kwargs
# ---------------------------------------------------------------------------


class TestToolAnnotations:
    def test_defaults_are_unknown(self):
        a = ToolAnnotations()
        assert (a.read_only, a.destructive, a.idempotent, a.open_world) == (
            None,
            None,
            None,
            None,
        )
        assert a.retry_safe is False

    @pytest.mark.parametrize(
        ("kwargs", "safe"),
        [
            ({"idempotent": True}, True),
            ({"read_only": True}, True),
            ({"idempotent": False, "read_only": False}, False),
            ({"destructive": True, "open_world": True}, False),
        ],
    )
    def test_retry_safe(self, kwargs, safe):
        assert ToolAnnotations(**kwargs).retry_safe is safe

    def test_unknown_field_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            ToolAnnotations(read_only=True, side_effect=True)  # type: ignore[call-arg]

    def test_frozen(self):
        with pytest.raises(pydantic.ValidationError):
            ToolAnnotations().idempotent = True  # type: ignore[misc]

    def test_definitions_do_not_share_an_instance(self):
        a = ToolDefinition(name="a", description="d")
        b = ToolDefinition(name="b", description="d")
        assert a.annotations == b.annotations == ToolAnnotations()

    def test_decorator_accepts_annotations_and_timeout(self):
        @tool(annotations=ToolAnnotations(read_only=True), timeout_s=2.5)
        def look(q: str) -> str:
            """Look."""
            return q

        definition = look._tool_definition
        assert definition.annotations.read_only is True
        assert definition.timeout_s == 2.5

    def test_register_function_accepts_annotations_and_timeout(self):
        registry = ToolRegistry()
        registry.register_function(
            lambda q: q,
            name="look",
            description="d",
            annotations=ToolAnnotations(idempotent=True),
            timeout_s=1.0,
        )
        definition = registry.get("look")
        assert definition.annotations.idempotent is True
        assert definition.timeout_s == 1.0

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_timeout_must_be_positive(self, bad):
        with pytest.raises(pydantic.ValidationError):
            ToolDefinition(name="a", description="d", timeout_s=bad)


# ---------------------------------------------------------------------------
# MCP annotations
# ---------------------------------------------------------------------------


def _mcp_tool(annotations: Any) -> Any:
    mcp_tool = Mock()
    mcp_tool.name = "remote"
    mcp_tool.description = "Remote tool"
    mcp_tool.inputSchema = None
    mcp_tool.annotations = annotations
    return mcp_tool


def _convert(mcp_tool: Any) -> ToolDefinition:
    from fsm_llm.agents.mcp import MCPToolProvider

    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._server_params = "mock_params"
    provider._server_url = None
    return provider._convert_mcp_tool(mcp_tool)


class TestMCPAnnotations:
    def test_camel_case_hints_mapped(self):
        hints = Mock(spec=["readOnlyHint", "destructiveHint", "idempotentHint"])
        hints.readOnlyHint = True
        hints.destructiveHint = False
        hints.idempotentHint = True
        definition = _convert(_mcp_tool(hints))
        assert definition.annotations == ToolAnnotations(
            read_only=True, destructive=False, idempotent=True
        )

    def test_snake_case_dict_mapped(self):
        definition = _convert(
            _mcp_tool({"open_world_hint": True, "idempotent_hint": False})
        )
        assert definition.annotations == ToolAnnotations(
            open_world=True, idempotent=False
        )

    def test_missing_annotations_are_unknown(self):
        assert _convert(_mcp_tool(None)).annotations == ToolAnnotations()

    def test_non_bool_hints_ignored(self):
        # A bare Mock answers every attribute with a (truthy) Mock.
        assert _convert(_mcp_tool(Mock())).annotations == ToolAnnotations()


# ---------------------------------------------------------------------------
# TOOL-07: RetryingToolRegistry retries only retry-safe, never gated calls
# ---------------------------------------------------------------------------


def _failing_registry(
    runs: list[str],
    *,
    annotations: ToolAnnotations | None = None,
    requires_approval: bool = False,
    max_retries: int = 3,
) -> RetryingToolRegistry:
    def pay(amount: str) -> str:
        runs.append(amount)
        raise RuntimeError("gateway down")

    registry = RetryingToolRegistry(max_retries=max_retries)
    registry.register_function(
        pay,
        name="pay",
        description="d",
        requires_approval=requires_approval,
        annotations=annotations,
    )
    return registry


_CALL = ToolCall(tool_name="pay", parameters={"amount": "9"})


class TestRetryRule:
    def test_unannotated_tool_is_not_retried(self):
        runs: list[str] = []
        result = _failing_registry(runs).execute(_CALL)
        assert result.success is False
        assert runs == ["9"]

    @pytest.mark.parametrize(
        "annotations",
        [ToolAnnotations(idempotent=True), ToolAnnotations(read_only=True)],
        ids=["idempotent", "read_only"],
    )
    def test_retry_safe_tool_is_retried(self, annotations):
        runs: list[str] = []
        _failing_registry(runs, annotations=annotations).execute(_CALL)
        assert runs == ["9"] * 4

    def test_flagged_idempotent_tool_is_not_retried(self):
        runs: list[str] = []
        _failing_registry(
            runs, annotations=ToolAnnotations(idempotent=True), requires_approval=True
        ).execute(_CALL)
        assert runs == ["9"]

    def test_gated_call_is_not_retried(self):
        runs: list[str] = []
        registry = _failing_registry(runs, annotations=ToolAnnotations(idempotent=True))
        registry.execute(_CALL, gated=True)
        assert runs == ["9"]
        registry.execute(_CALL)  # an ungated call is retried again
        assert runs == ["9"] * 5

    def test_unknown_tool_is_not_retried(self):
        registry = RetryingToolRegistry(max_retries=3)
        result = registry.execute(ToolCall(tool_name="ghost"))
        assert result.success is False

    def test_success_is_not_rerun(self):
        runs: list[str] = []

        def pay(amount: str) -> str:
            runs.append(amount)
            return "ok"

        registry = RetryingToolRegistry(max_retries=3)
        registry.register_function(
            pay,
            name="pay",
            description="d",
            annotations=ToolAnnotations(idempotent=True),
        )
        assert registry.execute(_CALL).success
        assert runs == ["9"]


def _approved_context() -> dict[str, Any]:
    """The executor's context on ``act`` entry after an approval of ``pay``."""
    return {
        ContextKeys.TASK: "pay",
        ContextKeys.TOOL_NAME: "pay",
        ContextKeys.TOOL_INPUT: {"amount": "9"},
        ContextKeys.APPROVAL_GRANTED: True,
        ContextKeys.APPROVAL_REQUIRED: False,
        ContextKeys.DRIVER_APPROVAL: {
            "tool_name": "pay",
            "parameters": {"amount": "9"},
        },
        ContextKeys.OBSERVATIONS: [],
        ContextKeys.AGENT_TRACE: [],
    }


def _policy_gates_everything(call: ToolCall, context: dict[str, Any]) -> bool:
    return True


class TestPolicyGatedToolIsNotRetried:
    """RED on the parent: a tool only a HITL *policy* gates (no
    ``requires_approval`` flag) was retried after its one approval."""

    def test_approved_policy_gated_call_runs_once(self):
        runs: list[str] = []
        registry = _failing_registry(runs)
        handlers = AgentHandlers(registry, requires_approval=_policy_gates_everything)
        delta = handlers.execute_tool(_approved_context())
        assert runs == ["9"], "the approved call was re-run by the registry"
        assert delta[ContextKeys.TOOL_STATUS] == "failed"

    def test_approved_idempotent_call_runs_once(self):
        runs: list[str] = []
        registry = _failing_registry(runs, annotations=ToolAnnotations(idempotent=True))
        handlers = AgentHandlers(registry, requires_approval=_policy_gates_everything)
        handlers.execute_tool(_approved_context())
        assert runs == ["9"]

    def test_ungated_idempotent_call_is_retried(self):
        runs: list[str] = []
        registry = _failing_registry(runs, annotations=ToolAnnotations(idempotent=True))
        context = _approved_context()
        for key in (
            ContextKeys.DRIVER_APPROVAL,
            ContextKeys.APPROVAL_GRANTED,
            ContextKeys.APPROVAL_REQUIRED,
        ):
            context.pop(key)
        AgentHandlers(registry).execute_tool(context)
        assert runs == ["9"] * 4


# ---------------------------------------------------------------------------
# `gated` is an explicit keyword on every execute (no context variable)
# ---------------------------------------------------------------------------


def _execute_implementations() -> list[type[ToolRegistry]]:
    from fsm_llm.agents.reasoning_react import _ReasonToolRegistry

    return [
        ToolRegistry,
        CachingToolRegistry,
        RetryingToolRegistry,
        _ReasonToolRegistry,
    ]


class TestGatedKeyword:
    @pytest.mark.parametrize(
        "cls", _execute_implementations(), ids=lambda c: c.__name__
    )
    def test_keyword_only_default_false(self, cls):
        param = inspect.signature(cls.execute).parameters["gated"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is False

    def test_no_context_variable_side_channel(self):
        import fsm_llm.agents.tool_registries as registries

        assert not hasattr(registries, "gated_execution")
        assert not hasattr(registries, "in_gated_execution")

    def test_caching_registry_passes_gated_on(self):
        class CachingRetrying(CachingToolRegistry, RetryingToolRegistry):
            pass

        runs: list[str] = []

        def pay(amount: str) -> str:
            runs.append(amount)
            raise RuntimeError("gateway down")

        registry = CachingRetrying()
        registry.register_function(
            pay,
            name="pay",
            description="d",
            annotations=ToolAnnotations(idempotent=True),
        )
        registry.execute(_CALL, gated=True)
        assert runs == ["9"]
        registry.execute(_CALL)
        assert runs == ["9"] * 4  # max_retries=2 by default: 3 more runs

    def test_reason_registry_passes_gated_to_the_callers_registry(self):
        from fsm_llm.agents.reasoning_react import _ReasonToolRegistry

        runs: list[str] = []
        base = _failing_registry(runs, annotations=ToolAnnotations(idempotent=True))
        view = _ReasonToolRegistry(base)
        view.execute(_CALL, gated=True)
        assert runs == ["9"]
        view.execute(_CALL)
        assert runs == ["9"] * 5

    def test_executor_passes_gated_only_for_a_granted_call(self):
        seen: list[bool] = []

        class Spy(ToolRegistry):
            def execute(self, tool_call: ToolCall, *, gated: bool = False):
                seen.append(gated)
                return super().execute(tool_call, gated=gated)

        def pay(amount: str) -> str:
            return "paid"

        registry = Spy()
        registry.register_function(pay, name="pay", description="d")
        AgentHandlers(
            registry, requires_approval=_policy_gates_everything
        ).execute_tool(_approved_context())
        context = _approved_context()
        for key in (
            ContextKeys.DRIVER_APPROVAL,
            ContextKeys.APPROVAL_GRANTED,
            ContextKeys.APPROVAL_REQUIRED,
        ):
            context.pop(key)
        AgentHandlers(registry).execute_tool(context)
        assert seen == [True, False]


# ---------------------------------------------------------------------------
# timeout_s is enforced by ToolRegistry.execute
# ---------------------------------------------------------------------------


def _capture_warnings() -> tuple[Any, list[str], int]:
    from fsm_llm.logging import logger

    logger.enable("fsm_llm")
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    return logger, captured, sink_id


def _tool_thread(name: str) -> threading.Thread:
    threads = [t for t in threading.enumerate() if t.name == f"fsm-llm-tool-{name}"]
    assert len(threads) == 1, threads
    return threads[0]


class TestTimeoutEnforced:
    def test_slow_tool_fails_fast(self):
        def slow() -> str:
            time.sleep(2.0)
            return "done"

        registry = ToolRegistry()
        registry.register_function(slow, name="slow", description="d", timeout_s=0.2)
        start = time.monotonic()
        result = registry.execute(ToolCall(tool_name="slow"))
        elapsed = time.monotonic() - start
        assert result.success is False
        assert "timed out after 0.2 s" in (result.error or "")
        assert elapsed < 1.0

    def test_fast_tool_returns_its_value(self):
        def fast(q: str) -> str:
            return q.upper()

        registry = ToolRegistry()
        registry.register_function(fast, name="fast", description="d", timeout_s=5.0)
        result = registry.execute(ToolCall(tool_name="fast", parameters={"q": "a"}))
        assert result.success is True
        assert result.result == "A"

    def test_tool_raising_before_the_timeout_is_a_failed_call(self):
        def boom() -> str:
            raise ValueError("bad input")

        registry = ToolRegistry()
        registry.register_function(boom, name="boom", description="d", timeout_s=5.0)
        start = time.monotonic()
        result = registry.execute(ToolCall(tool_name="boom"))
        assert time.monotonic() - start < 1.0
        assert result.success is False
        assert result.error == "bad input"

    def test_async_tool_is_timed_out(self):
        async def slow_async() -> str:
            await asyncio.sleep(2.0)
            return "done"

        registry = ToolRegistry()
        registry.register_function(
            slow_async, name="slow_async", description="d", timeout_s=0.2
        )
        start = time.monotonic()
        result = registry.execute(ToolCall(tool_name="slow_async"))
        assert time.monotonic() - start < 1.0
        assert result.success is False
        assert "timed out" in (result.error or "")

    def test_late_result_is_discarded_with_a_warning(self):
        release = threading.Event()
        ran: list[str] = []

        def late() -> str:
            release.wait(5.0)
            ran.append("side effect")
            return "late value"

        registry = ToolRegistry()
        registry.register_function(late, name="late", description="d", timeout_s=0.1)
        logger, captured, sink_id = _capture_warnings()
        try:
            result = registry.execute(ToolCall(tool_name="late"))
            worker = _tool_thread("late")
            release.set()
            worker.join(5.0)
            assert not worker.is_alive()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")
        assert result.success is False
        # The tool kept running: its side effect happened after the timeout.
        assert ran == ["side effect"]
        late_lines = [m for m in captured if "after its 0.1 s timeout" in m]
        assert len(late_lines) == 1
        assert "'late'" in late_lines[0]
        assert "late value" not in "".join(captured)

    def test_registry_lock_is_free_while_a_tool_runs(self):
        release = threading.Event()

        def hold() -> str:
            release.wait(5.0)
            return "ok"

        registry = ToolRegistry()
        registry.register_function(hold, name="hold", description="d", timeout_s=5.0)
        results: list[Any] = []
        caller = threading.Thread(
            target=lambda: results.append(registry.execute(ToolCall(tool_name="hold")))
        )
        caller.start()
        try:
            _wait_for_thread("fsm-llm-tool-hold")
            acquired = registry._tools_lock.acquire(timeout=1.0)
            assert acquired, "_tools_lock held across the tool call"
            registry._tools_lock.release()
            registry.register_function(lambda: "x", name="other", description="d")
        finally:
            release.set()
            caller.join(5.0)
        assert results[0].success is True

    def test_untimed_tool_runs_in_the_calling_thread(self):
        threads: list[threading.Thread] = []

        def where() -> str:
            threads.append(threading.current_thread())
            return "ok"

        registry = ToolRegistry()
        registry.register_function(where, name="where", description="d")
        assert registry.execute(ToolCall(tool_name="where")).success
        assert threads == [threading.current_thread()]


def _wait_for_thread(name: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(t.name == name for t in threading.enumerate()):
            return
        time.sleep(0.01)
    raise AssertionError(f"thread {name} never started")
