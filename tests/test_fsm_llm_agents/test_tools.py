from __future__ import annotations

"""Tests for fsm_llm.agents.tools module."""

import sys
import threading
import time
import typing
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from fsm_llm.agents.definitions import ToolCall, ToolDefinition
from fsm_llm.agents.exceptions import ToolNotFoundError
from fsm_llm.agents.tools import (
    ToolRegistry,
    normalize_tool_input,
    register_agent,
    tool,
)


def _add(params):
    """Add two numbers."""
    return params["a"] + params["b"]


def _greet():
    """Greet the user."""
    return "Hello!"


def _failing_tool(params):
    """Tool that always fails."""
    raise RuntimeError("Intentional failure")


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        registry = ToolRegistry()
        tool_def = ToolDefinition(
            name="add",
            description="Add numbers",
            execute_fn=_add,
        )
        registry.register(tool_def)
        assert "add" in registry
        assert len(registry) == 1

    def test_register_function(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add two numbers")
        assert "add" in registry

    def test_register_function_chaining(self):
        registry = ToolRegistry()
        result = registry.register_function(_add, name="add", description="Add")
        assert result is registry

    def test_register_duplicate_name_warns_and_last_wins(self):
        """FB-07: an overwrite is surfaced as a warning; behaviour stays last-wins."""
        from fsm_llm.logging import logger

        registry = ToolRegistry()
        first = ToolDefinition(name="add", description="first", execute_fn=_add)
        second = ToolDefinition(name="add", description="second", execute_fn=_add)
        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            registry.register(first)
            assert captured == []
            registry.register(second)
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert len(registry) == 1
        assert registry.get("add").description == "second"
        assert len(captured) == 1
        assert "add" in captured[0]
        assert "already registered" in captured[0]

    def test_register_without_execute_fn_raises(self):
        registry = ToolRegistry()
        tool_def = ToolDefinition(name="bad", description="No fn")
        with pytest.raises(ValueError, match="execute_fn"):
            registry.register(tool_def)

    def test_get_existing_tool(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        tool = registry.get("add")
        assert tool.name == "add"

    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        registry.register_function(_greet, name="greet", description="Greet")
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"add", "greet"}

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        assert registry.tool_names == ["add"]

    def test_contains(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        assert "add" in registry
        assert "missing" not in registry

    def test_execute_success(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")

        call = ToolCall(tool_name="add", parameters={"a": 2, "b": 3})
        result = registry.execute(call)

        assert result.success is True
        assert result.result == 5
        assert result.execution_time_ms > 0

    def test_execute_no_params(self):
        registry = ToolRegistry()
        registry.register_function(_greet, name="greet", description="Greet")

        call = ToolCall(tool_name="greet", parameters={})
        result = registry.execute(call)

        assert result.success is True
        assert result.result == "Hello!"

    def test_execute_nonexistent_tool(self):
        registry = ToolRegistry()
        call = ToolCall(tool_name="missing", parameters={})
        result = registry.execute(call)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_failing_tool(self):
        registry = ToolRegistry()
        registry.register_function(_failing_tool, name="fail", description="Fails")

        call = ToolCall(tool_name="fail", parameters={})
        result = registry.execute(call)

        assert result.success is False
        assert "Intentional failure" in result.error

    def test_to_prompt_description(self):
        registry = ToolRegistry()
        registry.register_function(
            _add,
            name="add",
            description="Add two numbers",
            parameter_schema={
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                }
            },
        )
        desc = registry.to_prompt_description()
        assert "add" in desc
        assert "Add two numbers" in desc
        assert "number" in desc

    def test_to_prompt_description_empty(self):
        registry = ToolRegistry()
        assert "No tools" in registry.to_prompt_description()

    def test_to_classification_schema(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        registry.register_function(_greet, name="greet", description="Greet")

        schema = registry.to_classification_schema()
        assert schema["fallback_intent"] == "none"
        intent_names = {i["name"] for i in schema["intents"]}
        assert "add" in intent_names
        assert "greet" in intent_names
        assert "none" in intent_names


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_basic_decorator(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert hasattr(add, "_tool_definition")
        defn = add._tool_definition
        assert defn.name == "add"
        assert defn.description == "Add two numbers"
        assert defn.execute_fn is add

    def test_decorator_with_custom_name(self):
        @tool(name="web_search", description="Search the internet")
        def search(query: str) -> str:
            return f"Results for: {query}"

        assert search._tool_definition.name == "web_search"

    def test_decorator_with_approval(self):
        @tool(description="Delete a file", requires_approval=True)
        def delete_file(path: str) -> bool:
            return True

        assert delete_file._tool_definition.requires_approval is True

    def test_register_decorated_function(self):
        @tool(description="Multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        registry = ToolRegistry()
        registry.register(multiply._tool_definition)
        assert "multiply" in registry


class TestToolDecoratorBare:
    """Tests for bare @tool decorator with auto-schema inference."""

    def test_bare_decorator(self):
        @tool
        def search(query: str) -> str:
            """Search the web for information."""
            return f"Results for: {query}"

        assert hasattr(search, "_tool_definition")
        defn = search._tool_definition
        assert defn.name == "search"
        assert defn.description == "Search the web for information."
        assert defn.execute_fn is search

    def test_bare_decorator_schema_inferred(self):
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search the web."""
            return "results"

        schema = search._tool_definition.parameter_schema
        assert "properties" in schema
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_bare_decorator_all_types(self):
        @tool
        def multi(
            name: str,
            count: int,
            ratio: float,
            active: bool,
            items: list,
            metadata: dict,
        ) -> str:
            """Test all types."""
            return "ok"

        schema = multi._tool_definition.parameter_schema
        props = schema["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["active"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
        assert props["metadata"]["type"] == "object"

    def test_bare_decorator_no_type_hints_single_param(self):
        @tool
        def bare_fn(x):
            """No types."""
            return x

        # Single param with no type hint → treated as legacy dict pattern
        schema = bare_fn._tool_definition.parameter_schema
        assert schema == {}

    def test_bare_decorator_no_type_hints_multi_param(self):
        @tool
        def bare_fn(x, y):
            """No types."""
            return f"{x}{y}"

        # Multi-param with no type hints → infers as string
        schema = bare_fn._tool_definition.parameter_schema
        assert schema["properties"]["x"]["type"] == "string"
        assert schema["properties"]["y"]["type"] == "string"

    def test_bare_decorator_legacy_dict_param(self):
        @tool
        def legacy(params: dict) -> str:
            """Legacy pattern."""
            return str(params)

        # Should return empty schema for legacy dict pattern
        schema = legacy._tool_definition.parameter_schema
        assert schema == {}

    def test_bare_decorator_docstring_first_line(self):
        @tool
        def my_tool(x: str) -> str:
            """First line description.

            This is extra detail that should not be in description.
            """
            return x

        assert my_tool._tool_definition.description == "First line description."

    def test_bare_decorator_no_docstring(self):
        @tool
        def nodoc(x: str) -> str:
            return x

        assert nodoc._tool_definition.description == "Tool: nodoc"

    def test_explicit_schema_overrides_inference(self):
        @tool(parameter_schema={"properties": {"q": {"type": "string"}}})
        def search(query: str) -> str:
            """Search."""
            return "results"

        schema = search._tool_definition.parameter_schema
        assert "q" in schema["properties"]
        assert "query" not in schema.get("properties", {})

    def test_bare_decorator_callable(self):
        """Verify bare-decorated function is still callable."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5

    def test_bare_decorator_with_registry_execute(self):
        """Bare @tool functions execute correctly via ToolRegistry."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        registry = ToolRegistry()
        registry.register(greet._tool_definition)
        result = registry.execute(
            ToolCall(tool_name="greet", parameters={"name": "Alice"})
        )
        assert result.success is True
        assert result.result == "Hello, Alice!"

    def test_annotated_descriptions(self):
        """Annotated[T, 'desc'] provides per-parameter descriptions."""

        @tool
        def search(
            query: Annotated[str, "The search query"],
            limit: Annotated[int, "Max results"] = 10,
        ) -> str:
            """Search the web."""
            return "results"

        schema = search._tool_definition.parameter_schema
        assert schema["properties"]["query"]["description"] == "The search query"
        assert schema["properties"]["limit"]["description"] == "Max results"

    def test_zero_param_function(self):
        @tool
        def get_time() -> str:
            """Get current time."""
            return "12:00"

        schema = get_time._tool_definition.parameter_schema
        assert schema == {}


class TestRegisterAgent:
    """Tests for ToolRegistry.register_agent()."""

    def test_register_agent_basic(self):
        class FakeAgent:
            def run(self, task):
                class FakeResult:
                    answer = f"Result for: {task}"

                return FakeResult()

        registry = ToolRegistry()
        registry.register_agent(
            FakeAgent(), name="helper", description="A helper agent"
        )
        assert "helper" in registry

    def test_register_agent_execute(self):
        class FakeAgent:
            def run(self, task):
                class FakeResult:
                    answer = f"Answer: {task}"

                return FakeResult()

        registry = ToolRegistry()
        registry.register_agent(
            FakeAgent(), name="helper", description="A helper agent"
        )

        result = registry.execute(
            ToolCall(tool_name="helper", parameters={"task": "What is 2+2?"})
        )
        assert result.success is True
        assert "Answer: What is 2+2?" in result.result

    def test_register_agent_no_run_method(self):
        class NotAnAgent:
            pass

        registry = ToolRegistry()
        with pytest.raises(ValueError, match="run"):
            registry.register_agent(NotAnAgent(), name="bad", description="Bad")

    def test_register_agent_chaining(self):
        class FakeAgent:
            def run(self, task):
                class R:
                    answer = "ok"

                return R()

        registry = ToolRegistry()
        result = registry.register_agent(
            FakeAgent(), name="a", description="A"
        ).register_agent(FakeAgent(), name="b", description="B")
        assert result is registry
        assert len(registry) == 2

    @staticmethod
    def _agent_returning(**fields):
        class FakeAgent:
            def run(self, task):
                return SimpleNamespace(**fields)

        return FakeAgent()

    def test_failed_sub_agent_is_a_failed_tool_result(self):
        """TOOL-14: a sub-agent reporting success=False must not look successful."""
        registry = ToolRegistry()
        registry.register_agent(
            self._agent_returning(answer="could not finish", success=False),
            name="helper",
            description="A helper agent",
        )

        result = registry.execute(
            ToolCall(tool_name="helper", parameters={"task": "do it"})
        )

        assert result.success is False
        assert "could not finish" in result.error
        assert "helper" in result.error

    def test_successful_sub_agent_returns_its_answer(self):
        registry = ToolRegistry()
        registry.register_agent(
            self._agent_returning(answer="all done", success=True),
            name="helper",
            description="A helper agent",
        )

        result = registry.execute(
            ToolCall(tool_name="helper", parameters={"task": "do it"})
        )

        assert result.success is True
        assert result.result == "all done"

    def test_module_register_agent_propagates_failure(self):
        registry = ToolRegistry()
        register_agent(
            registry,
            self._agent_returning(answer="nope", success=False),
            name="helper",
            description="A helper agent",
        )

        result = registry.execute(
            ToolCall(tool_name="helper", parameters={"task": "x"})
        )

        assert result.success is False


@pytest.fixture
def fine_grained_gil():
    """Force the interpreter to switch threads far more often than the default.

    CPython's default switch interval is 5ms, which is LONGER than a single
    iteration pass over a few dozen tools. Without this, a reader can complete
    its whole scan inside one GIL slice, the pre-fix race never lands, and the
    concurrency tests below pass against un-fixed source — a false green under
    SC-19 (measured: 5/5 pre-fix passes for the semantic registry at the default
    interval, 0/5 with this fixture). It widens the interleaving window; it does
    not weaken any assertion.
    """
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(original)


def _noop(params):
    """No-op tool."""
    return "ok"


def _make_tool(index: int) -> ToolDefinition:
    """Build a distinct, schema-carrying ToolDefinition for concurrency tests."""
    return ToolDefinition(
        name=f"tool_{index}",
        description=f"Concurrency probe tool {index}",
        parameter_schema={
            "properties": {"q": {"type": "string", "description": "query"}},
            "required": ["q"],
        },
        execute_fn=_noop,
    )


class TestToolRegistryConcurrency:
    """Registration must not race iteration (F-05, SC-9).

    Pre-fix these tests fail with
    ``RuntimeError: dictionary changed size during iteration`` raised out of
    ``to_prompt_description``/``get_json_schemas``/``to_classification_schema``,
    measured 20/20 trials. Both clauses matter: a lock that silently dropped
    registrations would satisfy "no error" while losing tools, so every test
    below also asserts the final read contains every tool that was registered.
    """

    TRIALS = 20
    TOOLS_PER_TRIAL = 40

    def _run_writer_reader(self, registry, reader_fns, n_tools):
        """Drive one writer thread against reader threads; return raised errors."""
        errors: list[BaseException] = []
        start = threading.Event()
        done = threading.Event()

        def writer():
            start.wait()
            try:
                for i in range(n_tools):
                    registry.register(_make_tool(i))
            except BaseException as exc:  # recorded for the assertion, not swallowed
                errors.append(exc)
            finally:
                done.set()

        def reader(fn):
            def _loop():
                start.wait()
                try:
                    while not done.is_set():
                        fn()
                    fn()
                except BaseException as exc:  # recorded for the assertion
                    errors.append(exc)

            return _loop

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader(fn)) for fn in reader_fns]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), (
            "writer/reader thread did not finish within 30s — likely a deadlock "
            "introduced by the registry lock"
        )
        return errors

    def test_register_never_races_iteration(self, fine_grained_gil):
        for _ in range(self.TRIALS):
            registry = ToolRegistry()
            errors = self._run_writer_reader(
                registry,
                [
                    registry.to_prompt_description,
                    registry.get_json_schemas,
                    registry.to_classification_schema,
                    registry.list_tools,
                ],
                self.TOOLS_PER_TRIAL,
            )
            assert errors == [], f"concurrent access raised: {errors!r}"

            # Second clause of SC-9: nothing was silently dropped.
            names = {t.name for t in registry.list_tools()}
            assert names == {f"tool_{i}" for i in range(self.TOOLS_PER_TRIAL)}
            assert len(registry.get_json_schemas()) == self.TOOLS_PER_TRIAL

    def test_execute_during_concurrent_registration(self, fine_grained_gil):
        """`execute` looks the tool up under the lock, so no KeyError/TOCTOU."""
        registry = ToolRegistry()
        registry.register(_make_tool(0))

        def _execute_known():
            result = registry.execute(
                ToolCall(tool_name="tool_0", parameters={"q": "x"})
            )
            assert result.success

        errors = self._run_writer_reader(
            registry, [_execute_known], self.TOOLS_PER_TRIAL
        )
        assert errors == [], f"concurrent execute raised: {errors!r}"

    def test_every_public_method_is_callable_from_a_worker_thread(self):
        """Watchdog: the non-reentrant lock must not self-deadlock.

        Each public method is driven from a worker thread with a hard join
        timeout. A nested acquisition (e.g. an iterating method re-entering a
        locked accessor) would hang here rather than fail a value assertion —
        which is precisely the failure the `_locked()`/snapshot split prevents
        and an `RLock` would hide.
        """
        registry = ToolRegistry()
        registry.register(_make_tool(0))

        class _Agent:
            def run(self, task):
                class R:
                    answer = "ok"

                return R()

        class _PythonTool:
            def to_tool_definition(self):
                return _make_tool(99)

        calls = [
            lambda: registry.register(_make_tool(1)),
            lambda: registry.register_function(_noop, name="fn", description="d"),
            lambda: registry.register_agent(_Agent(), name="ag", description="d"),
            lambda: registry.register_python_tool(_PythonTool()),
            lambda: registry.get("tool_0"),
            registry.list_tools,
            lambda: registry.tool_names,
            lambda: len(registry),
            lambda: "tool_0" in registry,
            registry.to_prompt_description,
            registry.get_json_schemas,
            registry.to_classification_schema,
            lambda: registry.execute(
                ToolCall(tool_name="tool_0", parameters={"q": "x"})
            ),
        ]

        for call in calls:
            errors: list[BaseException] = []

            def _target(fn=call, errors=errors):
                try:
                    fn()
                except BaseException as exc:  # recorded for the assertion
                    errors.append(exc)

            worker = threading.Thread(target=_target)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), f"deadlock in {call!r}"
            assert errors == [], f"{call!r} raised {errors!r}"


# 256 dims, not 3. `cosine_similarity` is pure Python, so the vector width sets
# how long a reader spends *inside* the `_embeddings` iteration. A 3-dim vector
# makes each scan so short that the pre-fix race almost never lands and the test
# passes against un-fixed source — a false green under SC-19. 256 dims keeps the
# iteration open across many bytecode switches and makes the pre-fix
# `RuntimeError` reliable.
_EMBEDDING_DIMS = 256


def _stub_embedding(text):
    """Deterministic offline embedding backend, injected as ``embed_fn``.

    Every embedding path (`register`, `retrieve`, the batched
    `rebuild_embeddings`) goes through it, so no test here reaches a provider.
    """
    seed = sum(map(ord, text))
    return [float((seed + i) % 13) + 1.0 for i in range(_EMBEDDING_DIMS)]


class TestSemanticToolRegistryConcurrency:
    """`SemanticToolRegistry._embeddings` must not race either (F-05, SC-9)."""

    TRIALS = 20
    TOOLS_PER_TRIAL = 40

    def _registry(self, monkeypatch):
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(top_k=5, embed_fn=_stub_embedding)
        # Guard the stub itself. `_embed_tool` swallows every exception from
        # the backend, so a stub that fails leaves `_embeddings` permanently
        # empty — `retrieve` then short-circuits to the all-tools fallback and
        # every concurrency assertion below becomes vacuous while still going
        # red for the wrong reason.
        assert len(registry._get_embedding("probe")) == _EMBEDDING_DIMS
        return registry

    def test_register_never_races_retrieve(self, monkeypatch, fine_grained_gil):
        for _ in range(self.TRIALS):
            registry = self._registry(monkeypatch)
            # Cross FALLBACK_THRESHOLD up front so `retrieve` actually reaches the
            # `_embeddings` scan instead of short-circuiting to the full list.
            for i in range(registry.FALLBACK_THRESHOLD):
                registry.register(_make_tool(i))

            errors: list[BaseException] = []
            start = threading.Event()
            done = threading.Event()

            def writer(registry=registry, start=start, done=done, errors=errors):
                start.wait()
                try:
                    for i in range(registry.FALLBACK_THRESHOLD, self.TOOLS_PER_TRIAL):
                        registry.register(_make_tool(i))
                        # Yield so registrations interleave with the reader's scan
                        # instead of the writer finishing inside one GIL slice.
                        time.sleep(0)
                except BaseException as exc:  # recorded for the assertion
                    errors.append(exc)
                finally:
                    done.set()

            def reader(registry=registry, start=start, done=done, errors=errors):
                start.wait()
                try:
                    while not done.is_set():
                        registry.retrieve("find something")
                        registry.to_prompt_description(query="find something")
                        registry.rebuild_embeddings()
                except BaseException as exc:  # recorded for the assertion
                    errors.append(exc)

            threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
            for t in threads:
                t.start()
            start.set()
            for t in threads:
                t.join(timeout=30)
            assert not any(t.is_alive() for t in threads), (
                "deadlock in semantic registry"
            )
            assert errors == [], f"concurrent access raised: {errors!r}"

            names = {t.name for t in registry.list_tools()}
            assert names == {f"tool_{i}" for i in range(self.TOOLS_PER_TRIAL)}
            assert registry.rebuild_embeddings() == self.TOOLS_PER_TRIAL
            assert registry.embedded_tool_count == self.TOOLS_PER_TRIAL

    def test_semantic_public_methods_callable_from_a_worker_thread(self, monkeypatch):
        """Watchdog for the second, separately-locked dict."""
        registry = self._registry(monkeypatch)
        for i in range(registry.FALLBACK_THRESHOLD):
            registry.register(_make_tool(i))

        calls = [
            lambda: registry.register(_make_tool(100)),
            lambda: registry.retrieve("q"),
            lambda: registry.to_prompt_description(query="q"),
            registry.to_prompt_description,
            registry.rebuild_embeddings,
            lambda: registry.embedded_tool_count,
            lambda: registry.embedding_model,
        ]
        for call in calls:
            errors: list[BaseException] = []

            def _target(fn=call, errors=errors):
                try:
                    fn()
                except BaseException as exc:  # recorded for the assertion
                    errors.append(exc)

            worker = threading.Thread(target=_target)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), f"deadlock in {call!r}"
            assert errors == [], f"{call!r} raised {errors!r}"


class _EmbeddingProvider:
    """A fake provider for both embedding bindings (core's and litellm's).

    Answers one ``_stub_embedding`` vector per input text in a real
    ``EmbeddingResponse``, or raises when ``fail``. ``requests`` holds the
    keyword arguments of every request.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[dict[str, Any]] = []

    @property
    def inputs(self) -> list[list[str]]:
        return [list(r["input"]) for r in self.requests]

    def __call__(self, **params: Any) -> Any:
        from litellm.types.utils import EmbeddingResponse

        self.requests.append(params)
        if self.fail:
            raise ConnectionError("embedding provider down")
        return EmbeddingResponse(
            model=params["model"],
            data=[
                {"object": "embedding", "index": i, "embedding": _stub_embedding(t)}
                for i, t in enumerate(params["input"])
            ],
        )


def _patch_embedding(monkeypatch, provider: _EmbeddingProvider) -> None:
    """Route core's binding and litellm's module function to ``provider``."""
    import litellm

    import fsm_llm.llm

    monkeypatch.setattr(fsm_llm.llm, "embedding", provider)
    monkeypatch.setattr(litellm, "embedding", provider)


class TestSemanticToolRegistryEmbedding:
    """Plan 944e2692 step 8 (D-005): the registry embeds through core's
    ``LiteLLMEmbedder`` or an injected ``embed_fn``; a rebuild is one batch."""

    def test_embed_fn_seam_is_accepted_and_used(self):
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        seen: list[str] = []

        def embed_fn(text: str) -> list[float]:
            seen.append(text)
            return _stub_embedding(text)

        registry = SemanticToolRegistry(embed_fn=embed_fn)
        registry.register(_make_tool(0))

        assert seen == ["tool_0: Concurrency probe tool 0"]
        assert registry.embedded_tool_count == 1

    def test_rebuild_embeddings_is_one_provider_request(self, monkeypatch):
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        provider = _EmbeddingProvider()
        _patch_embedding(monkeypatch, provider)
        registry = SemanticToolRegistry(
            embedding_model="ollama/qwen3-embedding:0.6b", auto_embed=False
        )
        for i in range(3):
            registry.register(_make_tool(i))

        assert registry.rebuild_embeddings() == 3
        assert provider.inputs == [
            [f"tool_{i}: Concurrency probe tool {i}" for i in range(3)]
        ]
        assert registry.embedded_tool_count == 3

    def test_default_backend_sends_the_registry_model(self, monkeypatch):
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        provider = _EmbeddingProvider()
        _patch_embedding(monkeypatch, provider)
        registry = SemanticToolRegistry(embedding_model="text-embedding-3-small")
        registry.register(_make_tool(0))

        assert [r["model"] for r in provider.requests] == ["text-embedding-3-small"]
        assert provider.inputs == [["tool_0: Concurrency probe tool 0"]]

    def test_failed_rebuild_degrades_to_the_full_list(self, monkeypatch):
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry
        from fsm_llm.logging import logger

        provider = _EmbeddingProvider(fail=True)
        _patch_embedding(monkeypatch, provider)
        registry = SemanticToolRegistry(auto_embed=False)
        for i in range(registry.FALLBACK_THRESHOLD):
            registry.register(_make_tool(i))

        messages: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            assert registry.rebuild_embeddings() == 0
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        assert registry.embedded_tool_count == 0
        assert any("Failed to embed 10 tools" in m for m in messages)
        assert len(registry.retrieve("anything")) == registry.FALLBACK_THRESHOLD


class TestNormalizeToolInput:
    """RA-01 / D-017: the typed ``any`` grammar can only carry an object as a JSON string."""

    def test_json_object_string_becomes_dict(self):
        assert normalize_tool_input('{"query": "cats", "limit": 3}') == {
            "query": "cats",
            "limit": 3,
        }

    def test_json_object_string_with_surrounding_whitespace(self):
        assert normalize_tool_input('  \n{"query": "cats"}\n ') == {"query": "cats"}

    def test_empty_json_object_string_is_empty_dict(self):
        assert normalize_tool_input("{}") == {}

    def test_malformed_json_keeps_input_wrapper(self):
        raw = '{"query": "cats", '
        assert normalize_tool_input(raw) == {"input": raw}

    def test_json_array_string_keeps_input_wrapper(self):
        raw = '["cats", "dogs"]'
        assert normalize_tool_input(raw) == {"input": raw}

    def test_bare_word_keeps_input_wrapper(self):
        assert normalize_tool_input("cats") == {"input": "cats"}

    def test_empty_string_keeps_input_wrapper(self):
        assert normalize_tool_input("") == {"input": ""}

    def test_none_dict_and_int_unchanged(self):
        d = {"a": 1}
        assert normalize_tool_input(None) == {}
        assert normalize_tool_input(d) is d
        assert normalize_tool_input(42) == {"input": "42"}

    def test_ra06_kwargs_tool_called_through_agent_handler(self):
        """Port of repro ra06: a kwargs tool driven by the grammar-forced JSON string."""
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.agents.handlers import AgentHandlers

        seen: dict = {}

        def search(query: str, limit: int = 5):
            seen["kwargs"] = {"query": query, "limit": limit}
            return "ok"

        registry = ToolRegistry()
        registry.register_function(search, name="search", description="search")
        out = AgentHandlers(registry).execute_tool(
            {
                ContextKeys.TOOL_NAME: "search",
                ContextKeys.TOOL_INPUT: '{"query": "cats", "limit": 3}',
                ContextKeys.TASK: "cats",
            }
        )
        assert out[ContextKeys.TOOL_STATUS] == "success", out
        assert seen["kwargs"] == {"query": "cats", "limit": 3}


class TestListToolInput:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-011: the core ``any`` union
    allows ``array``, so a model can emit a list ``tool_input``. It must reach
    the tool as a list, not as its ``str()`` repr."""

    def test_list_is_wrapped_as_is(self):
        raw = ["a", "b"]
        out = normalize_tool_input(raw)
        assert out == {"input": ["a", "b"]}
        assert out["input"] is raw

    def test_empty_list_is_wrapped_as_is(self):
        assert normalize_tool_input([]) == {"input": []}

    def test_single_list_param_tool_receives_a_list(self):
        seen: dict = {}

        @tool
        def total_length(words: list) -> int:
            """Sum the lengths of the words."""
            seen["words"] = words
            return sum(len(w) for w in words)

        registry = ToolRegistry()
        registry.register(total_length._tool_definition)
        result = registry.execute(
            ToolCall(
                tool_name="total_length",
                parameters=normalize_tool_input(["ab", "cde"]),
            )
        )
        assert result.success, result.error
        assert seen["words"] == ["ab", "cde"]
        assert result.result == 5

    def test_list_through_agent_handler(self):
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.agents.handlers import AgentHandlers

        seen: dict = {}

        @tool
        def count_items(items: list) -> int:
            """Count the items."""
            seen["items"] = items
            return len(items)

        registry = ToolRegistry()
        registry.register(count_items._tool_definition)
        out = AgentHandlers(registry).execute_tool(
            {
                ContextKeys.TOOL_NAME: "count_items",
                ContextKeys.TOOL_INPUT: ["x", "y", "z"],
                ContextKeys.TASK: "count",
            }
        )
        assert out[ContextKeys.TOOL_STATUS] == "success", out
        assert seen["items"] == ["x", "y", "z"]

    def test_single_str_param_tool_keeps_the_old_string_form(self):
        """A list reaching a string parameter gets ``str(list)``, exactly as
        before D-011 (review W2: ``.lower()`` on the raw list failed)."""
        from fsm_llm.agents.constants import ContextKeys
        from fsm_llm.agents.handlers import AgentHandlers

        @tool
        def search(query: str) -> str:
            """Search."""
            return "results for " + query.lower()

        registry = ToolRegistry()
        registry.register(search._tool_definition)
        out = AgentHandlers(registry).execute_tool(
            {
                ContextKeys.TOOL_NAME: "search",
                ContextKeys.TOOL_INPUT: ["Weather in Paris"],
                ContextKeys.TASK: "t",
            }
        )
        assert out[ContextKeys.TOOL_STATUS] == "success", out
        assert out[ContextKeys.TOOL_RESULT] == "results for ['weather in paris']"

    def test_str_param_with_optional_param_keeps_the_old_string_form(self):
        seen: dict = {}

        @tool
        def search(query: str, limit: int = 3) -> str:
            """Search."""
            seen["query"] = query
            return query.upper()

        registry = ToolRegistry()
        registry.register(search._tool_definition)
        result = registry.execute(
            ToolCall(tool_name="search", parameters=normalize_tool_input(["a", "b"]))
        )
        assert result.success, result.error
        assert seen["query"] == "['a', 'b']"

    @staticmethod
    def _list_tool(annotation: Any) -> tuple[Any, list]:
        seen: list = []

        def tag(tags):
            seen.append(tags)
            return f"{type(tags).__name__}:{tags}"

        tag.__annotations__ = {"tags": annotation, "return": str}
        return tool(tag), seen

    _LIST_LIKE = pytest.mark.parametrize(
        "annotation",
        [
            list[str],
            typing.List[str],  # noqa: UP006
            typing.Optional[list[str]],  # noqa: UP045
            list[int] | None,
            typing.Sequence[str],
            tuple[str, ...],
        ],
        ids=["list-str", "List-str", "Optional", "pipe-None", "Sequence", "tuple"],
    )

    @_LIST_LIKE
    def test_list_like_param_receives_a_list(self, annotation):
        """Review P2-W4: ``@tool`` inferred ``string`` for any generic, so a
        ``list[str]`` parameter got ``str(list)`` from the positional fallback
        (c6e8461 passed the list)."""
        fn, seen = self._list_tool(annotation)
        registry = ToolRegistry()
        registry.register(fn._tool_definition)
        for raw in (["a", "b"], {"input": ["a", "b"]}):
            result = registry.execute(
                ToolCall(tool_name="tag", parameters=normalize_tool_input(raw))
            )
            assert result.success, result.error
        assert seen == [["a", "b"], ["a", "b"]]

    @_LIST_LIKE
    def test_list_like_param_schema_is_an_array_with_items(self, annotation):
        fn, _ = self._list_tool(annotation)
        prop = fn._tool_definition.parameter_schema["properties"]["tags"]
        assert prop["type"] == "array"
        # OpenAI native function calling rejects an array without ``items``.
        assert prop["items"]["type"] in ("string", "integer")

    @pytest.mark.parametrize(
        "annotation",
        [list, typing.Optional[int], str],  # noqa: UP045
        ids=["bare-list", "Optional-int", "str"],
    )
    def test_other_annotations_keep_their_schema(self, annotation):
        fn, _ = self._list_tool(annotation)
        prop = fn._tool_definition.parameter_schema["properties"]["tags"]
        expected = {"type": "array"} if annotation is list else {"type": "string"}
        assert prop == expected

    def test_lone_dict_generic_param_is_the_dict_style_call(self):
        """A lone ``dict[str, int]`` parameter is the dict-style form (plan
        0068b055 D-009, the one rule ``register_function`` always applied and
        test_register_function_keeps_legacy_single_dict_param pins): the model
        sees an empty object schema and the whole arguments object is the
        dict. The parent's ``@tool`` showed the typed ``args_model`` property
        instead; a typed schema for this form is carried (F-AB-05b)."""
        fn, seen = self._list_tool(dict[str, int])
        definition = fn._tool_definition
        assert definition.parameter_schema == {}
        assert definition.args_model is None
        registry = ToolRegistry()
        registry.register(definition)
        assert registry.get_json_schemas()[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }
        result = registry.execute(
            ToolCall(tool_name="tag", parameters={"a": 1, "b": 2})
        )
        assert result.success, result.error
        assert seen == [{"a": 1, "b": 2}]

    def test_approval_grant_binds_the_list_consistently(self):
        """The driver writes the grant from an already-normalized input and the
        refusal recomputes it from the raw context value: both must agree."""
        from fsm_llm.agents.handlers import approval_grant

        raw = ["a", "b"]
        driver = approval_grant("t", normalize_tool_input(raw))
        refusal = approval_grant("t", raw)
        assert driver == refusal == {"tool_name": "t", "parameters": {"input": raw}}


_AB_SCHEMA = {"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}


def _run_ab(fn, schema=None):
    registry = ToolRegistry()
    registry.register_function(fn, name="t", parameter_schema=schema or _AB_SCHEMA)
    return registry.execute(ToolCall(tool_name="t", parameters={"a": 2, "b": 3}))


class TestDictParamToolDetection:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-024: a single-``params``-dict
    tool must be called with its dict also when the annotation is a ``dict[...]``
    generic or a string (``from __future__ import annotations``, as in this
    file). Keyword tools and a dict-typed parameter that the schema names keep
    their existing calling convention."""

    def test_string_generic_annotation(self):
        def add(params: dict[str, Any]) -> int:
            return params["a"] + params["b"]

        assert add.__annotations__["params"] == "dict[str, Any]"
        result = _run_ab(add)
        assert result.success, result.error
        assert result.result == 5

    def test_string_bare_dict_annotation(self):
        def add(params: dict) -> int:
            return params["a"] + params["b"]

        assert add.__annotations__["params"] == "dict"
        result = _run_ab(add)
        assert result.success, result.error
        assert result.result == 5

    @pytest.mark.parametrize(
        "annotation",
        [dict[str, Any], typing.Dict[str, int], "Dict[str, int]"],  # noqa: UP006
        ids=["builtin-generic", "typing-generic", "typing-string"],
    )
    def test_generic_annotation_objects(self, annotation):
        def add(params):
            return params["a"] + params["b"]

        add.__annotations__["params"] = annotation
        result = _run_ab(add)
        assert result.success, result.error
        assert result.result == 5

    def test_keyword_tools_unchanged(self):
        def add(a, b):
            return a + b

        def echo(text: str) -> str:
            return text

        assert _run_ab(add).result == 5
        registry = ToolRegistry()
        registry.register_function(
            echo, parameter_schema={"properties": {"text": {"type": "string"}}}
        )
        out = registry.execute(ToolCall(tool_name="echo", parameters={"text": "hi"}))
        assert out.success and out.result == "hi"

    def test_schema_named_dict_param_gets_its_value(self):
        """Pins the pre-existing behavior: a dict-typed parameter that the
        schema names (string or generic annotation) is called by keyword."""
        seen: list = []

        def configure(config: dict[str, Any]) -> str:
            seen.append(config)
            return "ok"

        def configure_bare(config: dict) -> str:
            seen.append(config)
            return "ok"

        schema = {"properties": {"config": {"type": "object"}}}
        for fn in (configure, configure_bare):
            registry = ToolRegistry()
            registry.register_function(fn, name="c", parameter_schema=schema)
            out = registry.execute(
                ToolCall(tool_name="c", parameters={"config": {"k": 1}})
            )
            assert out.success, out.error
        assert seen == [{"k": 1}, {"k": 1}]

    def test_class_dict_annotation_with_schema_named_param_unchanged(self):
        """Pins the pre-existing (older) behavior for an evaluated ``dict``
        class annotation: the whole parameters dict is passed, even when the
        schema names the parameter."""
        seen: list = []

        def configure(config):
            seen.append(config)
            return "ok"

        configure.__annotations__["config"] = dict
        schema = {"properties": {"config": {"type": "object"}}}
        registry = ToolRegistry()
        registry.register_function(configure, name="c", parameter_schema=schema)
        out = registry.execute(ToolCall(tool_name="c", parameters={"config": {"k": 1}}))
        assert out.success, out.error
        assert seen == [{"config": {"k": 1}}]


class TestKwargsOnlyTool:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-024 (review W5): a tool whose
    only parameter is ``**kwargs`` (a zero-argument MCP tool's ``execute``, the
    monitor's stub tools) is called with keyword arguments, not with the
    parameters dict as one positional argument."""

    @pytest.mark.parametrize("schema", [None, {"properties": {}}])
    @pytest.mark.parametrize("params", [{}, {"query": "x"}, {"input": "x"}])
    def test_sync_kwargs_only_tool(self, schema, params):
        seen: list[dict] = []

        def stub_fn(**kwargs: Any) -> str:
            seen.append(kwargs)
            return "STUB"

        registry = ToolRegistry()
        registry.register_function(
            stub_fn, name="stub", description="d", parameter_schema=schema
        )
        result = registry.execute(ToolCall(tool_name="stub", parameters=params))
        assert result.success, result.error
        assert result.result == "STUB"
        assert seen == [params]

    def test_async_zero_argument_mcp_style_tool(self):
        async def execute(**kwargs: Any) -> str:
            return f"time {sorted(kwargs)}"

        registry = ToolRegistry()
        registry.register_function(
            execute,
            name="get_time",
            description="d",
            parameter_schema={"properties": {}},
        )
        result = registry.execute(ToolCall(tool_name="get_time", parameters={}))
        assert result.success, result.error
        assert result.result == "time []"

    def test_legacy_single_dict_param_still_gets_the_dict(self):
        seen: list = []

        def legacy(params):
            seen.append(params)
            return "ok"

        registry = ToolRegistry()
        registry.register_function(legacy, name="legacy", description="d")
        result = registry.execute(
            ToolCall(tool_name="legacy", parameters={"input": "x"})
        )
        assert result.success, result.error
        assert seen == [{"input": "x"}]


class TestToolArgumentBinding:
    """TOOL-01/02/03 (plan-2026-09-29T103145-06a5ec0a step 7): arguments are
    bound against the signature before the call, a tool runs at most once per
    ``execute``, and a fallback never feeds a named optional to a missing
    required parameter."""

    @staticmethod
    def _query_limit_registry(got: list, schema: bool = True) -> ToolRegistry:
        def f(query: str, limit: int = 5) -> str:
            got.append((query, limit))
            return "ok"

        registry = ToolRegistry()
        registry.register_function(
            f,
            name="f",
            description="d",
            parameter_schema=(
                {
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                }
                if schema
                else {}
            ),
        )
        return registry

    def test_body_type_error_runs_tool_once_and_fails(self):
        calls: list[str] = []

        def writer(path: str) -> str:
            calls.append(path)
            raise TypeError("body bug after side effect")

        registry = ToolRegistry()
        registry.register_function(
            writer,
            name="writer",
            description="d",
            parameter_schema={
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        result = registry.execute(
            ToolCall(tool_name="writer", parameters={"path": "a"})
        )
        assert calls == ["a"]
        assert not result.success
        assert "body bug after side effect" in (result.error or "")

    def test_body_type_error_after_fallback_mapping_runs_once(self):
        calls: list[str] = []

        def writer(path: str) -> str:
            calls.append(path)
            raise TypeError("body bug")

        registry = ToolRegistry()
        registry.register_function(
            writer,
            name="writer",
            description="d",
            parameter_schema={"properties": {"path": {"type": "string"}}},
        )
        result = registry.execute(ToolCall(tool_name="writer", parameters={"p": "a"}))
        assert calls == ["a"]
        assert not result.success

    def test_async_body_type_error_runs_once(self):
        calls: list[str] = []

        async def writer(path: str, mode: str = "w") -> str:
            calls.append(path)
            raise TypeError("async body bug")

        registry = ToolRegistry()
        registry.register_function(
            writer,
            name="writer",
            description="d",
            parameter_schema={"properties": {"path": {"type": "string"}}},
        )
        result = registry.execute(
            ToolCall(tool_name="writer", parameters={"path": "a"})
        )
        assert calls == ["a"]
        assert not result.success

    def test_named_optional_is_never_mapped_to_missing_required(self):
        got: list = []
        registry = self._query_limit_registry(got)
        result = registry.execute(ToolCall(tool_name="f", parameters={"limit": 3}))
        assert got == []
        assert not result.success
        assert "query" in (result.error or "")

    def test_misnamed_single_value_still_maps_to_the_lone_required(self):
        got: list = []
        registry = self._query_limit_registry(got)
        result = registry.execute(ToolCall(tool_name="f", parameters={"q": "hello"}))
        assert result.success, result.error
        assert got == [("hello", 5)]

    def test_nested_tool_input_keeps_siblings(self):
        got: list = []
        registry = self._query_limit_registry(got)
        result = registry.execute(
            ToolCall(
                tool_name="f", parameters={"tool_input": {"query": "x"}, "limit": 9}
            )
        )
        assert result.success, result.error
        assert got == [("x", 9)]

    def test_nested_tool_input_wins_over_a_conflicting_sibling(self):
        got: list = []
        registry = self._query_limit_registry(got)
        result = registry.execute(
            ToolCall(
                tool_name="f",
                parameters={"tool_input": {"query": "x", "limit": 1}, "limit": 9},
            )
        )
        assert result.success, result.error
        assert got == [("x", 1)]

    def test_nested_tool_input_drops_tool_name_sibling_without_schema(self):
        got: list = []
        registry = self._query_limit_registry(got, schema=False)
        result = registry.execute(
            ToolCall(
                tool_name="f",
                parameters={"tool_name": "f", "tool_input": {"query": "x"}},
            )
        )
        assert result.success, result.error
        assert got == [("x", 5)]

    def test_register_function_infers_schema_for_annotated_param(self):
        seen: list = []

        def g(query: str) -> str:
            seen.append(query)
            return "ok"

        registry = ToolRegistry()
        registry.register_function(g, name="g")
        assert registry.get("g").parameter_schema == {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        result = registry.execute(ToolCall(tool_name="g", parameters={"query": "hi"}))
        assert result.success, result.error
        assert seen == ["hi"]

    @pytest.mark.parametrize("annotation", [None, dict, dict[str, Any]])
    def test_register_function_keeps_legacy_single_dict_param(self, annotation):
        seen: list = []

        def legacy(params):
            seen.append(params)
            return "ok"

        if annotation is not None:
            legacy.__annotations__ = {"params": annotation}
        registry = ToolRegistry()
        registry.register_function(legacy, name="legacy", description="d")
        assert registry.get("legacy").parameter_schema == {}
        result = registry.execute(
            ToolCall(tool_name="legacy", parameters={"input": "x"})
        )
        assert result.success, result.error
        assert seen == [{"input": "x"}]

    def test_register_function_var_keyword_passes_every_key(self):
        seen: list = []

        def fn(query: str, **extra: Any) -> str:
            seen.append((query, extra))
            return "ok"

        registry = ToolRegistry()
        registry.register_function(fn, name="fn", description="d")
        result = registry.execute(
            ToolCall(tool_name="fn", parameters={"query": "a", "k": 1})
        )
        assert result.success, result.error
        assert seen == [("a", {"k": 1})]


class _RuleModel(BaseModel):
    x: int


def _real(fn, **annotations):
    """Give ``fn`` real (not string) annotations: this module uses
    ``from __future__ import annotations``, and a string ``"dict"`` takes a
    different branch of the dispatch than the object ``dict``."""
    fn.__annotations__ = annotations
    return fn


def _positional_bare(seen):
    def f(payload: dict) -> str:
        seen.append(payload)
        return "ok"

    return _real(f, payload=dict)


def _positional_default(seen):
    def f(payload: dict = None) -> str:
        seen.append(payload)
        return "ok"

    return _real(f, payload=dict)


def _positional_only(seen):
    def f(payload: dict, /) -> str:
        seen.append(payload)
        return "ok"

    return _real(f, payload=dict)


def _positional_star(seen):
    def f(*args: dict) -> str:
        seen.append(args[0])
        return "ok"

    return _real(f, args=dict)


def _signature_cases():
    """(id, function) pairs: each signature in a different shape."""

    def dict_bare(p: dict) -> str: ...

    def dict_generic(p: dict[str, Any]) -> str: ...

    def one_int_kwargs(a: int, **kw) -> str: ...

    def kwonly_dict(*, payload: dict[str, Any]) -> str: ...

    def kwonly_dict_bare(*, payload: dict) -> str: ...

    def kwonly_dict_default(*, payload: dict = None) -> str: ...

    _real(kwonly_dict_bare, payload=dict)
    _real(kwonly_dict_default, payload=dict)

    def dict_beside_int(n: int, p: dict[str, Any]) -> str: ...

    def dict_beside_unannotated(p: dict[str, Any], q) -> str: ...

    def kwonly_optional_dict(*, p: typing.Optional[dict[str, Any]] = None) -> str: ...  # noqa: UP045

    def lone_kwargs(**kwargs: Any) -> str: ...

    def lone_args(*args) -> str: ...

    def int_and_args(a: int, *args: int) -> str: ...

    def optional_and_default(a: typing.Optional[int] = None, b: str = "x") -> str: ...  # noqa: UP045

    def list_int(items: list[int]) -> str: ...

    def pydantic_model(m: _RuleModel) -> str: ...

    return [
        ("dict", dict_bare),
        ("dict-generic", dict_generic),
        ("kwonly-dict", kwonly_dict),
        ("kwonly-dict-bare", kwonly_dict_bare),
        ("kwonly-dict-default", kwonly_dict_default),
        ("dict-beside-int", dict_beside_int),
        ("dict-beside-unannotated", dict_beside_unannotated),
        ("kwonly-optional-dict", kwonly_optional_dict),
        ("int-and-kwargs", one_int_kwargs),
        ("lone-kwargs", lone_kwargs),
        ("lone-args", lone_args),
        ("int-and-args", int_and_args),
        ("optional-and-default", optional_and_default),
        ("list-int", list_int),
        ("pydantic-model", pydantic_model),
    ]


def _registered(fn):
    registry = ToolRegistry()
    registry.register_function(fn, name="t")
    return registry.get("t")


class TestOneSchemaRule:
    """DECISION plan-2026-11-06T093029-0068b055/D-009 (F-AB-05): ``@tool`` and
    ``register_function`` infer a tool's schema by one rule, and a function
    that takes ``**kw`` receives every key the model sends."""

    @pytest.mark.parametrize(
        "fn", [f for _, f in _signature_cases()], ids=[i for i, _ in _signature_cases()]
    )
    def test_tool_and_register_function_agree_on_every_signature(self, fn):
        from_decorator = tool(fn)._tool_definition
        registry = ToolRegistry()
        registry.register_function(fn, name=from_decorator.name)
        from_register = registry.get(from_decorator.name)
        assert from_decorator.parameter_schema == from_register.parameter_schema
        assert (from_decorator.args_model is None) == (from_register.args_model is None)
        if from_decorator.args_model is not None:
            assert (
                from_decorator.args_model.model_json_schema()
                == from_register.args_model.model_json_schema()
            )

    @pytest.mark.parametrize(
        "shape", ["generic", "bare", "optional", "typing-dict", "beside-int"]
    )
    def test_dict_parameter_that_is_not_the_lone_positional_is_an_object(self, shape):
        """A ``dict[...]`` parameter that is named (keyword-only, or beside
        another parameter) is advertised as ``object``, never ``string``, by
        both the coarse schema and the args model (live finding D2)."""

        def generic(*, payload: dict[str, Any]) -> str: ...

        def bare(*, payload: dict) -> str: ...

        def optional(*, payload: typing.Optional[dict[str, Any]] = None) -> str: ...  # noqa: UP045

        def typing_dict(*, payload: typing.Dict[str, int]) -> str: ...  # noqa: UP006

        def beside_int(n: int, payload: dict[str, Any]) -> str: ...

        fn = {
            "generic": generic,
            "bare": bare,
            "optional": optional,
            "typing-dict": typing_dict,
            "beside-int": beside_int,
        }[shape]
        for definition in (
            tool(fn)._tool_definition,
            _registered(fn),
        ):
            assert definition.parameter_schema["properties"]["payload"] == {
                "type": "object"
            }

    def test_unannotated_sibling_keeps_the_object_schema_without_args_model(self):
        """No args model exists when a parameter has no annotation, so the
        coarse schema is what the model is shown."""

        def mixed(payload: dict[str, Any], note) -> str: ...

        definition = tool(mixed)._tool_definition
        assert definition.args_model is None
        assert definition.parameter_schema["properties"]["payload"] == {
            "type": "object"
        }
        assert definition.parameter_schema["properties"]["note"] == {"type": "string"}

    def test_keyword_only_dict_parameter_keeps_its_named_schema(self):
        """The lone-dict `{}` rule is for a parameter the dict can be passed to
        by position; a keyword-only one is named and called by keyword."""
        seen: list = []

        @tool
        def kwonly(*, payload: dict[str, Any]) -> str:
            """Take a payload."""
            seen.append(payload)
            return f"payload={payload}"

        schema = kwonly._tool_definition.parameter_schema
        assert list(schema["properties"]) == ["payload"]
        assert schema["required"] == ["payload"]
        registry = ToolRegistry()
        registry.register(kwonly._tool_definition)
        result = registry.execute(
            ToolCall(tool_name="kwonly", parameters={"payload": {"x": 1}})
        )
        assert result.success, result.error
        assert seen == [{"x": 1}]

    @pytest.mark.parametrize("via", ["tool", "register_function"])
    @pytest.mark.parametrize("shape", ["bare", "default", "generic", "method"])
    def test_keyword_only_dict_parameter_is_called_by_keyword(self, via, shape):
        """The schema names ``payload``, so the call must pass it by keyword,
        whatever the spelling of the annotation (D-018, pass-2 concern 1)."""
        seen: list = []

        def bare(*, payload: dict) -> str:
            seen.append(payload)
            return "ok"

        def with_default(*, payload: dict = None) -> str:
            seen.append(payload)
            return "ok"

        def generic(*, payload: dict[str, Any]) -> str:
            seen.append(payload)
            return "ok"

        class Holder:
            def method(self, *, payload: dict) -> str:
                seen.append(payload)
                return "ok"

        _real(bare, payload=dict)
        _real(with_default, payload=dict)
        _real(generic, payload=dict[str, Any])
        _real(Holder.method, payload=dict)
        fn = {
            "bare": bare,
            "default": with_default,
            "generic": generic,
            "method": Holder().method,
        }[shape]
        if via == "tool" and shape == "method":
            pytest.skip(
                "@tool sets an attribute on the function; a bound method has none"
            )
        registry = ToolRegistry()
        if via == "tool":
            registry.register(tool(fn)._tool_definition)
            name = tool(fn)._tool_definition.name
        else:
            registry.register_function(fn, name="kw", description="d")
            name = "kw"
        schema = registry.get(name).parameter_schema
        assert list(schema["properties"]) == ["payload"]
        result = registry.execute(
            ToolCall(tool_name=name, parameters={"payload": {"x": 1}})
        )
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_annotated_dict_parameter_is_a_named_property_called_by_keyword(self):
        """Unchanged by the keyword-only rule: ``Annotated[dict, ...]`` is not
        read as a dict-style annotation, so the schema names the parameter."""
        seen: list = []

        def run(payload: typing.Annotated[dict, "the payload"]) -> str:
            seen.append(payload)
            return "ok"

        _real(run, payload=typing.Annotated[dict, "the payload"])
        registry = ToolRegistry()
        registry.register_function(run, name="run", description="d")
        assert list(registry.get("run").parameter_schema["properties"]) == ["payload"]
        result = registry.execute(
            ToolCall(tool_name="run", parameters={"payload": {"x": 1}})
        )
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_lone_keyword_dict_kwargs_is_called_by_keyword(self):
        seen: list = []

        def run(**kw: dict) -> str:
            seen.append(kw)
            return "ok"

        _real(run, kw=dict)
        registry = ToolRegistry()
        registry.register_function(run, name="run", description="d")
        result = registry.execute(ToolCall(tool_name="run", parameters={"a": {"x": 1}}))
        assert result.success, result.error
        assert seen == [{"a": {"x": 1}}]

    def _run_kwonly(self, fn):
        """Register ``fn`` and call it with ``payload={"x": 1}``; the schema
        must name ``payload``."""
        registry = ToolRegistry()
        registry.register_function(fn, name="kw", description="d")
        assert list(registry.get("kw").parameter_schema["properties"]) == ["payload"]
        return registry.execute(
            ToolCall(tool_name="kw", parameters={"payload": {"x": 1}})
        )

    def test_pin_functools_wraps_wrapper_of_a_keyword_only_function(self):
        """PIN (step 12.9, passes at 11dd49d): ``inspect.signature`` follows
        ``__wrapped__``, so a ``functools.wraps`` wrapper of a keyword-only dict
        function is called by keyword."""
        import functools

        seen: list = []

        def inner(*, payload: dict) -> str:
            seen.append(payload)
            return "ok"

        _real(inner, payload=dict)

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        result = self._run_kwonly(wrapper)
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_pin_async_keyword_only_method(self):
        """PIN (step 12.9): an ``async`` method with a keyword-only dict
        parameter is called by keyword and awaited."""
        seen: list = []

        class Holder:
            async def method(self, *, payload: dict) -> str:
                seen.append(payload)
                return "ok"

        _real(Holder.method, payload=dict)
        result = self._run_kwonly(Holder().method)
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_pin_callable_instance_with_a_keyword_only_dict(self):
        """PIN (step 12.9): a callable instance is read through its
        ``__call__``; its keyword-only dict is called by keyword."""
        seen: list = []

        class Callable_:
            def __call__(self, *, payload: dict) -> str:
                seen.append(payload)
                return "ok"

        _real(Callable_.__call__, payload=dict)
        result = self._run_kwonly(Callable_())
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_pin_star_wrapper_with_a_keyword_only_dict_signature(self):
        """PIN (step 12.9): a wrapper declared ``(*a, **k)`` whose
        ``__signature__`` is a keyword-only dict signature is read through that
        signature, so it now receives ``payload=`` BY KEYWORD (the schema names
        ``payload``). The dict is never passed positionally to a keyword-only
        signature, so a wrapper that forwards ``*a`` positionally to its inner
        function must be given a ``__signature__`` that matches the inner
        function."""
        import inspect

        seen: list = []

        def inner(*, payload: dict) -> str:
            seen.append(payload)
            return "ok"

        _real(inner, payload=dict)

        def wrapper(*args, **kwargs):
            seen.append((args, tuple(kwargs)))
            return inner(**kwargs)

        wrapper.__signature__ = inspect.signature(inner)  # type: ignore[attr-defined]
        result = self._run_kwonly(wrapper)
        assert result.success, result.error
        assert seen[0] == ((), ("payload",))
        assert seen[1] == {"x": 1}

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda seen: _positional_bare(seen), id="bare"),
            pytest.param(lambda seen: _positional_default(seen), id="default"),
            pytest.param(lambda seen: _positional_only(seen), id="positional-only"),
            pytest.param(lambda seen: _positional_star(seen), id="lone-star-args"),
        ],
    )
    def test_positional_dict_style_calls_are_unchanged(self, make):
        """What the keyword-only rule must not move: a lone positional (or
        ``*args``) parameter still receives the whole arguments object."""
        seen: list = []
        fn = make(seen)
        registry = ToolRegistry()
        registry.register_function(fn, name="p", description="d")
        result = registry.execute(ToolCall(tool_name="p", parameters={"x": 1}))
        assert result.success, result.error
        assert seen == [{"x": 1}]

    def test_lone_kwargs_tool_receives_every_argument(self):
        seen: list[dict] = []

        @tool
        def echo(**kwargs: Any) -> str:
            seen.append(kwargs)
            return "ok"

        registry = ToolRegistry()
        registry.register(echo._tool_definition)
        result = registry.execute(ToolCall(tool_name="echo", parameters={"q": 1}))
        assert result.success, result.error
        assert seen == [{"q": 1}]

    def test_named_plus_kwargs_tool_receives_extras(self):
        seen: list = []

        @tool
        def search(query: str, **options: Any) -> str:
            seen.append((query, options))
            return "ok"

        definition = search._tool_definition
        assert list(definition.parameter_schema["properties"]) == ["query"]
        assert definition.parameter_schema["required"] == ["query"]
        registry = ToolRegistry()
        registry.register(definition)
        result = registry.execute(
            ToolCall(tool_name="search", parameters={"query": "x", "limit": 3})
        )
        assert result.success, result.error
        assert seen == [("x", {"limit": 3})]

    def test_var_positional_is_never_a_required_property(self):
        seen: list = []

        @tool
        def total(a: int, *args: int) -> int:
            seen.append(a)
            return a

        schema = total._tool_definition.parameter_schema
        assert list(schema["properties"]) == ["a"]
        assert schema["required"] == ["a"]
        registry = ToolRegistry()
        registry.register(total._tool_definition)
        result = registry.execute(ToolCall(tool_name="total", parameters={"a": 4}))
        assert result.success, result.error
        assert seen == [4]

    def test_model_visible_schema_of_a_lone_dict_tool_is_empty_object(self):
        @tool
        def run(params: dict) -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register(run._tool_definition)
        assert registry.get_json_schemas()[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_explicit_schema_still_filters_a_lone_kwargs_tool(self):
        """A lone ``**kw`` declares nothing, so an explicit schema (an MCP
        tool) stays its contract: unlisted keys are dropped as before."""
        seen: list[dict] = []

        def execute(**kwargs: Any) -> str:
            seen.append(kwargs)
            return "ok"

        registry = ToolRegistry()
        registry.register_function(
            execute,
            name="mcp",
            parameter_schema={"properties": {"city": {"type": "string"}}},
        )
        result = registry.execute(
            ToolCall(tool_name="mcp", parameters={"city": "x", "extra": 1})
        )
        assert result.success, result.error
        assert seen == [{"city": "x"}]
