from __future__ import annotations

"""Tests for fsm_llm_agents.tools module."""

import sys
import threading
import time
from typing import Annotated

import pytest

from fsm_llm_agents.definitions import ToolCall, ToolDefinition
from fsm_llm_agents.exceptions import ToolNotFoundError
from fsm_llm_agents.tools import ToolRegistry, tool


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

        class _Skill:
            def to_tool_definition(self):
                return _make_tool(99)

        calls = [
            lambda: registry.register(_make_tool(1)),
            lambda: registry.register_function(_noop, name="fn", description="d"),
            lambda: registry.register_agent(_Agent(), name="ag", description="d"),
            lambda: registry.register_skill(_Skill()),
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


# 256 dims, not 3. `_cosine_similarity` is pure Python, so the vector width sets
# how long a reader spends *inside* the `_embeddings` iteration. A 3-dim vector
# makes each scan so short that the pre-fix race almost never lands and the test
# passes against un-fixed source — a false green under SC-19. 256 dims keeps the
# iteration open across many bytecode switches and makes the pre-fix
# `RuntimeError` reliable.
_EMBEDDING_DIMS = 256


def _stub_embedding(self, text):
    """Deterministic offline stand-in for the litellm embedding call.

    Module-level (not a `staticmethod`/`classmethod` on the test class) so that
    `monkeypatch.setattr(SemanticToolRegistry, "_get_embedding", ...)` installs a
    plain function, which the descriptor protocol re-binds to the registry
    instance as `self`.
    """
    seed = sum(map(ord, text))
    return [float((seed + i) % 13) + 1.0 for i in range(_EMBEDDING_DIMS)]


class TestSemanticToolRegistryConcurrency:
    """`SemanticToolRegistry._embeddings` must not race either (F-05, SC-9)."""

    TRIALS = 20
    TOOLS_PER_TRIAL = 40

    def _registry(self, monkeypatch):
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry

        monkeypatch.setattr(SemanticToolRegistry, "_get_embedding", _stub_embedding)
        registry = SemanticToolRegistry(top_k=5)
        # Guard the stub itself. `_embed_tool` swallows every exception from
        # `_get_embedding`, so a stub that fails to bind as a descriptor (e.g. a
        # `classmethod`/bound method, which is NOT re-bound when set as a class
        # attribute) leaves `_embeddings` permanently empty — `retrieve` then
        # short-circuits to the all-tools fallback and every concurrency assertion
        # below becomes vacuous while still going red for the wrong reason.
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
