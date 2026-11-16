"""The tool layer (`fsm_llm.tools`) and its seam with `fsm_llm.agents`.

Step 85 of plan d454fd62 moved `ToolRegistry`, the tool models, the tool errors,
`require_int_limit` and the helpers around them from `fsm_llm.agents` down into
the layer package `fsm_llm.tools`. These tests pin what that move promised:

- the agents constants classes keep every attribute they had, and each tool-layer
  attribute is the SAME object (one source of value, `vars()` still lists it);
- `fsm_llm.agents` re-exports the tool-layer objects, and the old module paths no
  longer define them (no shim);
- an `except AgentError` still catches a tool error, `AgentError` lives in core;
- importing the layer loads no subpackage and no module above it.
"""

from __future__ import annotations

import ast
import importlib
import threading
from pathlib import Path

import pytest

import fsm_llm.agents as agents
import fsm_llm.agents.constants as agents_constants
import fsm_llm.tools as tools
import fsm_llm.tools.constants as tool_constants
from fsm_llm.core.definitions import AgentError
from tests.test_fsm_llm_agents.support import _layer_probe_output

SRC = Path(tools.__file__).resolve().parents[1]
_ATTR_CLASSES = ("ContextKeys", "Defaults", "ErrorMessages", "LogMessages")


def _public_attributes(cls: type) -> dict[str, object]:
    return {k: v for k, v in vars(cls).items() if not k.startswith("__")}


class TestConstantsKeepOneSource:
    @pytest.mark.parametrize("name", _ATTR_CLASSES)
    def test_every_tool_attribute_is_the_same_object_in_the_agents_class(
        self, name: str
    ) -> None:
        tool_cls = getattr(tool_constants, name)
        agents_cls = getattr(agents_constants, name)
        attributes = _public_attributes(tool_cls)
        assert attributes, f"{name} of the tool layer is empty"
        for attr, value in attributes.items():
            assert attr in vars(agents_cls), (
                f"{name}.{attr} left vars() of the agents class"
            )
            assert vars(agents_cls)[attr] is value, (
                f"{name}.{attr} is a copy, not the tool value"
            )

    @pytest.mark.parametrize("name", _ATTR_CLASSES)
    def test_the_agents_class_holds_nothing_the_tool_class_changed(
        self, name: str
    ) -> None:
        agents_cls = getattr(agents_constants, name)
        for attr, value in _public_attributes(getattr(tool_constants, name)).items():
            assert getattr(agents_cls, attr) == value

    @pytest.mark.parametrize("name", _ATTR_CLASSES)
    def test_the_agents_class_is_its_own_class_not_a_subclass(self, name: str) -> None:
        agents_cls = getattr(agents_constants, name)
        assert agents_cls is not getattr(tool_constants, name)
        assert agents_cls.__bases__ == (object,)

    def test_the_tool_values_are_the_documented_ones(self) -> None:
        # a golden copy of the values the registry, the models and the messages
        # depend on: a change here is a behaviour change, not a refactor
        assert tool_constants.ContextKeys.NO_TOOL == "none"
        assert tool_constants.ContextKeys.TOOL_NAME == "tool_name"
        assert tool_constants.ContextKeys.TOOL_INPUT == "tool_input"
        assert tool_constants.Defaults.MAX_INT_LIMIT == 2**63 - 1
        assert tool_constants.Defaults.MAX_OBSERVATION_LENGTH == 2000
        assert tool_constants.Defaults.TOOL_CONFIDENCE_THRESHOLD == 0.4
        assert tool_constants.Defaults.MAX_TOOL_TIMEOUT_S == threading.TIMEOUT_MAX
        assert tool_constants.ErrorMessages.TOOL_NOT_FOUND.format(name="x") == (
            "Tool 'x' not found in registry"
        )

    @pytest.mark.parametrize("name", ["ToolRunStatus", "ToolObservationPrefix"])
    def test_the_two_moved_classes_are_the_agents_names(self, name: str) -> None:
        assert getattr(agents_constants, name) is getattr(tool_constants, name)
        assert getattr(agents_constants, name).__module__ == "fsm_llm.tools.constants"
        assert name in agents_constants.__all__

    def test_the_observation_entry_is_one_string_defined_in_the_tool_layer(
        self,
    ) -> None:
        """Step 95 (D-042): the template moved down so `synthesis.tools` reads it without
        `agents`; the agents name is the same object, the text is the one the loop stores."""
        from fsm_llm.agents.constants import outcomes
        from fsm_llm.synthesis import tools as synthesis_tools

        assert agents_constants.OBSERVATION_ENTRY is tool_constants.OBSERVATION_ENTRY
        assert "OBSERVATION_ENTRY" in agents_constants.__all__
        assert not hasattr(outcomes, "OBSERVATION_ENTRY")
        assert synthesis_tools.OBSERVATION_ENTRY is tool_constants.OBSERVATION_ENTRY
        assert synthesis_tools.Defaults is tool_constants.Defaults
        assert tool_constants.OBSERVATION_ENTRY == (
            "[Step {step}] Tool: {tool} | Input: {input} | Result: {result}"
        )


class TestTheAgentsFacadeReexports:
    @pytest.mark.parametrize(
        "name",
        [
            "ToolRegistry",
            "ToolDefinition",
            "ToolCall",
            "ToolResult",
            "ToolAnnotations",
            "ToolExecutionError",
            "ToolNotFoundError",
            "tool",
            "registry_from",
        ],
    )
    def test_the_public_name_is_the_tool_layer_object(self, name: str) -> None:
        assert getattr(agents, name) is getattr(tools, name)
        # `registry_from` is bound by the package (create_agent uses it) but was never exported
        assert (name in agents.__all__) == (name != "registry_from")

    @pytest.mark.parametrize(
        "name, home",
        [
            ("ToolRegistry", "fsm_llm.tools.registry"),
            ("ToolDefinition", "fsm_llm.tools.models"),
            ("ToolCall", "fsm_llm.tools.models"),
            ("ToolResult", "fsm_llm.tools.models"),
            ("ToolAnnotations", "fsm_llm.tools.models"),
            ("ToolExecutionError", "fsm_llm.tools.errors"),
            ("ToolNotFoundError", "fsm_llm.tools.errors"),
        ],
    )
    def test_each_class_is_defined_in_its_module(self, name: str, home: str) -> None:
        assert getattr(tools, name).__module__ == home

    def test_agent_error_is_the_core_class(self) -> None:
        assert agents.AgentError is AgentError
        assert AgentError.__module__ == "fsm_llm.core.definitions.errors"
        assert (
            "AgentError" in importlib.import_module("fsm_llm.core.definitions").__all__
        )

    def test_except_agent_error_still_catches_a_tool_error(self) -> None:
        registry = tools.ToolRegistry()
        with pytest.raises(AgentError):
            registry.get("missing")
        with pytest.raises(AgentError):
            tools.check_timeout("t", -1)
        assert issubclass(tools.ToolTimedOut, tools.ToolExecutionError)
        assert issubclass(tools.ToolExecutionError, AgentError)


class TestOldPathsAreGone:
    """No shim: a moved name is not DEFINED at its old module (a re-binding for a
    pydantic annotation is allowed, a definition is not)."""

    @pytest.mark.parametrize(
        "module, names",
        [
            (
                "fsm_llm.agents.tools",
                ["ToolRegistry", "tool", "registry_from", "redact_secret_entries"],
            ),
            (
                "fsm_llm.agents.definitions",
                ["ToolDefinition", "ToolResult", "ToolAnnotations"],
            ),
            (
                "fsm_llm.agents.exceptions",
                ["ToolExecutionError", "ToolNotFoundError", "require_int_limit"],
            ),
        ],
    )
    def test_the_module_defines_none_of_them(
        self, module: str, names: list[str]
    ) -> None:
        path = Path(importlib.import_module(module).__file__ or "")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            n.name for n in tree.body if isinstance(n, ast.ClassDef | ast.FunctionDef)
        }
        assert not set(names) & defined

    def test_agents_tools_keeps_exactly_the_agent_specific_helpers(self) -> None:
        mod = importlib.import_module("fsm_llm.agents.tools")
        tree = ast.parse(Path(mod.__file__ or "").read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert defined == {
            "normalize_tool_input",
            "tool_call_from_context",
            "make_hitl_checker",  # step 86: the agent-side approval gate, D-037
            "refuse_execute_without_gated",
            "register_agent",
        }
        assert not hasattr(mod, "ToolRegistry")  # annotation-only, under TYPE_CHECKING

    def test_the_truncation_module_is_gone_from_agents(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("fsm_llm.agents.truncation")
        assert tools.smart_truncate.__module__ == "fsm_llm.tools.truncation"


class TestTheLayerIsBelowTheSubpackages:
    _PROBE = (
        "import sys\n"
        "import fsm_llm.tools\n"
        "import fsm_llm.tools.registry, fsm_llm.tools.models\n"
        "above = ('agents', 'reasoning', 'workflows', 'monitor', 'eval', 'synthesis',\n"
        "         'okf', 'skills', 'testing', 'runner')\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if any(m == f'fsm_llm.{a}' or m.startswith(f'fsm_llm.{a}.') for a in above))\n"
        "print('LOADED', bad)\n"
    )

    def test_importing_it_loads_no_subpackage_and_no_peer_layer(self) -> None:
        out = _layer_probe_output(self._PROBE)
        assert out.strip() == "LOADED []", out

    def test_no_module_of_the_layer_names_a_higher_module_in_its_source(self) -> None:
        forbidden = (
            "fsm_llm.agents",
            "fsm_llm.engine",
            "fsm_llm.handlers",
            "fsm_llm.llm",
            "fsm_llm.okf",
            "fsm_llm.skills",
            "fsm_llm.synthesis",
        )
        for path in sorted((SRC / "tools").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    assert not name.startswith(forbidden), f"{path.name}: {name}"

    def test_the_tools_layer_imports_no_context_module(self) -> None:
        edges = set()
        for path in sorted((SRC / "tools").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "fsm_llm.context"
                ):
                    edges.add((path.name, node.module))
        # the redaction walk moved down to core.redaction at step 95 (D-042): no edge is left
        assert edges == set()


class TestTheFacade:
    def test_every_name_of_all_resolves_and_nothing_else_is_public(self) -> None:
        for name in tools.__all__:
            assert hasattr(tools, name)
        public = {
            n for n in vars(tools) if not n.startswith("_") and n != "annotations"
        }
        submodules = {
            "arguments",
            "constants",
            "decorator",
            "errors",
            "hints",
            "limits",
            "models",
            "registry",
            "schema",
            "timeouts",
            "truncation",
        }
        assert public - submodules == set(tools.__all__)


class TestBoundariesOfTheMovedCode:
    """Two edges of the moved code that no earlier test pinned (found by the step-85 mutation
    proof); the values are the ones the old modules produced (a differential, old tree against new)."""

    def test_bounded_repr_keeps_a_text_of_exactly_the_limit(self) -> None:
        assert tools.bounded_repr("x" * 38) == repr("x" * 38)  # 40 characters
        assert tools.bounded_repr("x" * 39) == repr("x" * 39)[:40] + "..."
        assert tools.bounded_repr(1234, 4) == "1234"
        assert tools.bounded_repr(12345, 4) == "1234..."

    def test_a_misnamed_key_is_not_mapped_when_two_parameters_are_required(
        self,
    ) -> None:
        registry = tools.ToolRegistry()

        def two(a: str, b: str) -> str:
            return a + b

        registry.register_function(two, name="two")
        result = registry.execute(
            tools.ToolCall(tool_name="two", parameters={"zzz": "v"})
        )
        assert not result.success
        assert result.error == (
            "Tool requires parameters: ['a', 'b']. Model sent keys ['zzz'] which do not "
            "match. Provide values keyed by the expected parameter names."
        )
