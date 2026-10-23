"""
Python tool loading for FSM-LLM agents.

Enables loading external Python tool definitions as agent tools from
Python files, directories, or programmatic definitions.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fsm_llm.logging import logger

from .definitions import ToolDefinition
from .tools import ToolRegistry

# DECISION plan-2026-10-23T043147-4c2042f3/D-032: "skill" now means only Agent
# Skills (SKILL.md, never executed); this loader EXECUTES files, so it was
# renamed. Do NOT re-add the old Skill-prefixed names, the skill-named registry
# method or the all-caps list name as aliases, re-exports or shims (no legacy).


@dataclass
class PythonToolDefinition:
    """A structured Python tool that can be loaded as an agent tool.

    Python tools are a higher-level abstraction over tools, supporting
    categorization, trigger patterns, and directory-based discovery.

    Example::

        tool = PythonToolDefinition(
            name="web_search",
            description="Search the web for information",
            category="research",
            execute=search_function,
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    """

    name: str
    description: str
    execute: Callable[..., Any]
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_definition(self) -> ToolDefinition:
        """Convert this Python tool to a ``ToolDefinition`` for registration."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameter_schema=self.parameter_schema,
            requires_approval=self.requires_approval,
            execute_fn=self.execute,
        )


@dataclass(frozen=True)
class PythonToolLoadReport:
    """The tools ``load_directory`` found and one ``(file, message)`` pair per
    file or directory it could not load (message: ``ExcType: text`` or a fixed
    phrase)."""

    tools: tuple[PythonToolDefinition, ...] = ()
    errors: tuple[tuple[str, str], ...] = ()


class PythonToolLoader:
    """Load Python tools from various sources and convert to tool registries.

    Example::

        tools = PythonToolLoader.from_directory("./tools/")
        registry = PythonToolLoader.to_tool_registry(tools)
        agent = ReactAgent(tools=registry)
    """

    @staticmethod
    def load_directory(
        path: str | Path,
        pattern: str = "*.py",
        category: str | None = None,
    ) -> PythonToolLoadReport:
        """Load Python tools from a directory, reporting each file it cannot.

        Each file defines a ``PYTHON_TOOLS`` list of ``PythonToolDefinition``
        objects or ``@tool`` functions. Files starting with ``_`` and non-``.py``
        files are ignored, not errors; ``SystemExit`` and ``KeyboardInterrupt``
        from a file propagate. Never raises for a bad file.

        Args:
            path: Directory path containing Python tool files.
            pattern: Glob pattern for Python tool files (default: ``*.py``).
            category: Override category for all loaded Python tools.
        """
        tool_dir = Path(path)
        if not tool_dir.is_dir():
            logger.warning(f"Python tool directory not found: {path}")
            return PythonToolLoadReport(errors=((str(path), "directory not found"),))

        tools: list[PythonToolDefinition] = []
        errors: list[tuple[str, str]] = []
        for file_path in sorted(tool_dir.glob(pattern)):
            if file_path.name.startswith("_"):
                continue
            try:
                loaded = PythonToolLoader._load_from_file(file_path, category)
                tools.extend(loaded)
            except Exception as e:
                logger.warning(f"Failed to load Python tools from {file_path}: {e!s}")
                errors.append((str(file_path), f"{type(e).__name__}: {e}"))

        logger.info(f"Loaded {len(tools)} Python tools from {path}")
        return PythonToolLoadReport(tuple(tools), tuple(errors))

    @staticmethod
    def from_directory(
        path: str | Path,
        pattern: str = "*.py",
        category: str | None = None,
    ) -> list[PythonToolDefinition]:
        """``load_directory(...).tools`` as a list, with one warning per error.

        Each failed file is also reported with ``warnings.warn`` (a
        ``UserWarning`` attributed to the caller, so ``disable_warnings()``
        cannot hide it). Arguments as for ``load_directory``.
        """
        report = PythonToolLoader.load_directory(path, pattern, category)
        for file, message in report.errors:
            warnings.warn(
                f"Python tools not loaded from {file}: {message}",
                UserWarning,
                stacklevel=2,
            )
        return list(report.tools)

    @staticmethod
    def from_functions(
        *functions: Callable[..., Any],
        category: str = "general",
    ) -> list[PythonToolDefinition]:
        """Create Python tools from ``@tool``-decorated functions.

        Each function must have a ``_tool_definition`` attribute
        (set by the ``@tool`` decorator).

        Args:
            functions: Decorated functions to convert.
            category: Category to assign to all Python tools.

        Returns:
            List of PythonToolDefinition objects.
        """
        tools = []
        for fn in functions:
            tool_def = getattr(fn, "_tool_definition", None)
            if tool_def is None:
                logger.warning(
                    f"Function {fn.__name__} has no _tool_definition; skipping"
                )
                continue
            tools.append(
                PythonToolDefinition(
                    name=tool_def.name,
                    description=tool_def.description,
                    execute=tool_def.execute_fn,
                    parameter_schema=tool_def.parameter_schema,
                    requires_approval=tool_def.requires_approval,
                    category=category,
                )
            )
        return tools

    @staticmethod
    def to_tool_registry(
        python_tools: list[PythonToolDefinition],
        registry: ToolRegistry | None = None,
    ) -> ToolRegistry:
        """Convert Python tools to a ``ToolRegistry``.

        Args:
            python_tools: Python tools to register.
            registry: Existing registry to add to.
                If None, creates a new one.

        Returns:
            ToolRegistry with all Python tools registered.
        """
        reg = registry or ToolRegistry()
        for tool in python_tools:
            reg.register(tool.to_tool_definition())
        return reg

    @staticmethod
    def by_category(
        python_tools: list[PythonToolDefinition],
    ) -> dict[str, list[PythonToolDefinition]]:
        """Group Python tools by category.

        Useful for building hierarchical classification schemas
        where categories map to domains.

        Returns:
            Dict mapping category names to Python tool lists.
        """
        groups: dict[str, list[PythonToolDefinition]] = {}
        for tool in python_tools:
            groups.setdefault(tool.category, []).append(tool)
        return groups

    @staticmethod
    def _load_from_file(
        file_path: Path,
        category_override: str | None = None,
    ) -> list[PythonToolDefinition]:
        """Load Python tools from a single Python file.

        Raises for a file it cannot load (``load_directory`` records it).

        .. warning::

           This method executes arbitrary Python code from *file_path* via
           ``importlib``.  Only load Python tool files from **trusted sources**.
        """
        if file_path.suffix != ".py":
            logger.warning(f"Skipping non-Python tool file: {file_path}")
            return []
        if file_path.stat().st_size > 1_000_000:
            raise ValueError("file is larger than 1 MB")

        path_hash = hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()[
            :12
        ]
        module_name = f"_fsm_python_tool_{path_hash}_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot build a module spec for the file")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        tools: list[PythonToolDefinition] = []

        # Check for explicit PYTHON_TOOLS list
        if hasattr(module, "PYTHON_TOOLS"):
            for tool in module.PYTHON_TOOLS:
                if isinstance(tool, PythonToolDefinition):
                    if category_override:
                        tool.category = category_override
                    tools.append(tool)
        # Names already provided via PYTHON_TOOLS win; the @tool scan skips them so
        # one function listed in both places is not loaded twice (PT-07).
        explicit_names = {tool.name for tool in tools}
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: dedupe the @tool scan
        # on the tool name against the @tool skills already loaded too, not
        # only against PYTHON_TOOLS: dir() lists one function once per attribute
        # name, so an alias loaded it twice. The first attribute in dir()
        # (alphabetical) order wins; a DIFFERENT function claiming the same
        # name is skipped with a warning (an alias of the same function is
        # silent). Do NOT make this last-wins to mirror ToolRegistry.register:
        # that would make the loaded function depend on attribute spelling in
        # the other direction and still hide the conflict.
        tool_attrs: dict[str, tuple[str, Any]] = {}

        # Check for @tool decorated functions
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool_definition"):
                tool_def = obj._tool_definition
                if tool_def.name in explicit_names:
                    continue
                if tool_def.name in tool_attrs:
                    first_attr, first_fn = tool_attrs[tool_def.name]
                    if tool_def.execute_fn is not first_fn:
                        logger.warning(
                            f"Python tool file {file_path.name}: '{attr_name}' "
                            f"declares tool '{tool_def.name}', already loaded "
                            f"from '{first_attr}'; skipping '{attr_name}'"
                        )
                    continue
                tool_attrs[tool_def.name] = (attr_name, tool_def.execute_fn)
                tools.append(
                    PythonToolDefinition(
                        name=tool_def.name,
                        description=tool_def.description,
                        execute=tool_def.execute_fn,
                        parameter_schema=tool_def.parameter_schema,
                        requires_approval=tool_def.requires_approval,
                        category=category_override or "general",
                    )
                )

        # Clean up
        sys.modules.pop(module_name, None)

        return tools
