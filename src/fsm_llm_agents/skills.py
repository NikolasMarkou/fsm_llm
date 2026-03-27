from __future__ import annotations

"""
Skill loading system for FSM-LLM agents.

Enables loading external skill definitions as agent tools from
Python files, directories, or programmatic definitions.
"""

import importlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fsm_llm.logging import logger

from .definitions import ToolDefinition
from .tools import ToolRegistry


@dataclass
class SkillDefinition:
    """A structured skill that can be loaded as an agent tool.

    Skills are a higher-level abstraction over tools, supporting
    categorization, trigger patterns, and directory-based discovery.

    Example::

        skill = SkillDefinition(
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
        """Convert this skill to a ``ToolDefinition`` for registration."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameter_schema=self.parameter_schema,
            requires_approval=self.requires_approval,
            execute_fn=self.execute,
        )


class SkillLoader:
    """Load skills from various sources and convert to tool registries.

    Example::

        skills = SkillLoader.from_directory("./skills/")
        registry = SkillLoader.to_tool_registry(skills)
        agent = ReactAgent(tools=registry)
    """

    @staticmethod
    def from_directory(
        path: str | Path,
        pattern: str = "*.py",
        category: str | None = None,
    ) -> list[SkillDefinition]:
        """Load skills from Python files in a directory.

        Each Python file should define a ``SKILLS`` list of
        ``SkillDefinition`` objects, or functions decorated with
        ``@tool`` that have a ``_tool_definition`` attribute.

        Args:
            path: Directory path containing skill files.
            pattern: Glob pattern for skill files (default: ``*.py``).
            category: Override category for all loaded skills.
                If None, uses each skill's own category.

        Returns:
            List of discovered SkillDefinition objects.
        """
        skill_dir = Path(path)
        if not skill_dir.is_dir():
            logger.warning(f"Skill directory not found: {path}")
            return []

        skills: list[SkillDefinition] = []
        for file_path in sorted(skill_dir.glob(pattern)):
            if file_path.name.startswith("_"):
                continue
            try:
                loaded = SkillLoader._load_from_file(file_path, category)
                skills.extend(loaded)
            except Exception as e:
                logger.warning(f"Failed to load skills from {file_path}: {e!s}")

        logger.info(f"Loaded {len(skills)} skills from {path}")
        return skills

    @staticmethod
    def from_functions(
        *functions: Callable[..., Any],
        category: str = "general",
    ) -> list[SkillDefinition]:
        """Create skills from ``@tool``-decorated functions.

        Each function must have a ``_tool_definition`` attribute
        (set by the ``@tool`` decorator).

        Args:
            functions: Decorated functions to convert.
            category: Category to assign to all skills.

        Returns:
            List of SkillDefinition objects.
        """
        skills = []
        for fn in functions:
            tool_def = getattr(fn, "_tool_definition", None)
            if tool_def is None:
                logger.warning(
                    f"Function {fn.__name__} has no _tool_definition; skipping"
                )
                continue
            skills.append(
                SkillDefinition(
                    name=tool_def.name,
                    description=tool_def.description,
                    execute=tool_def.execute_fn,
                    parameter_schema=tool_def.parameter_schema,
                    requires_approval=tool_def.requires_approval,
                    category=category,
                )
            )
        return skills

    @staticmethod
    def to_tool_registry(
        skills: list[SkillDefinition],
        registry: ToolRegistry | None = None,
    ) -> ToolRegistry:
        """Convert skills to a ``ToolRegistry``.

        Args:
            skills: Skills to register.
            registry: Existing registry to add to.
                If None, creates a new one.

        Returns:
            ToolRegistry with all skills registered.
        """
        reg = registry or ToolRegistry()
        for skill in skills:
            reg.register(skill.to_tool_definition())
        return reg

    @staticmethod
    def by_category(
        skills: list[SkillDefinition],
    ) -> dict[str, list[SkillDefinition]]:
        """Group skills by category.

        Useful for building hierarchical classification schemas
        where categories map to domains.

        Returns:
            Dict mapping category names to skill lists.
        """
        groups: dict[str, list[SkillDefinition]] = {}
        for skill in skills:
            groups.setdefault(skill.category, []).append(skill)
        return groups

    @staticmethod
    def _load_from_file(
        file_path: Path,
        category_override: str | None = None,
    ) -> list[SkillDefinition]:
        """Load skills from a single Python file.

        .. warning::

           This method executes arbitrary Python code from *file_path* via
           ``importlib``.  Only load skill files from **trusted sources**.
        """
        if file_path.suffix != ".py":
            logger.warning(f"Skipping non-Python skill file: {file_path}")
            return []
        try:
            if file_path.stat().st_size > 1_000_000:
                logger.warning(f"Skipping oversized skill file (>1 MB): {file_path}")
                return []
        except OSError:
            return []

        module_name = f"_fsm_skill_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        skills: list[SkillDefinition] = []

        # Check for explicit SKILLS list
        if hasattr(module, "SKILLS"):
            for skill in module.SKILLS:
                if isinstance(skill, SkillDefinition):
                    if category_override:
                        skill.category = category_override
                    skills.append(skill)

        # Check for @tool decorated functions
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool_definition"):
                tool_def = obj._tool_definition
                skills.append(
                    SkillDefinition(
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

        return skills
