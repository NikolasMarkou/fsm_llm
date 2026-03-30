from __future__ import annotations

"""
Agent SOPs — Standard Operating Procedures.

Loads YAML/JSON SOP definitions and configures agents from them.
SOPs define reusable task configurations: prompt templates, required tools,
output schemas, and agent patterns.
"""

import json
from pathlib import Path
from typing import Any

from fsm_llm.logging import logger

from .definitions import AgentConfig

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class SOPDefinition:
    """A Standard Operating Procedure definition.

    Defines a reusable agent configuration template.

    Attributes:
        name: Unique SOP identifier.
        description: What this SOP does.
        agent_pattern: Agent pattern to use (e.g., "react", "plan_execute").
        task_template: Prompt template with {variable} placeholders.
        required_tools: List of tool names needed.
        output_schema: Optional Pydantic model name or JSON schema for structured output.
        config_overrides: AgentConfig field overrides.
        metadata: Additional metadata.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        agent_pattern: str = "react",
        task_template: str = "",
        required_tools: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.agent_pattern = agent_pattern
        self.task_template = task_template
        self.required_tools = required_tools or []
        self.output_schema = output_schema
        self.config_overrides = config_overrides or {}
        self.metadata = metadata or {}

    def render_task(self, **kwargs: Any) -> str:
        """Render the task template with provided variables.

        Args:
            **kwargs: Template variables to substitute.

        Returns:
            Rendered task string.
        """
        try:
            return self.task_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"SOP '{self.name}' task template requires variable {e}"
            ) from None

    def to_agent_config(self, **overrides: Any) -> AgentConfig:
        """Create an AgentConfig from SOP settings.

        Args:
            **overrides: Additional overrides applied after SOP defaults.
        """
        config_dict = {**self.config_overrides, **overrides}
        return AgentConfig(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent_pattern": self.agent_pattern,
            "task_template": self.task_template,
            "required_tools": self.required_tools,
            "output_schema": self.output_schema,
            "config_overrides": self.config_overrides,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SOPDefinition:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            agent_pattern=data.get("agent_pattern", "react"),
            task_template=data.get("task_template", ""),
            required_tools=data.get("required_tools", []),
            output_schema=data.get("output_schema"),
            config_overrides=data.get("config_overrides", {}),
            metadata=data.get("metadata", {}),
        )


class SOPRegistry:
    """Registry for managing Standard Operating Procedures.

    Example::

        from fsm_llm_agents.sop import SOPRegistry

        registry = SOPRegistry()
        registry.register_directory("./sops")
        sop = registry.get("code-review")
        task = sop.render_task(code="def foo(): pass", language="python")
    """

    def __init__(self) -> None:
        self._sops: dict[str, SOPDefinition] = {}

    def register(self, sop: SOPDefinition) -> SOPRegistry:
        """Register an SOP. Returns self for chaining.

        Validates config_overrides by attempting AgentConfig construction.
        """
        if sop.config_overrides:
            try:
                AgentConfig(**sop.config_overrides)
            except Exception as e:
                raise ValueError(
                    f"SOP '{sop.name}' has invalid config_overrides: {e}"
                ) from e
        self._sops[sop.name] = sop
        logger.debug(f"Registered SOP: {sop.name}")
        return self

    def register_from_dict(self, data: dict[str, Any]) -> SOPRegistry:
        """Register an SOP from a dictionary."""
        return self.register(SOPDefinition.from_dict(data))

    def register_from_file(self, path: str | Path) -> SOPRegistry:
        """Register an SOP from a JSON or YAML file.

        Args:
            path: Path to the SOP definition file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SOP file not found: {path}")

        data = self._load_file(path)
        return self.register_from_dict(data)

    def register_directory(self, directory: str | Path) -> int:
        """Register all SOP files from a directory.

        Args:
            directory: Path to directory containing SOP files.

        Returns:
            Number of SOPs registered.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        count = 0
        for path in sorted(directory.iterdir()):
            if path.suffix in (".json", ".yaml", ".yml"):
                try:
                    self.register_from_file(path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load SOP from {path}: {e}")
        return count

    def get(self, name: str) -> SOPDefinition:
        """Get an SOP by name.

        Raises:
            KeyError: If SOP not found.
        """
        if name not in self._sops:
            raise KeyError(
                f"SOP '{name}' not found. Available: {sorted(self._sops.keys())}"
            )
        return self._sops[name]

    def list_sops(self) -> list[SOPDefinition]:
        """List all registered SOPs."""
        return list(self._sops.values())

    def list_names(self) -> list[str]:
        """List all registered SOP names."""
        return sorted(self._sops.keys())

    def has(self, name: str) -> bool:
        """Check if an SOP is registered."""
        return name in self._sops

    def remove(self, name: str) -> bool:
        """Remove an SOP. Returns True if it existed."""
        return self._sops.pop(name, None) is not None

    def __len__(self) -> int:
        return len(self._sops)

    def __contains__(self, name: str) -> bool:
        return name in self._sops

    @staticmethod
    def _load_file(path: Path) -> dict[str, Any]:
        """Load a JSON or YAML file."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return json.loads(text)
        elif path.suffix in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise ImportError(
                    "YAML support requires PyYAML. Install with: pip install pyyaml"
                )
            return yaml.safe_load(text)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


# ---------------------------------------------------------------------------
# Built-in SOP templates
# ---------------------------------------------------------------------------

BUILTIN_SOPS: list[dict[str, Any]] = [
    {
        "name": "code-review",
        "description": "Review code for bugs, style issues, and improvements",
        "agent_pattern": "react",
        "task_template": (
            "Review the following {language} code for bugs, style issues, "
            "and potential improvements. Provide specific, actionable feedback.\n\n"
            "```{language}\n{code}\n```"
        ),
        "required_tools": [],
        "config_overrides": {"temperature": 0.3, "max_iterations": 5},
        "metadata": {"category": "development"},
    },
    {
        "name": "summarize",
        "description": "Summarize text into key points",
        "agent_pattern": "react",
        "task_template": (
            "Summarize the following text into {num_points} key points. "
            "Be concise and focus on the most important information.\n\n"
            "{text}"
        ),
        "required_tools": [],
        "config_overrides": {"temperature": 0.3, "max_iterations": 3},
        "metadata": {"category": "analysis"},
    },
    {
        "name": "data-extraction",
        "description": "Extract structured data from unstructured text",
        "agent_pattern": "react",
        "task_template": (
            "Extract the following fields from the text: {fields}\n\n"
            "Text:\n{text}\n\n"
            "Return the extracted data as a structured response."
        ),
        "required_tools": [],
        "config_overrides": {"temperature": 0.1, "max_iterations": 3},
        "metadata": {"category": "extraction"},
    },
]


def load_builtin_sops() -> SOPRegistry:
    """Create a registry pre-loaded with built-in SOPs."""
    registry = SOPRegistry()
    for sop_data in BUILTIN_SOPS:
        registry.register_from_dict(sop_data)
    return registry
