"""``AgentArtifactBuilder``: incrementally builds an agent configuration (plan d454fd62 step 89; the
class body is the old ``agents/meta_builders.py`` one). The ``MetaDefaults.AGENT_MODEL`` template value
it seeds the config with and the anchor that says not to switch it stay in this one module."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import ValidationError

from fsm_llm.agents.constants import MetaDefaults
from fsm_llm.agents.definitions import (
    PLACEHOLDER_MODEL,
    AgentConfig,
    ArtifactType,
)
from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.core.definitions import ModelNotConfiguredError

from .base import ArtifactBuilder


class AgentArtifactBuilder(ArtifactBuilder):
    """Incrementally builds an agent configuration."""

    VALID_AGENT_TYPES: ClassVar[set[str]] = {
        "react",
        "plan_execute",
        "reflexion",
        "rewoo",
        "evaluator_optimizer",
        "maker_checker",
        "prompt_chain",
        "self_consistency",
        "debate",
        "orchestrator",
        "adapt",
    }

    def __init__(self, agent_types: Iterable[str] = ()) -> None:
        super().__init__()
        # the built-in set plus the ids of ``--executables DIR`` (instance state, never the class set)
        self.valid_agent_types = frozenset({*self.VALID_AGENT_TYPES, *agent_types})
        self.agent_type: str | None = None
        self.name: str | None = None
        self.description: str | None = None
        self.config: dict[str, Any] = {
            "model": MetaDefaults.AGENT_MODEL,
            "max_iterations": MetaDefaults.AGENT_MAX_ITERATIONS,
            "timeout_seconds": MetaDefaults.AGENT_TIMEOUT_SECONDS,
            "temperature": MetaDefaults.AGENT_TEMPERATURE,
            "max_tokens": MetaDefaults.AGENT_MAX_TOKENS,
        }
        self.tools: list[dict[str, Any]] = []

    @property
    def artifact_type(self) -> ArtifactType:
        return ArtifactType.AGENT

    # -- Mutation methods ---------------------------------------------------

    # Common LLM misspellings/variations → canonical type
    _AGENT_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "react_agent": "react",
        "reactagent": "react",
        "plan_and_execute": "plan_execute",
        "plan_execute_agent": "plan_execute",
        "planexecute": "plan_execute",
        "self_consistent": "self_consistency",
        "eval_opt": "evaluator_optimizer",
        "eval_optimize": "evaluator_optimizer",
        "maker_check": "maker_checker",
        "reflexion_agent": "reflexion",
        "debate_agent": "debate",
        "orchestrate": "orchestrator",
        "adapt_agent": "adapt",
        "prompt_chaining": "prompt_chain",
        "rewoo_agent": "rewoo",
    }

    def set_agent_type(self, agent_type: str) -> AgentArtifactBuilder:
        """Set the agent pattern type."""
        agent_type = agent_type.strip().lower()
        # Resolve common LLM variations
        agent_type = self._AGENT_TYPE_ALIASES.get(agent_type, agent_type)
        if agent_type not in self.valid_agent_types:
            raise BuilderError(
                f"Unknown agent type '{agent_type}'. "
                f"Valid: {', '.join(sorted(self.valid_agent_types))}",
                action="set_agent_type",
            )
        self.agent_type = agent_type
        return self

    def set_overview(
        self,
        name: str,
        description: str,
    ) -> AgentArtifactBuilder:
        """Set basic agent metadata."""
        self.name = name.strip()
        self.description = description.strip()
        return self

    # Allowed config fields and their expected types
    _CONFIG_ALLOWED: ClassVar[set[str]] = {
        "model",
        "max_iterations",
        "timeout_seconds",
        "temperature",
        "max_tokens",
    }
    _CONFIG_NUMERIC: ClassVar[set[str]] = {"timeout_seconds", "temperature"}
    _CONFIG_INT: ClassVar[set[str]] = {"max_iterations", "max_tokens"}

    @classmethod
    def _config_problems(cls, config: dict[str, Any]) -> dict[str, str]:
        """Map each invalid config key to a reason; an empty dict means valid.

        Contract: ``config`` is the whole artifact config dict. The rules come
        from ``AgentConfig``'s own validators (one source: ranges, finite
        floats, no bool, non-blank ``model``, an int field at most
        ``Defaults.MAX_INT_LIMIT`` so the artifact is always JSON-serialisable:
        ``json.dumps`` raises on an int of more than 4300 digits). Never
        raises; a non-validation error from pydantic is not expected.
        """
        # DECISION plan-2026-11-06T093029-0068b055/D-011: validate through
        # ``AgentConfig`` (the agent this artifact describes), NOT through
        # ``MetaBuilderConfig``: that class refuses fields the meta-builder
        # never reads (step 9) and would reject keys that are valid data about
        # the generated agent. Do NOT copy the ranges here. Do NOT switch
        # ``MetaDefaults.AGENT_MODEL`` in this change: it is a fixed template
        # value, not the runtime default (``resolve_agent_model()``), and the
        # change is carried for the owner.
        # DECISION plan-2026-11-12T073150-d454fd62/D-005: a config without a
        # ``model`` and no ``LLM_MODEL`` makes ``AgentConfig``'s default
        # factory raise ``ModelNotConfiguredError``, not a ``ValidationError``.
        # Do NOT let it escape (this method never raises) and do NOT drop the
        # other keys' problems: report ``model`` and validate the rest with a
        # placeholder model.
        problems: dict[str, str] = {}
        try:
            AgentConfig.model_validate(config)
        except ModelNotConfiguredError as exc:
            problems["model"] = str(exc)
            for key, reason in cls._config_problems(
                {**config, "model": PLACEHOLDER_MODEL}
            ).items():
                problems.setdefault(key, reason)
        except ValidationError as exc:
            for err in exc.errors():
                loc = err["loc"]
                key = str(loc[0]) if loc else "?"
                reason = err["msg"].removeprefix("Value error, ")
                problems.setdefault(key, reason)
        return problems

    def set_config(self, **kwargs: Any) -> AgentArtifactBuilder:
        """Update agent config fields.

        A value of the wrong type or outside the ``AgentConfig`` range logs a
        warning and is skipped (the config keeps its previous value). The
        default ``model`` is a fixed template value, not the runtime default.
        """
        warnings: list[str] = []
        for key, value in kwargs.items():
            if key not in self._CONFIG_ALLOWED:
                warnings.append(f"Ignoring unknown config field '{key}'")
                continue
            if key == "model":
                if not isinstance(value, str):
                    warnings.append(
                        f"Config 'model' expects str, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            elif key in self._CONFIG_INT:
                if not isinstance(value, int):
                    warnings.append(
                        f"Config '{key}' expects int, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            elif key in self._CONFIG_NUMERIC:
                if not isinstance(value, int | float):
                    warnings.append(
                        f"Config '{key}' expects number, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            reason = self._config_problems({**self.config, key: value}).get(key)
            if reason is not None:
                warnings.append(f"Config '{key}' rejected ({reason}); skipping")
                continue
            self.config[key] = value
        self._warnings.extend(warnings)
        return self

    def add_tool(
        self,
        name: str,
        description: str,
        parameter_schema: dict[str, Any] | None = None,
    ) -> AgentArtifactBuilder:
        """Add a tool definition."""
        warnings: list[str] = []
        name = name.strip()
        if not name:
            raise BuilderError("Tool name cannot be empty", action="add_tool")

        # Check for duplicates
        for t in self.tools:
            if t["name"] == name:
                warnings.append(f"Tool '{name}' already exists; overwriting")
                self.tools = [t for t in self.tools if t["name"] != name]
                break

        tool_def: dict[str, Any] = {
            "name": name,
            "description": description.strip(),
        }
        if parameter_schema:
            tool_def["parameter_schema"] = parameter_schema
        self.tools.append(tool_def)
        self._warnings.extend(warnings)
        return self

    def remove_tool(self, name: str) -> AgentArtifactBuilder:
        """Remove a tool. A missing item is a no-op; returns self."""
        self.tools = [t for t in self.tools if t["name"] != name]
        return self

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce agent configuration dict."""
        return {
            "name": self.name or "unnamed_agent",
            "description": self.description or "",
            "agent_type": self.agent_type or "",
            "config": dict(self.config),
            "tools": list(self.tools),
        }

    # -- Validation ---------------------------------------------------------

    def validate_partial(self) -> list[str]:
        warnings: list[str] = []
        if not self.agent_type:
            warnings.append("Agent type not set")
        if not self.tools:
            warnings.append("No tools defined yet")
        return warnings

    def validate_complete(self) -> list[str]:
        """Return structural errors; an empty list means a complete *spec*.

        An agent artifact is not directly loadable: each tool needs an
        ``execute_fn`` that a JSON spec cannot carry, so a caller still wires
        the tools by hand. This checks the agent type, the name, that every
        tool has a name and a description, and that ``config`` passes
        ``AgentConfig`` (``set_config`` already refuses a bad value; this
        catches a dict edited directly).
        """
        errors: list[str] = []
        if not self.agent_type:
            errors.append("Agent type is required")
        if not self.name:
            errors.append("Agent name is required")
        if not self.tools:
            errors.append("At least one tool is required")
        for t in self.tools:
            if not t.get("name"):
                errors.append("Tool name is required")
            if not t.get("description"):
                errors.append(f"Tool '{t.get('name', '?')}' needs a description")
        for key, reason in self._config_problems(self.config).items():
            errors.append(f"Config '{key}' is invalid: {reason}")
        return errors

    def get_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.agent_type:
            missing.append("Agent type")
        if not self.name:
            missing.append("Agent name")
        if not self.tools:
            missing.append("At least one tool")
        return missing

    _TOTAL_REQUIRED: ClassVar[int] = 3  # agent type, name, a tool

    def _get_minimal_summary(self) -> str:
        parts: list[str] = [
            "=== Agent Builder Status ===",
            f"Name: {self.name or '(not set)'} | Type: {self.agent_type or '(not set)'}",
        ]
        if self.tools:
            tool_names = [t["name"] for t in self.tools]
            parts.append(f"Tools ({len(self.tools)}): {', '.join(tool_names)}")
        else:
            parts.append("Tools: none yet")

        return "\n".join([*parts, *self._missing_line()])

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== Agent Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Agent type: {self.agent_type or '(not set)'}",
            f"Tools ({len(self.tools)}):",
        ]
        for t in self.tools:
            parts.append(
                f"  - {t['name']}: {t['description'][: MetaDefaults.SUMMARY_LINE_WIDTH]}"
            )

        return "\n".join([*parts, *self._missing_line()])

    def _get_full_summary(self) -> str:
        parts: list[str] = []
        parts.append("=== Agent Builder Status ===")
        parts.append(f"Name: {self.name or '(not set)'}")
        parts.append(f"Description: {self.description or '(not set)'}")
        parts.append(f"Agent type: {self.agent_type or '(not set)'}")
        parts.append(
            f"Config: model={self.config.get('model')}, "
            f"max_iter={self.config.get('max_iterations')}, "
            f"temp={self.config.get('temperature')}"
        )
        parts.append(f"Tools ({len(self.tools)}):")
        for t in self.tools:
            schema_str = ""
            if t.get("parameter_schema"):
                params = t["parameter_schema"].get("properties", {})
                schema_str = f" (params: {', '.join(params.keys())})" if params else ""
            parts.append(f"  - {t['name']}: {t['description']}{schema_str}")

        return "\n".join([*parts, *self._still_missing_lines()])
