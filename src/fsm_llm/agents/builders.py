"""``ConfiguredAgentBuilder``: a fluent front for ``create_agent``.

Follows the shared builder convention: ``set_<x>`` records one value,
``add_<x>`` appends, mutators only record and return ``self``, and ``build()``
is the only validator. ``build()`` calls ``create_agent`` unchanged with only
the values that were set, so every default and rule lives there.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import BuildError

from . import create_agent
from .definitions import AgentConfig, ToolDefinition
from .exceptions import AgentError
from .tools import ToolRegistry


class ConfiguredAgentBuilder:
    """Builds an agent of any ``create_agent`` pattern.

    Interface contract: every ``set_*`` / ``add_*`` returns ``self``;
    ``build()`` returns a new agent or raises ``BuildError`` (chained from the
    ``ValueError``/``TypeError``/``ValidationError``/``AgentError`` raised
    below it) with ``.errors``. The builder's own rule: ``add_tool`` together
    with ``set_tool_registry`` is an error. No pattern set means
    ``create_agent``'s default.

    Typed setters (``set_model``, ``set_temperature``, ``set_max_tokens``,
    ``set_max_iterations``, ``set_timeout_seconds``) and ``set_config_option``
    write ``AgentConfig`` fields; ``set_option(name, value)`` passes any other
    ``create_agent`` keyword (pattern parameters, API passthrough such as
    ``seed``, ``handlers``, ``llm_interface``, ``hitl``) unfiltered.

    Isolation: the ``AgentConfig``, the option dict and the tool list are
    copied at ``build()``, and ``add_tool`` entries go into a fresh
    ``ToolRegistry`` each build. A registry from ``set_tool_registry``, the
    tools themselves, and objects passed as options stay shared by reference.
    """

    def __init__(self) -> None:
        self._pattern: str | None = None
        self._tools: list[Any] = []
        self._registry: ToolRegistry | None = None
        self._config: AgentConfig | None = None
        self._config_options: dict[str, Any] = {}
        self._system_prompt: str | None = None
        self._options: dict[str, Any] = {}

    def set_pattern(self, pattern: str) -> ConfiguredAgentBuilder:
        self._pattern = pattern
        return self

    def add_tool(self, tool: Any | ToolDefinition) -> ConfiguredAgentBuilder:
        """Append a ``@tool`` function, plain function or ``ToolDefinition``."""
        self._tools.append(tool)
        return self

    def set_tool_registry(self, registry: ToolRegistry) -> ConfiguredAgentBuilder:
        self._registry = registry
        return self

    def set_config(self, config: AgentConfig) -> ConfiguredAgentBuilder:
        self._config = config
        return self

    def set_config_option(self, name: str, value: Any) -> ConfiguredAgentBuilder:
        """Record one ``AgentConfig`` field; ``AgentConfig`` validates at build."""
        self._config_options[name] = value
        return self

    def set_model(self, model: str) -> ConfiguredAgentBuilder:
        return self.set_config_option("model", model)

    def set_temperature(self, temperature: float) -> ConfiguredAgentBuilder:
        return self.set_config_option("temperature", temperature)

    def set_max_tokens(self, max_tokens: int) -> ConfiguredAgentBuilder:
        return self.set_config_option("max_tokens", max_tokens)

    def set_max_iterations(self, max_iterations: int) -> ConfiguredAgentBuilder:
        return self.set_config_option("max_iterations", max_iterations)

    def set_timeout_seconds(self, seconds: float) -> ConfiguredAgentBuilder:
        return self.set_config_option("timeout_seconds", seconds)

    def set_system_prompt(self, system_prompt: str) -> ConfiguredAgentBuilder:
        self._system_prompt = system_prompt
        return self

    def set_hitl(self, hitl: Any) -> ConfiguredAgentBuilder:
        return self.set_option("hitl", hitl)

    def set_option(self, name: str, value: Any) -> ConfiguredAgentBuilder:
        """Record one ``create_agent`` keyword; repeatable, never filtered."""
        self._options[name] = value
        return self

    def build(self) -> Any:
        if self._tools and self._registry is not None:
            raise BuildError(
                "Cannot build agent: add_tool() and set_tool_registry() are "
                "mutually exclusive"
            )
        try:
            args: dict[str, Any] = {}
            config = self._config_for_build()
            if config is not None:
                args["config"] = config
            if self._system_prompt is not None:
                args["system_prompt"] = self._system_prompt
            if self._pattern is not None:
                args["pattern"] = self._pattern
            if self._registry is not None:
                args["tools"] = self._registry
            elif self._tools:
                registry = ToolRegistry()
                for fn in self._tools:
                    if isinstance(fn, ToolDefinition):
                        registry.register(fn)
                    elif hasattr(fn, "_tool_definition"):
                        registry.register(fn._tool_definition)
                    else:
                        registry.register_function(fn)
                args["tools"] = registry
            # DECISION plan-2026-10-02T052921-89b03f61/D-008
            # Options go to create_agent exactly as set. Do NOT filter or
            # rename them here: the BaseAgent denylist (misplaced and
            # config-owned kwargs) decides, and its TypeError is wrapped.
            return create_agent(**args, **dict(self._options))
        except (ValueError, TypeError, AgentError) as exc:
            raise BuildError(f"Cannot build agent: {exc}", errors=[str(exc)]) from exc

    def _config_for_build(self) -> AgentConfig | None:
        """A private ``AgentConfig`` (set config overlaid by the options), or ``None``."""
        if self._config is None and not self._config_options:
            return None
        base = self._config.model_copy(deep=True) if self._config else AgentConfig()
        fields = {name: getattr(base, name) for name in type(base).model_fields}
        return type(base)(**{**fields, **self._config_options})
