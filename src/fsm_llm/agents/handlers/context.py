"""``HandlerContext``: what one run hands to the handler factories of the library.

Interface contract (plan a555dcfe step 7, D-013):

- ``HandlerContext(tools=None, config=None, hitl=None, hitl_active=False, approval_predicate=None,
  options=(), callables=(), scope=<fresh dict>)`` is a frozen value built ONCE PER RUN by the agent
  that runs a pattern. ``tools`` is the run's ``ToolRegistry`` (or ``None``), ``config`` its
  ``AgentConfig``, ``hitl`` the agent's own ``HumanInTheLoop`` (a file never builds one),
  ``hitl_active`` and ``approval_predicate`` the two halves of the ONE predicate (the agent's
  ``_hitl_active`` and ``_approval_predicate``): the predicate is set exactly when the run is
  HITL-active, anything else raises ``AgentError`` at construction. ``options`` and ``callables`` are
  read-only copies (resolved option values; named callables a factory may call), ``scope`` is the run's
  mutable bag (the shared ``AgentHandlers`` of a run lives in it).
- ``next_run()`` returns the same context with a new empty ``scope``: two runs never share one.
- A factory reads a context and never stores it, or anything built from it, on a registry or on
  ``self``: the registry holds factories, a run holds its handlers (D-014 of plan 089d0ec7).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fsm_llm.core.definitions import AgentError

if TYPE_CHECKING:
    from fsm_llm.agents.definitions import AgentConfig
    from fsm_llm.policy import ApprovalPolicy, HumanInTheLoop
    from fsm_llm.tools import ToolRegistry

__all__ = ["HandlerContext"]


@dataclass(frozen=True)
class HandlerContext:
    """The per-run inputs of the handler factories; see the module docstring."""

    tools: ToolRegistry | None = None
    config: AgentConfig | None = None
    hitl: HumanInTheLoop | None = None
    hitl_active: bool = False
    approval_predicate: ApprovalPolicy | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    callables: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.approval_predicate is not None) != bool(self.hitl_active):
            raise AgentError(
                "HandlerContext: approval_predicate must be set exactly when hitl_active "
                f"is True (got hitl_active={self.hitl_active!r}, "
                f"approval_predicate={'set' if self.approval_predicate else 'None'}): the "
                "await_approval state, the gate and the refusal share one predicate"
            )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(self, "callables", MappingProxyType(dict(self.callables)))

    def next_run(self) -> HandlerContext:
        """The same context for another run: a new empty ``scope``, nothing shared."""
        return replace(self, scope={})
