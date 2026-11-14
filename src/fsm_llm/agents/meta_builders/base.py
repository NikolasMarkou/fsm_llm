"""The abstract ``ArtifactBuilder`` (plan d454fd62 step 89; the class body is the old
``agents/meta_builders.py`` one).

Every artifact builder keeps partial state and provides:
- Mutation methods (add/remove/update)
- Incremental validation
- Progress tracking
- Summary generation for LLM context injection
- Final serialization to validated definitions
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from fsm_llm.agents.definitions import ArtifactType, BuildProgress
from fsm_llm.core.definitions import BuildError


class ArtifactBuilder(ABC):
    """Base class for incrementally building artifacts.

    Interface contract: every ``set_*``/``add_*``/``update_*``/``remove_*``
    mutator returns ``self`` (fluent) and records soft warnings, read once
    with ``take_warnings()``. A mutator raises ``BuilderError`` at call time
    only for an unusable argument (empty id, unknown target); this is the one
    carve-out from "validate in ``build()``", kept so the LLM tools give
    per-call feedback. ``build()`` is strict: ``BuildError`` unless
    ``validate_complete()`` is empty, else a deep copy of ``to_dict()``.
    ``MetaBuilderAgent`` uses the non-raising ``validate_complete()`` instead.
    Subclasses need not call ``super().__init__()``: the warnings list is lazy.
    """

    # DECISION plan-2026-10-02T105536-042b086f/D-012
    # Do NOT make this a class-level list (all instances would share it) and do NOT
    # drop the property to require super().__init__() (the tools must work for
    # subclasses that skip it). See decisions.md D-012.
    @property
    def _warnings(self) -> list[str]:
        # Lazy so a subclass that skips ``super().__init__()`` still works. Kept
        # per instance in ``__dict__`` (not a class-level list, which all
        # instances would share), so deepcopy and pickle see a plain list.
        warnings: list[str] = self.__dict__.setdefault("_warnings", [])
        return warnings

    @_warnings.setter
    def _warnings(self, value: list[str]) -> None:
        self.__dict__["_warnings"] = value

    def take_warnings(self) -> list[str]:
        """Return the warnings recorded since the last call, and clear them."""
        warnings, self._warnings = self._warnings, []
        return warnings

    def build(self) -> dict[str, Any]:
        """Return a deep copy of ``to_dict()``; ``BuildError`` if incomplete."""
        errors = self.validate_complete()
        if errors:
            raise BuildError(
                f"Incomplete {self.artifact_type.value}: {'; '.join(errors)}",
                errors=errors,
            )
        return copy.deepcopy(self.to_dict())

    @property
    @abstractmethod
    def artifact_type(self) -> ArtifactType: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def validate_partial(self) -> list[str]: ...

    @abstractmethod
    def validate_complete(self) -> list[str]: ...

    @abstractmethod
    def get_missing_fields(self) -> list[str]: ...

    # How many items ``get_missing_fields`` can name: the denominator of
    # ``get_progress`` (a subclass sets it).
    _TOTAL_REQUIRED: ClassVar[int]

    def get_progress(self) -> BuildProgress:
        """Progress: ``_TOTAL_REQUIRED`` items, one short per missing field."""
        missing = self.get_missing_fields()
        total = self._TOTAL_REQUIRED
        return BuildProgress(
            total_required=total,
            completed=total - min(len(missing), total),
            missing=missing,
            warnings=self.validate_partial(),
        )

    def get_summary(self, detail_level: str = "full") -> str:
        """Human-readable summary for LLM context injection.

        Args:
            detail_level: One of ``"minimal"`` (names + counts),
                ``"standard"`` (names + short descriptions + targets),
                or ``"full"`` (everything including instructions); any other
                value reads as ``"full"``.
        """
        if detail_level == "minimal":
            return self._get_minimal_summary()
        if detail_level == "standard":
            return self._get_standard_summary()
        return self._get_full_summary()

    @abstractmethod
    def _get_minimal_summary(self) -> str: ...

    @abstractmethod
    def _get_standard_summary(self) -> str: ...

    @abstractmethod
    def _get_full_summary(self) -> str: ...

    def _missing_line(self) -> list[str]:
        """The ``Missing: ...`` tail of the minimal and standard summaries."""
        missing = self.get_missing_fields()
        return [f"Missing: {', '.join(missing)}"] if missing else []

    def _still_missing_lines(self) -> list[str]:
        """The ``Still missing:`` tail of the full summary."""
        missing = self.get_missing_fields()
        if not missing:
            return []
        return ["\nStill missing:", *(f"  - {m}" for m in missing)]

    @staticmethod
    def _node_markers(node_id: str, node: dict[str, Any], initial: str | None) -> str:
        """``[INITIAL]`` / ``[TERMINAL]`` for a state or step of a graph."""
        marker = " [INITIAL]" if node_id == initial else ""
        return marker + (" [TERMINAL]" if not node["transitions"] else "")

    @classmethod
    def _graph_minimal_lines(
        cls,
        nodes: dict[str, dict[str, Any]],
        initial: str | None,
        noun: str,
        edge_noun: str,
    ) -> list[str]:
        """The node list and edge count of a graph's minimal summary."""
        if not nodes:
            return [f"{noun}: none yet"]
        labels = [nid + cls._node_markers(nid, n, initial) for nid, n in nodes.items()]
        edges = sum(len(n["transitions"]) for n in nodes.values())
        return [
            f"{noun} ({len(nodes)}): {', '.join(labels)}",
            f"{edge_noun}: {edges} defined",
        ]
