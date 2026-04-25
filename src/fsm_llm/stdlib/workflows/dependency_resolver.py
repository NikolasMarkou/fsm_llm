from __future__ import annotations

"""
Dependency-Based Workflow Parallelism.

Computes execution waves from declared step dependencies using topological
sort. Steps within the same wave have no dependencies on each other and
can execute in parallel.
"""

from collections import defaultdict, deque

from fsm_llm.logging import logger

from .exceptions import WorkflowValidationError


class DependencyResolver:
    """Resolves step dependencies into parallel execution waves.

    Steps are organized into waves where all steps within a wave are
    independent and can execute concurrently. Waves execute sequentially.

    Example::

        resolver = DependencyResolver()
        resolver.add_step("fetch_data")
        resolver.add_step("fetch_config")
        resolver.add_step("process", depends_on=["fetch_data", "fetch_config"])
        resolver.add_step("notify", depends_on=["process"])

        waves = resolver.resolve()
        # waves = [["fetch_data", "fetch_config"], ["process"], ["notify"]]
    """

    def __init__(self) -> None:
        self._steps: set[str] = set()
        self._dependencies: dict[str, set[str]] = defaultdict(set)
        self._dependents: dict[str, set[str]] = defaultdict(set)

    def add_step(
        self,
        step_id: str,
        depends_on: list[str] | None = None,
    ) -> DependencyResolver:
        """Register a step with optional dependencies.

        Args:
            step_id: Unique identifier for the step.
            depends_on: List of step IDs this step depends on.

        Returns:
            Self for chaining.
        """
        self._steps.add(step_id)
        if depends_on:
            for dep in depends_on:
                self._dependencies[step_id].add(dep)
                self._dependents[dep].add(step_id)
        return self

    def add_dependency(self, step_id: str, depends_on: str) -> DependencyResolver:
        """Add a single dependency for a step.

        Args:
            step_id: The dependent step.
            depends_on: The step it depends on.

        Returns:
            Self for chaining.
        """
        self._steps.add(step_id)
        self._steps.add(depends_on)
        self._dependencies[step_id].add(depends_on)
        self._dependents[depends_on].add(step_id)
        return self

    def resolve(self) -> list[list[str]]:
        """Compute execution waves using Kahn's algorithm (topological sort).

        Returns:
            List of waves, where each wave is a list of step IDs that
            can execute in parallel.

        Raises:
            WorkflowValidationError: If a cycle is detected.
        """
        # Validate all dependencies reference known steps
        for step_id, deps in self._dependencies.items():
            unknown = deps - self._steps
            if unknown:
                raise WorkflowValidationError(
                    [f"Step '{step_id}' depends on unknown steps: {sorted(unknown)}"]
                )

        # Kahn's algorithm
        in_degree: dict[str, int] = {s: 0 for s in self._steps}
        for step_id, deps in self._dependencies.items():
            in_degree[step_id] = len(deps)

        # Find all steps with no dependencies
        queue: deque[str] = deque(s for s in self._steps if in_degree[s] == 0)

        waves: list[list[str]] = []
        processed = 0

        while queue:
            # Current wave: all steps with in_degree 0
            wave = sorted(queue)  # Sorted for deterministic output
            waves.append(wave)
            queue.clear()

            for step_id in wave:
                processed += 1
                for dependent in sorted(self._dependents.get(step_id, [])):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if processed < len(self._steps):
            # Cycle detected — find the cycle members
            remaining = [s for s in self._steps if in_degree[s] > 0]
            raise WorkflowValidationError(
                [f"Dependency cycle detected involving steps: {sorted(remaining)}"]
            )

        logger.debug(f"Resolved {len(self._steps)} steps into {len(waves)} waves")
        return waves

    def has_cycles(self) -> bool:
        """Check if the dependency graph contains cycles."""
        try:
            self.resolve()
            return False
        except WorkflowValidationError:
            return True

    def get_dependencies(self, step_id: str) -> set[str]:
        """Get direct dependencies for a step."""
        return set(self._dependencies.get(step_id, set()))

    def get_dependents(self, step_id: str) -> set[str]:
        """Get steps that depend on the given step."""
        return set(self._dependents.get(step_id, set()))

    def get_all_steps(self) -> set[str]:
        """Get all registered step IDs."""
        return set(self._steps)

    @property
    def step_count(self) -> int:
        """Return the number of registered steps."""
        return len(self._steps)

    @property
    def dependency_count(self) -> int:
        """Return the total number of dependency edges."""
        return sum(len(deps) for deps in self._dependencies.values())

    def clear(self) -> None:
        """Remove all steps and dependencies."""
        self._steps.clear()
        self._dependencies.clear()
        self._dependents.clear()

    @classmethod
    def from_dict(
        cls,
        dependencies: dict[str, list[str]],
    ) -> DependencyResolver:
        """Create a resolver from a dependency dictionary.

        Args:
            dependencies: Mapping of step_id to list of dependencies.
                Steps with no dependencies should map to an empty list.

        Example::

            resolver = DependencyResolver.from_dict({
                "fetch": [],
                "process": ["fetch"],
                "notify": ["process"],
            })
        """
        resolver = cls()
        for step_id, deps in dependencies.items():
            resolver.add_step(step_id, depends_on=deps if deps else None)
        return resolver
