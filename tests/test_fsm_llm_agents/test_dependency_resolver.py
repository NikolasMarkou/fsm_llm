"""DependencyResolver of ``fsm_llm.workflows`` (a workflows class, tested from the agents suite).

It stays in this suite because a move would change the pinned (suite, class::test) ids.
"""

from __future__ import annotations

import pytest


class TestDependencyResolver:
    """Tests for DependencyResolver."""

    def test_simple_linear_deps(self):
        """Linear A -> B -> C produces 3 waves."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.add_step("b", depends_on=["a"])
        resolver.add_step("c", depends_on=["b"])

        waves = resolver.resolve()
        assert waves == [["a"], ["b"], ["c"]]

    def test_parallel_steps(self):
        """Independent steps are in the same wave."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.add_step("b")
        resolver.add_step("c", depends_on=["a", "b"])

        waves = resolver.resolve()
        assert waves[0] == ["a", "b"]  # Parallel
        assert waves[1] == ["c"]

    def test_diamond_dependency(self):
        """Diamond: A -> B,C -> D."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver.from_dict(
            {
                "a": [],
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            }
        )
        waves = resolver.resolve()
        assert waves[0] == ["a"]
        assert waves[1] == ["b", "c"]
        assert waves[2] == ["d"]

    def test_cycle_detection(self):
        """Cycles raise WorkflowValidationError."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver
        from fsm_llm.workflows.exceptions import WorkflowValidationError

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["b"])
        resolver.add_step("b", depends_on=["a"])

        with pytest.raises(WorkflowValidationError):
            resolver.resolve()

    def test_has_cycles(self):
        """has_cycles returns True for cyclic graph."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["b"])
        resolver.add_step("b", depends_on=["a"])
        assert resolver.has_cycles()

    def test_unknown_dependency_raises(self):
        """Dependency on non-existent step raises error."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver
        from fsm_llm.workflows.exceptions import WorkflowValidationError

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["nonexistent"])

        with pytest.raises(WorkflowValidationError, match="unknown"):
            resolver.resolve()

    def test_from_dict(self):
        """from_dict creates resolver correctly."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver.from_dict(
            {
                "fetch": [],
                "process": ["fetch"],
            }
        )
        assert resolver.step_count == 2
        assert resolver.resolve() == [["fetch"], ["process"]]

    def test_clear(self):
        """clear() removes all state."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.clear()
        assert resolver.step_count == 0

    def test_single_step(self):
        """Single step with no deps produces one wave."""
        from fsm_llm.workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("only")
        waves = resolver.resolve()
        assert waves == [["only"]]
