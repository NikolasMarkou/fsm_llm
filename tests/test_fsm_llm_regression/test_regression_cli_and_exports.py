"""
Regression tests for Plan 13 fixes.
Ensures professionalization and operational fixes remain in place.
"""

import os
import pytest


# ----------------------------------------------------------------
# H-1: CLI entry points must have main_cli()
# ----------------------------------------------------------------

class TestCLIEntryPoints:
    """Verify CLI entry points resolve to valid functions."""

    def test_visualizer_has_main_cli(self):
        """H-1: fsm-llm-visualize entry point requires main_cli."""
        from fsm_llm.visualizer import main_cli
        assert callable(main_cli)

    def test_validator_has_main_cli(self):
        """H-1: fsm-llm-validate entry point requires main_cli."""
        from fsm_llm.validator import main_cli
        assert callable(main_cli)

    def test_main_module_has_main_cli(self):
        """Baseline: fsm-llm entry point requires main_cli."""
        from fsm_llm.__main__ import main_cli
        assert callable(main_cli)

    def test_visualizer_main_still_exists(self):
        """Ensure adding main_cli didn't break existing main()."""
        from fsm_llm.visualizer import main
        assert callable(main)

    def test_validator_main_still_exists(self):
        """Ensure adding main_cli didn't break existing main()."""
        from fsm_llm.validator import main
        assert callable(main)


# ----------------------------------------------------------------
# H-2: ContextMergeStrategy must be exported
# ----------------------------------------------------------------

class TestContextMergeStrategyExport:
    """Verify ContextMergeStrategy is importable from top-level."""

    def test_import_from_fsm_llm(self):
        """H-2: Must be importable from fsm_llm."""
        from fsm_llm import ContextMergeStrategy
        assert ContextMergeStrategy is not None

    def test_in_all(self):
        """H-2: Must be in __all__."""
        import fsm_llm
        assert "ContextMergeStrategy" in fsm_llm.__all__

    def test_enum_values(self):
        """H-2: Enum must have expected values."""
        from fsm_llm import ContextMergeStrategy
        assert hasattr(ContextMergeStrategy, "UPDATE")
        assert hasattr(ContextMergeStrategy, "PRESERVE")


# ----------------------------------------------------------------
# H-3: Phantom workflow CLI entry point removed
# ----------------------------------------------------------------

class TestPhantomEntryPointRemoved:
    """Verify phantom workflow CLI config is gone."""

    def test_no_workflow_cli_module(self):
        """H-3: fsm_llm_workflows.cli should not exist."""
        with pytest.raises(ModuleNotFoundError):
            import fsm_llm_workflows.cli  # noqa: F401


# ----------------------------------------------------------------
# M-1: README Python version badge
# ----------------------------------------------------------------

class TestReadmeBadge:
    """Verify README has correct Python version badge."""

    def test_no_python_38_39_in_badge(self):
        """M-1: Badge should not reference Python 3.8 or 3.9."""
        readme_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "README.md"
        )
        with open(readme_path) as f:
            content = f.read()
        # Check badge line specifically
        for line in content.split("\n"):
            if "img.shields.io/badge/python" in line:
                assert "3.8" not in line, "Badge still references Python 3.8"
                assert "3.9" not in line, "Badge still references Python 3.9"
                assert "3.10" in line, "Badge missing Python 3.10"
                break


# ----------------------------------------------------------------
# M-2: quickstart.md references
# ----------------------------------------------------------------

class TestQuickstartReferences:
    """Verify quickstart.md has correct example references."""

    def test_no_stale_references(self):
        """M-2: No references to non-existent examples or main.py."""
        qs_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "quickstart.md"
        )
        with open(qs_path) as f:
            content = f.read()
        assert "examples/basic/quiz" not in content
        assert "examples/intermediate/customer_service" not in content
        assert "python main.py" not in content


# ----------------------------------------------------------------
# M-3: py.typed files exist
# ----------------------------------------------------------------

class TestPyTypedFiles:
    """Verify PEP 561 py.typed marker files exist."""

    @pytest.mark.parametrize("package", [
        "fsm_llm", "fsm_llm_reasoning", "fsm_llm_workflows"
    ])
    def test_py_typed_exists(self, package):
        """M-3: py.typed must exist in each package."""
        src_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", package
        )
        py_typed = os.path.join(src_dir, "py.typed")
        assert os.path.exists(py_typed), f"Missing py.typed in {package}"


# ----------------------------------------------------------------
# M-5: No emoji in README
# ----------------------------------------------------------------

class TestReadmeNoEmoji:
    """Verify README has no emoji characters."""

    def test_no_emoji_in_headings(self):
        """M-5: Section headings should not contain emoji."""
        readme_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "README.md"
        )
        with open(readme_path) as f:
            content = f.read()
        for line in content.split("\n"):
            if line.startswith("#"):
                # Check for common emoji ranges
                for char in line:
                    code = ord(char)
                    assert not (0x1F300 <= code <= 0x1FFFF), \
                        f"Emoji found in heading: {line.strip()}"


# ----------------------------------------------------------------
# L-3: Unified versioning
# ----------------------------------------------------------------

class TestUnifiedVersioning:
    """Verify all packages share the same version."""

    def test_reasoning_version_matches(self):
        """L-3: Reasoning extension version must match main package."""
        from fsm_llm.__version__ import __version__ as main_version
        from fsm_llm_reasoning.__version__ import __version__ as reasoning_version
        assert reasoning_version == main_version

    def test_workflows_version_matches(self):
        """Baseline: Workflows extension version must match main package."""
        from fsm_llm.__version__ import __version__ as main_version
        from fsm_llm_workflows import __version__ as workflows_version
        assert workflows_version == main_version
