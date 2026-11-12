"""
Regression tests for Plan 13 fixes.
Ensures professionalization and operational fixes remain in place.
"""

from __future__ import annotations

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
        """H-3: fsm_llm.workflows.cli should not exist."""
        with pytest.raises(ModuleNotFoundError):
            import fsm_llm.workflows.cli  # noqa: F401


# ----------------------------------------------------------------
# M-1: README Python version badge
# ----------------------------------------------------------------


class TestReadmeBadge:
    """Verify README has correct Python version badge."""

    def test_no_python_38_39_in_badge(self):
        """M-1: Badge should not reference Python 3.8 or 3.9."""
        readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
        with open(readme_path, encoding="utf-8") as f:
            content = f.read()
        # Check badge line specifically
        for line in content.split("\n"):
            if "img.shields.io/badge/python" in line:
                assert "3.8" not in line, "Badge still references Python 3.8"
                assert "3.9" not in line, "Badge still references Python 3.9"
                assert "3.10" not in line, "Badge still references Python 3.10"
                assert "3.11" in line, "Badge missing Python 3.11"
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
        with open(qs_path, encoding="utf-8") as f:
            content = f.read()
        assert "examples/basic/quiz" not in content
        assert "examples/intermediate/customer_service" not in content
        assert "python main.py" not in content


# ----------------------------------------------------------------
# M-3: py.typed files exist
# ----------------------------------------------------------------


class TestPyTypedFiles:
    """Verify PEP 561 py.typed marker files exist."""

    # D-010: one top-level marker covers every subpackage (PEP 561).
    @pytest.mark.parametrize("package", ["fsm_llm"])
    def test_py_typed_exists(self, package):
        """M-3: py.typed must exist in the top-level package."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        py_typed = os.path.join(src_dir, "py.typed")
        assert os.path.exists(py_typed), f"Missing py.typed in {package}"


# ----------------------------------------------------------------
# M-5: No emoji in README
# ----------------------------------------------------------------


class TestReadmeNoEmoji:
    """Verify README has no emoji characters."""

    def test_no_emoji_in_headings(self):
        """M-5: Section headings should not contain emoji."""
        readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
        with open(readme_path, encoding="utf-8") as f:
            content = f.read()
        for line in content.split("\n"):
            if line.startswith("#"):
                # Check for common emoji ranges
                for char in line:
                    code = ord(char)
                    assert not (0x1F300 <= code <= 0x1FFFF), (
                        f"Emoji found in heading: {line.strip()}"
                    )


# ----------------------------------------------------------------
# L-3: Unified versioning
# ----------------------------------------------------------------


class TestUnifiedVersioning:
    """Verify all packages share the same version."""

    def test_reasoning_version_matches(self):
        """L-3: Reasoning extension version must match main package."""
        from fsm_llm.__version__ import __version__ as main_version
        from fsm_llm.reasoning.__version__ import __version__ as reasoning_version

        assert reasoning_version == main_version

    def test_workflows_version_matches(self):
        """Baseline: Workflows extension version must match main package."""
        from fsm_llm.__version__ import __version__ as main_version
        from fsm_llm.workflows import __version__ as workflows_version

        assert workflows_version == main_version


class TestVersionAlignment:
    """C1: pyproject.toml and __version__.py must agree."""

    def test_version_is_0_5_0(self):
        from fsm_llm.__version__ import __version__

        assert __version__ == "0.12.0"

    def test_init_exports_correct_version(self):
        import fsm_llm

        assert fsm_llm.__version__ == "0.12.0"


class TestRequirementsAlignment:
    """H5: requirements.txt should match pyproject.toml core deps.

    Note: requirements.txt was removed in favor of pyproject.toml as the
    single source of truth for dependencies. These tests are skipped.
    """

    @pytest.mark.skip(
        reason="requirements.txt removed; deps managed via pyproject.toml"
    )
    def test_requirements_has_correct_dotenv_version(self):
        import pathlib

        req_path = pathlib.Path(__file__).parents[2] / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        assert "python-dotenv>=1.0.0" in content
        # Should not have dev deps
        assert "pytest" not in content
        assert "pytest-mock" not in content

    @pytest.mark.skip(
        reason="requirements.txt removed; deps managed via pyproject.toml"
    )
    def test_requirements_has_litellm_upper_bound(self):
        import pathlib

        req_path = pathlib.Path(__file__).parents[2] / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        assert "<2.0" in content

    def test_litellm_excludes_compromised_versions(self):
        """Verify pyproject.toml structurally excludes compromised litellm versions.

        litellm 1.82.7 and 1.82.8 were compromised with credential-stealing
        malware via .pth file injection (March 2026 PyPI supply-chain
        incident). These must remain unreachable by ANY environment that
        respects this dependency spec -- either via explicit `!=` excludes,
        or via a version floor at/above 1.83.0 (litellm's first release
        built on its post-incident, isolated CI/CD v2 pipeline; see
        docs.litellm.ai/blog/security-update-march-2026), which excludes
        them structurally without needing to name them.
        """
        import pathlib
        import tomllib

        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        pyproject_path = pathlib.Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        litellm_deps = [d for d in deps if d.startswith("litellm")]
        assert len(litellm_deps) == 1, "Expected exactly one litellm dependency"
        litellm_dep = litellm_deps[0]
        spec: SpecifierSet = Requirement(litellm_dep).specifier
        for compromised in ("1.82.7", "1.82.8"):
            assert not spec.contains(Version(compromised)), (
                f"litellm dependency spec {litellm_dep!r} must not permit the "
                f"compromised version {compromised}"
            )


class TestConsolidatedAll:
    """M3: __all__ should be defined once, not extended dynamically."""

    def test_no_dynamic_all_extension(self):
        import pathlib

        init_path = pathlib.Path(__file__).parents[2] / "src" / "__init__.py"
        source = init_path.read_text(encoding="utf-8")
        assert source.count("__all__.extend") == 0, (
            "__all__ should not be extended dynamically"
        )
        assert source.count("__all__.append") == 0, (
            "__all__ should not be appended to dynamically"
        )

    def test_all_exports_are_importable(self):
        import fsm_llm

        for name in fsm_llm.__all__:
            assert hasattr(fsm_llm, name), f"{name} in __all__ but not importable"

    def test_okf_and_skill_errors_are_exported_with_the_other_exceptions(self):
        import fsm_llm
        from fsm_llm import definitions

        for name in ("OKFError", "SkillError"):
            assert name in fsm_llm.__all__
            assert getattr(fsm_llm, name) is getattr(definitions, name)
            assert issubclass(getattr(fsm_llm, name), fsm_llm.FSMError)


class TestNoDeadToxDocs:
    """M6: tox.ini should not have unused docs environment."""

    def test_no_sphinx_in_tox(self):
        import pathlib

        tox_path = pathlib.Path(__file__).parents[2] / "tox.ini"
        content = tox_path.read_text(encoding="utf-8")
        assert "sphinx" not in content.lower()
        assert "[testenv:docs]" not in content
