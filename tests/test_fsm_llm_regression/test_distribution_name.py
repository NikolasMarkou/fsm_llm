"""The distribution is published as `llm-fsm` (step 113 of plan d454fd62, D-020, roadmap row 0.1).

`fsm-llm` on PyPI is another project, so only the DISTRIBUTION name moves. The import package
`fsm_llm`, the eight console scripts `fsm-llm*`, the GitHub URLs and the OKF product string
`fsm-llm/<version>` (a protocol identifier written into bundles) stay as they were.

`tests/test_packaging.py` holds the build-level facts (wheel, sdist and `.dist-info` names, the
`all` extra extraction, the ban of the other project's install hint). This file holds the
rest: the name in `pyproject.toml`, what did NOT change, and the nine install hints that
said "From a clone" and now name the PyPI distribution with the extra the module needs.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import tomllib

import pytest
from packaging.requirements import Requirement

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "fsm_llm"

#: The eight console scripts: command names, not the distribution. A rename of the
#: distribution must leave this table byte-identical.
SCRIPTS = {
    "fsm-llm": "fsm_llm.__main__:main_cli",
    "fsm-llm-visualize": "fsm_llm.core.visualize.cli:main_cli",
    "fsm-llm-validate": "fsm_llm.core.verify.cli:main_cli",
    "fsm-llm-monitor": "fsm_llm.monitor.__main__:main_cli",
    "fsm-llm-meta": "fsm_llm.agents.meta_cli:main_cli",
    "fsm-llm-eval": "fsm_llm.eval.__main__:run",
    "fsm-llm-okf": "fsm_llm.okf.cli:main_cli",
    "fsm-llm-skills": "fsm_llm.skills.cli:main_cli",
}

#: `(module path, extras its install hint must name)`; the extra a hint names is the one
#: the module needs (the `testing`, `eval` and `synthesis` extras add no dependency, but a
#: user asked for the subpackage by that name).
HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("src/fsm_llm/__init__.py", ("workflows", "reasoning", "agents")),
    ("src/fsm_llm/monitor/__init__.py", ("monitor",)),
    ("src/fsm_llm/monitor/__main__.py", ("monitor",)),
    ("src/fsm_llm/agents/mcp.py", ("mcp",)),
    ("src/fsm_llm/agents/remote.py", ("a2a", "monitor")),
    ("src/fsm_llm/monitor/otel.py", ("otel",)),
    ("src/fsm_llm/synthesis/__init__.py", ("synthesis",)),
    ("src/fsm_llm/eval/__init__.py", ("eval",)),
    ("src/fsm_llm/testing/__init__.py", ("testing",)),
]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _hint(extra: str) -> re.Pattern[str]:
    """`pip install "llm-fsm[extra]"`, optionally with `--force-reinstall` after `install`."""
    return re.compile(
        r"pip install (?:--force-reinstall )?\\?\"llm-fsm\["
        + re.escape(extra)
        + r"\]\\?\""
    )


class TestTheProjectName:
    def test_the_name_is_llm_fsm(self):
        assert _pyproject()["project"]["name"] == "llm-fsm"

    def test_every_all_entry_is_a_self_reference_under_the_new_name(self):
        entries = _pyproject()["project"]["optional-dependencies"]["all"]
        # Ten extras today; the floor keeps a failed extraction from passing on nothing.
        assert len(entries) >= 10, entries
        for entry in entries:
            req = Requirement(entry)
            assert req.name == "llm-fsm", entry
            assert len(req.extras) == 1, entry

    def test_the_all_extra_names_only_declared_extras(self):
        extras = _pyproject()["project"]["optional-dependencies"]
        requested = {e for entry in extras["all"] for e in Requirement(entry).extras}
        assert requested <= set(extras) - {"all", "dev"}
        assert requested == set(extras) - {"all", "dev"}

    def test_the_old_name_is_not_a_requirement_anywhere(self):
        data = _pyproject()
        found: list[str] = list(data["project"].get("dependencies", []))
        for reqs in data["project"].get("optional-dependencies", {}).values():
            found += reqs
        for reqs in data.get("dependency-groups", {}).values():
            found += [r for r in reqs if isinstance(r, str)]
        assert [r for r in found if Requirement(r).name == "fsm-llm"] == []

    def test_the_urls_still_point_at_the_github_repository(self):
        urls = _pyproject()["project"]["urls"]
        assert set(urls.values()) == {
            "https://github.com/NikolasMarkou/fsm_llm",
            "https://github.com/NikolasMarkou/fsm_llm/issues",
        }


class TestWhatDidNotChange:
    def test_the_console_scripts_keep_their_names_and_targets(self):
        assert _pyproject()["project"]["scripts"] == SCRIPTS

    def test_the_import_package_is_still_fsm_llm(self):
        find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
        assert find["where"] == ["src"] and find["include"] == ["fsm_llm*"]
        assert (SRC / "__init__.py").is_file()
        assert not (REPO / "src" / "llm_fsm").exists()

    @pytest.mark.parametrize(
        "rel", ["src/fsm_llm/agents/okf.py", "src/fsm_llm/okf/memory_export.py"]
    )
    def test_the_okf_product_string_is_untouched(self, rel: str):
        # `generated.by` is `fsm-llm/<version>`: a product identifier in written bundles,
        # pinned by the OKF tests; the distribution rename does not reach it.
        text = (REPO / rel).read_text(encoding="utf-8")
        assert 'f"fsm-llm/{__version__}"' in text
        assert "llm-fsm/" not in text

    def test_the_default_otel_service_name_is_the_product_name(self):
        text = (SRC / "monitor" / "otel.py").read_text(encoding="utf-8")
        assert 'service_name: str = "fsm-llm"' in text


class TestTheInstallHints:
    @pytest.mark.parametrize(
        "rel,extra",
        [(rel, extra) for rel, extras in HINTS for extra in extras],
    )
    def test_the_module_names_the_pypi_distribution_with_its_extra(
        self, rel: str, extra: str
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert _hint(extra).search(text), (
            f'{rel} must say pip install "llm-fsm[{extra}]"'
        )

    def test_no_python_source_still_says_to_install_from_a_clone(self):
        hits = [
            f"{p.relative_to(REPO)}:{n}"
            for p in sorted(SRC.rglob("*.py"))
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if "From a clone" in line or "pip install -e" in line
        ]
        assert hits == []

    @pytest.mark.parametrize("extra", ["mcp", "a2a", "monitor", "otel"])
    def test_the_hint_pattern_rejects_the_wrong_extra(self, extra: str):
        wrong = 'pip install "llm-fsm[agents]"'
        assert not _hint(extra).search(wrong)
        assert _hint(extra).search(f'pip install "llm-fsm[{extra}]"')
        assert _hint(extra).search(f'pip install --force-reinstall "llm-fsm[{extra}]"')
        assert not _hint(extra).search(f'pip install "fsm-llm[{extra}]"')


class TestTheHintsAreWhatTheCodeRaises:
    """The error texts, run: a missing optional package names the extra to install."""

    @pytest.mark.parametrize("name", ["workflows", "reasoning", "agents"])
    def test_a_missing_extension_package_names_its_extra(self, name, monkeypatch):
        import fsm_llm

        def refuse(module: str, *args, **kwargs):
            raise ModuleNotFoundError(f"No module named {module!r}", name=module)

        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(ImportError) as caught:
            getattr(fsm_llm, f"get_{name}")()
        assert _hint(name).search(str(caught.value)), str(caught.value)

    @pytest.mark.parametrize(
        "module,flag,function,extra",
        [
            ("fsm_llm.agents.mcp", "_HAS_MCP", "_require_mcp", "mcp"),
            ("fsm_llm.agents.remote", "_HAS_HTTPX", "_require_httpx", "a2a"),
            ("fsm_llm.agents.remote", "_HAS_FASTAPI", "_require_fastapi", "monitor"),
            ("fsm_llm.monitor.otel", "_HAS_OTEL", "_require_otel", "otel"),
        ],
    )
    def test_a_missing_optional_dependency_names_its_extra(
        self, module, flag, function, extra, monkeypatch
    ):
        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, flag, False)
        with pytest.raises(ImportError) as caught:
            getattr(mod, function)()
        assert _hint(extra).search(str(caught.value)), str(caught.value)
