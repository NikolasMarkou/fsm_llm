"""`import fsm_llm` loads a pinned set of packages and no more.

Plan d454fd62, step 96. The package is lazy on purpose: `fsm_llm/__init__.py` never imports a
subpackage (they import `from fsm_llm import API`, and `monitor` would drag fastapi into a core
install), and the layer packages above the core (`okf`, `skills`, `tools`, `policy`) and the
shared kernel modules that only they use (`text`, `frontmatter`, `cli_support`) load when a
caller imports them by name. The public API needs the core layers, so those load.

The allowed set is measured, not argued: `EAGER` is what `import fsm_llm` loads in a fresh
`python -I` process (first-level names below `fsm_llm`). The test pins it exactly, so a new
eager import, in either direction, is a visible diff of this file. A second test plants an eager
import of a subpackage, of a layer package and of a kernel module in a scratch copy of the tree
and checks the same probe catches each, so the pin is not vacuous.

Each probe is a child process (`python -I`, model-server variables on a closed port, nothing
inherited): a session that already holds a module cannot hide a leak, and no model or network
is touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_fsm_llm_regression.workflow_support import passthrough_env

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "fsm_llm"

#: First-level names below `fsm_llm` that `import fsm_llm` loads today (a module or a package).
EAGER = frozenset(
    {
        "__version__",
        "constants",
        "context",
        "core",
        "engine",
        "env",
        "handlers",
        "llm",
        "logging",
        "memory",
        "security",
    }
)
#: The seven extras-backed subpackages: never loaded by `import fsm_llm`.
SUBPACKAGES = (
    "agents",
    "reasoning",
    "workflows",
    "monitor",
    "eval",
    "synthesis",
    "testing",
)
#: Layer packages and kernel modules that load only on request.
ON_REQUEST = (
    "okf",
    "skills",
    "tools",
    "policy",
    "text",
    "frontmatter",
    "cli_support",
    "runner",
)
#: Third-party packages a core import must not pull in (the extras bring them).
HEAVY_THIRD_PARTY = ("pytest", "fastapi", "uvicorn", "mcp", "opentelemetry")

# The child code: import the package from `root`, print the first-level names loaded.
_PROBE = (
    "import sys\n"
    "sys.path.insert(0, {root!r})\n"
    "import fsm_llm\n"
    "assert fsm_llm.__file__.startswith({root!r}), fsm_llm.__file__\n"
    "names = sorted({{m.split('.')[1] for m in sys.modules if m.startswith('fsm_llm.')}})\n"
    "heavy = [m for m in {heavy!r} if m in sys.modules]\n"
    "print(','.join(names)); print(','.join(heavy))\n"
)


def loaded_by_import(src_root: Path = REPO / "src") -> tuple[set[str], list[str]]:
    """First-level `fsm_llm.<name>` modules and heavy third-party packages that
    `import fsm_llm` loads from the tree whose `src` directory is *src_root*.

    Interface contract: *src_root* contains the `fsm_llm` package directory. Runs one
    `python -I` child (so no user site, no `PYTHONPATH`, no cwd on the path) with the model-server
    variables on a closed port and a 120 s timeout; raises `AssertionError` with the child's stderr
    when the import fails. Returns `(names, heavy)`.
    """
    code = _PROBE.format(root=str(src_root), heavy=HEAVY_THIRD_PARTY)
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
        env={
            "PATH": os.environ.get("PATH", ""),
            "OLLAMA_API_BASE": "http://127.0.0.1:9",
            "OLLAMA_HOST": "http://127.0.0.1:9",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            **passthrough_env(),  # Windows: `import asyncio` needs SYSTEMROOT
        },
    )
    assert done.returncode == 0, done.stderr
    names_line, heavy_line = [*done.stdout.strip().split("\n"), ""][:2]
    return (
        {n for n in names_line.split(",") if n},
        [h for h in heavy_line.split(",") if h],
    )


class TestImportFsmLlmIsLazy:
    def test_it_loads_exactly_the_pinned_set(self) -> None:
        names, _ = loaded_by_import()

        assert names == EAGER, (
            f"unexpected eager imports: {sorted(names - EAGER)}; "
            f"no longer loaded: {sorted(EAGER - names)}. "
            "A new eager import is a layout decision: update EAGER with the reason."
        )

    def test_no_subpackage_loads(self) -> None:
        names, _ = loaded_by_import()

        assert not names & set(SUBPACKAGES)

    def test_no_request_only_layer_or_kernel_module_loads(self) -> None:
        names, _ = loaded_by_import()

        assert not names & set(ON_REQUEST)

    def test_no_extras_only_third_party_package_loads(self) -> None:
        _, heavy = loaded_by_import()

        assert heavy == []

    def test_the_pinned_names_are_real_and_the_lists_do_not_overlap(self) -> None:
        on_disk = {
            p.name[:-3] if p.suffix == ".py" else p.name
            for p in PKG.iterdir()
            if p.suffix == ".py" or (p / "__init__.py").is_file()
        }

        assert EAGER <= on_disk
        assert set(SUBPACKAGES) <= on_disk and set(ON_REQUEST) <= on_disk
        assert not EAGER & set(SUBPACKAGES) and not EAGER & set(ON_REQUEST)
        assert not set(SUBPACKAGES) & set(ON_REQUEST)

    def test_every_top_level_package_is_classified(self) -> None:
        """A new top-level package or module has to be put in one of the three lists."""
        on_disk = {
            p.name[:-3] if p.suffix == ".py" else p.name
            for p in PKG.iterdir()
            if (p.suffix == ".py" and p.name != "__init__.py")
            or (p / "__init__.py").is_file()
        }

        unclassified = (
            on_disk - EAGER - set(SUBPACKAGES) - set(ON_REQUEST) - {"__main__"}
        )

        assert unclassified == set(), unclassified


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scratch copy of the package (`<tmp>/src/fsm_llm`); the tests plant into it and restore."""
    root = tmp_path_factory.mktemp("lazy") / "src"
    shutil.copytree(
        PKG, root / "fsm_llm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return root


class TestThePinCatchesAPlantedEagerImport:
    """The same probe, run against a scratch copy of the package with one eager import
    appended to `fsm_llm/__init__.py`, reports it: the pin is not vacuous."""

    @pytest.mark.parametrize(
        "planted",
        [
            "fsm_llm.agents",  # a subpackage
            "fsm_llm.okf",  # a layer package above the core
            "fsm_llm.text",  # a kernel module that loads on request
        ],
    )
    def test_a_planted_eager_import_is_reported(self, tree: Path, planted: str) -> None:
        init = tree / "fsm_llm" / "__init__.py"
        original = init.read_text(encoding="utf-8")
        try:
            init.write_text(
                original + f"\nimport {planted}  # planted by the test\n",
                encoding="utf-8",
            )
            names, _ = loaded_by_import(tree)
        finally:
            init.write_text(original, encoding="utf-8")

        assert planted.split(".")[1] in names
        assert names != EAGER

    def test_the_unplanted_copy_matches_the_pin(self, tree: Path) -> None:
        names, _ = loaded_by_import(tree)

        assert names == EAGER
