"""`import fsm_llm` loads a pinned set of packages and no more.

Plan d454fd62, step 96; merged into one child process by plan ace7a8c8, step 3 (D-037). The
package is lazy on purpose: `fsm_llm/__init__.py` never imports a subpackage (they import
`from fsm_llm import API`, and `monitor` would drag fastapi into a core install), and the layer
packages above the core (`okf`, `skills`, `tools`, `policy`) and the shared kernel modules that
only they use (`text`, `frontmatter`, `cli_support`) load when a caller imports them by name.
The public API needs the core layers, so those load.

The allowed set is measured, not argued: `EAGER` is what `import fsm_llm` loads in a fresh
`python -I` process (first-level names below `fsm_llm`). The test pins it exactly, so a new
eager import, in either direction, is a visible diff of this file. That exact pin is also what
catches a planted eager import of a subpackage, a layer package or a kernel module (an earlier
form of this file proved it with a scratch copy of the tree per mutation; one child now serves
every row).

`MUST_NOT_LOAD_AT_IMPORT` names deeper modules that must stay out of `sys.modules` after
`import fsm_llm`: a module that a package would otherwise pull in by a re-export or a top-level
import (the envelope and its loader, the run context, the executable registry). A row is added
in the step that creates the module.

The probe is one child process (`python -I`, model-server variables on a closed port, nothing
inherited): a session that already holds a module cannot hide a leak, and no model or network
is touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "fsm_llm"

#: First-level names below `fsm_llm` that `import fsm_llm` loads today (a module or a package).
EAGER = frozenset(
    {
        "__version__",
        # plan 51fde95b step 4.3: a kernel module (standard library only); the LLM layer
        # imports it at module top (`providers/litellm.py` drives its bodies with `run_sync`)
        "aio",
        "constants",
        "context",
        "core",
        "engine",
        "env",
        "handlers",
        "llm",
        "logging",
        "memory",
        # plan 21e6f00d step 7: `engine` imports `runtime` at module top (D-007: the opt-in
        # `API(event_log=)` and the turn boundary of `FSMManager` need it, and a function-local
        # import would hide a real dependency from this pin). Cost, measured in step 7: about
        # 5 ms and 9 small modules on top of `import fsm_llm` (standard library, `core`, `llm`).
        "runtime",
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
#: Dotted modules (a package name stands for itself and everything below it) that must not be
#: in `sys.modules` after `import fsm_llm`, each with the reason. The steps that
#: create the envelope, its loader, the run context and the executable registry add their row.
MUST_NOT_LOAD_AT_IMPORT: dict[str, str] = {
    "fsm_llm.core.results": "the result envelope (C13) loads when a runner or a caller imports it",
    "fsm_llm.core.definitions.executable": (
        "the executable envelope loads when a loader imports it; "
        "`core/definitions/__init__.py` does not re-export it"
    ),
    "fsm_llm.core.definitions.executable_instance": (
        "the identities and the instantiation of an envelope (split from `executable`, a lazy "
        "module) load when a loader imports them; `core/definitions/__init__.py` does not re-export it"
    ),
    "fsm_llm.core.definitions.executable_pattern": (
        "the body of a pattern executable (holes, options, variants; the envelope module imports "
        "it) loads when a loader imports it; `core/definitions/__init__.py` does not re-export it"
    ),
    "fsm_llm.core.verify.executable_pattern": (
        "the pattern checks of the load sequence (instances, names, the hitl rule, V4, V7) load "
        "when `core/verify/executable.py` is imported; `core/verify/__init__.py` does not"
    ),
    "fsm_llm.core.verify.executable": (
        "the load sequence loads when a caller imports it; `core/verify/cli.py` imports it "
        "inside the function, `core/verify/__init__.py` not at all"
    ),
    "fsm_llm.engine.executables": (
        "the executable protocol, adapter, registry and aliases load when a runner imports them; "
        "the engine packages below import none of them (an import contract)"
    ),
    "fsm_llm.engine.context": (
        "the run context (C12) loads when a runner imports it; `engine/__init__.py` imports nothing"
    ),
}

# The child code: import the package from `root`, print every loaded `fsm_llm` module and the
# heavy third-party packages that came with it, as one JSON object.
_PROBE = (
    "import json, sys\n"
    "sys.path.insert(0, {root!r})\n"
    "import fsm_llm\n"
    "assert fsm_llm.__file__.startswith({root!r}), fsm_llm.__file__\n"
    "mods = sorted(m for m in sys.modules if m == 'fsm_llm' or m.startswith('fsm_llm.'))\n"
    "heavy = [m for m in {heavy!r} if m in sys.modules]\n"
    "print(json.dumps({{'modules': mods, 'heavy': heavy}}))\n"
)


def loaded_by_import(src_root: Path = REPO / "src") -> dict[str, list[str]]:
    """What `import fsm_llm` loads from the tree whose `src` directory is *src_root*.

    Interface contract: *src_root* contains the `fsm_llm` package directory. Runs one
    `python -I` child (so no user site, no `PYTHONPATH`, no cwd on the path) with the model-server
    variables on a closed port and a 120 s timeout; raises `AssertionError` with the child's stderr
    when the import fails. Returns `{"modules": [...], "heavy": [...]}`: every loaded dotted
    `fsm_llm` module (sorted) and the heavy third-party packages that were loaded.
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
            # Windows: `import asyncio` needs SYSTEMROOT
            **(
                {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
                if "SYSTEMROOT" in os.environ
                else {}
            ),
        },
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def first_level(modules: list[str]) -> set[str]:
    """The first-level names below `fsm_llm` in a list of dotted module names."""
    return {m.split(".")[1] for m in modules if m.startswith("fsm_llm.")}


@pytest.fixture(scope="module")
def loaded() -> dict[str, list[str]]:
    """The one child process of this file."""
    return loaded_by_import()


class TestImportFsmLlmIsLazy:
    def test_it_loads_exactly_the_pinned_set(
        self, loaded: dict[str, list[str]]
    ) -> None:
        names = first_level(loaded["modules"])

        assert names == EAGER, (
            f"unexpected eager imports: {sorted(names - EAGER)}; "
            f"no longer loaded: {sorted(EAGER - names)}. "
            "A new eager import is a layout decision: update EAGER with the reason."
        )

    def test_no_subpackage_and_no_request_only_module_loads(
        self, loaded: dict[str, list[str]]
    ) -> None:
        names = first_level(loaded["modules"])

        assert not names & set(SUBPACKAGES)
        assert not names & set(ON_REQUEST)

    def test_no_extras_only_third_party_package_loads(
        self, loaded: dict[str, list[str]]
    ) -> None:
        assert loaded["heavy"] == []

    def test_a_listed_module_stays_out(self, loaded: dict[str, list[str]]) -> None:
        for name, reason in MUST_NOT_LOAD_AT_IMPORT.items():
            leaked = [
                m for m in loaded["modules"] if m == name or m.startswith(f"{name}.")
            ]
            assert leaked == [], f"{name} must not load at import ({reason}): {leaked}"

    def test_the_listed_modules_are_real(self) -> None:
        for name in MUST_NOT_LOAD_AT_IMPORT:
            relative = Path(*name.split(".")[1:])
            assert (PKG / relative).is_dir() or (PKG / relative).with_suffix(
                ".py"
            ).is_file(), name

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
