"""The import contract of the package (Section 5.2 of the roadmap) runs with the tests.

Plan d454fd62, step 95. ``.importlinter`` holds the layered contract (a layer package imports
only the layers below it), the kernel contract and the contracts that say what the layers
contract cannot. ``make lint-imports`` runs it by hand; this module runs the same command as one
test, so pytest gates it: a layer violation fails the default non-slow selection.

Four things are held here:

* the command itself: it exits 0, and the number of kept contracts it prints equals the number of
  ``[importlinter:contract:*]`` sections of the file (read from the file, never a literal);
* every ``ignore_imports`` line is a named debt: it carries a ``# retired by: <row>`` or a
  ``# permanent: <reason>`` comment, and the set of debts equals ``NAMED_DEBTS`` below, so adding
  one is a visible diff of this file (import-linter itself fails on an ignore that matches no
  import, so a stale line is caught by the run);
* the shape of the layers contract (the roadmap rows, the amendment of D-037: ``policy`` above
  ``tools``) and that no package on disk is left out of the layers or the kernel contract;
* the mechanism is not vacuous: a forbidden import planted in a copy of the tree makes the same
  command fail, naming the contract.

The test skips, with its reason, only when import-linter (the ``dev`` extra) is not installed.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".importlinter"
SRC = ROOT / "src" / "fsm_llm"

NO_LINTER = importlib.util.find_spec("importlinter") is None
needs_linter = pytest.mark.skipif(
    NO_LINTER,
    reason="import-linter is not installed (the `dev` extra: pip install -e '.[dev]')",
)

# The named debts: every ignore line of `.importlinter` that is not a package's own modules
# importing each other. Edge -> how it ends. A new line is a new debt: it needs a row that
# retires it or a written reason, and an entry here, in the same commit.
NAMED_DEBTS = {
    # D-016 a: the G6b simulator and the repair statistics read the eval case driver
    "fsm_llm.synthesis.simulation -> fsm_llm.eval.cases": "retired by: roadmap 6.6 and P7.6",
    "fsm_llm.synthesis.simulation -> fsm_llm.eval.config": "retired by: roadmap 6.6 and P7.6",
    "fsm_llm.synthesis.simulation -> fsm_llm.eval.stats": "retired by: roadmap 6.6 and P7.6",
    "fsm_llm.synthesis.repair -> fsm_llm.eval.stats": "retired by: roadmap 6.6 and P7.6",
    # D-016 b: the embedding helpers live in agents.semantic_tools
    "fsm_llm.synthesis.library -> fsm_llm.agents.semantic_tools": "retired by: roadmap 3.2",
    "fsm_llm.synthesis.synthesizer -> fsm_llm.agents.semantic_tools": "retired by: roadmap 3.2",
    # D-016 c: no roadmap row retires it; reported to the user for a decision
    "fsm_llm.agents.reasoning_react -> fsm_llm.reasoning": "permanent",
    # D-033, D-042: the engine's one module on OKF
    "fsm_llm.engine.okf_bridge -> fsm_llm.okf": "permanent",
    "fsm_llm.engine.okf_bridge -> fsm_llm.okf.memory_export": "permanent",
    # the one LLM layer, and the one entry that hands over to the runner
    "fsm_llm.llm.providers.litellm -> litellm": "permanent",
    "fsm_llm.__main__ -> fsm_llm.runner": "permanent",
}
# `a.** -> a.**` lines: the modules of one package importing each other, no debt
PACKAGE_OWN_IGNORES = {
    "fsm_llm.core.** -> fsm_llm.core.**",
    "fsm_llm.reasoning.** -> fsm_llm.reasoning.**",
    "fsm_llm.security.** -> fsm_llm.security.**",
}

# Kernel modules (D-014, D-016), entry modules and the test doubles: the top-level names that are
# not layer packages. Every other top-level module or package must be in the layers contract.
KERNEL = {
    "cli_support",
    "constants",
    "env",
    "frontmatter",
    "logging",
    "memory",
    "security",
    "text",
}
ENTRY = {"__main__", "runner"}
OUTSIDE_THE_LAYERS = {"testing"}
IGNORED_NAMES = {"__init__", "__version__", "__pycache__"}
# the roadmap's Section 5.2 packages that do not exist yet: optional in the layers contract
ROADMAP_ABSENT = {"serve", "simulation", "oo", "retrieval", "runtime"}

_TAG = re.compile(r"^(retired by: [^(]+?|permanent)(?: \([^)]*\))?(?::| \()")


def _sections() -> dict[str, dict]:
    """The sections of `.importlinter`: ``{name: {"type", "fields": {key: [items]},
    "ignores": [(edge, comment_group)]}}``; the global section is ``"importlinter"``."""
    sections: dict[str, dict] = {}
    section: dict | None = None
    key: str | None = None
    group: list[str] = []
    previous_was_comment = False
    for raw in CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            previous_was_comment = False
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if key == "ignore_imports":
                group = [*group, text] if previous_was_comment else [text]
            previous_was_comment = True
            continue
        previous_was_comment = False
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].removeprefix("importlinter:contract:")
            section = sections.setdefault(
                name, {"type": None, "fields": {}, "ignores": []}
            )
            key, group = None, []
            continue
        assert section is not None, line
        if not line[0].isspace():
            key, _, rest = line.partition("=")
            key = key.strip()
            section["fields"][key] = [rest.strip()] if rest.strip() else []
            group = []
            if key == "type":
                section["type"] = rest.strip()
            continue
        assert key is not None, line
        section["fields"][key].append(stripped)
        if key == "ignore_imports":
            section["ignores"].append((" ".join(stripped.split()), " ".join(group)))
    return sections


def _on_disk() -> set[str]:
    """The top-level modules and packages of ``src/fsm_llm`` (``__main__`` included)."""
    names = set()
    for path in SRC.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            names.add(path.name)
        elif path.is_file() and path.suffix == ".py":
            names.add(path.stem)
    return names - IGNORED_NAMES


def _contracts() -> dict[str, dict]:
    return {n: s for n, s in _sections().items() if n != "importlinter"}


def _run(cwd: Path, extra_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """``lint-imports --no-cache`` run as a Python call (the console script is not portable)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(extra_path or cwd),
            *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []),
        ]
    )
    code = (
        "import sys; from importlinter.cli import lint_imports_command; "
        "sys.argv = ['lint-imports', '--no-cache']; sys.exit(lint_imports_command())"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@needs_linter
class TestTheContractRuns:
    def test_lint_imports_exits_zero_and_every_contract_is_kept(self) -> None:
        result = _run(ROOT, ROOT / "src")

        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
        match = re.search(r"Contracts: (\d+) kept, (\d+) broken", result.stdout)
        assert match, result.stdout[-2000:]
        assert int(match.group(2)) == 0
        # the count comes from the file, so adding a contract needs no edit here
        assert int(match.group(1)) == len(_contracts())

    def test_a_planted_forbidden_import_makes_the_command_fail_and_names_the_contract(
        self, tmp_path: Path
    ) -> None:
        """The mechanism is not vacuous (RED): the same command on a copy of the tree with
        one forbidden import per contract family fails, and names that contract."""
        tree = tmp_path / "tree"
        tree.mkdir()
        shutil.copytree(
            SRC, tree / "fsm_llm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        shutil.copy(CONFIG, tree / ".importlinter")
        plants = [
            # (file, line, the contract that must be named)
            (
                "core/coerce.py",
                "import fsm_llm.agents  # planted",
                "Section 5.2 layers",
            ),
            (
                "tools/limits.py",
                "import fsm_llm.context  # planted",
                "Section 5.2 layers",
            ),
            (
                "constants.py",
                "import fsm_llm.core  # planted",
                "kernel modules import no layer package",
            ),
            (
                "okf/models.py",
                "import fsm_llm.llm  # planted",
                "okf, skills, tools and policy import no llm module",
            ),
            (
                "policy/approval.py",
                "import fsm_llm.skills  # planted",
                "policy imports no context, okf or skills module",
            ),
        ]
        for rel, line, contract in plants:
            path = tree / "fsm_llm" / rel
            original = path.read_text(encoding="utf-8")
            marker = "from __future__ import annotations\n"
            at = original.index(marker) + len(marker) if marker in original else 0
            path.write_text(
                original[:at] + line + "\n" + original[at:], encoding="utf-8"
            )
            try:
                result = _run(tree)
            finally:
                path.write_text(original, encoding="utf-8")
            assert result.returncode == 1, (rel, result.stdout[-1500:])
            broken = [ln for ln in result.stdout.splitlines() if "BROKEN" in ln]
            assert any(contract in ln or contract in result.stdout for ln in broken), (
                rel,
                broken,
            )
            assert re.search(r"Contracts: \d+ kept, [1-9]\d* broken", result.stdout), (
                rel
            )


class TestTheDebts:
    def test_every_ignore_line_is_a_tagged_named_debt(self) -> None:
        untagged = []
        for name, section in _contracts().items():
            for edge, group in section["ignores"]:
                if edge in PACKAGE_OWN_IGNORES:
                    continue
                if not _TAG.match(group):
                    untagged.append((name, edge, group[:60]))
        assert untagged == []

    def test_the_set_of_debts_is_the_pinned_one(self) -> None:
        found: dict[str, str] = {}
        for name, section in _contracts().items():
            for edge, group in section["ignores"]:
                if edge in PACKAGE_OWN_IGNORES:
                    continue
                match = _TAG.match(group)
                assert match, (name, edge)
                kind = match.group(1).strip()
                # the same edge listed by two contracts carries the same tag
                assert found.setdefault(edge, kind) == kind, (name, edge)
        assert found == NAMED_DEBTS

    def test_the_package_own_ignores_are_exactly_the_pinned_ones(self) -> None:
        found = {
            edge
            for section in _contracts().values()
            for edge, _ in section["ignores"]
            if re.fullmatch(r"(\S+)\.\*\* -> \1\.\*\*", edge)
        }
        assert found == PACKAGE_OWN_IGNORES

    def test_no_ignore_line_is_a_blanket(self) -> None:
        """A debt names its two modules: a wildcard on either side is only a package importing
        its own modules."""
        blanket = [
            (name, edge)
            for name, section in _contracts().items()
            for edge, _ in section["ignores"]
            if "*" in edge and edge not in PACKAGE_OWN_IGNORES
        ]
        assert blanket == []

    def test_no_forbidden_contract_allows_indirect_imports(self) -> None:
        # a chain through another module counts, so a re-export cannot launder an import
        assert [
            name
            for name, section in _contracts().items()
            if "allow_indirect_imports" in section["fields"]
        ] == []


class TestTheLayersContract:
    def _layers(self) -> list[list[str]]:
        section = _contracts()["layers-section-5-2"]
        assert section["type"] == "layers"
        rows = []
        for item in section["fields"]["layers"]:
            rows.append(
                [name.strip().strip("()") for name in item.replace(":", "|").split("|")]
            )
        return rows

    def test_the_global_settings(self) -> None:
        fields = _sections()["importlinter"]["fields"]
        assert fields["root_package"] == ["fsm_llm"]
        assert fields["include_external_packages"] == ["True"]
        assert fields["exclude_type_checking_imports"] == ["True"]
        assert _contracts()["layers-section-5-2"]["fields"]["containers"] == ["fsm_llm"]

    def test_the_rows_are_the_roadmap_rows_with_the_decided_deviations(self) -> None:
        assert self._layers() == [
            ["serve", "monitor", "eval"],
            ["agents", "synthesis", "workflows", "reasoning"],
            ["simulation", "oo"],
            ["engine"],
            ["handlers"],
            ["policy"],
            ["context", "retrieval", "okf", "skills", "tools"],
            ["runtime"],
            ["llm"],
            ["core"],
        ]

    def test_policy_sits_above_tools(self) -> None:
        flat = [n for row in self._layers() for n in row]
        assert flat.index("policy") < flat.index("tools")  # D-037: policy imports tools

    def test_the_absent_roadmap_packages_are_optional_and_the_others_exist(
        self,
    ) -> None:
        items = [
            n.strip()
            for item in _contracts()["layers-section-5-2"]["fields"]["layers"]
            for n in re.split(r"[|:]", item)
        ]
        optional = {n.strip("()") for n in items if n.startswith("(")}
        required = {n for n in items if not n.startswith("(")}
        # a roadmap package that gets created loses its parentheses in the same commit
        assert optional == ROADMAP_ABSENT
        assert required <= _on_disk(), sorted(required - _on_disk())
        assert not optional & _on_disk(), (
            "created: drop the parentheses and ROADMAP_ABSENT"
        )

    def test_every_package_on_disk_is_placed(self) -> None:
        in_layers = {n for row in self._layers() for n in row}
        layer_packages = _on_disk() - KERNEL - ENTRY - OUTSIDE_THE_LAYERS
        assert layer_packages <= in_layers, (
            f"not placed in the layers contract: {sorted(layer_packages - in_layers)}"
        )
        assert not in_layers & (KERNEL | ENTRY | OUTSIDE_THE_LAYERS)
        assert KERNEL | ENTRY | OUTSIDE_THE_LAYERS <= _on_disk() | {"__main__"}

    def test_the_kernel_contract_forbids_every_layer_entry_and_testing_module(
        self,
    ) -> None:
        section = _contracts()["kernel-imports-no-layer"]
        sources = {
            m.removeprefix("fsm_llm.") for m in section["fields"]["source_modules"]
        }
        forbidden = {
            m.removeprefix("fsm_llm.") for m in section["fields"]["forbidden_modules"]
        }
        assert sources == KERNEL
        assert forbidden == (_on_disk() - KERNEL) | ENTRY
        assert not sources & forbidden
