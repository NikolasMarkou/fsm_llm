"""`fsm_llm.simulation` is a layer package that imports no consumer (plan c720601d, step 2, R1).

Rows: the new contract has the pinned shape, the kernel and core contracts name the package, and
a consumer import planted in a copy of the tree turns the contract BROKEN (never vacuous).
`import fsm_llm` and `import fsm_llm.simulation` load no consumer: see `test_import_laziness.py`.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from tests.test_fsm_llm_regression.test_import_contract import (
    CONFIG,
    SRC,
    _contracts,
    _run,
    needs_linter,
)

CONTRACT = "simulation-imports-no-consumer"
CONSUMERS = {
    "__main__",
    "agents",
    "eval",
    "monitor",
    "reasoning",
    "runner",
    "synthesis",
    "testing",
    "workflows",
}


def _names(section: dict, field: str) -> set[str]:
    return {m.removeprefix("fsm_llm.") for m in section["fields"][field]}


class TestTheContractShape:
    def test_the_contract_forbids_every_consumer_from_the_package(self) -> None:
        section = _contracts()[CONTRACT]

        assert section["type"] == "forbidden"
        assert _names(section, "source_modules") == {"simulation"}
        assert _names(section, "forbidden_modules") == CONSUMERS
        assert "ignore_imports" not in section["fields"]
        assert "allow_indirect_imports" not in section["fields"]

    def test_the_kernel_and_core_contracts_name_the_package(self) -> None:
        for name in ("kernel-imports-no-layer", "core-imports-no-higher-layer"):
            forbidden = _names(_contracts()[name], "forbidden_modules")
            assert "simulation" in forbidden, name

    def test_the_layers_contract_places_it_below_the_consumers(self) -> None:
        rows = [
            [n.strip().strip("()") for n in item.split("|")]
            for item in _contracts()["layers-section-5-2"]["fields"]["layers"]
        ]
        flat = [name for row in rows for name in row]

        for above in ("eval", "synthesis", "agents"):
            assert flat.index(above) < flat.index("simulation")
        assert flat.index("simulation") < flat.index("engine")
        assert "(simulation)" not in "".join(
            _contracts()["layers-section-5-2"]["fields"]["layers"]
        )


@needs_linter
class TestAPlantedConsumerImportIsRefused:
    def test_each_consumer_import_breaks_the_contract(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        tree.mkdir()
        shutil.copytree(
            SRC, tree / "fsm_llm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        shutil.copy(CONFIG, tree / ".importlinter")
        path = tree / "fsm_llm" / "simulation" / "__init__.py"
        original = path.read_text(encoding="utf-8")
        for target in ("eval", "synthesis", "agents", "testing"):
            path.write_text(
                original + f"import fsm_llm.{target}  # planted\n", encoding="utf-8"
            )
            try:
                result = _run(tree)
            finally:
                path.write_text(original, encoding="utf-8")
            assert result.returncode == 1, (target, result.stdout[-1500:])
            assert "fsm_llm.simulation imports no eval" in result.stdout, target
            assert re.search(r"Contracts: \d+ kept, [1-9]\d* broken", result.stdout)
