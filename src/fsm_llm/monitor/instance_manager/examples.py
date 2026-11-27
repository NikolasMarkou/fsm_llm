"""The example FSM presets: where they live and how an id is read (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``find_examples_dir`` (the ``examples/`` directory of a source checkout, ``None`` in a wheel install),
``validate_preset_id`` (a preset id may not escape that directory), ``read_json_file`` and ``load_preset_json``.
``find_examples_dir`` counts the directories between THIS file and the repository root: the file is
``src/fsm_llm/monitor/instance_manager/examples.py``, four directories below the repo root, so the read is
``parents[4]`` (the old single module was one level higher, ``parents[3]``). A wrong depth returns ``None``,
not an error, and the dashboard then lists no presets; ``tests/test_fsm_llm_monitor/test_instance_manager.py``
(``TestFindExamplesDir``) pins the resolved directory. Imports no other module of the package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fsm_llm.security import resolve_within


def find_examples_dir() -> Path | None:
    """Locate the examples/ directory."""
    # parents[4] is the repo root: src/fsm_llm/monitor/instance_manager/examples.py
    # (parents[3] was the root for the old single module instance_manager.py).
    base = Path(__file__).resolve().parents[4] / "examples"
    return base if base.exists() else None


def validate_preset_id(preset_id: str, base: Path) -> Path:
    """Validate a preset ID against path traversal and resolve to a file path.

    Raises ``ValueError`` if the ID is invalid or escapes the base directory.
    Raises ``FileNotFoundError`` if the resolved path is not an existing ``.json`` file.
    """
    if ".." in preset_id or preset_id.startswith("/"):
        raise ValueError("Invalid preset ID")
    # The confinement test is the shared `security.resolve_within`; the lexical
    # pre-check above stays, as it also refuses `sub/../x.json`, which would
    # land inside.
    try:
        confined = resolve_within(base.resolve(), preset_id)
    except (OSError, RuntimeError, ValueError) as e:
        # an embedded NUL byte, or a symlink loop or unreadable link on the way
        # (`OSError`, `RuntimeError` on Python 3.11 and 3.12): an invalid id, not a 500
        raise ValueError("Invalid preset ID") from e
    if confined is None:
        raise ValueError("Invalid preset ID")
    file_path = base / preset_id
    # a directory, the empty id (the base itself) and a non-JSON file are no preset
    # (`_scan_fsm_presets` lists `*/*/*.json`): a 404, not a failed read
    if file_path.suffix != ".json" or not file_path.is_file():
        raise FileNotFoundError(f"Preset not found: {preset_id}")
    return file_path


def read_json_file(path: Path) -> Any:
    """Parse a JSON file as UTF-8, whatever the locale (the one reader of the
    monitor's presets and ``flows.json``)."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_preset_json(preset_id: str) -> dict[str, Any]:
    """Read an example FSM preset by ID.

    Raises ``FileNotFoundError`` (no examples directory, or no such preset),
    ``ValueError`` (an ID that escapes the examples directory) or the JSON and
    OS errors of reading the file.
    """
    base = find_examples_dir()
    if base is None:
        raise FileNotFoundError("Examples directory not found")
    result: dict[str, Any] = read_json_file(validate_preset_id(preset_id, base))
    return result
