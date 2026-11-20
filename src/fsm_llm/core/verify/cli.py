"""The ``fsm-llm-validate`` console script: ``main`` (the run) and ``main_cli`` (the argparse entry).

Layer ``core`` (plan d454fd62, step 75): split out of the old ``fsm_llm.validator``.
"""

from __future__ import annotations

import json

from fsm_llm.constants import CLI_EXIT_FAILURE, CLI_EXIT_OK
from fsm_llm.core.definitions.schema import (
    EXECUTABLE_SCHEMA_ID,
    SCHEMA_VERSION_DEFAULT,
    SCHEMA_VERSION_OLDEST,
)
from fsm_llm.core.loaders import read_json_object
from fsm_llm.logging import enable_library_logging, logger, setup_cli_logging

from .result import FSMValidationResult
from .upgrade import upgrade_fsm_file
from .validator import validate_fsm_from_file


def _executable_document(path: str) -> dict | None:
    """The parsed object when the file's header names an executable, else ``None``.

    The route is decided by the ``schema`` header, never by the extension. A file that is
    not readable JSON or not an object is not an executable here: it takes the FSM path, which
    keeps its own messages.
    """
    try:
        data = read_json_object(path)
    except Exception:  # OSError, ValueError, RecursionError, UnicodeDecodeError, ...
        return None
    return data if data.get("schema") == EXECUTABLE_SCHEMA_ID else None


def _validate_executable(
    path: str, data: dict, upgrade: bool
) -> tuple[FSMValidationResult, bool]:
    """``(report, upgrade refused)`` of an executable document.

    The loader is imported here, not at module top: ``fsm_llm.core.verify.executable`` stays
    out of ``sys.modules`` for an FSM file (the laziness pin). No resolver is given, so names
    are not resolved. ``--upgrade`` never applies: an executable carries its own
    ``schema_version`` and no file is rewritten.
    """
    from .executable import load_executable

    refused = upgrade
    if upgrade:
        logger.warning(
            f"Not upgrading '{path}': an executable document is never rewritten "
            "(--upgrade rewrites an older FSM definition file only)"
        )
    return load_executable(data)[1], refused


def main(fsm_path, upgrade: bool = False):
    # DECISION plan-2026-07-19T191147-4b664252/D-014 [STALE]
    # NOT redundant — do not delete. `logging.py` calls `logger.disable("fsm_llm")`
    # at import time, which is correct LIBRARY hygiene (a library must not hijack a
    # host application's logging). But this function IS the application: it is the
    # body of the `fsm-llm-validate` console script and of `fsm-llm --mode validate`,
    # and every line it produces goes out through `logger`. Without this opt-in the
    # tool writes ZERO bytes on success AND on failure, and only the exit code
    # speaks. Mirrors `runner.py` (file logging enables it). See decisions.md
    # D-014. The enable goes through the one `enable_library_logging`; the
    # console script and `fsm-llm --mode validate` also call
    # `setup_cli_logging("INFO")` before this, so no DEBUG record is printed.
    enable_library_logging()

    executable = _executable_document(fsm_path)
    if executable is not None:
        validation_result, upgrade_refused = _validate_executable(
            fsm_path, executable, upgrade
        )
    else:
        upgrade_refused = upgrade and upgrade_fsm_file(fsm_path).refusal is not None
        # Run validation
        validation_result = validate_fsm_from_file(fsm_path)

    # Format output based on user preference
    output = json.dumps(validation_result.as_dict(), indent=2)

    # DECISION plan-2026-09-21T203800-8a03483a/D-037
    # The report is the tool's PRIMARY output: it goes to stdout so
    # `fsm-llm-validate ... | jq` works (C8). Do NOT route it back through
    # `logger` (stderr, stdout stayed empty). The logger keeps the diagnostics:
    # the verdict line below and each error/warning as it is found, which is
    # why the D-014 `logger.enable` above is still required.
    print(output)
    verdict = "valid" if validation_result.is_valid else "invalid"
    logger.info(f"Validation result: {verdict}")

    # DECISION plan-2026-11-12T023759-9013b0e3/D-016: an asked upgrade that did
    # not happen is a failure, whatever the validation said, so a script sees
    # that nothing was written. Nothing to do (current, newer or omitted
    # version) is not a refusal.
    if validation_result.is_valid and not upgrade_refused:
        return CLI_EXIT_OK
    return CLI_EXIT_FAILURE


def main_cli():
    """CLI entry point for fsm-llm-validate."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate an FSM definition file or an executable document"
    )
    parser.add_argument(
        "--fsm",
        "-f",
        required=True,
        help=(
            "Path to an FSM definition JSON file, or to an executable document "
            f'(its header says "schema": "{EXECUTABLE_SCHEMA_ID}"; routed by the header, '
            "never by the extension)"
        ),
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help=(
            "Validate the file, then rewrite it in place to the current "
            f"definition version ({SCHEMA_VERSION_DEFAULT}) when it is an FSM "
            f"declaring {SCHEMA_VERSION_OLDEST} up to the current one. "
            "Never downgrades or touches another version (an omitted version is "
            "left omitted); exits 1 when the upgrade was refused or failed (an "
            "executable document is never rewritten, so --upgrade on one exits 1)"
        ),
    )
    args = parser.parse_args()
    setup_cli_logging("INFO")
    sys.exit(main(fsm_path=args.fsm, upgrade=args.upgrade))
