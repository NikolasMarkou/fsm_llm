"""``fsm-llm run <target>`` implementation.

Target detection rules (deliberately simple to keep the surface
predictable):

1. Target ends in ``.json`` OR is an existing file path → **FSM JSON
   mode**. Delegates to :func:`fsm_llm.runner.main` with the target as
   ``fsm_path``. ``--turns`` enables a non-interactive smoke loop that
   reads up to N lines from stdin (one per turn).

2. Target contains exactly one ``:`` and is not a file path →
   **factory mode**. Resolved as ``pkg.mod:factory_name``; the named
   attribute must be callable. We invoke
   :meth:`fsm_llm.program.Program.from_factory` with ``--factory-arg``
   kwargs (best-effort coerced to int/float/bool/string), then call
   :meth:`fsm_llm.program.Program.run` with ``--env`` kwargs. The
   reduction result is printed to stdout.

3. Anything else → :class:`SystemExit` with a descriptive message.

The factory arg / env coercion is intentionally minimal: ``true / false``
→ bool, anything that parses as ``int`` → int, then ``float``, else the
raw string. This is sufficient for smoke tests and the example
scenarios; richer typing belongs to a dedicated config-driven runner.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any


def _coerce(value: str) -> Any:
    """Best-effort coerce a CLI string to bool/int/float/str."""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_kv_list(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise SystemExit(
                f"fsm-llm run: --env / --factory-arg expects KEY=VALUE, got {raw!r}"
            )
        k, v = raw.split("=", 1)
        out[k.strip()] = _coerce(v)
    return out


def _is_factory_string(target: str) -> bool:
    """Return True iff target looks like ``pkg.mod:attr`` and is not a file."""
    if os.path.exists(target):
        return False
    if target.count(":") != 1:
        return False
    pkg, attr = target.split(":", 1)
    # Both halves must be non-empty and look like valid python identifiers
    # (allow dots in pkg).
    if not pkg or not attr:
        return False
    return all(p.isidentifier() for p in pkg.split(".")) and attr.isidentifier()


def _resolve_factory(target: str):
    """Resolve ``pkg.mod:attr`` to a callable. Raises SystemExit on failure."""
    pkg, attr = target.split(":", 1)
    try:
        module = importlib.import_module(pkg)
    except ImportError as e:
        raise SystemExit(
            f"fsm-llm run: cannot import factory module {pkg!r}: {e}"
        ) from e
    try:
        factory = getattr(module, attr)
    except AttributeError as e:
        raise SystemExit(
            f"fsm-llm run: module {pkg!r} has no attribute {attr!r}"
        ) from e
    if not callable(factory):
        raise SystemExit(
            f"fsm-llm run: {target!r} is not callable (got {type(factory).__name__})"
        )
    return factory


def _run_fsm(args: argparse.Namespace) -> int:
    """Delegate to fsm_llm.runner.main with sensible defaults."""
    from ..constants import DEFAULT_MAX_HISTORY_SIZE, DEFAULT_MAX_MESSAGE_LENGTH
    from ..runner import main as runner_main

    history = args.history_size or DEFAULT_MAX_HISTORY_SIZE
    message_length = args.message_length or DEFAULT_MAX_MESSAGE_LENGTH

    # ``--turns N`` provides a non-interactive smoke path. The runner
    # reads from stdin via ``input()``; for a smoke test, we connect a
    # pre-canned stdin (N empty lines + ``exit``) so the loop exits
    # cleanly without hanging.
    if args.turns is not None:
        if args.turns < 0:
            raise SystemExit("fsm-llm run: --turns must be >= 0")
        # Replace stdin with N blank turns followed by ``exit``.
        canned = "\n".join([""] * args.turns + ["exit"]) + "\n"
        import io

        sys.stdin = io.StringIO(canned)

    return int(
        runner_main(
            fsm_path=args.target,
            max_history_size=history,
            max_message_length=message_length,
        )
        or 0
    )


def _run_factory(args: argparse.Namespace) -> int:
    """Build a Program via from_factory, call .run(**env), print result."""
    from ..program import Program

    factory = _resolve_factory(args.target)
    factory_kwargs = _parse_kv_list(args.factory_arg)
    env_kwargs = _parse_kv_list(args.env)

    program = Program.from_factory(factory, factory_kwargs=factory_kwargs)
    try:
        result = program.run(**env_kwargs)
    except Exception as e:  # pragma: no cover — surfaced to user
        print(f"fsm-llm run: term evaluation failed: {e}", file=sys.stderr)
        return 1

    # Render the result. Pydantic models / dicts → JSON; otherwise str.
    try:
        if hasattr(result, "model_dump"):
            print(json.dumps(result.model_dump(), indent=2, default=str))
        elif isinstance(result, dict | list):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
    except (TypeError, ValueError):
        print(repr(result))
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch on target type."""
    target: str = args.target
    if _is_factory_string(target):
        return _run_factory(args)
    # Otherwise treat as FSM JSON path — let the runner emit the
    # canonical "missing FSM file" error if the path is bogus.
    return _run_fsm(args)
