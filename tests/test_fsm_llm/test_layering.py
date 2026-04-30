"""Layer partition + import-audit tests for ``fsm_llm`` (M2 — merge spec §6 G3).

Two flavours of assertion:

1. ``__all__`` is partitioned by ``_LAYER_L1.._LAYER_L4`` plus a Legacy
   complement (``set(__all__) - (L1 | L2 | L3 | L4)``). The four named
   frozensets must be pairwise-disjoint and a subset of ``__all__``.

2. **Import-audit (I4)** — AST-walks every ``.py`` file under
   ``src/fsm_llm/`` and asserts no upward edge against the documented
   layer rules. A small ``_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST`` covers
   the five legacy kernel↔dialog couplings that exist at HEAD; new code
   must not extend the allow-list silently.

Both audits are PURE static checks — no network, no LLM. They run as
fast unit tests in the regular ``pytest -m 'not slow and not real_llm'``
collection.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm import _LAYER_L1, _LAYER_L2, _LAYER_L3, _LAYER_L4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PKG_ROOT: Path = Path(fsm_llm.__file__).parent

# Pre-existing kernel↔dialog couplings. Each entry is a relative path
# under ``src/fsm_llm/``.
#
# **History.** This allow-list peaked at 5 entries through 0.6.x:
# ``runtime/_litellm.py``, ``runtime/oracle.py``, ``runtime/errors.py``,
# ``stdlib/workflows/exceptions.py``, and ``handlers.py`` — each importing
# the FSMError class or request/response Pydantic models from
# ``fsm_llm.dialog.definitions``. The "Decouple kernel from
# dialog/definitions.py" follow-up landed in 0.7.0 (Phase 2 of the deep
# cleanup): the FSMError hierarchy + the runtime-touching request/response
# models moved to a neutral ``fsm_llm.types`` layer; ``dialog/definitions``
# now re-exports those names back-compat for the dialog-callers. The
# allow-list shrinks to **zero** — every kernel and L2 module sources its
# FSMError + extraction Pydantic models from ``fsm_llm.types`` directly.
#
# **Hard rule**: if this set must grow during a future PR, surface it as a
# D-NNN-SURPRISE in the active plan's decisions.md and seek explicit
# approval. Do NOT silently extend.
_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST: frozenset[str] = frozenset()

# The lam/ shim was deleted in 0.6.0 cleanup (R13 removal). The allow-list
# is now empty for shim entries.
_LAM_SHIM_ALLOWLIST: frozenset[str] = frozenset()

# Combined allow-list for the L1 audit.
_L1_AUDIT_ALLOWLIST: frozenset[str] = (
    _PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST | _LAM_SHIM_ALLOWLIST
)

# Forbidden upward edges by layer. Each value is a tuple of dotted-prefix
# strings; an import whose module starts with any of these is a violation
# unless the importing file is allow-listed.
_L1_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "fsm_llm.dialog",
    "fsm_llm.stdlib",
    "fsm_llm.handlers",
    "fsm_llm.program",
)

# L2 (handlers.py) may import only from the runtime substrate. Top-level
# utilities (logging, constants, etc.) are layer-neutral.
_L2_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "fsm_llm.dialog",
    "fsm_llm.stdlib",
    "fsm_llm.program",
)

# L3 (stdlib + dialog/compile_fsm) may import from runtime + handlers,
# plus layer-neutral top-level utilities. Forbid program (top of stack).
_L3_FORBIDDEN_PREFIXES: tuple[str, ...] = ("fsm_llm.program",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_py_files() -> list[Path]:
    """Every ``.py`` file under ``src/fsm_llm/``."""
    return sorted(p for p in PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    """POSIX-style relative path from ``src/fsm_llm/`` (e.g. ``runtime/oracle.py``)."""
    return path.relative_to(PKG_ROOT).as_posix()


def _module_imports(path: Path) -> list[str]:
    """Return a list of fully-qualified imported module names.

    For ``from X.Y import a, b`` we yield ``"X.Y"`` (one entry per source
    statement). For ``import X.Y`` we yield ``"X.Y"``. Relative imports
    are resolved against the package the file lives in.
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        # Should not happen on a healthy package, but never blow up the
        # audit on a parse error — let pytest surface a clearer error.
        return []
    rel_parts = path.relative_to(PKG_ROOT.parent).with_suffix("").parts
    # rel_parts is e.g. ("fsm_llm", "runtime", "oracle"); the parent
    # package is rel_parts[:-1].
    pkg_parts = list(rel_parts[:-1])
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    out.append(node.module)
            else:
                # Relative import — resolve against the file's package.
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)] or pkg_parts
                if node.module:
                    out.append(".".join([*base, node.module]))
                else:
                    out.append(".".join(base))
    return out


# ---------------------------------------------------------------------------
# Test 1 — __all__ partition (disjoint + cover)
# ---------------------------------------------------------------------------


class TestAllPartition:
    def test_layers_are_pairwise_disjoint(self) -> None:
        layers = {
            "L1": _LAYER_L1,
            "L2": _LAYER_L2,
            "L3": _LAYER_L3,
            "L4": _LAYER_L4,
        }
        names = list(layers.items())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, a = names[i]
                b_name, b = names[j]
                overlap = a & b
                assert not overlap, (
                    f"Layers {a_name} and {b_name} are not disjoint: {sorted(overlap)}"
                )

    def test_layered_subset_of_all(self) -> None:
        layered = _LAYER_L1 | _LAYER_L2 | _LAYER_L3 | _LAYER_L4
        all_set = set(fsm_llm.__all__)
        missing = layered - all_set
        assert not missing, (
            f"Names in _LAYER_L1..L4 missing from __all__: {sorted(missing)}"
        )

    def test_all_partitions_cleanly(self) -> None:
        """Every name in ``__all__`` is in either a layer set OR Legacy."""
        layered = _LAYER_L1 | _LAYER_L2 | _LAYER_L3 | _LAYER_L4
        all_set = set(fsm_llm.__all__)
        legacy = all_set - layered
        # Sanity — Legacy must not overlap any layer set.
        for layer_name, layer in [
            ("L1", _LAYER_L1),
            ("L2", _LAYER_L2),
            ("L3", _LAYER_L3),
            ("L4", _LAYER_L4),
        ]:
            assert not (legacy & layer), (
                f"Legacy block overlaps {layer_name}: {sorted(legacy & layer)}"
            )
        # Sanity — version sentinel is in legacy.
        assert "__version__" in legacy

    def test_layer_l4_exact(self) -> None:
        """L4 INVOKE = Program facade exactly (4 names)."""
        assert _LAYER_L4 == frozenset(
            {"Program", "Result", "ExplainOutput", "ProgramModeError"}
        )

    def test_layer_l2_exact(self) -> None:
        """L2 COMPOSE — full handler surface + Profiles.

        The underlying handler types (``HandlerSystem``, ``FSMHandler``,
        ``BaseHandler``, ``create_handler``) live in L2 alongside the
        ``compose`` / ``HandlerTiming`` / ``HandlerBuilder`` surface so
        that L2 is self-contained — users wiring handlers into a Program
        do not reach into the Legacy block. Profiles (``HarnessProfile``,
        ``ProviderProfile``, ``register_*``, ``get_*``) are L2 too —
        pure construction-time data + AST→AST application. See
        ``src/fsm_llm/profiles.py``.

        0.8.0: the back-compat ``Handler`` alias was removed —
        ``FSMHandler`` is the only protocol name. ``BUILTIN_OPS`` was
        also dropped from L1 (closed registry, internal-only).
        """
        assert _LAYER_L2 == frozenset(
            {
                "compose",
                "FSMHandler",
                "BaseHandler",
                "HandlerTiming",
                "HandlerBuilder",
                "HandlerSystem",
                "create_handler",
                "HarnessProfile",
                "ProviderProfile",
                "register_harness_profile",
                "register_provider_profile",
                "get_harness_profile",
                "get_provider_profile",
            }
        )

    def test_handler_surface_lifted_to_l2(self) -> None:
        """0.6.0: full handler surface lives in L2, not Legacy."""
        for name in ("HandlerSystem", "FSMHandler", "BaseHandler", "create_handler"):
            assert name in _LAYER_L2, f"{name!r} should be in L2 (lifted in 0.6.0)"


# ---------------------------------------------------------------------------
# Test 2 — Import-audit (I4)
# ---------------------------------------------------------------------------


class TestImportAudit:
    @pytest.fixture(scope="class")
    def runtime_files(self) -> list[Path]:
        return [p for p in _iter_py_files() if _rel(p).startswith("runtime/")]

    @pytest.fixture(scope="class")
    def stdlib_files(self) -> list[Path]:
        return [p for p in _iter_py_files() if _rel(p).startswith("stdlib/")]

    def test_l1_runtime_no_upward_edge(self, runtime_files: list[Path]) -> None:
        """``runtime/*.py`` imports nothing from dialog/stdlib/handlers/program.

        Exception: the ``_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST`` documents
        four kernel modules that legitimately couple to
        ``fsm_llm.dialog.definitions`` at HEAD (FSMError relocation is a
        follow-up plan).
        """
        violations: list[tuple[str, str]] = []
        for path in runtime_files:
            rel = _rel(path)
            for mod in _module_imports(path):
                if any(mod.startswith(p) for p in _L1_FORBIDDEN_PREFIXES):
                    if rel not in _L1_AUDIT_ALLOWLIST:
                        violations.append((rel, mod))
        assert not violations, (
            "L1 (runtime/) upward-edge violations:\n"
            + "\n".join(f"  {f} → {m}" for f, m in violations)
            + "\n\nIf an entry must be added to the allow-list, surface it "
            "as a D-NNN-SURPRISE per the active plan's pre-mortem (Scenario B)."
        )

    def test_l2_handlers_imports_only_runtime(self) -> None:
        """``handlers.py`` imports nothing from dialog/stdlib/program.

        Exception: ``handlers.py`` is itself in
        ``_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST`` for its
        ``HandlerSystemError(FSMError)`` MRO import from
        ``dialog/definitions``. Only that specific import path is
        tolerated; any other dialog/stdlib/program import would fail.
        """
        path = PKG_ROOT / "handlers.py"
        rel = _rel(path)
        violations: list[str] = []
        for mod in _module_imports(path):
            if any(mod.startswith(p) for p in _L2_FORBIDDEN_PREFIXES):
                # Tolerate the documented FSMError-MRO import only.
                if (
                    rel in _PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST
                    and mod == "fsm_llm.dialog.definitions"
                ):
                    continue
                violations.append(mod)
        assert not violations, f"L2 (handlers.py) upward-edge violations: {violations}"

    def test_l3_stdlib_no_program_dependency(self, stdlib_files: list[Path]) -> None:
        """``stdlib/*`` modules do not import from ``fsm_llm.program``.

        ``stdlib/workflows/exceptions.py`` is allow-listed for its
        ``fsm_llm.dialog.definitions.FSMError`` import; that is a
        kernel-decoupling follow-up, not an L3 audit failure.
        """
        violations: list[tuple[str, str]] = []
        for path in stdlib_files:
            rel = _rel(path)
            for mod in _module_imports(path):
                if any(mod.startswith(p) for p in _L3_FORBIDDEN_PREFIXES):
                    violations.append((rel, mod))
        assert not violations, "L3 (stdlib/) upward-edge violations:\n" + "\n".join(
            f"  {f} → {m}" for f, m in violations
        )

    def test_l3_compile_fsm_no_program_dependency(self) -> None:
        """``dialog/compile_fsm.py`` (L3) does not import from program."""
        path = PKG_ROOT / "dialog" / "compile_fsm.py"
        violations: list[str] = []
        for mod in _module_imports(path):
            if any(mod.startswith(p) for p in _L3_FORBIDDEN_PREFIXES):
                violations.append(mod)
        assert not violations, (
            f"L3 (dialog/compile_fsm.py) upward-edge violations: {violations}"
        )

    def test_allowlist_entries_actually_exist(self) -> None:
        """Every allow-list entry must be a real file under ``src/fsm_llm/``.

        Prevents typo-rot: if a kernel-decoupling refactor lands and removes
        one of these files, the allow-list shrinks naturally; this test
        catches the removal.
        """
        for entry in _L1_AUDIT_ALLOWLIST:
            assert (PKG_ROOT / entry).is_file(), (
                f"Allow-list entry {entry!r} does not exist under "
                f"src/fsm_llm/ — remove it from the allow-list."
            )

    def test_allowlist_size_pinned(self) -> None:
        """Hard-pin the allow-list size to surface silent growth.

        History:
        - 4 entries (plan v1 assumption A2; M2 EXECUTE).
        - Grew to 5 when ``handlers.py``'s ``HandlerSystemError(FSMError)``
          MRO coupling was surfaced as D-007-SURPRISE in
          plan_2026-04-28_6597e394.
        - Shrunk to **0** in 0.7.0 when ``FSMError`` and the runtime-
          touching request/response models moved to ``fsm_llm.types``.
          ``dialog/definitions`` now re-exports those names for back-compat
          but is no longer the canonical home.
        """
        assert len(_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST) == 0, (
            "Pre-existing dialog-import allow-list grew/shrunk silently. "
            f"Got {len(_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST)} entries; "
            "expected 0 (post-fsm_llm.types decoupling). See docstring."
        )
        # 0.6.0: the lam/ shim was deleted. Allow-list stays at 0.
        assert len(_LAM_SHIM_ALLOWLIST) == 0
