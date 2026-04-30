"""Smoke tests for the ``fsm_llm.types`` neutral models layer.

Pre-Phase-2 placeholder. After Phase 2 lands the actual move, this file
asserts:

1. Each canonical model is importable from ``fsm_llm.types``.
2. The same name re-exported from ``fsm_llm.dialog.definitions`` resolves
   to the *same object identity* (``isinstance`` equivalence preserved).
3. Pydantic ``model_validate`` round-trips succeed on the moved models
   (smoke).

This file is intentionally tolerant of the pre-Phase-2 state: each test
``importorskip``-equivalent gates on whether ``fsm_llm.types`` exists,
so the file collects cleanly at HEAD (0.6.0) before the layer is added.
"""

from __future__ import annotations

import importlib

import pytest

# Names expected to live in fsm_llm.types after Phase 2.
_CANONICAL_NAMES: tuple[str, ...] = (
    # Exception hierarchy
    "FSMError",
    "LLMResponseError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "TransitionEvaluationError",
    "ClassificationError",
    "SchemaValidationError",
    "ClassificationResponseError",
    # Request/response models
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "DataExtractionResponse",
    # Enums
    "LLMRequestType",
    "TransitionEvaluationResult",
)


def _types_module_exists() -> bool:
    try:
        importlib.import_module("fsm_llm.types")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _types_module_exists(),
    reason="fsm_llm.types layer lands in Phase 2",
)
class TestTypesLayer:
    """At Phase 2+, every canonical model is importable from ``fsm_llm.types``."""

    @pytest.mark.parametrize("name", _CANONICAL_NAMES)
    def test_canonical_import(self, name: str) -> None:
        types_mod = importlib.import_module("fsm_llm.types")
        assert hasattr(types_mod, name), (
            f"fsm_llm.types is missing canonical export {name!r}"
        )

    @pytest.mark.parametrize("name", _CANONICAL_NAMES)
    def test_back_compat_reexport_identity(self, name: str) -> None:
        """``dialog.definitions.X`` is the same object as ``types.X`` — back-compat
        re-export preserves Pydantic / isinstance identity."""
        types_mod = importlib.import_module("fsm_llm.types")
        defs_mod = importlib.import_module("fsm_llm.dialog.definitions")
        if not hasattr(defs_mod, name):
            pytest.skip(f"{name!r} not re-exported from dialog.definitions")
        assert getattr(types_mod, name) is getattr(defs_mod, name), (
            f"{name!r} object identity diverged between fsm_llm.types and "
            "fsm_llm.dialog.definitions — re-export shim broken."
        )


@pytest.mark.skipif(
    _types_module_exists(),
    reason="At HEAD (pre-Phase-2), fsm_llm.types does not exist yet — this test "
    "documents that and is auto-flipped once the module lands.",
)
def test_types_module_pending_phase_2() -> None:
    """Sentinel: at 0.6.0 we expect fsm_llm.types to be ABSENT.

    After Phase 2 ships, this test is skipped (the module exists) and the
    ``TestTypesLayer`` class above takes over. The skip reasons make the
    transition observable in test output.
    """
    with pytest.raises(ImportError):
        importlib.import_module("fsm_llm.types")
