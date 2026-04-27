"""
R4 module-shim tests (plan v3 step 24).

Verifies the back-compat sys.modules shims installed in plan v3 steps 20
and 22 + 23. These tests cover SC29-SC32 from plan v3:

- SC29: `from fsm_llm.lam import Executor, Planner, Oracle, LiteLLMOracle,
        compile_fsm, compile_fsm_cached` works.
- SC30: `fsm_llm.lam.executor` ↔ `fsm_llm.runtime.executor` identity
        (deep-submodule path resolves to the same module object).
- SC31: 9 identity assertions across the dialog + runtime shims —
        api ↔ dialog.api, fsm ↔ dialog.fsm, pipeline ↔ dialog.pipeline,
        prompts ↔ dialog.prompts, classification ↔ dialog.classification,
        transition_evaluator ↔ dialog.transition_evaluator,
        definitions ↔ dialog.definitions, session ↔ dialog.session,
        llm ↔ runtime._litellm.
- SC32: `from fsm_llm import API, FSMManager, LiteLLMInterface, Program`
        — top-level public surface intact post-R4.

Plus negative-control tests: deprecation-warning silence (D-004 / D-PLAN-10
silent-shim policy in 0.4.x) and submodule-aliasing for the 9 lam
submodules.
"""

from __future__ import annotations

import importlib
import sys
import warnings

# ---------------------------------------------------------------------------
# SC29 — `from fsm_llm.lam import …` smoke
# ---------------------------------------------------------------------------


class TestSC29LamImports:
    """`from fsm_llm.lam import …` keeps resolving the post-R4 surface."""

    def test_lam_imports_resolve(self) -> None:
        from fsm_llm.lam import (  # noqa: F401
            Executor,
            LiteLLMOracle,
            Oracle,
            Plan,
            PlanInputs,
            compile_fsm,
            compile_fsm_cached,
            plan,
        )

    def test_lam_top_level_is_runtime(self) -> None:
        import fsm_llm.lam as lam
        import fsm_llm.runtime as runtime

        assert lam is runtime, "fsm_llm.lam must alias fsm_llm.runtime in 0.4.x"


# ---------------------------------------------------------------------------
# SC30 — submodule identity for the renamed kernel
# ---------------------------------------------------------------------------


class TestSC30LamSubmoduleIdentity:
    """Each `fsm_llm.lam.<sub>` resolves to the same module object as
    `fsm_llm.runtime.<sub>` (Scenario 9 from plan v3)."""

    def test_lam_executor_identity(self) -> None:
        import fsm_llm.lam.executor as A
        import fsm_llm.runtime.executor as B

        assert A is B

    def test_lam_planner_identity(self) -> None:
        import fsm_llm.lam.planner as A
        import fsm_llm.runtime.planner as B

        assert A is B

    def test_lam_oracle_identity(self) -> None:
        import fsm_llm.lam.oracle as A
        import fsm_llm.runtime.oracle as B

        assert A is B

    def test_lam_ast_identity(self) -> None:
        import fsm_llm.lam.ast as A
        import fsm_llm.runtime.ast as B

        assert A is B

    def test_lam_dsl_identity(self) -> None:
        import fsm_llm.lam.dsl as A
        import fsm_llm.runtime.dsl as B

        assert A is B

    def test_lam_fsm_compile_identity(self) -> None:
        """`fsm_llm.lam.fsm_compile` must resolve to the post-R4 home —
        the `fsm_llm.dialog.compile_fsm` submodule — via the runtime
        alias. Note: `import fsm_llm.dialog.compile_fsm as B` would bind
        `B` to the *function* `compile_fsm` rather than the submodule
        because `dialog/__init__.py` re-exports the function under that
        name; the canonical way to reach the submodule is via
        `sys.modules` (or `importlib.import_module`)."""
        import sys

        import fsm_llm.lam.fsm_compile as A  # noqa: F401  trigger import

        B = sys.modules["fsm_llm.dialog.compile_fsm"]
        assert sys.modules["fsm_llm.lam.fsm_compile"] is B

    def test_lam_executor_class_identity(self) -> None:
        from fsm_llm.lam.executor import Executor as E1
        from fsm_llm.runtime.executor import Executor as E2

        assert E1 is E2


# ---------------------------------------------------------------------------
# SC31 — dialog + llm shims, 9 identity assertions
# ---------------------------------------------------------------------------


class TestSC31DialogShimIdentity:
    """All 8 dialog shims + the llm/_litellm shim resolve to the same
    module object as the new home."""

    def test_api_identity(self) -> None:
        import fsm_llm.api as A
        import fsm_llm.dialog.api as B

        assert A is B

    def test_fsm_identity(self) -> None:
        import fsm_llm.dialog.fsm as B
        import fsm_llm.fsm as A

        assert A is B

    def test_pipeline_identity(self) -> None:
        import fsm_llm.dialog.pipeline as B
        import fsm_llm.pipeline as A

        assert A is B

    def test_prompts_identity(self) -> None:
        import fsm_llm.dialog.prompts as B
        import fsm_llm.prompts as A

        assert A is B

    def test_classification_identity(self) -> None:
        import fsm_llm.classification as A
        import fsm_llm.dialog.classification as B

        assert A is B

    def test_transition_evaluator_identity(self) -> None:
        import fsm_llm.dialog.transition_evaluator as B
        import fsm_llm.transition_evaluator as A

        assert A is B

    def test_definitions_identity(self) -> None:
        import fsm_llm.definitions as A
        import fsm_llm.dialog.definitions as B

        assert A is B

    def test_session_identity(self) -> None:
        import fsm_llm.dialog.session as B
        import fsm_llm.session as A

        assert A is B

    def test_llm_identity(self) -> None:
        import fsm_llm.llm as A
        import fsm_llm.runtime._litellm as B

        assert A is B


# ---------------------------------------------------------------------------
# SC32 — top-level public surface intact
# ---------------------------------------------------------------------------


class TestSC32TopLevelPublicImports:
    """`from fsm_llm import API, FSMManager, LiteLLMInterface, Program`
    keeps working at the end of R4."""

    def test_top_level_public_imports(self) -> None:
        from fsm_llm import API, FSMManager, LiteLLMInterface, Program  # noqa: F401

    def test_class_identity_via_top_level(self) -> None:
        """Classes reached via `fsm_llm.<C>` are the same objects reached
        via the new dialog/runtime homes."""
        import fsm_llm
        from fsm_llm.dialog.api import API as DialogAPI
        from fsm_llm.dialog.fsm import FSMManager as DialogFSMManager
        from fsm_llm.runtime._litellm import LiteLLMInterface as RuntimeLiteLLM

        assert fsm_llm.API is DialogAPI
        assert fsm_llm.FSMManager is DialogFSMManager
        assert fsm_llm.LiteLLMInterface is RuntimeLiteLLM


# ---------------------------------------------------------------------------
# Submodule shim coverage — the 9 fsm_llm.lam.<sub> paths captured in
# findings/r4-import-sites.md must all import without ModuleNotFoundError.
# ---------------------------------------------------------------------------


class TestLamSubmoduleCoverage:
    """All 10 documented `fsm_llm.lam.<sub>` paths import cleanly through
    the shim (Scenario 9 falsifier)."""

    SUBMODULES = (
        "ast",
        "combinators",
        "constants",
        "cost",
        "dsl",
        "errors",
        "executor",
        "fsm_compile",
        "oracle",
        "planner",
    )

    def test_all_lam_submodules_import(self) -> None:
        for sub in self.SUBMODULES:
            mod = importlib.import_module(f"fsm_llm.lam.{sub}")
            assert mod is not None, f"fsm_llm.lam.{sub} did not import"


# ---------------------------------------------------------------------------
# SC37 — silent-shim policy (no DeprecationWarning emitted in 0.4.x).
# ---------------------------------------------------------------------------


class TestSC37ShimDeprecationWarning:
    """0.5.0: the 10 module shims emit DeprecationWarning at import time
    (D-PIVOT-1-R13, plan_2026-04-27_32652286 step 13). Inverted from the
    prior 0.4.x silent-shim guarantee per D-PLAN-10."""

    SHIM_PATHS = (
        "fsm_llm.lam",
        "fsm_llm.api",
        "fsm_llm.fsm",
        "fsm_llm.pipeline",
        "fsm_llm.prompts",
        "fsm_llm.classification",
        "fsm_llm.transition_evaluator",
        "fsm_llm.definitions",
        "fsm_llm.session",
        "fsm_llm.llm",
    )

    def test_all_shim_files_contain_deprecation_emitter(self) -> None:
        """Each of the 10 shim files MUST contain a
        ``warnings.warn(..., DeprecationWarning)`` call (post-R13)."""
        from pathlib import Path

        import fsm_llm

        pkg_root = Path(fsm_llm.__file__).parent
        shim_files = [
            pkg_root / "lam" / "__init__.py",
            pkg_root / "api.py",
            pkg_root / "fsm.py",
            pkg_root / "pipeline.py",
            pkg_root / "prompts.py",
            pkg_root / "classification.py",
            pkg_root / "transition_evaluator.py",
            pkg_root / "definitions.py",
            pkg_root / "session.py",
            pkg_root / "llm.py",
        ]

        missing = []
        for f in shim_files:
            text = f.read_text()
            # The canonical `_warnings.warn(..., DeprecationWarning, ...)`
            # shape used by the R13 emitter.
            if "warnings.warn" not in text or "DeprecationWarning" not in text:
                missing.append(str(f))

        assert missing == [], (
            f"Shim files MISSING DeprecationWarning emitter "
            f"(R13 / D-PIVOT-1-R13 contract): {missing}"
        )

    def test_shims_emit_deprecation_warning_on_fresh_import(self) -> None:
        """Force a fresh import of each shim and verify a
        DeprecationWarning is emitted with the expected message shape."""
        for path in self.SHIM_PATHS:
            # Drop from sys.modules so the import re-runs (warning is at
            # module-import time, not on every attribute access).
            sys.modules.pop(path, None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                importlib.import_module(path)
                dep = [
                    w
                    for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and path.split(".", 1)[1] in str(w.message)
                ]
                assert dep, (
                    f"Shim {path!r} did not emit DeprecationWarning on "
                    f"fresh import. Caught: "
                    f"{[(w.category.__name__, str(w.message)) for w in caught]}"
                )
