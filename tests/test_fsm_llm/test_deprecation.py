"""Tests for ``fsm_llm._api.deprecation``.

Covers:
* basic emission (DeprecationWarning category, message format).
* replacement-vs-no-replacement message variants.
* stacklevel attribution (defensive lower bound).
* per-target dedupe.
* ``reset_deprecation_dedupe()`` behaviour (all-targets / specific targets).
* concurrent dedupe is process-local and thread-safe (smoke test).
"""

from __future__ import annotations

import threading
import warnings

import pytest

from fsm_llm._api.deprecation import reset_deprecation_dedupe, warn_deprecated


@pytest.fixture(autouse=True)
def _clean_dedupe() -> None:
    """Each test starts with an empty dedupe registry."""
    reset_deprecation_dedupe()
    yield
    reset_deprecation_dedupe()


class TestWarnDeprecatedEmission:
    def test_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "pkg.foo" in str(caught[0].message)
        assert "0.6.0" in str(caught[0].message)
        assert "0.7.0" in str(caught[0].message)

    def test_message_with_replacement(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated(
                "pkg.foo",
                since="0.6.0",
                removal="0.7.0",
                replacement="pkg.bar",
            )
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "pkg.foo" in msg
        assert "Use pkg.bar instead." in msg

    def test_message_without_replacement_has_no_use_clause(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
        msg = str(caught[0].message)
        assert "Use " not in msg
        assert " instead." not in msg


class TestStacklevel:
    def test_default_stacklevel_attributes_to_caller(self) -> None:
        # Calling warn_deprecated directly from this test frame; with
        # stacklevel=2 default the helper bumps internally so the warning
        # filename is this test file (or pytest's runner).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
        assert len(caught) == 1
        # Most importantly: not attributed to deprecation.py itself.
        assert "deprecation.py" not in caught[0].filename

    def test_stacklevel_below_one_is_coerced(self) -> None:
        # Should not raise; defensive coerce keeps stacklevel >= 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated(
                "pkg.foo",
                since="0.6.0",
                removal="0.7.0",
                stacklevel=0,
            )
        assert len(caught) == 1


class TestDedupe:
    def test_repeated_calls_emit_only_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
        assert len(caught) == 1

    def test_distinct_targets_emit_independently(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.bar", since="0.6.0", removal="0.7.0")
        assert len(caught) == 2

    def test_changing_removal_version_re_emits(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.8.0")
        assert len(caught) == 2


class TestResetDedupe:
    def test_reset_all_clears_registry(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            reset_deprecation_dedupe()
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
        assert len(caught) == 2

    def test_reset_specific_target(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.bar", since="0.6.0", removal="0.7.0")
            # Reset only pkg.foo; pkg.bar should remain deduped.
            reset_deprecation_dedupe("pkg.foo")
            warn_deprecated("pkg.foo", since="0.6.0", removal="0.7.0")
            warn_deprecated("pkg.bar", since="0.6.0", removal="0.7.0")
        # Two original + one re-emit of pkg.foo = 3.
        assert len(caught) == 3
        names = [str(w.message) for w in caught]
        assert sum("pkg.foo" in n for n in names) == 2
        assert sum("pkg.bar" in n for n in names) == 1


class TestThreadSafety:
    def test_concurrent_emission_is_deduped(self) -> None:
        # Smoke: many threads racing on the same target → exactly one warning.
        n_threads = 16

        def _hit() -> None:
            warn_deprecated("pkg.race", since="0.6.0", removal="0.7.0")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            threads = [threading.Thread(target=_hit) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # Exactly one emission survived dedupe.
        relevant = [w for w in caught if "pkg.race" in str(w.message)]
        assert len(relevant) == 1
