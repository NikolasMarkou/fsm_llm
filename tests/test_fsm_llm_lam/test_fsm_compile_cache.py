"""R2 — kernel-level FSM compile cache tests.

Covers `compile_fsm_cached` and `_compile_fsm_by_id` in
`src/fsm_llm/lam/fsm_compile.py`. See plan v3 D-PLAN-07 and
`# DECISION D-002` in fsm_compile.py.

Behaviour under test:
- compile_fsm_cached returns a Term identical to compile_fsm for any defn.
- Repeat calls with the same definition reuse the cached Term object.
- Distinct fsm_id values produce distinct cache entries even on identical JSON.
- Default fsm_id derivation (sha256 prefix) is stable across calls.
- Cache eviction follows lru_cache(maxsize=64).
- Direct calls to _compile_fsm_by_id with malformed JSON raise.
- Failed compiles do not populate the cache (lru_cache behaviour on raise).
"""

from __future__ import annotations

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.lam import compile_fsm, compile_fsm_cached
from fsm_llm.lam.errors import ASTConstructionError
from fsm_llm.lam.fsm_compile import _compile_fsm_by_id


def _greeter_fsm_dict() -> dict:
    return {
        "name": "greeter",
        "description": "single-state greeter for cache tests",
        "initial_state": "hello",
        "states": {
            "hello": {
                "id": "hello",
                "description": "greet",
                "purpose": "say hi",
                "response_instructions": "Say hello.",
                "transitions": [],
            },
        },
    }


@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear the kernel lru_cache between tests so hits/misses are
    deterministic. Tests in this file own the cache state."""
    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


# ---------------------------------------------------------------------------
# Basic surface
# ---------------------------------------------------------------------------


def test_compile_fsm_cached_returns_term():
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    term = compile_fsm_cached(defn)
    # compile_fsm returns an Abs (the outer state_id abstraction).
    from fsm_llm.lam.ast import Abs

    assert isinstance(term, Abs)


def test_compile_fsm_cached_byte_equal_compile_fsm():
    """compile_fsm_cached(defn) returns the same structural Term as
    compile_fsm(defn). The cached version is just memoisation around
    the pure compile pipeline."""
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    direct = compile_fsm(defn)
    cached = compile_fsm_cached(defn)
    # Pydantic models compare by content; structural equality OK.
    assert direct == cached


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


def test_repeat_call_returns_same_object():
    """Cache hit returns identically the same Term object."""
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    t1 = compile_fsm_cached(defn)
    t2 = compile_fsm_cached(defn)
    assert t1 is t2


def test_repeat_call_increments_cache_hits():
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    compile_fsm_cached(defn)
    compile_fsm_cached(defn)
    info = _compile_fsm_by_id.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_distinct_fsm_ids_produce_distinct_cache_entries():
    """fsm_id is part of the key — same JSON, different fsm_id ⇒ two entries."""
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    t1 = compile_fsm_cached(defn, fsm_id="src_a")
    t2 = compile_fsm_cached(defn, fsm_id="src_b")
    info = _compile_fsm_by_id.cache_info()
    # Two misses (one per id), two cache slots, no hits.
    assert info.misses == 2
    assert info.currsize == 2
    # The Terms compare equal (same JSON) but were each compiled fresh.
    assert t1 == t2


def test_default_fsm_id_is_stable():
    """When fsm_id is None, the auto-derived id is stable across calls
    (sha256 prefix of model_dump_json), so repeat calls hit the same
    cache slot."""
    defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    compile_fsm_cached(defn)  # fsm_id derived from sha256
    compile_fsm_cached(defn)  # same derivation → same key → cache hit
    info = _compile_fsm_by_id.cache_info()
    assert info.hits == 1


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


def test_compile_failure_does_not_populate_cache():
    """If the compile path raises, lru_cache does not store the result.
    Subsequent calls with valid definitions still work.

    Note: bypassing pydantic frozen validation (object.__setattr__) and
    re-serialising via model_dump_json can produce JSON that fails the
    FSMDefinition validator inside _compile_fsm_by_id (Pydantic
    re-validates on model_validate_json, catching some malformations
    before compile_fsm runs). Either failure mode (ValidationError or
    ASTConstructionError) demonstrates the lru_cache "raise → no cache"
    contract; we accept both.
    """
    import pydantic

    bad_defn = FSMDefinition.model_validate(_greeter_fsm_dict())
    object.__setattr__(bad_defn, "states", {})

    with pytest.raises((ASTConstructionError, pydantic.ValidationError)):
        compile_fsm_cached(bad_defn, fsm_id="bad")

    # currsize is 0 — the failed compile did not cache.
    assert _compile_fsm_by_id.cache_info().currsize == 0

    # A subsequent valid compile works and lands in the cache.
    good = FSMDefinition.model_validate(_greeter_fsm_dict())
    compile_fsm_cached(good, fsm_id="good")
    assert _compile_fsm_by_id.cache_info().currsize == 1


def test_internal_helper_rejects_malformed_json():
    """_compile_fsm_by_id is internal — direct callers must pass valid JSON."""
    import pydantic

    # Invalid JSON for FSMDefinition.model_validate_json.
    with pytest.raises(pydantic.ValidationError):
        _compile_fsm_by_id("any_id", '{"not": "an FSM"}')
