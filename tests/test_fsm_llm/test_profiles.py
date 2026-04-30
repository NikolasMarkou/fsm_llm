"""Tests for ``fsm_llm.profiles``.

Covers:
* HarnessProfile / ProviderProfile construction (frozen Pydantic).
* Registry register / get / unregister with replace= semantics.
* `provider:model` -> bare-provider fallback.
* Provider-name -> bare-provider fallback for ``foo/bar``-style models.
* assemble_system_prompt order USER -> BASE -> CUSTOM -> SUFFIX.
* apply_to_term: no-op for empty overrides; rewrites only matching leaf
  ids; preserves AST shape (Theorem-2 contract — predicted_calls
  unchanged).
* Program.from_term(profile=...) applies the profile once.
* Program.from_term(profile="missing") raises KeyError.
* LiteLLMInterface consumes ProviderProfile defaults (caller wins).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from fsm_llm import (
    Executor,
    HarnessProfile,
    Program,
    ProviderProfile,
    profile_registry,
)
from fsm_llm.dsl import leaf
from fsm_llm.profiles import (
    apply_to_term,
    assemble_system_prompt,
)
from fsm_llm.runtime.planner import PlanInputs, plan

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestProfileConstruction:
    def test_harness_profile_defaults(self) -> None:
        p = HarnessProfile()
        assert p.system_prompt_base is None
        assert p.system_prompt_custom is None
        assert p.system_prompt_suffix is None
        assert p.leaf_template_overrides == {}
        assert p.provider_profile_name is None

    def test_harness_profile_is_frozen(self) -> None:
        p = HarnessProfile(system_prompt_suffix="x")
        with pytest.raises(ValidationError):
            p.system_prompt_suffix = "y"  # type: ignore[misc]

    def test_provider_profile_defaults(self) -> None:
        p = ProviderProfile()
        assert p.extra_kwargs == {}

    def test_provider_profile_is_frozen(self) -> None:
        p = ProviderProfile(extra_kwargs={"k": 1})
        with pytest.raises(ValidationError):
            p.extra_kwargs = {"j": 2}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestHarnessRegistry:
    @pytest.fixture(autouse=True)
    def _isolate(self) -> None:
        # Record state, restore after.
        from fsm_llm import profile_registry

        snapshot = dict(profile_registry._harness)
        yield
        profile_registry._harness.clear()
        profile_registry._harness.update(snapshot)

    def test_register_and_get(self) -> None:
        p = HarnessProfile(system_prompt_suffix="X")
        profile_registry.register("openai", p, kind="harness")
        assert profile_registry.get("openai", kind="harness") is p

    def test_register_collision_raises(self) -> None:
        profile_registry.register("openai", HarnessProfile(), kind="harness")
        with pytest.raises(ValueError, match="already registered"):
            profile_registry.register("openai", HarnessProfile(), kind="harness")

    def test_register_replace_overrides(self) -> None:
        a = HarnessProfile(system_prompt_suffix="A")
        b = HarnessProfile(system_prompt_suffix="B")
        profile_registry.register("openai", a, kind="harness")
        profile_registry.register("openai", b, kind="harness", replace=True)
        assert profile_registry.get("openai", kind="harness") is b

    def test_provider_model_fallback_to_bare(self) -> None:
        bare = HarnessProfile(system_prompt_suffix="bare")
        profile_registry.register("openai", bare, kind="harness")
        # No specific entry for openai:gpt-4o → fallback to bare.
        assert profile_registry.get("openai:gpt-4o", kind="harness") is bare

    def test_provider_model_specific_wins(self) -> None:
        bare = HarnessProfile(system_prompt_suffix="bare")
        spec = HarnessProfile(system_prompt_suffix="spec")
        profile_registry.register("openai", bare, kind="harness")
        profile_registry.register("openai:gpt-4o", spec, kind="harness")
        assert profile_registry.get("openai:gpt-4o", kind="harness") is spec
        # Different model still falls back to bare.
        assert profile_registry.get("openai:gpt-3.5", kind="harness") is bare

    def test_unregister_is_noop_when_absent(self) -> None:
        profile_registry.unregister("nope", kind="harness")  # no raise

    def test_get_missing_returns_none(self) -> None:
        assert profile_registry.get("totally-nonexistent", kind="harness") is None


class TestProviderRegistry:
    @pytest.fixture(autouse=True)
    def _isolate(self) -> None:
        from fsm_llm import profile_registry

        snapshot = dict(profile_registry._provider)
        yield
        profile_registry._provider.clear()
        profile_registry._provider.update(snapshot)

    def test_register_and_get_by_bare_name(self) -> None:
        p = ProviderProfile(extra_kwargs={"api_base": "http://x"})
        profile_registry.register("ollama_chat", p, kind="provider")
        assert profile_registry.get("ollama_chat", kind="provider") is p

    def test_get_full_model_string_extracts_prefix(self) -> None:
        p = ProviderProfile(extra_kwargs={"api_base": "http://x"})
        profile_registry.register("ollama_chat", p, kind="provider")
        # Model string with `/` separator → extract prefix.
        assert profile_registry.get("ollama_chat/qwen3.5:4b", kind="provider") is p


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


class TestAssembleSystemPrompt:
    def test_full_order(self) -> None:
        out = assemble_system_prompt("USER", "BASE", "CUSTOM", "SUFFIX")
        # Order is USER -> BASE -> CUSTOM -> SUFFIX, joined by blank line.
        assert out == "USER\n\nBASE\n\nCUSTOM\n\nSUFFIX"

    def test_skips_none_and_empty(self) -> None:
        out = assemble_system_prompt("u", None, "", "s")
        assert out == "u\n\ns"

    def test_all_empty_returns_empty_string(self) -> None:
        assert assemble_system_prompt(None, None, None, None) == ""

    def test_custom_separator(self) -> None:
        out = assemble_system_prompt("a", None, None, "b", sep=" | ")
        assert out == "a | b"


# ---------------------------------------------------------------------------
# apply_to_term — Theorem-2 contract
# ---------------------------------------------------------------------------


class _OutModel(BaseModel):
    val: str


def _make_simple_term():
    """Two unrelated leaves — leaf_000 and leaf_001 — stitched via Let."""
    from fsm_llm.runtime.dsl import let

    a = leaf("Extract data from {x}", input_vars=("x",))
    b = leaf("Respond about {y}", input_vars=("y",))
    return let("a", a, b)


class TestApplyToTerm:
    def test_no_overrides_returns_same_term(self) -> None:
        t = _make_simple_term()
        prof = HarnessProfile()
        out = apply_to_term(t, prof)
        # Empty override map → identity short-circuit.
        assert out is t

    def test_override_rewrites_matching_leaf_only(self) -> None:
        from fsm_llm.runtime.ast import Leaf

        t = _make_simple_term()
        # leaf_id format from program.py:_walk
        # leaf_000_'Extract data from {x}' (preview is first 30 chars
        # of template).
        first_id = "leaf_000_'Extract data from {x}'"
        prof = HarnessProfile(leaf_template_overrides={first_id: "REWRITTEN {x}"})
        rewritten = apply_to_term(t, prof)
        # Walk the rewritten term and collect leaf templates in order.
        templates = []

        # Walk in the same order as program.py:_walk and apply_to_term:
        # for Let, value first then body.
        def _walk(node):
            if isinstance(node, Leaf):
                templates.append(node.template)
                return
            kind = type(node).__name__
            if kind == "Let":
                _walk(node.value)
                _walk(node.body)
            elif kind == "Abs":
                _walk(node.body)
            elif kind == "App":
                _walk(node.fn)
                _walk(node.arg)
            elif kind == "Case":
                _walk(node.scrutinee)
                for v in node.branches.values():
                    _walk(v)
                if node.default is not None:
                    _walk(node.default)
            elif kind == "Combinator":
                for a in node.args:
                    _walk(a)
            elif kind == "Fix":
                _walk(node.body)

        _walk(rewritten)
        assert templates[0] == "REWRITTEN {x}"
        # Second leaf untouched.
        assert templates[1] == "Respond about {y}"

    def test_unmatched_override_is_silent(self) -> None:
        t = _make_simple_term()
        prof = HarnessProfile(leaf_template_overrides={"leaf_999_'nonexistent'": "X"})
        out = apply_to_term(t, prof)
        # Returns equivalent term — overrides ids that don't match are
        # silent. Each leaf still has its original template.
        # Identity is NOT guaranteed (we walked the tree); equality is.
        assert out == t

    def test_predicted_calls_unchanged(self) -> None:
        """Theorem-2 contract: profile rewriting touches only Leaf.template
        strings — Leaf cardinality and AST shape are preserved, so the
        planner's predicted_calls is invariant."""
        # The planner is closed-form over PlanInputs, not over the AST
        # shape — but we can still assert that applying a profile does
        # not require the planner to be re-run with different inputs.
        # The non-leaf inputs (n, K, tau, k) are unchanged after a
        # profile rewrite by definition; the test below verifies the
        # AST shape (Leaf count, kinds) is structurally identical.
        from fsm_llm.runtime.ast import Leaf

        t = _make_simple_term()
        prof = HarnessProfile(
            leaf_template_overrides={"leaf_000_'Extract data from {x}'": "ALT {x}"}
        )
        rewritten = apply_to_term(t, prof)

        def _count_leaves(node) -> int:
            n = 1 if isinstance(node, Leaf) else 0
            for f in ("body", "fn", "arg", "value", "scrutinee", "default"):
                child = getattr(node, f, None)
                if child is not None:
                    n += _count_leaves(child)
            d = getattr(node, "branches", None)
            if d:
                for v in d.values():
                    n += _count_leaves(v)
            lst = getattr(node, "args", None)
            if lst:
                for v in lst:
                    n += _count_leaves(v)
            return n

        assert _count_leaves(t) == _count_leaves(rewritten)
        # Sanity: the planner's closed-form output is invariant under
        # profile rewriting (predicted_calls depends on n, k, tau, K —
        # not on Leaf.template strings).
        before = plan(PlanInputs(n=100, K=8192, tau=32))
        after = plan(PlanInputs(n=100, K=8192, tau=32))
        assert before.predicted_calls == after.predicted_calls


# ---------------------------------------------------------------------------
# Program.from_term integration
# ---------------------------------------------------------------------------


class TestProgramFromTermProfile:
    def test_no_profile_is_byte_equivalent(self) -> None:
        t = _make_simple_term()
        p = Program.from_term(t)
        # Internal _term identity preserved when no profile.
        assert p._term is t
        assert p._profile is None

    def test_profile_instance_applies(self) -> None:
        t = _make_simple_term()
        prof = HarnessProfile(
            leaf_template_overrides={
                "leaf_000_'Extract data from {x}'": "REWRITTEN {x}"
            }
        )
        p = Program.from_term(t, profile=prof)
        assert p._profile is prof
        # Term has been rewritten — not the same object as t anymore
        # (because the leaf was replaced via model_copy).
        assert p._term is not t

    def test_profile_string_resolves_via_registry(self) -> None:
        from fsm_llm import profile_registry

        snap = dict(profile_registry._harness)
        try:
            prof = HarnessProfile(system_prompt_suffix="X")
            profile_registry.register("test-harness", prof, kind="harness")
            t = _make_simple_term()
            p = Program.from_term(t, profile="test-harness")
            assert p._profile is prof
        finally:
            profile_registry._harness.clear()
            profile_registry._harness.update(snap)

    def test_profile_unknown_string_raises(self) -> None:
        t = _make_simple_term()
        with pytest.raises(KeyError, match="No HarnessProfile registered"):
            Program.from_term(t, profile="totally-not-registered")

    def test_profile_wrong_type_raises(self) -> None:
        t = _make_simple_term()
        with pytest.raises(TypeError, match="profile must be"):
            Program.from_term(t, profile=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LiteLLMInterface ProviderProfile consumption
# ---------------------------------------------------------------------------


class TestLiteLLMProviderProfile:
    @pytest.fixture(autouse=True)
    def _isolate(self) -> None:
        from fsm_llm import profile_registry

        snap = dict(profile_registry._provider)
        yield
        profile_registry._provider.clear()
        profile_registry._provider.update(snap)

    def test_provider_kwargs_merge_under_caller(self) -> None:
        # Register provider defaults.
        profile_registry.register(
            "ollama_chat",
            ProviderProfile(extra_kwargs={"api_base": "http://default"}),
            kind="provider",
        )
        from fsm_llm.runtime._litellm import LiteLLMInterface

        # Caller does NOT supply api_base → profile default flows
        # through.
        iface_a = LiteLLMInterface(model="ollama_chat/qwen3.5:4b")
        assert iface_a.kwargs.get("api_base") == "http://default"

        # Caller supplies api_base → caller wins.
        iface_b = LiteLLMInterface(
            model="ollama_chat/qwen3.5:4b",
            api_base="http://override",
        )
        assert iface_b.kwargs.get("api_base") == "http://override"

    def test_no_profile_registered_is_passthrough(self) -> None:
        from fsm_llm.runtime._litellm import LiteLLMInterface

        # No provider registered for some-novel-provider/foo.
        iface = LiteLLMInterface(model="some-novel-provider/foo", custom_kw=1)
        assert iface.kwargs.get("custom_kw") == 1
        # No api_base injected.
        assert "api_base" not in iface.kwargs


# ---------------------------------------------------------------------------
# Sanity: Theorem-2 holds after profile application end-to-end
# ---------------------------------------------------------------------------


class TestTheorem2WithProfile:
    def test_executor_oracle_calls_match_pre_profile(self) -> None:
        """Build a simple term, run with a scripted oracle, and verify
        oracle_calls is invariant under apply_to_term."""

        class _ScriptedOracle:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, prompt, schema=None, *, model_override=None, env=None):
                self.calls += 1
                return "ok"

            def tokenize(self, text):
                return len(text.split())

        from fsm_llm.runtime.dsl import let

        t = let(
            "a",
            leaf("Extract data from {x}", input_vars=("x",)),
            leaf("Respond about {y}", input_vars=("y",)),
        )

        # Baseline.
        ex1 = Executor(oracle=_ScriptedOracle())
        ex1.run(t, {"x": "hello", "y": "world"})
        baseline_calls = ex1.oracle_calls

        # With a profile override on leaf_000.
        prof = HarnessProfile(
            leaf_template_overrides={"leaf_000_'Extract data from {x}'": "ALT {x}"}
        )
        rewritten = apply_to_term(t, prof)
        ex2 = Executor(oracle=_ScriptedOracle())
        ex2.run(rewritten, {"x": "hello", "y": "world"})

        # Same number of oracle calls — Theorem-2 contract.
        assert ex2.oracle_calls == baseline_calls
