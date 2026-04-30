"""Harness and provider profiles — L2 COMPOSE surface.

Profiles are **construction-time** data bundles that customise how a
:class:`fsm_llm.Program` and its underlying :class:`LiteLLMInterface`
behave without touching reduction semantics. There are two kinds:

* :class:`HarnessProfile` — system-prompt assembly (BASE / CUSTOM /
  SUFFIX), per-leaf template overrides, and an optional
  :class:`ProviderProfile` reference. Applied **once** at
  ``Program.from_*`` construction time via :func:`apply_to_term`.

* :class:`ProviderProfile` — extra kwargs to merge into
  :class:`LiteLLMInterface` at construction (e.g. ``api_base``,
  ``custom_llm_provider``, retry settings). Caller-supplied kwargs
  always win on collision.

Both profiles are :mod:`pydantic` ``frozen=True`` models — values are
immutable once registered. Registries are module-level dicts keyed on
the provider / harness name; the ``register_*`` helpers raise on
duplicate registration unless ``replace=True`` is passed.

Theorem-2 contract
------------------

Profiles touch ONLY:

* :attr:`Leaf.template` strings (via :func:`apply_to_term`, using
  :meth:`pydantic.BaseModel.model_copy`),
* the system-prompt string assembled by :func:`assemble_system_prompt`,
* the kwargs passed to :class:`LiteLLMInterface`.

They DO NOT add or remove AST nodes, change Leaf cardinality, or alter
reduction order. ``Executor.run(...).oracle_calls`` therefore equals
``plan(...).predicted_calls`` for τ·k^d-aligned inputs both before and
after a profile is applied.

Apply-once principle
--------------------

A profile is resolved and applied at the boundary of
:meth:`Program.from_fsm` / :meth:`Program.from_term` /
:meth:`Program.from_factory`. Subsequent calls to ``invoke`` see only
the rewritten term. There is no ``invoke``-time profile injection — that
would conflict with Theorem-2 cost prediction (the planner is closed
over the term's static shape, including its Leaf templates).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .runtime.ast import Term

__all__ = [
    "HarnessProfile",
    "ProviderProfile",
    "ProfileRegistry",
    "profile_registry",
    "apply_to_term",
    "assemble_system_prompt",
]


# ---------------------------------------------------------------------------
# ProviderProfile
# ---------------------------------------------------------------------------


class ProviderProfile(BaseModel):
    """Provider-side defaults consulted by :class:`LiteLLMInterface`.

    Stores extra kwargs to merge into the litellm call. Caller-supplied
    kwargs on :class:`LiteLLMInterface.__init__` always win on conflict
    (we are providing **defaults**, not overrides).

    Examples
    --------
    >>> register_provider_profile(
    ...     "ollama",
    ...     ProviderProfile(
    ...         extra_kwargs={"api_base": "http://localhost:11434"},
    ...     ),
    ... )
    """

    model_config = ConfigDict(frozen=True)

    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Kwargs merged into LiteLLMInterface.kwargs at construction. "
            "Caller-supplied kwargs win on collision."
        ),
    )


# ---------------------------------------------------------------------------
# HarnessProfile
# ---------------------------------------------------------------------------


class HarnessProfile(BaseModel):
    """Construction-time profile for a Program.

    The fields below all accept ``None`` (or empty defaults) so a bare
    ``HarnessProfile()`` is a no-op rewrite. Profiles compose by being
    applied — there is no ``HarnessProfile.merge`` because composition
    of two custom prompts has no canonical answer; the caller decides.

    Fields
    ------
    system_prompt_base:
        Replaces the framework's default system-prompt scaffolding when
        non-None. ``None`` keeps the existing assembly.
    system_prompt_custom:
        Operator-supplied prompt fragment, inserted between BASE and
        SUFFIX during :func:`assemble_system_prompt`.
    system_prompt_suffix:
        Trailing fragment appended after BASE/CUSTOM. Useful for
        per-deployment policy preambles ("respond in Spanish",
        "you are deployed at acme.example", etc.).
    leaf_template_overrides:
        ``{leaf_id: replacement_template}``. ``leaf_id`` is the
        synthesised id used by :meth:`Program.explain` (e.g.
        ``"leaf_001_'Extract data from'"``). The override replaces the
        Leaf's :attr:`template` field via
        :meth:`pydantic.BaseModel.model_copy`. Missing leaf ids are
        silently ignored — overrides are best-effort.
    provider_profile_name:
        Optional registry key looked up against
        :func:`get_provider_profile` when this harness is applied at
        construction. ``None`` → no provider-side default merge.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt_base: str | None = None
    system_prompt_custom: str | None = None
    system_prompt_suffix: str | None = None
    leaf_template_overrides: dict[str, str] = Field(default_factory=dict)
    provider_profile_name: str | None = None


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_REGISTRY_LOCK = threading.RLock()

ProfileKind = Literal["harness", "provider"]


class ProfileRegistry:
    """Single registry for both harness and provider profiles.

    Replaces the six module-level functions
    (``register_harness_profile`` / ``register_provider_profile`` /
    ``unregister_*`` / ``get_*``) used through 0.8.0. The kind is now
    a parameter rather than a name suffix:

        from fsm_llm import profile_registry, HarnessProfile, ProviderProfile

        profile_registry.register("openai", HarnessProfile(...), kind="harness")
        profile_registry.register("ollama_chat", ProviderProfile(...), kind="provider")

        prof = profile_registry.get("openai", kind="harness")
        provider = profile_registry.get("ollama_chat/qwen3.5:4b", kind="provider")

        profile_registry.unregister("openai", kind="harness")
        profile_registry.list(kind="harness")  # -> [...]
        profile_registry.list()                # -> {"harness": [...], "provider": [...]}

    Lookup conventions match the 0.8.0 functions exactly:

    * Harness lookup falls back from ``"provider:model"`` to bare
      ``"provider"`` (deepagents convention).
    * Provider lookup falls back from ``"provider/model"`` to the bare
      ``"provider"`` prefix.

    Module-level singleton: ``profile_registry``. There is one shared
    registry per process (the previous module-level globals
    ``_HARNESS_REGISTRY`` / ``_PROVIDER_REGISTRY`` are now instance
    attributes of that singleton).
    """

    def __init__(self) -> None:
        self._harness: dict[str, HarnessProfile] = {}
        self._provider: dict[str, ProviderProfile] = {}

    def register(
        self,
        name: str,
        profile: HarnessProfile | ProviderProfile,
        *,
        kind: ProfileKind | None = None,
        replace: bool = False,
    ) -> None:
        """Register a profile under ``name``.

        ``kind`` may be omitted when ``profile`` is one of the concrete
        profile classes — it is inferred from the instance type.

        Raises :class:`ValueError` on duplicate name unless
        ``replace=True``.
        """
        resolved_kind = self._infer_kind(profile, kind)
        store = self._store_for(resolved_kind)
        with _REGISTRY_LOCK:
            if not replace and name in store:
                raise ValueError(
                    f"{resolved_kind.capitalize()}Profile already registered under "
                    f"{name!r}; pass replace=True to override."
                )
            store[name] = profile  # type: ignore[assignment]

    def unregister(self, name: str, *, kind: ProfileKind) -> None:
        """Remove a profile registration. No-op if absent."""
        with _REGISTRY_LOCK:
            self._store_for(kind).pop(name, None)

    def get(
        self, name: str, *, kind: ProfileKind
    ) -> HarnessProfile | ProviderProfile | None:
        """Look up a profile by name.

        For ``kind="harness"``: exact match → fall back to bare
        ``"provider"`` prefix when ``name`` contains ``":"``.

        For ``kind="provider"``: exact match → fall back to bare
        ``"provider"`` prefix when ``name`` contains ``"/"``.

        Returns ``None`` on miss.
        """
        with _REGISTRY_LOCK:
            store = self._store_for(kind)
            if name in store:
                return store[name]
            if kind == "harness" and ":" in name:
                return store.get(name.split(":", 1)[0])
            if kind == "provider" and "/" in name:
                return store.get(name.split("/", 1)[0])
        return None

    def list(
        self, *, kind: ProfileKind | Literal["all"] = "all"
    ) -> list[str] | dict[str, list[str]]:
        """List registered names.

        ``kind="harness"`` / ``"provider"`` returns a list of names;
        ``kind="all"`` (default) returns a dict with both kinds.
        """
        with _REGISTRY_LOCK:
            if kind == "harness":
                return sorted(self._harness)
            if kind == "provider":
                return sorted(self._provider)
            return {
                "harness": sorted(self._harness),
                "provider": sorted(self._provider),
            }

    def clear(self, *, kind: ProfileKind | Literal["all"] = "all") -> None:
        """Drop all registrations. Used by tests."""
        with _REGISTRY_LOCK:
            if kind in ("harness", "all"):
                self._harness.clear()
            if kind in ("provider", "all"):
                self._provider.clear()

    # ----- internal helpers -----

    def _store_for(
        self, kind: ProfileKind
    ) -> dict[str, HarnessProfile] | dict[str, ProviderProfile]:
        if kind == "harness":
            return self._harness
        if kind == "provider":
            return self._provider
        raise ValueError(f"kind must be 'harness' or 'provider', got {kind!r}")

    @staticmethod
    def _infer_kind(
        profile: HarnessProfile | ProviderProfile, kind: ProfileKind | None
    ) -> ProfileKind:
        if kind is not None:
            return kind
        if isinstance(profile, HarnessProfile):
            return "harness"
        if isinstance(profile, ProviderProfile):
            return "provider"
        raise TypeError(
            f"Cannot infer kind from profile of type {type(profile).__name__}; "
            "pass kind='harness' or kind='provider' explicitly."
        )


# Module-level singleton — the canonical registry.
profile_registry = ProfileRegistry()


# ---------------------------------------------------------------------------
# AST application — non-mutating Leaf.template rewriting.
# ---------------------------------------------------------------------------


def _walk_and_rewrite_leaves(
    term: Term,
    overrides: dict[str, str],
    counter: list[int],
) -> Term:
    """Walk ``term``; for every Leaf, if its synthesised id is in
    ``overrides``, rewrite its ``template`` via ``model_copy``.

    Returns a new Term (Pydantic frozen models cannot be mutated). For
    nodes whose subtree contains no overridden Leaf, the original node
    is returned unchanged so structural identity is preserved where
    possible.

    The synthesised leaf id matches :meth:`Program.explain` exactly:
    ``f"leaf_{idx:03d}_{template[:30].replace(chr(10), ' ')!r}"`` —
    see ``program.py:_walk`` for the canonical implementation.
    """
    # Local import — avoids surfacing runtime AST types as L2 imports
    # at module load time.
    from .runtime.ast import (
        Abs,
        App,
        Case,
        Combinator,
        Fix,
        Leaf,
        Let,
        Var,
    )

    if isinstance(term, Var):
        return term
    if isinstance(term, Abs):
        new_body = _walk_and_rewrite_leaves(term.body, overrides, counter)
        if new_body is term.body:
            return term
        return term.model_copy(update={"body": new_body})
    if isinstance(term, App):
        new_fn = _walk_and_rewrite_leaves(term.fn, overrides, counter)
        new_arg = _walk_and_rewrite_leaves(term.arg, overrides, counter)
        if new_fn is term.fn and new_arg is term.arg:
            return term
        return term.model_copy(update={"fn": new_fn, "arg": new_arg})
    if isinstance(term, Let):
        new_value = _walk_and_rewrite_leaves(term.value, overrides, counter)
        new_body = _walk_and_rewrite_leaves(term.body, overrides, counter)
        if new_value is term.value and new_body is term.body:
            return term
        return term.model_copy(update={"value": new_value, "body": new_body})
    if isinstance(term, Case):
        new_scrutinee = _walk_and_rewrite_leaves(term.scrutinee, overrides, counter)
        new_branches = {
            k: _walk_and_rewrite_leaves(v, overrides, counter)
            for k, v in term.branches.items()
        }
        new_default = (
            _walk_and_rewrite_leaves(term.default, overrides, counter)
            if term.default is not None
            else None
        )
        if (
            new_scrutinee is term.scrutinee
            and all(new_branches[k] is term.branches[k] for k in term.branches)
            and new_default is term.default
        ):
            return term
        update: dict[str, Any] = {
            "scrutinee": new_scrutinee,
            "branches": new_branches,
        }
        if term.default is not None:
            update["default"] = new_default
        return term.model_copy(update=update)
    if isinstance(term, Combinator):
        new_args = [_walk_and_rewrite_leaves(a, overrides, counter) for a in term.args]
        if all(new_args[i] is term.args[i] for i in range(len(term.args))):
            return term
        return term.model_copy(update={"args": new_args})
    if isinstance(term, Fix):
        new_body = _walk_and_rewrite_leaves(term.body, overrides, counter)
        if new_body is term.body:
            return term
        return term.model_copy(update={"body": new_body})
    if isinstance(term, Leaf):
        idx = counter[0]
        counter[0] += 1
        tpl_preview = term.template[:30].replace("\n", " ")
        leaf_id = f"leaf_{idx:03d}_{tpl_preview!r}"
        if leaf_id in overrides:
            return term.model_copy(update={"template": overrides[leaf_id]})
        return term

    # Unknown kind — defensive passthrough.
    return term


def apply_to_term(term: Term, profile: HarnessProfile) -> Term:
    """Apply a :class:`HarnessProfile` to ``term``, returning a new term.

    Currently rewrites only :attr:`Leaf.template` per
    :attr:`HarnessProfile.leaf_template_overrides`. Returns ``term``
    unchanged when the profile carries no leaf-level overrides.

    This is the ONLY AST-side application point for a profile, by
    design. System-prompt assembly is handled separately by
    :func:`assemble_system_prompt` and provider-side defaults by
    :func:`get_provider_profile` consumers (see
    :class:`LiteLLMInterface`).
    """
    if not profile.leaf_template_overrides:
        return term
    counter = [0]
    return _walk_and_rewrite_leaves(
        term, dict(profile.leaf_template_overrides), counter
    )


# ---------------------------------------------------------------------------
# Prompt assembly — USER → (BASE | CUSTOM) → SUFFIX
# ---------------------------------------------------------------------------


def assemble_system_prompt(
    user_prompt: str | None,
    base: str | None,
    custom: str | None,
    suffix: str | None,
    *,
    sep: str = "\n\n",
) -> str:
    """Assemble a system prompt in the canonical
    ``USER → (BASE | CUSTOM) → SUFFIX`` order.

    Joins the supplied fragments with ``sep`` (default: blank line).
    ``None`` and empty strings are skipped. Returns the concatenated
    prompt — never ``None``; an all-empty input returns ``""``.

    The order is fixed by the deepagents reference and intentionally
    documented: caller content is treated as primary; framework BASE
    and operator CUSTOM are layered after; SUFFIX is the trailing
    policy/permissions block.
    """
    parts = [p for p in (user_prompt, base, custom, suffix) if p]
    return sep.join(parts)
