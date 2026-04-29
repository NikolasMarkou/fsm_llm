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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .runtime.ast import Term

__all__ = [
    "HarnessProfile",
    "ProviderProfile",
    "register_harness_profile",
    "register_provider_profile",
    "get_harness_profile",
    "get_provider_profile",
    "unregister_harness_profile",
    "unregister_provider_profile",
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

_HARNESS_REGISTRY: dict[str, HarnessProfile] = {}
_PROVIDER_REGISTRY: dict[str, ProviderProfile] = {}
_REGISTRY_LOCK = threading.RLock()


def register_harness_profile(
    name: str,
    profile: HarnessProfile,
    *,
    replace: bool = False,
) -> None:
    """Register a HarnessProfile under ``name``.

    ``name`` follows the deepagents convention: either a bare provider
    string (``"openai"``) or a ``provider:model`` pair
    (``"openai:gpt-4o"``). Resolution by :func:`get_harness_profile`
    falls back from model-specific to bare-provider when the
    ``provider:model`` key is missing.

    Parameters
    ----------
    name:
        Registry key.
    profile:
        Frozen :class:`HarnessProfile` instance.
    replace:
        If ``True``, an existing registration under ``name`` is
        overwritten silently. Default ``False`` raises
        :class:`ValueError` on collision.
    """
    if not isinstance(profile, HarnessProfile):  # pragma: no cover — defensive
        raise TypeError(
            f"profile must be a HarnessProfile instance, got {type(profile).__name__}"
        )
    with _REGISTRY_LOCK:
        if not replace and name in _HARNESS_REGISTRY:
            raise ValueError(
                f"HarnessProfile already registered under {name!r}; "
                "pass replace=True to override."
            )
        _HARNESS_REGISTRY[name] = profile


def register_provider_profile(
    name: str,
    profile: ProviderProfile,
    *,
    replace: bool = False,
) -> None:
    """Register a ProviderProfile under ``name``.

    ``name`` is the provider portion of a litellm model spec — for
    ``"ollama_chat/qwen3.5:4b"`` the provider is ``"ollama_chat"``.
    Lookup by :func:`get_provider_profile` accepts either the full
    model string (and extracts the prefix) or the bare provider name.
    """
    if not isinstance(profile, ProviderProfile):  # pragma: no cover — defensive
        raise TypeError(
            f"profile must be a ProviderProfile instance, got {type(profile).__name__}"
        )
    with _REGISTRY_LOCK:
        if not replace and name in _PROVIDER_REGISTRY:
            raise ValueError(
                f"ProviderProfile already registered under {name!r}; "
                "pass replace=True to override."
            )
        _PROVIDER_REGISTRY[name] = profile


def unregister_harness_profile(name: str) -> None:
    """Remove a HarnessProfile registration. No-op if absent."""
    with _REGISTRY_LOCK:
        _HARNESS_REGISTRY.pop(name, None)


def unregister_provider_profile(name: str) -> None:
    """Remove a ProviderProfile registration. No-op if absent."""
    with _REGISTRY_LOCK:
        _PROVIDER_REGISTRY.pop(name, None)


def get_harness_profile(name: str) -> HarnessProfile | None:
    """Look up a HarnessProfile by ``name``.

    Resolution order:

    1. Exact match on ``name`` (e.g. ``"openai:gpt-4o"``).
    2. If ``name`` contains ``":"``, fall back to the bare provider
       prefix (``"openai"``).
    3. Otherwise, ``None``.

    The fallback is the deepagents convention — operators register a
    single profile under ``"openai"`` and have it apply to every model
    routed through that provider, while still being able to override
    for specific models via ``register_harness_profile("openai:gpt-4o", ...)``.
    """
    with _REGISTRY_LOCK:
        if name in _HARNESS_REGISTRY:
            return _HARNESS_REGISTRY[name]
        if ":" in name:
            prefix = name.split(":", 1)[0]
            return _HARNESS_REGISTRY.get(prefix)
    return None


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a ProviderProfile by provider name or full model string.

    Accepts:

    * ``"ollama_chat"`` — bare provider name.
    * ``"ollama_chat/qwen3.5:4b"`` — full litellm model string. The
      provider portion (before the first ``"/"``) is extracted.

    Returns ``None`` on miss.
    """
    with _REGISTRY_LOCK:
        if name in _PROVIDER_REGISTRY:
            return _PROVIDER_REGISTRY[name]
        if "/" in name:
            prefix = name.split("/", 1)[0]
            return _PROVIDER_REGISTRY.get(prefix)
    return None


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
        new_scrutinee = _walk_and_rewrite_leaves(
            term.scrutinee, overrides, counter
        )
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
        new_args = [
            _walk_and_rewrite_leaves(a, overrides, counter) for a in term.args
        ]
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
