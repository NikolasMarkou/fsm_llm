"""The state kinds: which kinds of state exist and the rules each kind owns.

A state is one of three kinds today: ``conversation`` (the default: field,
bulk and classification extraction, then a reply), ``completion`` (Pass 1 is
one model completion over a consumer-owned transcript) and ``synthesize`` (a
runner outside core pushes a compiled procedure). Until this module the rules
of the two non-default kinds were spelled out in the State validator, the
definition loader, the data-flow and liveness analyses, the dry-run gate, the
pipeline and ``API.push_fsm``. Row 0.14 of the roadmap gathers them here, one
spec per kind, and each later step of plan d454fd62 moves one rule family to
read its spec instead of testing ``state.completion`` or ``state.synthesize``.

A leaf core module, like ``schema_registry`` and ``codes``: it imports only the
standard library, never logs and never raises on a loaded state. A state is read
through ``getattr`` (its kind config by field name), so this module needs no
``State`` import and ``definitions`` can import it. ``fsm_llm/__init__.py`` does
not import it; the registry is internal until the kind extension point is
public (roadmap C10).

Interface contract:

- ``KINDS``: the ordered tuple of specs, the default kind first. The order is
  the detection order: ``kind_of`` returns the first config-bearing spec whose
  config field is set, else the default. A state holding two configs is refused
  by the State validator; ``model_construct`` and ``model_copy(update=)`` can
  bypass that, which is why the order is fixed.
- ``kind_of(state)``: the ``KindSpec`` of a state or any object with the same
  attributes. Never raises; an object without a config attribute is the default
  kind.
- ``get_kind(name)``: the spec with that name, ``KeyError`` for an unknown one.
- ``KindSpec`` fields (all declarative; the engine's executors are a separate
  module, the verifier hooks that need the evaluator are added with the steps
  that move them):

  - ``name``: the kind name, the value of ``State.kind``.
  - ``config_field``: the ``State`` field holding the kind's config model, or
    ``None`` for the default kind (declared by the absence of any config).
  - ``result_key(state)``: the context key the kind writes its result to, or
    ``None``. Such a key is handler-only: no extraction may write it.
  - ``inherits_result_on_push``: ``False`` when ``API.push_fsm`` must drop the
    parent's value of the result key from the child's initial context.
  - ``forbidden_declarations``: the ``State`` fields the kind cannot also
    declare (they would be dead config); the loader refuses them (see
    ``state_problem``). Data, not a hook: a field is declared when its value is
    truthy.
  - ``check_state(state)``: the kind's own structural rule over one state, a
    message (without the ``State '<id>':`` prefix) or ``None`` when the state is
    fine. Called after the forbidden declarations; default accepts.
  - ``dryrun_judged``: ``False`` when gate G6a does not judge the edges that
    leave a state of the kind.
  - ``minted_keys(state)``: the ordered names the pipeline mints a plain
    extraction for in the state: the auto-minted keys (``required_context_keys``
    then each edge's ``requires_context_keys``, first occurrence order, minus
    the keys a ``classification_extractions`` entry owns, minus the names an
    explicit extraction overrides) followed by the explicit
    ``field_extractions`` names, in list order. Empty for a kind that
    mints nothing (completion, synthesize). The ONE minting rule: the pipeline
    builds its extraction configs from it and the data-flow and liveness
    analyses read the same tuple (never re-spell it elsewhere).
  - ``produced_keys(state, handler_only)``: the keys the state can write on
    arrival or while the conversation sits in it: ``minted_keys`` plus its
    classification fields, minus ``handler_only``, plus its own ``result_key``
    (only that state writes it). A ``frozenset``. The one arrival rule of G3
    and G4 (``flow_analysis.declared_available``).

- ``state_problem(state)``: the one State-level refusal rule: for every kind
  whose config the state holds, in registry order, the first forbidden
  declaration found (``a <kind> state cannot also declare [...]``), then the
  kind's ``check_state``; ``None`` when nothing is wrong. The State validator
  raises ``ValueError(f"State '<id>': {problem}")`` from it, so the loader and
  ``fsm-llm-validate`` keep one verdict. Every set config is judged, not only
  ``kind_of``'s: a state built around the validator with two configs is judged
  by both.

``State.kind`` is a plain property over ``kind_of`` (never a field, a computed
field or a private attribute: any of them would reach ``model_dump`` and move
``fsm_id`` and the library digest).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

__all__ = [
    "COMPLETION",
    "CONVERSATION",
    "KINDS",
    "SYNTHESIZE",
    "CompletionStateKind",
    "KindSpec",
    "get_kind",
    "kind_of",
    "state_problem",
]

_E = TypeVar("_E", bound=Enum)


class CompletionStateKind(str, Enum):
    """The names of the three built-in state kinds (the values of ``State.kind``).

    Named for the state kinds, not ``codes.CompletionKind`` (the ``kind`` of a
    model reply: calls, final, malformed). Use ``.value`` in messages.
    """

    CONVERSATION = "conversation"
    COMPLETION = "completion"
    SYNTHESIZE = "synthesize"

    @classmethod
    def from_string(cls: type[_E], value: str) -> _E:
        """The member for *value*; ``ValueError`` for an unknown string."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown state kind {value!r}; expected one of {[m.value for m in cls]}"
        )


def _no_result_key(state: Any) -> str | None:
    return None


# DECISION plan-2026-11-10T194139-2cefeb61/D-017 (item 7)
# This is the ONE minting rule. It used to be spelled twice (the pipeline's
# `_build_field_configs_from_state` and a hand mirror `_minted_keys` in
# flow_analysis.py, kept equal by a parity test); both now read it through
# `KindSpec.minted_keys`. Do NOT re-spell it in either place and do NOT change
# it without the corpus: the completion/synthesize exclusion (those kinds use
# `_no_minted_keys`), the classification-owned filter (plan 21cd7f8e D-006)
# and explicit `field_extractions` winning over an auto-minted key.
def _conversation_minted_keys(state: Any) -> tuple[str, ...]:
    """The conversation kind's ``minted_keys`` (see the module contract)."""
    auto: list[str] = list(getattr(state, "required_context_keys", None) or [])
    for transition in getattr(state, "transitions", None) or []:
        for condition in getattr(transition, "conditions", None) or []:
            for key in getattr(condition, "requires_context_keys", None) or []:
                if key not in auto:
                    auto.append(key)
    owned = {
        c.field_name for c in getattr(state, "classification_extractions", None) or []
    }
    explicit = [f.field_name for f in getattr(state, "field_extractions", None) or []]
    # An explicit extraction of the same name overrides the auto-minted one.
    return (*(k for k in auto if k not in owned and k not in explicit), *explicit)


def _no_minted_keys(state: Any) -> tuple[str, ...]:
    return ()


def _produced_keys(state: Any, handler_only: frozenset[str]) -> frozenset[str]:
    """The default ``produced_keys``: see the module contract."""
    spec = kind_of(state)
    keys = set(spec.minted_keys(state)) | {
        c.field_name for c in getattr(state, "classification_extractions", None) or []
    }
    keys -= handler_only
    own = spec.result_key(state)
    return frozenset(keys | {own}) if own else frozenset(keys)


def _no_state_problem(state: Any) -> str | None:
    return None


def _completion_check_state(state: Any) -> str | None:
    # A terminal state runs no turn, so its completion call would never run.
    if not getattr(state, "transitions", None):
        return "a completion state needs a transition (a terminal state runs no turn)"
    return None


def _synthesize_check_state(state: Any) -> str | None:
    on_failure = state.synthesize.on_failure
    if on_failure not in {t.target_state for t in state.transitions or []}:
        return (
            f"synthesize on_failure {on_failure!r} must be one of the state's "
            "transition targets (a terminal state has none)"
        )
    return None


def _result_key_of(config_field: str) -> Callable[[Any], str | None]:
    """The hook reading ``<config_field>.result_key`` (``None`` when unset)."""

    def result_key(state: Any) -> str | None:
        config = getattr(state, config_field, None)
        return None if config is None else config.result_key

    return result_key


@dataclass(frozen=True)
class KindSpec:
    """One state kind: its name, config field and declarative rules.

    See the module docstring for the field contract. Frozen: a spec is shared
    by every state of its kind.
    """

    name: str
    config_field: str | None = None
    result_key: Callable[[Any], str | None] = _no_result_key
    inherits_result_on_push: bool = True
    forbidden_declarations: tuple[str, ...] = ()
    dryrun_judged: bool = True
    minted_keys: Callable[[Any], tuple[str, ...]] = _no_minted_keys
    produced_keys: Callable[[Any, frozenset[str]], frozenset[str]] = _produced_keys
    check_state: Callable[[Any], str | None] = _no_state_problem


# The extraction channels a kind that replaces Pass 1 cannot also declare: they
# would never run (the loader refuses them, in this order).
_EXTRACTION_DECLARATIONS = (
    "field_extractions",
    "classification_extractions",
    "required_context_keys",
    "extraction_instructions",
)

CONVERSATION = KindSpec(
    name=CompletionStateKind.CONVERSATION.value,
    minted_keys=_conversation_minted_keys,
)

COMPLETION = KindSpec(
    name=CompletionStateKind.COMPLETION.value,
    config_field="completion",
    result_key=_result_key_of("completion"),
    # An inherited completion result made the child's completion state skip its
    # call (skip-if-set) and answer with the parent's reply, so the key is
    # dropped on push (the rule stays anchored in `API.push_fsm`).
    inherits_result_on_push=False,
    forbidden_declarations=_EXTRACTION_DECLARATIONS,
    dryrun_judged=False,
    check_state=_completion_check_state,
)

# The synthesize result key is NOT dropped on push today (an inherited
# `synthesis_result` makes a nested synthesize state skip itself): the flag
# keeps that behaviour until its own fix (plan d454fd62 step 55).
SYNTHESIZE = KindSpec(
    name=CompletionStateKind.SYNTHESIZE.value,
    config_field="synthesize",
    result_key=_result_key_of("synthesize"),
    inherits_result_on_push=True,
    forbidden_declarations=("completion", *_EXTRACTION_DECLARATIONS),
    dryrun_judged=False,
    check_state=_synthesize_check_state,
)

KINDS: tuple[KindSpec, ...] = (CONVERSATION, COMPLETION, SYNTHESIZE)


def get_kind(name: str) -> KindSpec:
    """The spec named *name*; ``KeyError`` for an unknown kind."""
    for spec in KINDS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown state kind {name!r}")


def kind_of(state: Any) -> KindSpec:
    """The kind of *state*: the first spec in ``KINDS`` whose config is set,
    else the default kind. Reads ``state`` by ``getattr`` only; never raises."""
    for spec in KINDS:
        if spec.config_field and getattr(state, spec.config_field, None) is not None:
            return spec
    return CONVERSATION


def state_problem(state: Any) -> str | None:
    """The first rule of a kind that *state* breaks, as a message, else ``None``.

    Applies to every kind whose config the state holds, in ``KINDS`` order
    (completion before synthesize): the forbidden declarations first, then the
    kind's ``check_state``. The message has no state id; the State validator
    prefixes it. Reads ``state`` by ``getattr`` only.
    """
    for spec in KINDS:
        if not spec.config_field or getattr(state, spec.config_field, None) is None:
            continue
        conflicts = [
            name for name in spec.forbidden_declarations if getattr(state, name, None)
        ]
        if conflicts:
            return f"a {spec.name} state cannot also declare {conflicts}"
        problem = spec.check_state(state)
        if problem is not None:
            return problem
    return None
