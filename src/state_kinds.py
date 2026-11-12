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
  - ``sample_results(state)``: the sample results the loader runs the state's
    edges on, a tuple of ``(label, result)`` (``None`` is "no result yet");
    empty for a kind that has none. Shaped as the kind's runner writes its
    result key.
  - ``check_definition(definition, state, evaluator, logic_keys)``: the kind's
    cross-state rule, which needs the whole definition: a message (without the
    ``State '<id>':`` prefix) or ``None``. The loader hands in the two services
    this leaf may not import, the real ``TransitionEvaluator`` (its
    ``transition_passes(transition, context)`` judges the sample results) and
    ``definitions.logic_referenced_keys``; ``definition`` is read by ``getattr``
    (``states``, ``handler_only_keys``). Default accepts. The edge check stays
    in the loader, never in ``validator.py`` (it may not be stricter than the
    loader): see the anchor at ``_synthesize_check_definition``.

  - ``mentioned_keys(state)``: the key names the state declares or requires
    (``required_context_keys``, each extraction's ``field_name``, each
    classification's ``field_name``, each edge condition's
    ``requires_context_keys``), in that order, duplicates kept; a gate that only
    reads a key in its logic does not mention it, and a ``result_key`` is not a
    mention (it is handler-only). Gate G6a narrows a key to a type only when no
    other state mentions it. The default reads the fields by ``getattr``, so a
    kind that replaces Pass 1 (no extraction fields) still mentions the keys its
    edges require.
  - ``typed_keys(state)``: ``{key: (kind, intents)}`` for the keys whose value the
    state's own extraction types: ``kind`` is ``"intent"`` (a classification,
    ``intents`` its intent names), ``"bool"`` or ``"number"`` (a typed field,
    ``intents`` empty) or ``"any"`` (every other field type, a later entry of the
    same name wins). Plain data; G6a drops the ``"any"`` entries.
  - ``lint_raw(state_id, state_dict, fsm_data, services)``: the kind's lint over
    the RAW dict of one state (``FSMValidator`` never builds a ``State``, so it
    cannot use the typed hooks above): a list of WARNING messages, empty when
    there is nothing to say. Called only for a state whose raw ``config_field``
    value is a dict. Never stricter than the loader: a warning, never an error.
    ``services`` is a ``RawLintServices`` (see below).
  - ``raw_result_key(state_dict, services)``: the raw twin of ``result_key``: the
    key a state of the kind writes its result to, or ``None``; the validator warns
    on an extraction channel naming such a key (it is never extracted).

- ``RawLintServices``: the three facts the raw lint needs and this leaf may not
  import (``result_kinds`` the reply kinds of a completion reply read off
  ``CompletionResponse``, ``default_result_key`` the completion result key a
  state without one gets, ``string_literals`` the string leaves of a JsonLogic
  node); ``validator`` builds one from its own imports.

- ``raw_kinds(state_dict)``: the specs a raw state dict holds a config for, in
  registry order (the raw twin of ``kind_of``, used by the validator's lint).
- ``state_problem(state)``: the one State-level refusal rule: for every kind
  whose config the state holds, in registry order, the first forbidden
  declaration found (``a <kind> state cannot also declare [...]``), then the
  kind's ``check_state``; ``None`` when nothing is wrong. The State validator
  raises ``ValueError(f"State '<id>': {problem}")`` from it, so the loader and
  ``fsm-llm-validate`` keep one verdict. Every set config is judged, not only
  ``kind_of``'s: a state built around the validator with two configs is judged
  by both.
- ``definition_problem(definition, state, evaluator, logic_keys)``: the one
  definition-level refusal rule: the ``check_definition`` of every kind whose
  config the state holds, in registry order; the first message, else ``None``.
  ``FSMDefinition._check_synthesize_states`` raises
  ``ValueError(f"State '<id>': {problem}")`` from it for each state.
- ``owned_result_keys(states)``: ``{result key: "result_key of a <kind>
  state"}`` over the given states, in registry order (a later kind wins a shared
  key); the labels of the loader's collision refusal.

``State.kind`` is a plain property over ``kind_of`` (never a field, a computed
field or a private attribute: any of them would reach ``model_dump`` and move
``fsm_id`` and the library digest).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

__all__ = [
    "COMPLETION",
    "CONVERSATION",
    "KINDS",
    "SYNTHESIZE",
    "SYNTHESIZE_SAMPLES",
    "CompletionStateKind",
    "KindSpec",
    "RawLintServices",
    "definition_problem",
    "get_kind",
    "kind_of",
    "owned_result_keys",
    "raw_kinds",
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


def _mentioned_keys(state: Any) -> tuple[str, ...]:
    """The default ``mentioned_keys``: see the module contract."""
    return (
        *(getattr(state, "required_context_keys", None) or []),
        *(f.field_name for f in getattr(state, "field_extractions", None) or []),
        *(
            c.field_name
            for c in getattr(state, "classification_extractions", None) or []
        ),
        *(
            key
            for t in getattr(state, "transitions", None) or []
            for c in getattr(t, "conditions", None) or []
            for key in getattr(c, "requires_context_keys", None) or []
        ),
    )


# The extraction value types a gate can reason about: a field type to the kind
# of value the pool of gate G6a holds for it. Any other field type is "any".
_TYPED_KINDS = {"bool": "bool", "int": "number", "float": "number"}


def _typed_keys(state: Any) -> dict[str, tuple[str, tuple[str, ...]]]:
    """The default ``typed_keys``: see the module contract."""
    found: dict[str, tuple[str, tuple[str, ...]]] = {
        c.field_name: ("intent", tuple(i.name for i in c.intents))
        for c in getattr(state, "classification_extractions", None) or []
    }
    for f in getattr(state, "field_extractions", None) or []:
        found[f.field_name] = (_TYPED_KINDS.get(f.field_type, "any"), ())
    return found


@dataclass(frozen=True)
class RawLintServices:
    """What a raw lint needs and this leaf may not import (see the module
    contract). Built once per validation by ``validator``."""

    result_kinds: tuple[str, ...]
    default_result_key: str
    string_literals: Callable[[Any], set[str]]


def _no_lint(
    state_id: str, state: dict[str, Any], fsm_data: dict[str, Any], services: Any
) -> list[str]:
    return []


def _no_raw_result_key(state: dict[str, Any], services: Any) -> str | None:
    return None


def _completion_lint_raw(
    state_id: str, state: dict[str, Any], fsm_data: dict[str, Any], services: Any
) -> list[str]:
    """A completion state whose transitions do not cover every result kind it
    can produce (``calls``, ``final``, ``malformed`` with tools; ``final``,
    ``malformed`` for a structured turn): no unconditional transition and a kind
    no condition names as a string literal. An uncovered kind stalls the state,
    and skip-if-set makes no further call, so the step budget burns."""
    kinds = (
        services.result_kinds
        if state["completion"].get("tools")
        else tuple(k for k in services.result_kinds if k != "calls")
    )
    transitions = state.get("transitions") or []
    if any(not t.get("conditions") for t in transitions):
        return []
    named: set[str] = set()
    for transition in transitions:
        for condition in transition.get("conditions") or []:
            named |= services.string_literals(condition.get("logic"))
    missing = [kind for kind in kinds if kind not in named]
    if not missing:
        return []
    return [
        f"Completion state '{state_id}' has no transition for result "
        f"kind(s) {missing} and no unconditional transition: such a "
        "turn stays in the state and makes no further call"
    ]


def _completion_raw_result_key(state: dict[str, Any], services: Any) -> str | None:
    key: str = state["completion"].get("result_key") or services.default_result_key
    return key


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


def _no_samples(state: Any) -> tuple[tuple[str, dict[str, Any] | None], ...]:
    return ()


def _no_definition_problem(
    definition: Any, state: Any, evaluator: Any, logic_keys: Callable[[Any], set[str]]
) -> str | None:
    return None


# Sample results the loader runs a synthesize state's edges on (D-035). Shaped as
# the runner writes the result key (`status`, `procedure_id`, `version`, `reason`;
# producer: `fsm_llm.synthesis.runner._result`, which core may not import). `None`
# is "no result yet". The strings are the wire values of `codes.Status` and
# `codes.Reason`: this leaf cannot import `codes`, so `test_codes` pins them
# against the enums (and `_RUNNING` below). `definitions._SYNTHESIZE_SAMPLES` is
# this tuple under the name `test_codes` reads.
SYNTHESIZE_SAMPLES: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("absent", None),
    ("running", {"status": "running"}),
    ("failed", {"status": "failed", "reason": "compile_failed"}),
    ("aborted", {"status": "aborted", "reason": "budget_spent"}),
)
_RUNNING = "running"


def _synthesize_samples(state: Any) -> tuple[tuple[str, dict[str, Any] | None], ...]:
    return SYNTHESIZE_SAMPLES


def owned_result_keys(states: Iterable[Any]) -> dict[str, str]:
    """``{result key: "result_key of a <kind> state"}`` over *states*.

    Kinds in registry order: when two kinds share a key the later label wins.
    """
    states = list(states)
    return {
        key: f"result_key of a {spec.name} state"
        for spec in KINDS
        for state in states
        if kind_of(state) is spec and (key := spec.result_key(state)) is not None
    }


# DECISION plan-2026-11-10T194139-2cefeb61/D-034 and
# plan-2026-11-10T194139-2cefeb61/D-035: an edge of a synthesize state may
# read the result key OR a handler-only key (the runner's injected abort
# edge, handler-only `procedure_abort`, sits on every state of a compiled
# procedure, nested synthesize states included), and is then RUN on four
# sample results by the real evaluator. Do NOT
# require the result key alone, do NOT move this check to validator.py (it may
# never be stricter than the loader; the loader calls it, see
# `FSMDefinition._check_synthesize_states`), and do NOT fall back to a
# structural "mentions the key" rule alone: review pass 2 reproduced both holes
# (`status == "failed"` strands the parent after an abort; a `!` edge on
# the result fires only when there is none).
def _synthesize_check_definition(
    definition: Any, state: Any, evaluator: Any, logic_keys: Callable[[Any], set[str]]
) -> str | None:
    """Cross-state rules for a synthesize state (they need the whole FSM).

    1. Every transition must read the state's ``result_key`` (a condition that
       requires it or reads it in its logic) or a ``handler_only_keys`` entry
       (only a handler or the runner writes one; the runner's injected abort
       edge is of this kind). This is necessary, not enough (rule 2 runs the
       edges); it also catches an edge on an unrelated key, which the samples
       of rule 2 cannot see.
    2. The real evaluator (``TransitionEvaluator.transition_passes``, never a
       copy of its logic) runs the state's edges on the four sample results,
       shaped as the runner writes them (``status`` of ``running``, ``failed``
       or ``aborted``, or no result at all). No edge may pass the absent or the
       ``running`` sample (it would fire before the runner has decided), and an
       edge to ``on_failure`` must pass the ``failed`` sample and an edge to it
       the ``aborted`` sample (or the parent is stranded after a failure). An
       edge on a handler-only key alone passes no sample: fine, but it cannot
       be the failure edge.
    3. Host keys a synthesize state returns or takes may not be runner-owned: a
       return key merged by ``pop_fsm`` would write a value the child's model
       extracted over a handler-only key, a synthesize result or a completion
       result; an input named like a result would be overwritten by the runner.
       (A completion ``messages_key`` is internal-prefixed and so can never be a
       public return key.)
    """
    config = state.synthesize
    handler_only = frozenset(getattr(definition, "handler_only_keys", None) or ())
    reads = {config.result_key, *handler_only}
    for transition in state.transitions:
        if not any(
            reads.intersection(condition.requires_context_keys or ())
            or reads.intersection(logic_keys(condition.logic))
            for condition in transition.conditions or ()
        ):
            return (
                f"the transition to {transition.target_state!r} must read the "
                f"synthesize result key {config.result_key!r} (a condition that "
                "requires it or reads it in its logic), or it fires "
                "before the runner has produced a result"
            )
    # other keys the edges require are present: an edge also gated on
    # one (an AND) must not escape the samples for lacking it
    others = {
        key: "x"
        for transition in state.transitions
        for condition in transition.conditions or ()
        for key in condition.requires_context_keys or ()
        if key != config.result_key and key not in handler_only
    }
    for label, result in SYNTHESIZE.sample_results(state):
        context: dict[str, Any] = dict(others)
        if result is not None:
            context[config.result_key] = result
        passing = [
            t for t in state.transitions if evaluator.transition_passes(t, context)
        ]
        if result is None or result["status"] == _RUNNING:
            for transition in passing:
                return (
                    f"the transition to {transition.target_state!r} fires before "
                    f"the runner has produced a result (it passes when "
                    f"{config.result_key!r} is {label})"
                )
        elif not any(t.target_state == config.on_failure for t in passing):
            return (
                f"no transition to {config.on_failure!r} fires when the "
                f"procedure failed or aborted (it does not pass when "
                f"{config.result_key!r} is {label}, {result!r})"
            )
    owned = owned_result_keys(getattr(definition, "states", {}).values())
    for entry, names in (
        ("return_keys", config.return_keys),
        ("inputs", config.inputs),
    ):
        for name in names:
            if entry == "return_keys" and name in handler_only:
                reason = "a handler_only_keys entry"
            elif name in owned:
                reason = owned[name]
            else:
                continue
            return f"synthesize {entry} entry {name!r} collides with {reason}"
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
    sample_results: Callable[[Any], tuple[tuple[str, dict[str, Any] | None], ...]] = (
        _no_samples
    )
    check_definition: Callable[
        [Any, Any, Any, Callable[[Any], set[str]]], str | None
    ] = _no_definition_problem
    mentioned_keys: Callable[[Any], tuple[str, ...]] = _mentioned_keys
    typed_keys: Callable[[Any], dict[str, tuple[str, tuple[str, ...]]]] = _typed_keys
    lint_raw: Callable[[str, dict[str, Any], dict[str, Any], Any], list[str]] = _no_lint
    raw_result_key: Callable[[dict[str, Any], Any], str | None] = _no_raw_result_key


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
    lint_raw=_completion_lint_raw,
    raw_result_key=_completion_raw_result_key,
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
    sample_results=_synthesize_samples,
    check_definition=_synthesize_check_definition,
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


def raw_kinds(state: dict[str, Any]) -> tuple[KindSpec, ...]:
    """The specs a RAW state dict holds a config for, in ``KINDS`` order.

    The raw twin of ``kind_of`` (a spec per kind, not only the first): a kind is
    held when the dict's ``config_field`` value is a dict (a non-dict value is
    not a config; the loader refuses it). The default kind has no config field
    and is never returned. ``FSMValidator`` reads raw dicts through it.
    """
    return tuple(
        spec
        for spec in KINDS
        if spec.config_field and isinstance(state.get(spec.config_field), dict)
    )


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


def definition_problem(
    definition: Any, state: Any, evaluator: Any, logic_keys: Callable[[Any], set[str]]
) -> str | None:
    """The first cross-state rule of a kind that *state* breaks, else ``None``.

    Applies to every kind whose config the state holds, in ``KINDS`` order, as
    ``state_problem`` does. The message has no state id; the loader prefixes it.
    *evaluator* is the real ``TransitionEvaluator`` and *logic_keys* is
    ``definitions.logic_referenced_keys``, both passed in because this leaf
    cannot import them.
    """
    for spec in KINDS:
        if not spec.config_field or getattr(state, spec.config_field, None) is None:
            continue
        problem = spec.check_definition(definition, state, evaluator, logic_keys)
        if problem is not None:
            return problem
    return None
