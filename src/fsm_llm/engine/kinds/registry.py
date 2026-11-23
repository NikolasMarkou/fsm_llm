"""The executor table, the per-instance map check and the lookup.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the table,
``register_executor``, ``check_executors`` and ``executor_for`` (with the D-026 anchor) are moved
unchanged. Interface contract: the package docstring of ``fsm_llm.engine.kinds``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

from fsm_llm.core.definitions import MissingExecutorError, State
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.state_kinds import get_kind, kind_of

from .base import StateExecutor
from .call import CallExecutor
from .completion import CompletionExecutor
from .conversation import ConversationExecutor

# The ``executors=`` argument of ``API``, ``FSMManager`` and ``MessagePipeline``: a
# ``{kind_name: executor}`` mapping or a list (or tuple) of executors, each naming its ``kind``.
ExecutorsArg: TypeAlias = Mapping[str, StateExecutor] | Sequence[StateExecutor]

_CONVERSATION = ConversationExecutor()

_CONVERSATION_KIND = "conversation"

# Kind name -> executor: the ``state_kinds`` namespace of the name registry (plan ace7a8c8,
# D-045; it names no entry-point group yet, that wiring is roadmap E2.3 and 5.5).
# ``synthesize`` has no entry on purpose: its executor lives in the subpackage
# ``fsm_llm.synthesis`` (an engine layer may not import it) and a host passes it through
# ``executors=``; the kind sets ``requires_executor``, so a bare engine refuses it with
# ``MissingExecutorError`` (plan baa37f40 D-041). A kind with no executor and no flag still runs
# as conversation through ``executor_for``'s fallback, which honours a per-instance override of
# ``conversation``.
_EXECUTORS: NameRegistry[StateExecutor] = NameRegistry("state_kinds")
_EXECUTORS.register(_CONVERSATION_KIND, _CONVERSATION)
_EXECUTORS.register("completion", CompletionExecutor())
_EXECUTORS.register("call", CallExecutor())


def register_executor(kind_name: str, executor: StateExecutor) -> None:
    """Register (or replace) the executor of the kind named ``kind_name``.

    One executor instance cannot serve two kind names (``NameCollisionError``).
    """
    _EXECUTORS.register(kind_name, executor, override=True)


def _by_kind(executors: Sequence[StateExecutor]) -> dict[str, StateExecutor]:
    """The ``{kind name: executor}`` mapping of the list form of ``executors=``.

    Each entry must be a ``StateExecutor`` with a non-blank ``kind`` (``TypeError`` naming the
    index). A kind named twice raises ``NameCollisionError``: the entries go through one local
    ``NameRegistry``, the same primitive and the same refusal as the module table.
    """
    by_kind: NameRegistry[StateExecutor] = NameRegistry("executors")
    for index, executor in enumerate(executors):
        if not isinstance(executor, StateExecutor):
            raise TypeError(
                f"executors[{index}] must be a StateExecutor, got "
                f"{type(executor).__name__}"
            )
        kind = executor.kind
        if not isinstance(kind, str) or not kind.strip():
            raise TypeError(
                f"executors[{index}] ({type(executor).__name__}) names no kind: set its "
                "`kind` attribute to the state kind it serves, or pass a "
                "{kind_name: executor} mapping"
            )
        by_kind.register(kind, executor)
    return dict(by_kind.items())


def check_executors(
    executors: ExecutorsArg | None,
) -> Mapping[str, StateExecutor]:
    """Validate an ``executors=`` argument and return a read-only copy.

    ``executors`` is a ``{kind_name: executor}`` mapping, a list or tuple of executors (each
    registered under its ``kind``), or ``None`` (empty).

    Raises:
        TypeError: ``executors`` is neither a mapping nor a list or tuple, a key is not a
            ``str``, a value or list entry is not a ``StateExecutor``, or a list entry names
            no ``kind``.
        ValueError: a key (or an entry's ``kind``) names no registered state kind.
        NameCollisionError: a list names one kind twice.
    """
    if executors is None:
        return MappingProxyType({})
    if isinstance(executors, (list, tuple)):
        executors = _by_kind(executors)
    if not isinstance(executors, Mapping):
        raise TypeError(
            "executors must be a mapping of kind name to StateExecutor, or a list of "
            f"StateExecutor, got {type(executors).__name__}"
        )
    checked: dict[str, StateExecutor] = {}
    for name, executor in executors.items():
        if not isinstance(name, str):
            raise TypeError(
                f"executors keys must be kind names (str), got {type(name).__name__}"
            )
        if not isinstance(executor, StateExecutor):
            raise TypeError(
                f"executors[{name!r}] must be a StateExecutor, got "
                f"{type(executor).__name__}"
            )
        try:
            get_kind(name)
        except KeyError:
            raise ValueError(
                f"executors names unknown state kind {name!r}: register its "
                "KindSpec first"
            ) from None
        checked[name] = executor
    return MappingProxyType(checked)


def executor_for(
    state: State, executors: Mapping[str, StateExecutor] | None = None
) -> StateExecutor:
    """The executor of ``state``'s kind.

    Interface contract (callers: the turn pipeline, three sites):
        - Resolution order: the pipeline's own ``executors`` map for the
          kind's name, then the module table for it, then the ``conversation``
          entry of the same two tables in the same order (a kind with no
          executor of its own, a runtime kind nobody registered, runs
          as conversation, so an override of ``conversation``
          in the per-instance map applies to it too), then the built-in
          conversation executor.
        - A kind whose spec says ``requires_executor`` has no such fallback: with no entry of
          its own name in either table, ``MissingExecutorError`` (naming the state id, the kind
          and the remedy). This is the one rule; ``_require_state_executors`` of ``engine/api``
          asks it for every such state of a definition. ``synthesize`` sets the flag (plan baa37f40
          D-041).
        - Raises nothing else; ``kind_of`` raises nothing for a state.
    """
    # DECISION plan-2026-11-12T073150-d454fd62/D-026
    # A passive kind falls back through the per-instance "conversation" entry.
    # Do NOT register ``synthesize`` in the module table as an alias of the
    # built-in conversation executor (it was one): the alias bypassed an
    # ``API(executors={"conversation": X})`` override for a synthesize state
    # (review gate 1, concern 11). A real synthesize executor registers under
    # its own name and then wins.
    # DECISION plan-2026-11-23T122249-baa37f40/D-041
    # [Supersedes plan 2cefeb61 D-002: "a bare converse on a synthesize state stays BLOCKED".]
    # A kind whose spec says ``requires_executor`` (``synthesize``) takes NO conversation
    # fallback: with no executor of its own name it raises ``MissingExecutorError``. Do NOT
    # restore the fallback (a bare engine then BLOCKED forever and wrote nothing, with no word
    # to the host), do NOT put ``SynthesizeExecutor`` in the module table (a subpackage, and one
    # table entry would bind every ``API`` of the process to one synthesizer: a host passes it
    # through ``executors=``) and do NOT make the loader or ``FSMValidator`` refuse the state
    # (core only loads the field; the validator is never stricter than the loader).
    spec = kind_of(state)
    name = spec.name
    for kind in (name,) if spec.requires_executor else (name, _CONVERSATION_KIND):
        if executors is not None and kind in executors:
            return executors[kind]
        module_executor = _EXECUTORS.get(kind)
        if module_executor is not None:
            return module_executor
    if spec.requires_executor:
        state_id = getattr(state, "id", None)
        raise MissingExecutorError(
            f"State '{state_id}' is of kind '{name}', which needs an executor and none is "
            f"registered for it: pass executors={{'{name}': <executor>}} (or a list of "
            f"executors naming their kind) to API, FSMManager or MessagePipeline, or call "
            f"register_executor('{name}', <executor>)",
            state_id=state_id,
            kind=name,
        )
    return _CONVERSATION
