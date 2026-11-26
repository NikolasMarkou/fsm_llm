"""Module-level helpers the ``API`` mixins share: the error wrapping, the ended-cache read, the
``handle_conversation_errors`` decorator, the classifier-interface check and ``llm_settings_for``.

Layer ``engine`` (plan d454fd62, step 81): split out of the old ``fsm_llm.api``; the function
bodies are unchanged. The helpers that took the class (``api: API``) take ``_ApiSurface`` (the
mypy-only declarations of ``surface.py``) because a mixin passes ``self``, which is not an ``API``;
annotation only, a runtime no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fsm_llm.core.definitions import (
    ConversationBusyError,
    FSMDefinition,
    FSMError,
    MissingExecutorError,
    RunBudgetExceededError,
)
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import StateExecutor, executor_for
from fsm_llm.engine.kinds.targets import TargetResolver
from fsm_llm.llm import LLMInterface, implements_complete
from fsm_llm.logging import logger

if TYPE_CHECKING:
    from .surface import _ApiSurface

# Result type of one `_read_or_ended` read.
_ReadT = TypeVar("_ReadT")


def _unexpected_error(error: Exception, log_text: str, error_text: str) -> FSMError:
    """The ``FSMError`` that stands for an unexpected ``error`` at an ``API`` entry.

    Contract: logs ``"<log_text>: <error>"`` at ERROR and returns
    ``FSMError("<error_text>: <error>")``; the caller raises it ``from error``.
    The one place that defines this wrapping for ``_turn_errors``,
    ``handle_conversation_errors``, ``push_fsm``, ``pop_fsm`` and
    ``start_conversation`` (each keeps its own pass-through clause; ``ValueError``
    and ``FSMError`` pass through unchanged at every one of them).
    """
    logger.error(f"{log_text}: {error!s}")
    return FSMError(f"{error_text}: {error!s}")


def _read_or_ended(
    api: _ApiSurface,
    conversation_id: str,
    read: Callable[[FSMManager, str], _ReadT],
    cache_key: str,
    default: _ReadT,
) -> _ReadT:
    """Read from the live conversation, else from the ended-conversation cache.

    Contract (``get_data``, ``get_current_state``, ``get_conversation_history``):
    ``read(api.fsm_manager, frame_id)`` is the manager call for the top frame.
    Its result is returned. ``ConversationBusyError`` always re-raises. Any other
    ``ValueError``/``KeyError``/``FSMError`` is answered from the cache entry's
    ``cache_key`` (``default`` when absent) only when the conversation is gone
    and an entry exists; otherwise it re-raises. ``has_conversation_ended`` keeps
    its own ladder: it answers ``False``, not an error, for a gone conversation
    with no entry. A module function, not a method (07ad3f8c/D-021).
    """
    # DECISION plan-2026-09-22T080837-8b258a25/D-004
    # All four getters: busy re-raises first; any other error answers from
    # the ended cache ONLY if `_conversation_gone` (stack or instance torn
    # down). Do NOT fall back on a live conversation: a StateNotFoundError
    # there is a real failure, not "ended". Do NOT drop the catch-all:
    # teardown mid-read raises a not-found FSMError. See D-004, D-039.
    try:
        current_fsm_id = api._get_current_fsm_conversation_id(conversation_id)
        return read(api.fsm_manager, current_fsm_id)
    except ConversationBusyError:
        raise
    except (ValueError, KeyError, FSMError):
        cached = api._ended_cache_entry(conversation_id)
        if cached and api._conversation_gone(conversation_id):
            return cast(_ReadT, cached.get(cache_key, default))
        raise


def handle_conversation_errors(
    method_or_error_msg: Callable[..., Any] | str | None = None,
):
    """Decorator for handling common conversation-related errors in FSM_LLM methods.

    Can be used in two ways:
    1. @handle_conversation_errors - uses the method name in error messages
    2. @handle_conversation_errors("Custom error message") - uses the provided message
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, conversation_id, *args, **kwargs):
            error_message = (
                method_or_error_msg
                if isinstance(method_or_error_msg, str)
                else f"Failed in {method.__name__}"
            )

            try:
                return method(self, conversation_id, *args, **kwargs)
            except (ValueError, FSMError):
                raise
            except Exception as e:
                raise _unexpected_error(
                    e, f"Error in {method.__name__}", error_message
                ) from e

        return wrapper

    # Handle both @handle_conversation_errors and @handle_conversation_errors("msg")
    if callable(method_or_error_msg):
        return decorator(method_or_error_msg)
    return decorator


def _require_classifier_interface(
    fsm_def: FSMDefinition, llm_interface: LLMInterface
) -> None:
    """Refuse a definition whose classification the interface cannot send.

    Contract: raises ``ValueError`` when ``fsm_def`` has a
    ``classification_extractions`` entry that classifies through the
    conversation's interface (no ``model`` override, or one equal to the
    interface's ``model``) and ``llm_interface`` does not implement
    ``complete`` (``llm.implements_complete``). Returns ``None`` otherwise.
    Checked once per definition (construction, push) instead of failing
    every turn.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-031: refuse here, once. Do
    # NOT downgrade this to a logged ERROR (library logging is off by
    # default, so it is silent) or rely on the per-turn catch (a missing
    # `complete` raises NotImplementedError, a RuntimeError the pipeline's
    # soft-fail tuple turns into a stay). See D-031.
    if implements_complete(llm_interface):
        return
    injected_model = getattr(llm_interface, "model", None)
    for state in fsm_def.states.values():
        for entry in state.classification_extractions or []:
            if entry.model is None or entry.model == injected_model:
                raise ValueError(
                    f"State '{state.id}' classifies '{entry.field_name}' "
                    f"through the conversation's interface, and "
                    f"{type(llm_interface).__name__} does not implement "
                    "complete (LLMInterface.complete)"
                )


def _require_state_executors(
    fsm_def: FSMDefinition, executors: Mapping[str, StateExecutor]
) -> None:
    """Refuse a definition holding a state whose kind needs an executor this API lacks.

    Contract: ``executors`` is the checked per-instance map (``check_executors``). For every
    state whose kind has ``KindSpec.requires_executor`` set, asks ``executor_for``, the one
    lookup rule, which raises ``MissingExecutorError`` naming the state id, the kind and the
    remedy when neither the map nor the module table holds an executor of the kind. Returns
    ``None`` otherwise; ``synthesize`` is the one built-in kind that sets the flag (D-041 of plan
    baa37f40), so a bare ``API`` refuses a definition holding a synthesize state.
    Checked once per definition (construction, push, the frames of fork and resume) beside
    ``_require_classifier_interface``, instead of failing at the first turn; a bare
    ``FSMManager`` has no such site and fails at its first Pass 1 through ``executor_for``.
    """
    for state in fsm_def.states.values():
        if kind_of(state).requires_executor:
            executor_for(state, executors)


def _require_target_resolver(
    fsm_def: FSMDefinition, targets: TargetResolver | None
) -> None:
    """Refuse a definition holding a state that resolves its target when this API has no resolver.

    Contract: ``targets`` is the checked resolver of the ``API`` (``API(targets=)``) or
    ``None``. For the first state whose kind has ``KindSpec.requires_targets`` set (``call``)
    and no resolver, raises ``MissingExecutorError`` naming the state id, the kind, the
    resolver and the remedy. Returns ``None`` otherwise. Called beside
    ``_require_state_executors`` at its four sites (construction, ``push_fsm``, the frames of
    ``fork`` and ``resume``); a bare ``FSMManager`` has no such site and the call executor
    refuses at its first Pass 1.
    """
    if targets is not None:
        return
    for state in fsm_def.states.values():
        spec = kind_of(state)
        if spec.requires_targets:
            raise MissingExecutorError(
                f"State '{state.id}' is of kind '{spec.name}', which resolves its target "
                "through a TargetResolver and this API has none: pass "
                "targets=<a TargetResolver, for example an ExecutableRegistry> to API",
                state_id=state.id,
                kind=spec.name,
            )


def llm_settings_for(api_kwargs: Mapping[str, Any], **settings: Any) -> dict[str, Any]:
    """The LLM settings a caller may pass to ``API`` beside ``api_kwargs``.

    Interface contract (callers: every subpackage that builds an ``API`` from
    its own config plus a caller's ``API`` keyword arguments: the agents'
    ``BaseAgent._create_api``, the meta-builder, the reasoning engine):
        - ``api_kwargs``: the caller's extra ``API`` keyword arguments.
        - ``settings``: the subpackage's own LLM settings for the interface
          ``API`` builds (``model``, ``temperature``, ``max_tokens``, ...).
        - Returns a copy of ``settings`` when ``api_kwargs`` injects no
          ``llm_interface``; ``{}`` when it does: the injected interface owns
          its model and sampling settings, and ``API`` refuses them beside it.
        - Never raises.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-038: a subpackage's own
    # config defaults (AgentConfig.model/temperature/max_tokens, which always
    # carry a value) configure only the interface core builds. Do NOT pass
    # them beside an injected interface (API refuses them, D-038), and do NOT
    # copy this rule into each subpackage. A setting the CALLER passes beside
    # its interface still reaches API and is refused there. See D-038.
    if api_kwargs.get("llm_interface") is not None:
        return {}
    return dict(settings)


_CONTEXT_NAMES = "resolve_llm step_hook event_log registry cancel budget max_steps max_seconds".split()


# DECISION plan-2026-11-26T022222-7efd5cb4/D-013
# `API` reads a context ONLY by these names (the `check_targets` precedent) and imports nothing of
# `engine.context`. A context beside a keyword naming the same thing is REFUSED, never merged. Do NOT
# import `RunContext` at run time, read `callables` (no host object, no grant) or fold its limits
# into the lifetime `TurnBudget` (they narrow each run-loop call).
def adopt_run_context(
    context: Any, *, llm_interface: Any, event_log: Any, targets: Any
) -> tuple[Any, ...]:
    """Check *context*; return the ``(llm_interface, event_log, targets)`` ``API`` uses (D-013)."""
    if context is None:
        return llm_interface, event_log, targets
    if missing := [n for n in _CONTEXT_NAMES if not hasattr(context, n)]:
        raise TypeError(f"context must provide {', '.join(missing)}, got {context!r}")
    offered = (context.resolve_llm(), context.event_log, context.registry)
    names = ("llm_interface", "event_log", "targets")
    given = (llm_interface, event_log, targets)
    pairs = {n: (a, b) for n, a, b in zip(names, offered, given, strict=True)}
    both = [n for n, (a, b) in pairs.items() if None not in (a, b)]
    if both:
        raise ValueError(f"API(context=...) already supplies {both}: drop the keyword")
    return tuple(b if b is not None else a for a, b in pairs.values())


def refuse_stopped_context(context: Any) -> None:
    """A set token or a spent ledger of *context* raises before a model call; nothing is charged."""
    if context is None:
        return
    context.cancel.raise_if_cancelled()
    if (budget := context.budget) and (spent := budget.spent()):
        limit = getattr(budget, f"max_{spent}")
        raise RunBudgetExceededError(spent, limit, budget.steps_taken)


def _narrowed(name: str, given: Any, ceiling: Any) -> Any:
    if None not in (given, ceiling) and given > ceiling:
        raise ValueError(f"{name}={given!r} is above the context's {ceiling!r}")
    return ceiling if given is None else given


def context_run(context: Any, max_steps: int, max_seconds: float | None) -> Any:
    """``(max_steps, max_seconds, hook)`` of ONE run-loop call: each limit the smaller of the
    caller's and the context's (a larger caller value is a ``ValueError``), the hook a fresh
    ``context.step_hook()``; ``None`` passes the limits through with no hook."""
    if context is None:
        return max_steps, max_seconds, None
    seconds = _narrowed("max_seconds", max_seconds, context.max_seconds)
    return (
        _narrowed("max_steps", max_steps, context.max_steps),
        seconds,
        context.step_hook(),
    )
