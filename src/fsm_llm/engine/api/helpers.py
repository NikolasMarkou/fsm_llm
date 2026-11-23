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

from fsm_llm.core.definitions import ConversationBusyError, FSMDefinition, FSMError
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import StateExecutor, executor_for
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
    ``None`` otherwise; no built-in kind sets the flag yet, so nothing is refused today.
    Checked once per definition (construction, push, the frames of fork and resume) beside
    ``_require_classifier_interface``, instead of failing at the first turn; a bare
    ``FSMManager`` has no such site and fails at its first Pass 1 through ``executor_for``.
    """
    for state in fsm_def.states.values():
        if kind_of(state).requires_executor:
            executor_for(state, executors)


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
