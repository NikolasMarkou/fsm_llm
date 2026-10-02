"""Fluent builders for the two core constructors: ``APIBuilder``, ``FSMManagerBuilder``.

Both follow the shared builder convention: ``set_<x>`` records one value,
``add_<x>`` appends, mutators only record and return ``self``, and ``build()``
is the only validator. ``build()`` hands the real constructor ONLY the values
that were set, so every default and every rule (the ``llm_interface`` refusal,
``max_fsm_cache_size >= 1``, definition validation) lives in one place.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import Any, TypeVar

from .api import API
from .definitions import BuildError, FSMDefinition
from .fsm import FSMManager

_T = TypeVar("_T")


def _construct(product: Callable[..., _T], what: str, **kwargs: Any) -> _T:
    """Call a real constructor; wrap its ``ValueError``/``TypeError`` in ``BuildError``."""
    # DECISION plan-2026-10-02T052921-89b03f61/D-008
    # The builder calls the real constructor with only the values that were
    # set and wraps its error. Do NOT copy the llm_interface-vs-LLM-settings
    # refusal, range checks or defaults here, and do NOT call
    # `llm_settings_for`: a builder-side copy would drift from the constructor.
    try:
        return product(**kwargs)
    except (ValueError, TypeError) as exc:
        raise BuildError(f"Cannot build {what}: {exc}", errors=[str(exc)]) from exc


class APIBuilder:
    """Builds an :class:`API`.

    Interface contract: every ``set_*`` / ``add_*`` returns ``self``;
    ``build()`` returns a new ``API`` or raises ``BuildError`` (chained from the
    constructor's ``ValueError``/``TypeError``) with ``.errors``. A missing
    definition raises ``BuildError``. ``set_llm_option(name, value)`` passes
    open-ended litellm kwargs (``seed``, ``timeout``, ...) unfiltered.

    Isolation: a dict definition, an ``FSMDefinition``, the handler list and
    the option dict are copied at ``build()``; objects passed by reference (an
    ``LLMInterface``, a session store, the handlers themselves, a transition
    config) stay shared with the caller. A second ``build()`` is independent.
    """

    def __init__(self) -> None:
        self._definition: FSMDefinition | dict[str, Any] | str | None = None
        self._handlers: list[Any] = []
        self._llm_options: dict[str, Any] = {}
        self._set: dict[str, Any] = {}

    def _put(self, name: str, value: Any) -> APIBuilder:
        self._set[name] = value
        return self

    def set_definition(
        self, definition: FSMDefinition | dict[str, Any] | str | os.PathLike[str]
    ) -> APIBuilder:
        self._definition = (
            os.fspath(definition) if isinstance(definition, os.PathLike) else definition
        )
        return self

    def set_llm_interface(self, llm_interface: Any) -> APIBuilder:
        return self._put("llm_interface", llm_interface)

    def set_model(self, model: str) -> APIBuilder:
        return self._put("model", model)

    def set_api_key(self, api_key: str) -> APIBuilder:
        return self._put("api_key", api_key)

    def set_temperature(self, temperature: float) -> APIBuilder:
        return self._put("temperature", temperature)

    def set_max_tokens(self, max_tokens: int) -> APIBuilder:
        return self._put("max_tokens", max_tokens)

    def set_llm_option(self, name: str, value: Any) -> APIBuilder:
        """Record one open-ended litellm kwarg; repeatable, never filtered."""
        self._llm_options[name] = value
        return self

    def add_handler(self, handler: Any) -> APIBuilder:
        self._handlers.append(handler)
        return self

    def set_handler_error_mode(self, mode: str) -> APIBuilder:
        return self._put("handler_error_mode", mode)

    def set_transition_config(self, config: Any) -> APIBuilder:
        return self._put("transition_config", config)

    def set_session_store(self, store: Any) -> APIBuilder:
        return self._put("session_store", store)

    def set_handler_timeout(self, seconds: float | None) -> APIBuilder:
        return self._put("handler_timeout", seconds)

    def set_max_history_size(self, size: int) -> APIBuilder:
        return self._put("max_history_size", size)

    def set_max_message_length(self, length: int) -> APIBuilder:
        return self._put("max_message_length", length)

    def set_max_fsm_cache_size(self, size: int) -> APIBuilder:
        return self._put("max_fsm_cache_size", size)

    def build(self) -> API:
        if self._definition is None:
            raise BuildError("Cannot build API: no FSM definition set")
        definition = self._definition
        if isinstance(definition, FSMDefinition):
            definition = definition.model_copy(deep=True)
        elif isinstance(definition, dict):
            definition = copy.deepcopy(definition)
        kwargs = dict(self._set)
        if self._handlers:
            kwargs["handlers"] = list(self._handlers)
        kwargs.update(self._llm_options)
        return _construct(API, "API", fsm_definition=definition, **kwargs)


class FSMManagerBuilder:
    """Builds an :class:`FSMManager`.

    Interface contract: one ``set_*`` per constructor parameter, each returning
    ``self``; ``build()`` returns a new ``FSMManager`` or raises ``BuildError``
    chained from the constructor's error. ``llm_interface`` is required (the
    constructor refuses ``None``, so ``build()`` without it raises
    ``BuildError``). Objects passed (loader, interface, evaluator, prompt
    builders, handler system) stay shared with the caller; each ``build()``
    makes a fresh manager with its own caches and locks.
    """

    def __init__(self) -> None:
        self._set: dict[str, Any] = {}

    def _put(self, name: str, value: Any) -> FSMManagerBuilder:
        self._set[name] = value
        return self

    def set_fsm_loader(
        self, loader: Callable[[str], FSMDefinition]
    ) -> FSMManagerBuilder:
        return self._put("fsm_loader", loader)

    def set_llm_interface(self, llm_interface: Any) -> FSMManagerBuilder:
        return self._put("llm_interface", llm_interface)

    def set_data_extraction_prompt_builder(self, builder: Any) -> FSMManagerBuilder:
        return self._put("data_extraction_prompt_builder", builder)

    def set_response_generation_prompt_builder(self, builder: Any) -> FSMManagerBuilder:
        return self._put("response_generation_prompt_builder", builder)

    def set_field_extraction_prompt_builder(self, builder: Any) -> FSMManagerBuilder:
        return self._put("field_extraction_prompt_builder", builder)

    def set_transition_evaluator(self, evaluator: Any) -> FSMManagerBuilder:
        return self._put("transition_evaluator", evaluator)

    def set_max_history_size(self, size: int) -> FSMManagerBuilder:
        return self._put("max_history_size", size)

    def set_max_message_length(self, length: int) -> FSMManagerBuilder:
        return self._put("max_message_length", length)

    def set_handler_system(self, handler_system: Any) -> FSMManagerBuilder:
        return self._put("handler_system", handler_system)

    def set_handler_error_mode(self, mode: str) -> FSMManagerBuilder:
        return self._put("handler_error_mode", mode)

    def set_max_fsm_cache_size(self, size: int) -> FSMManagerBuilder:
        return self._put("max_fsm_cache_size", size)

    def build(self) -> FSMManager:
        return _construct(FSMManager, "FSMManager", **self._set)
