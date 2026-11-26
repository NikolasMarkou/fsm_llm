"""
Enhanced API Module for FSM-LLM: Stateful Conversational AI

This module implements the main API interface for the FSM-LLM library, providing developers
with a powerful framework for building stateful conversational AI applications. The enhanced
implementation features a sophisticated 2-pass architecture that separates data extraction,
transition evaluation, and response generation for improved conversation quality and consistency.

Core Architecture
-----------------
The enhanced API implements a 2-pass processing model:

1. **First Pass - Analysis & Transition**:
   - Data extraction from user input using specialized prompts
   - Transition evaluation using configurable logic
   - State management and context updates

2. **Second Pass - Response Generation**:
   - Response generation based on new state and updated context
   - Enhanced prompt building with rich context awareness
   - Consistent, contextually-appropriate responses

Key Features
------------
- **FSM Stacking**: Modular conversation design with push/pop FSM operations
- **Enhanced Context Management**: Sophisticated context inheritance and merging strategies
- **Flexible Handler System**: Extensible event-driven handler architecture
- **Advanced Transition Logic**: JsonLogic-based conditional transitions with custom evaluators
- **Multi-LLM Support**: Pluggable LLM interfaces with default LiteLLM integration
- **Conversation Persistence**: Comprehensive conversation state and history management
- **Error Handling**: Robust error handling with detailed logging and recovery mechanisms

Usage Examples
--------------
Basic conversation with single FSM:

.. code-block:: python

    from fsm_llm import API

    # Initialize from FSM definition file
    api = API.from_file("conversation_fsm.json", model="gpt-4")

    # Start conversation
    conversation_id, initial_response = api.start_conversation()
    print(f"Bot: {initial_response}")

    # Process user messages
    response = api.converse("Hello there!", conversation_id)
    print(f"Bot: {response}")

Advanced FSM stacking for modular conversations:

.. code-block:: python

    # Start main conversation
    conversation_id, response = api.start_conversation()

    # Push specialized FSM for address collection
    address_response = api.push_fsm(
        conversation_id=conversation_id,
        new_fsm_definition="address_collection_fsm.json",
        shared_context_keys=["user_name", "email"],
        preserve_history=True
    )

    # Collect address information...
    # When done, pop back to main FSM
    resume_response = api.pop_fsm(
        conversation_id=conversation_id,
        merge_strategy="update"  # Merge collected address data
    )

Custom handler integration:

.. code-block:: python

    # Create and register custom handler
    validation_handler = (api.create_handler("AddressValidator")
                         .at(HandlerTiming.POST_TRANSITION)
                         .on_state("address_confirmation")
                         .do(validate_address_function))

    api.register_handler(validation_handler)

Layer ``engine`` (plan d454fd62, step 81): this module is the shell of the old ``fsm_llm.api``:
``API`` (the constructor, the factories, ``close``), composed of mixins: ``TurnEntriesMixin``
(``turn_entries.py``: ``start_conversation``, ``converse``, ``advance`` and their streams, moved
out at plan f150ec28 step 1.1 to make room for the async entries), ``RunMixin`` (``run.py``),
``StackMixin`` (``stack.py``), ``LifecycleMixin`` (``lifecycle.py``), ``SessionsMixin``
(``sessions.py``), ``HandlerRegistrationMixin`` (``registration.py``, moved out at plan
5a6a1f80 step 3) and ``ForkMixin`` (``forks.py``, ``fork``, plan 5a6a1f80 step 4). The method
bodies are unchanged.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fsm_llm.constants import (
    DEFAULT_MAX_FSM_CACHE_SIZE,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    DEFAULT_TEMPERATURE,
)
from fsm_llm.core.definitions import ConversationBusyError, FSMDefinition, fsm_id_of
from fsm_llm.core.loaders import load_fsm_definition, load_fsm_from_file
from fsm_llm.core.transition_evaluator import (
    TransitionEvaluator,
    TransitionEvaluatorConfig,
)
from fsm_llm.engine.fsm.budget import check_turn_limits
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import ExecutorsArg, check_executors
from fsm_llm.engine.kinds.targets import TargetResolver, check_targets
from fsm_llm.engine.session import SessionStore
from fsm_llm.handlers import (
    FSMHandler,
    HandlerErrorMode,
    HandlerSystem,
)
from fsm_llm.llm import (
    DEFAULT_MAX_TOKENS,
    LiteLLMInterface,
    LLMInterface,
    interface_model,
    resolve_model,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import EventLog

from .calls import CallMarks
from .events import (
    close_event_log,
    open_event_log,
)
from .forks import ForkMixin
from .helpers import (
    _require_classifier_interface,
    _require_state_executors,
    _require_target_resolver,
    adopt_run_context,
)
from .lifecycle import MAX_ENDED_CONVERSATIONS_CACHE, LifecycleMixin
from .registration import HandlerRegistrationMixin
from .run import RunMixin
from .sessions import SessionsMixin
from .stack import FSMStackFrame, StackMixin
from .turn_entries import TurnEntriesMixin

# --------------------------------------------------------------
# Main API Class
# --------------------------------------------------------------


class API(
    TurnEntriesMixin,
    RunMixin,
    StackMixin,
    LifecycleMixin,
    SessionsMixin,
    HandlerRegistrationMixin,
    ForkMixin,
):
    """
    Enhanced API for Improved 2-Pass FSM-LLM Architecture.

    This class is the public entry point; internally it runs
    the 2-pass architecture for better conversation quality
    and response generation after transition evaluation.
    """

    def __init__(
        self,
        fsm_definition: FSMDefinition | dict[str, Any] | str | os.PathLike[str],
        llm_interface: LLMInterface | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        handlers: list[FSMHandler] | None = None,
        handler_error_mode: HandlerErrorMode | str = HandlerErrorMode.CONTINUE,
        transition_config: TransitionEvaluatorConfig | None = None,
        session_store: SessionStore | None = None,
        handler_timeout: float | None = None,
        max_fsm_cache_size: int = DEFAULT_MAX_FSM_CACHE_SIZE,
        *,
        executors: ExecutorsArg | None = None,
        targets: TargetResolver | None = None,
        event_log: EventLog | str | os.PathLike[str] | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        context: Any = None,
        **llm_kwargs,
    ):
        """
        Initialize API with the 2-pass architecture.

        Args:
            fsm_definition: FSM definition (object, dict, or file path)
            llm_interface: Optional custom LLM interface
            model: LLM model name (if using default interface)
            api_key: API key (if using default interface)
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens for LLM responses
            max_history_size: Exchanges stored per conversation; the prompt shows the
                last ``max_history_size`` messages (default 5)
            max_message_length: Maximum message length
            handlers: Optional list of handlers
            handler_error_mode: What a failing handler does: ``HandlerErrorMode``
                or its string (``"continue"``, the default, or ``"raise"``)
            transition_config: Configuration for transition evaluation
            session_store: Optional store used by save_session/restore_session
            handler_timeout: Seconds a handler may run before it times out
                (``None``, the default, disables the timeout). A timed-out
                handler's thread keeps running as a straggler; the
                ``constants.MAX_TIMED_HANDLER_STRAGGLERS`` cap is shared by
                every conversation of this API (one ``HandlerSystem``).
            max_fsm_cache_size: Bound of the FSM definition LRU cache
                (at least 1; a smaller value raises ``ValueError``)
            executors: Optional ``{kind_name: StateExecutor}`` map, or a list of
                executors each naming its ``kind``, for this ``API`` only: it overrides
                the module table of ``engine/kinds/registry.py`` and never changes
                another ``API``. ``TypeError`` for a value that is neither or holds a
                non-``StateExecutor``, ``ValueError`` for a kind name no ``KindSpec``
                carries, ``NameCollisionError`` for a kind named twice in a list. A
                definition holding a state of a kind that requires an executor and has
                none here or in the module table is refused with
                ``MissingExecutorError`` (at construction, ``push_fsm`` and ``fork``).
                Internal until the kind extension point is public.
            targets: Optional ``TargetResolver`` (an object with
                ``resolve_target(target_id, *, conversation_id=None)``, for example an
                ``ExecutableRegistry``), keyword-only beside ``executors``: how a ``call``
                state resolves its target id to a definition (plan baa37f40 D-021).
                ``TypeError`` for a value with no callable ``resolve_target``. A definition
                holding a state of a kind that resolves a target (``call``) and no resolver
                here is refused with ``MissingExecutorError`` naming the resolver (at
                construction, ``push_fsm`` and ``fork``). Internal until the kind
                extension point is public.
            event_log: Opt-in durable event log: an ``EventLog`` (shared, left
                open by ``close``) or a directory path (the API owns the log and
                closes it). Off by default, and then nothing changes. See
                ``engine/api/events.py``.
            max_steps: Default lifetime turn budget of the root conversation and of
                every frame pushed on it (``FSMManager.start_conversation`` says what
                counts); ``None`` (default) is no limit. A snapshot restore, a fork and
                a resume start a fresh budget from it. An int of at least 1.
            max_seconds: Default lifetime wall-clock budget, as ``max_steps``; a
                number above 0.
            context: Optional ``RunContext`` (read by attribute): its interface, log and registry
                (as ``targets``) fill what the caller left unset (a clash is a ``ValueError``);
                each run loop takes its step hook and its limits narrow the call, not the lifetime.
            **llm_kwargs: Additional LLM parameters
        """
        self._context = context
        llm_interface, event_log, targets = adopt_run_context(
            context, llm_interface=llm_interface, event_log=event_log, targets=targets
        )
        # Handle LLM interface initialization
        if llm_interface is not None:
            if not isinstance(llm_interface, LLMInterface):
                raise ValueError("llm_interface must be an instance of LLMInterface")
            # DECISION plan-2026-10-01T093600-944e2692/D-029 (completed by
            # D-038): every LLM setting beside an injected interface is
            # refused: connection kwargs, `api_key`, `temperature`,
            # `max_tokens`, and a `model` other than the interface's own `str`
            # model (the rule `Classifier(llm_interface=...)` applies). Do NOT accept
            # and ignore any of them: a `seed` (or `caching`, `timeout`, a
            # per-sample `temperature`, ...) was dropped silently while the
            # caller believed it applied. Subpackages omit their own config
            # defaults when an interface is injected (`llm_settings_for`).
            given = sorted(llm_kwargs)
            given += [
                name
                for name, value in (
                    ("api_key", api_key),
                    ("temperature", temperature),
                    ("max_tokens", max_tokens),
                )
                if value is not None
            ]
            if model is not None and model != interface_model(llm_interface):
                given.append("model")
            if given:
                raise ValueError(
                    f"API(llm_interface=...) takes no LLM settings {given}: "
                    "the injected interface owns its model and settings"
                )
            self.llm_interface = llm_interface
            logger.info(
                f"API initialized with custom LLM interface: {type(llm_interface).__name__}"
            )
        else:
            # Create default interface
            model = resolve_model(model)
            temperature = (
                temperature if temperature is not None else DEFAULT_TEMPERATURE
            )
            max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

            self.llm_interface = LiteLLMInterface(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **llm_kwargs,
            )
            logger.info(
                f"API initialized with default LiteLLM interface, model={model}"
            )

        # Process FSM definition
        self.fsm_definition, self.fsm_id = self.process_fsm_definition(fsm_definition)
        _require_classifier_interface(self.fsm_definition, self.llm_interface)
        # The checked map the manager is given and the push and fork sites read.
        self._executors = check_executors(executors)
        _require_state_executors(self.fsm_definition, self._executors)
        self._targets = check_targets(targets)
        _require_target_resolver(self.fsm_definition, self._targets)

        # DECISION plan-2026-11-06T093029-0068b055/D-004
        # No prompt builders are built here: FSMManager builds the defaults with
        # `max_history_messages=max_history_size` (one site, see engine/fsm/manager.py). Do NOT
        # construct config-less builders here again: they would pin the window
        # to 5 messages and make `max_history_size` inert for the model.

        # Create transition evaluator
        evaluator_config = transition_config or TransitionEvaluatorConfig()
        transition_evaluator = TransitionEvaluator(evaluator_config)

        # FSM stacking support (initialized before FSMManager so closure can access it)
        self._temp_fsm_definitions: dict[str, FSMDefinition] = {}
        # DECISION plan-2026-07-21T045419-9925aa3a/D-011
        # In-flight-push guard: ids registered in `_temp_fsm_definitions` but not
        # yet referenced by a live FSMStackFrame. `push_fsm` releases `_stack_lock`
        # between registering the temp def and appending the frame (to run
        # `start_conversation`, which may make a greeting LLM call). Without this
        # set, a concurrent `pop_fsm`/`end_conversation` on ANOTHER conversation
        # calls `_release_unreferenced_temp_definitions` in that window, sees the
        # id as unreferenced (no frame yet), and evicts it — bricking the in-flight
        # sub-conversation with `FSMDefinitionNotFoundError` (a `ValueError`,
        # "Unknown FSM ID ..."). Cleanup treats
        # pending ids as referenced. See D-011.
        self._pending_push_ids: set[str] = set()
        # What `settle` keeps in memory (the one-driver mark per root, refused child ends).
        self._call_marks = CallMarks()

        # Create custom FSM loader
        def custom_fsm_loader(fsm_id: str) -> FSMDefinition:
            if fsm_id == self.fsm_id:
                return self.fsm_definition
            elif fsm_id in self._temp_fsm_definitions:
                return self._temp_fsm_definitions[fsm_id]
            else:
                return load_fsm_definition(fsm_id)

        # Initialize handler system (single instance shared with FSMManager)
        self.handler_system = HandlerSystem(
            handler_error_mode=handler_error_mode, handler_timeout=handler_timeout
        )

        # Initialize enhanced FSM manager with the 2-pass architecture
        self.fsm_manager = FSMManager(
            fsm_loader=custom_fsm_loader,
            llm_interface=self.llm_interface,
            transition_evaluator=transition_evaluator,
            max_history_size=max_history_size,
            max_message_length=max_message_length,
            handler_system=self.handler_system,
            max_fsm_cache_size=max_fsm_cache_size,
            executors=self._executors,
            targets=self._targets,
        )

        # Register provided handlers
        if handlers:
            for handler in handlers:
                self.register_handler(handler)

        # FSM stacking support
        self._stack_lock = threading.Lock()
        self.conversation_stacks: dict[str, list[FSMStackFrame]] = {}
        self._last_accessed: dict[str, float] = {}
        self._ended_conversations: dict[str, dict[str, Any]] = {}
        self._MAX_ENDED_CACHE: int = MAX_ENDED_CONVERSATIONS_CACHE

        # Default lifetime turn budget (``engine/api/budget.py``)
        check_turn_limits(max_steps, max_seconds)
        self._turn_budget = (max_steps, max_seconds)

        # Session persistence
        self._session_store = session_store
        self._event_log, self._owns_event_log = open_event_log(event_log)

        logger.info("Enhanced API fully initialized with the 2-pass architecture")

    @classmethod
    def process_fsm_definition(
        cls, fsm_definition: FSMDefinition | dict[str, Any] | str | os.PathLike[str]
    ) -> tuple[FSMDefinition, str]:
        """Process FSM definition input and return standardized format.

        ``fsm_id`` identifies WHAT the FSM is, never HOW it was constructed
        or loaded -- restore_session's fsm_id-mismatch WARNING (D-011)
        depends on this.
        """
        if isinstance(fsm_definition, FSMDefinition):
            fsm_def = fsm_definition

        elif isinstance(fsm_definition, dict):
            try:
                fsm_def = FSMDefinition(**fsm_definition)
            except Exception as e:
                raise ValueError(f"Invalid FSM definition dictionary: {e!s}") from e

        elif isinstance(fsm_definition, (str, os.PathLike)):
            try:
                fsm_def = load_fsm_from_file(os.fspath(fsm_definition))
            except Exception as e:
                raise ValueError(
                    f"Failed to load FSM from file '{fsm_definition}': {e!s}"
                ) from e

        else:
            raise ValueError(
                f"Invalid FSM definition type: {type(fsm_definition)}. "
                f"Must be FSMDefinition, dict, str or a path"
            )

        # DECISION plan-2026-09-20T114608-a8e47b88/D-016 (its dumped form
        # refined by D-030 of plan 944e2692, now in definition_canonical)
        # ONE content hash, computed from fsm_def's model dump AFTER
        # construction, shared by all three input shapes (dict/
        # FSMDefinition/file path). Do NOT go back to a per-branch id
        # (`fsm_def_`/`fsm_dict_`/`fsm_file_{path}`): the old file-path
        # branch hashed the PATH STRING, not the file's content, so
        # editing a loaded FSM file in place into a semantically
        # different FSM kept the OLD fsm_id and restore_session's
        # mismatch check (D-011) stayed silent -- the exact live-probe
        # failure `findings/review-iter-2.md` CRITICAL 1 reproduced. The
        # old dict/FSMDefinition branches also hashed different inputs
        # (raw dict vs. model_dump()) under different prefixes, so the
        # SAME FSM content loaded two different ways got two different
        # ids -- a false-positive mismatch warning on every restore
        # across construction paths (WARNING 2). Hashing the parsed
        # model's own canonical model_dump() after the if/elif chain
        # fixes both: identical content -> identical id, regardless of
        # dict/object/file/relative-vs-absolute-path construction. A
        # session saved under the OLD path-derived id will trigger a
        # one-time mismatch WARNING against a NEW content-derived id on
        # first restore post-upgrade -- an accepted, correct transition
        # cost per D-011's own "WARNING, not hard-fail" philosophy, not
        # a bug to work around. See decisions.md D-016.
        fsm_id = fsm_id_of(fsm_def)

        return fsm_def, fsm_id

    @classmethod
    def from_file(cls, path: Path | str, **kwargs: Any) -> API:
        """Create API instance from FSM definition file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")
        return cls(fsm_definition=path, **kwargs)

    @classmethod
    def from_definition(
        cls,
        fsm_definition: FSMDefinition | dict[str, Any] | None = None,
        *,
        definition: FSMDefinition | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> API:
        """Create API instance from FSM definition object or dictionary.

        Args:
            fsm_definition: The definition, positionally or by keyword.
            definition: The same argument under the keyword the shipped
                examples use. Give exactly one of the two.
            **kwargs: Passed to the constructor.

        Raises:
            TypeError: both ``fsm_definition`` and ``definition`` were given,
                or neither.
        """
        chosen = fsm_definition if fsm_definition is not None else definition
        if chosen is None or (fsm_definition is not None and definition is not None):
            raise TypeError(
                "from_definition() takes exactly one of 'fsm_definition' and "
                "'definition'"
            )
        return cls(fsm_definition=chosen, **kwargs)

    def get_llm_interface(self) -> LLMInterface:
        """Get current LLM interface."""
        return self.llm_interface

    def close(self) -> None:
        """Clean up all active conversations and release resources.

        A conversation whose turn is still running refuses its end
        (``ConversationBusyError``, a WARNING here): then an owned event log stays OPEN, so the
        turn can write its end; call ``close()`` again after the turn.
        """
        busy = False
        for conversation_id in self.list_active_conversations():
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                busy = busy or isinstance(e, ConversationBusyError)
                logger.warning(
                    f"Error ending conversation {conversation_id} during cleanup: {e!s}"
                )
        if busy:
            logger.warning(
                "Event log left open: a turn is still running; close() again"
            )
        else:
            close_event_log(self)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
