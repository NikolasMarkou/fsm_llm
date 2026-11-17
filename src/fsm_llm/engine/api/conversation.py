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
``API`` (the constructor, the factories, ``start_conversation``, the turn entries ``converse`` and
``advance`` with their streams, the handler registration, ``close``), composed of four mixins:
``RunMixin`` (``run.py``), ``StackMixin`` (``stack.py``), ``LifecycleMixin`` (``lifecycle.py``) and
``SessionsMixin`` (``sessions.py``). The method bodies are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from fsm_llm.constants import (
    DEFAULT_MAX_FSM_CACHE_SIZE,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    DEFAULT_TEMPERATURE,
    FSM_ID_HASH_LENGTH,
)
from fsm_llm.core.definitions import AdvanceResult, FSMDefinition, FSMError
from fsm_llm.core.loaders import load_fsm_definition, load_fsm_from_file
from fsm_llm.core.transition_evaluator import (
    TransitionEvaluator,
    TransitionEvaluatorConfig,
)
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import StateExecutor
from fsm_llm.engine.session import SessionStore
from fsm_llm.handlers import (
    BaseHandler,
    FSMHandler,
    HandlerBuilder,
    HandlerErrorMode,
    HandlerSystem,
    HandlerTiming,
    create_handler,
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

from .events import (
    close_event_log,
    event_scope,
    event_stream,
    open_event_log,
    record_started,
)
from .helpers import _require_classifier_interface, _unexpected_error
from .lifecycle import MAX_ENDED_CONVERSATIONS_CACHE, LifecycleMixin
from .run import RunMixin
from .sessions import SessionsMixin
from .stack import FSMStackFrame, StackMixin

# --------------------------------------------------------------
# Main API Class
# --------------------------------------------------------------


# DECISION plan-2026-09-30T062855-07ad3f8c/D-021
# The two helpers below are what every turn entry of ``API`` shares (sync and
# stream). They are MODULE-LEVEL functions on purpose: do NOT turn them into
# ``API`` methods called through ``self``. Under the ``Mock(spec=API)``
# unbound-self tests a ``self._helper()`` call resolves to a Mock, so the error
# mapping and the auto-save would silently not run (the same trap the
# plan-2026-07-21T082818-4c63deac/D-002 anchor in ``converse_stream`` records).
# Do NOT merge them into one wrapper either: the sync turn saves only after a
# successful turn, the stream saves in a ``finally`` (also when abandoned).


@contextmanager
def _turn_errors(doing: str, do: str) -> Iterator[None]:
    """Map what escapes a turn entry of ``API`` to the public error contract.

    Contract: ``doing`` / ``do`` name the entry in the log line and the error
    message (``"processing"`` / ``"process"``, ``"streaming"`` / ``"stream"``).
    ``ValueError`` and ``FSMError`` pass through unchanged; any other
    ``Exception`` is logged and re-raised as ``FSMError`` chained from it.
    ``BaseException`` (``GeneratorExit``, ``KeyboardInterrupt``) is untouched.
    """
    try:
        yield
    except (ValueError, FSMError):
        raise
    except Exception as e:
        raise _unexpected_error(
            e, f"Error {doing} message", f"Failed to {do} message"
        ) from e


def _auto_save_session(api: API, conversation_id: str) -> None:
    """Save the session after a turn when ``api`` has a session store.

    Contract: no-op without a store. Never raises for an ``Exception``: a
    failed save is logged at WARNING and the turn's result stands.
    """
    if api._session_store is None:
        return
    try:
        api.save_session(conversation_id)
    except Exception as e:
        logger.warning(f"Auto-save session failed: {e!s}")


class API(RunMixin, StackMixin, LifecycleMixin, SessionsMixin):
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
        executors: Mapping[str, StateExecutor] | None = None,
        event_log: EventLog | str | os.PathLike[str] | None = None,
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
            executors: Optional ``{kind_name: StateExecutor}`` map for this
                ``API`` only: it overrides the module table of ``engine/kinds/registry.py``
                and never changes another ``API``. ``TypeError`` for a value
                that is not a mapping of ``StateExecutor``, ``ValueError`` for
                a kind name no ``KindSpec`` carries. Internal until the kind
                extension point is public.
            event_log: Opt-in durable event log: an ``EventLog`` (shared, left
                open by ``close``) or a directory path (the API owns the log and
                closes it). Off by default, and then nothing changes. See
                ``engine/api/events.py``.
            **llm_kwargs: Additional LLM parameters
        """
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
            executors=executors,
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
        # refined by D-030 of plan 944e2692, below)
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
        # DECISION plan-2026-10-01T093600-944e2692/D-030 (refines D-016): the
        # hash covers `model_dump(exclude_defaults=True)`, so a field left at
        # its default (given explicitly or omitted) is not in it and adding an
        # optional field to the models changes no id. Do NOT hash the full
        # dump (every new optional field re-ids every FSM, as `completion`
        # and `handler_only_keys` did), and do NOT use `exclude_unset` (an
        # explicit default and an omitted field then hash differently) or
        # `exclude_none` (an explicit None merges into a non-None default).
        # See D-030.
        content_hash = hashlib.sha256(
            json.dumps(
                fsm_def.model_dump(exclude_defaults=True), sort_keys=True
            ).encode()
        ).hexdigest()[:FSM_ID_HASH_LENGTH]
        fsm_id = f"fsm_{fsm_def.name}_{content_hash}"

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

    def start_conversation(
        self,
        initial_context: dict[str, Any] | None = None,
        *,
        _suppress_start: bool = False,
    ) -> tuple[str, str]:
        """
        Start new conversation with the 2-pass architecture.

        Args:
            initial_context: Optional initial context data
            _suppress_start: Internal resume flag forwarded to
                ``FSMManager.start_conversation(suppress_start=...)``. When True
                the START_CONVERSATION handlers and the Pass-2 greeting are
                skipped (used by ``restore_session``). Not part of the public
                API; default False preserves normal start behavior.

        Returns:
            Tuple of (conversation_id, initial_response). The response is
            the empty string when the initial state is silent (empty
            ``response_instructions``): no LLM call, nothing in the history.

        Raises:
            FSMError: ``initial_context`` is neither ``None`` nor a dict, or the
                start failed (any other unexpected error is wrapped in it).
            ValueError: passes through unchanged, as at ``push_fsm`` and
                ``pop_fsm``.
        """
        try:
            # Start conversation using enhanced FSM manager
            conversation_id, response = self.fsm_manager.start_conversation(
                self.fsm_id,
                initial_context=initial_context,
                suppress_start=_suppress_start,
            )

            # Track conversation and initialize stack
            with self._stack_lock:
                self.conversation_stacks[conversation_id] = [
                    FSMStackFrame(
                        fsm_definition=self.fsm_definition,
                        conversation_id=conversation_id,
                        fsm_id=self.fsm_id,
                    )
                ]
                self._last_accessed[conversation_id] = time.monotonic()

            if not _suppress_start:  # a restore writes its own record (restore_session)
                record_started(self, conversation_id)
            return conversation_id, response

        except (ValueError, FSMError):
            raise
        except Exception as e:
            raise _unexpected_error(
                e, "Error starting conversation", "Failed to start conversation"
            ) from e

    def converse(self, user_message: str, conversation_id: str) -> str:
        """
        Process message using the 2-pass architecture.

        Args:
            user_message: User's message
            conversation_id: Existing conversation ID

        Returns:
            System response; the empty string when the turn ended on a silent
            state (empty ``response_instructions``), which makes no Pass-2
            call and records the user message only.
        """
        with _turn_errors("processing", "process"):
            # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            with event_scope(self, conversation_id):
                response: str = self.fsm_manager.process_message(
                    current_fsm_id, user_message
                )
            # Auto-save session if store is configured
            _auto_save_session(self, conversation_id)
            return response

    def advance(self, conversation_id: str) -> AdvanceResult:
        """Run one step of the current state with no user message.

        The same turn as ``converse`` (one turn at a time per conversation,
        rollback on failure, every handler timing, extraction, transition
        evaluation, Pass 2 from the post-transition state, session auto-save),
        resolved on the top of the FSM stack. Differences from ``converse``:
        nothing is appended to the history for a user and prompts carry no
        user message.

        Args:
            conversation_id: Existing conversation ID.

        Returns:
            ``AdvanceResult`` with the state before and after, the transition
            outcome, the reply (``None`` for a silent state) and ``ended``.

        Raises:
            ValueError: unknown conversation ID.
            FSMError: the conversation has ended (terminal state), a turn is
                already running for it (a handler calling back, or another
                thread), or the step failed. A failed step is rolled back as a
                ``converse`` turn is: a Pass-2 or POST_PROCESSING failure
                restores the whole turn; a failure in a Pass-1 handler
                (CONTEXT_UPDATE, PRE_TRANSITION, POST_TRANSITION) follows the
                same partial-commit rules as ``converse`` (keys extracted
                before the failure can stay).
        """
        with _turn_errors("advancing without a", "advance without a"):
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            with event_scope(self, conversation_id):
                result: AdvanceResult = self.fsm_manager.advance(current_fsm_id)
            _auto_save_session(self, conversation_id)
            return result

    def converse_stream(self, user_message: str, conversation_id: str) -> Iterator[str]:
        """Process message, streaming the response tokens.

        Pass 1 (extraction + transitions) runs fully.  Pass 2 yields
        response tokens as they arrive from the LLM.

        This is a thin NON-generator wrapper: it resolves the current FSM
        conversation id at CALL time — which validates existence AND refreshes
        ``_last_accessed`` — then returns a lazy nested-closure generator that
        streams the reply.  Resolving eagerly means a created-but-not-yet-iterated
        stream cannot skip validation or be reaped as stale mid-flight by
        ``cleanup_stale_conversations``.

        Args:
            user_message: User's message.
            conversation_id: Existing conversation ID.

        Yields:
            String chunks of the response as they arrive; nothing when the
            turn ended on a silent state.
        """
        with _turn_errors("streaming", "stream"):
            # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
            # Runs at CALL time (this is a plain function, not a generator).
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)

        # DECISION plan-2026-07-21T082818-4c63deac/D-002
        # conv_lock acquisition stays LAZY inside this nested-closure generator
        # (via ``process_message_stream``'s own inner generator), so an abandoned
        # (never-iterated) stream cannot leak a lock and the ``_lock -> conv_lock``
        # order is preserved.  The generator is a NESTED CLOSURE (capturing
        # ``self``/args from scope) and deliberately NOT a ``self._..._inner``
        # method: a bound-method self-dispatch resolves to a Mock under
        # ``Mock(spec=API)`` unbound-self tests and regressed the agents
        # auto-save suite (D-003/CF1).  Do NOT hoist conv_lock into the eager
        # prologue and do NOT reintroduce a ``self.``-attribute inner generator.
        def _stream() -> Iterator[str]:
            with _turn_errors("streaming", "stream"):
                try:
                    yield from event_stream(
                        self,
                        conversation_id,
                        self.fsm_manager.process_message_stream(
                            current_fsm_id, user_message
                        ),
                    )
                finally:
                    # Auto-save session after stream completes or is abandoned
                    _auto_save_session(self, conversation_id)

        return _stream()

    def advance_stream(self, conversation_id: str) -> Iterator[str]:
        """Run one step of the current state with no user message, streaming
        the reply.

        The stream form of ``advance``: the same turn, resolved on the top of
        the FSM stack at CALL time, run lazily. Only reply text is yielded; a
        silent state (empty ``response_instructions``) yields no chunk, makes
        no Pass-2 LLM call and appends nothing to the history.
        Read the outcome of the step after the stream is exhausted with
        ``get_current_state`` and ``has_conversation_ended``. The conversation
        is claimed at the first ``next()`` and released when the stream
        finishes, fails or is closed; a stream closed early keeps the partial
        reply in the history and keeps the step (as ``converse_stream`` does).

        Args:
            conversation_id: Existing conversation ID.

        Yields:
            String chunks of the reply as they arrive.

        Raises:
            ValueError: unknown conversation ID (at call time).
            FSMError: at the first ``next()``, when the conversation has ended
                (terminal state) or a turn is already running for it; or when
                the step failed, after its rollback.
        """
        with _turn_errors("streaming a step without a", "stream a step without a"):
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)

        # Lazy nested closure, as in ``converse_stream`` (anchor
        # plan-2026-07-21T082818-4c63deac/D-002 there): no lock is taken here
        # and the generator is not a ``self.`` attribute.
        def _stream() -> Iterator[str]:
            with _turn_errors("streaming a step without a", "stream a step without a"):
                try:
                    yield from event_stream(
                        self,
                        conversation_id,
                        self.fsm_manager.advance_stream(current_fsm_id),
                    )
                finally:
                    _auto_save_session(self, conversation_id)

        return _stream()

    # ==========================================
    # HANDLER MANAGEMENT METHODS
    # ==========================================

    def register_handler(self, handler: FSMHandler) -> None:
        """Register handler with the system."""
        self.handler_system.register_handler(handler)

    def register_handlers(self, handlers: list[FSMHandler]) -> None:
        """Register multiple handlers."""
        for handler in handlers:
            self.register_handler(handler)

    def create_handler(
        self,
        name: str = "CustomHandler",
        timing: HandlerTiming | None = None,
        action: Any | None = None,
    ) -> HandlerBuilder:
        """Create new handler using fluent builder.

        When *timing* and *action* are both provided the handler is built
        and registered automatically, providing a convenient shorthand::

            fsm.create_handler(
                name="on_start",
                timing=HandlerTiming.START_CONVERSATION,
                action=lambda ctx: print("started"),
            )
        """
        # mypy: `do()` returns a built BaseHandler (not the builder), so the local
        # is annotated as the union. register_handler receives a BaseHandler at
        # runtime (only reachable after `.do()` ran); cast narrows for the Protocol.
        # The declared `-> HandlerBuilder` return is pre-existing and unchanged;
        # cast preserves the exact runtime object returned. Annotation-only.
        builder: HandlerBuilder | BaseHandler = create_handler(name)
        if timing is not None:
            builder = cast(HandlerBuilder, builder).at(timing)
        if action is not None:
            builder = cast(HandlerBuilder, builder).do(action)
        if timing is not None and action is not None:
            self.register_handler(cast(FSMHandler, builder))
        return cast(HandlerBuilder, builder)

    def get_llm_interface(self) -> LLMInterface:
        """Get current LLM interface."""
        return self.llm_interface

    def close(self) -> None:
        """Clean up all active conversations and release resources."""
        for conversation_id in self.list_active_conversations():
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                logger.warning(
                    f"Error ending conversation {conversation_id} during cleanup: {e!s}"
                )
        close_event_log(self)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
