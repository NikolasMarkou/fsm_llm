"""MessagePipeline: The 2-pass message processing engine.

Encapsulates all LLM-driven processing logic extracted from FSMManager:
- Pass 1: Data extraction + transition evaluation + state transition
- Pass 2: Response generation from final state
- Handler execution bridge (deep-copy context, merge deltas)

FSMManager delegates to this class for all message processing.
The pipeline does not own instances or locks — those remain in FSMManager.

Layer ``engine`` (plan d454fd62, step 79): the shell of the old ``fsm_llm.pipeline``: the
constructor, ``get_state`` and the handler bridge ``execute_handlers``. Everything else is a
mixin (``driver``, ``pass1``, ``transitions`` and ``pass2``); the method bodies are unchanged. The
conversation kind's Pass-1 methods are not here since step 80: the conversation executor
(``engine/kinds/conversation/``) owns them and reads this class through ``_PipelineServices``.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from typing import Any

from fsm_llm.constants import (
    PROVENANCE_METADATA_KEY,
    RESERVED_CONTEXT_KEYS,
    has_internal_prefix,
)
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.core.definitions import (
    FSMDefinition,
    FSMInstance,
    State,
    StateNotFoundError,
)
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.classification import Classifier
from fsm_llm.engine.kinds import StateExecutor, check_executors
from fsm_llm.handlers import HandlerSystem, HandlerTiming
from fsm_llm.llm import LLMInterface
from fsm_llm.logging import logger

from .driver import DriverMixin
from .pass1 import Pass1Mixin
from .pass2 import Pass2Mixin
from .transitions import TransitionsMixin


class MessagePipeline(
    DriverMixin,
    Pass1Mixin,
    TransitionsMixin,
    Pass2Mixin,
):
    """2-pass message processing pipeline.

    Handles data extraction, transition evaluation, state transitions,
    response generation, and handler execution. Stateless with respect
    to conversation instances — all state is passed as parameters.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        data_extraction_prompt_builder: DataExtractionPromptBuilder,
        response_generation_prompt_builder: ResponseGenerationPromptBuilder,
        transition_evaluator: TransitionEvaluator,
        handler_system: HandlerSystem,
        fsm_resolver: Callable[[str], FSMDefinition],
        field_extraction_prompt_builder: FieldExtractionPromptBuilder | None = None,
        *,
        executors: Mapping[str, StateExecutor] | None = None,
    ):
        self.llm_interface = llm_interface
        self.data_extraction_prompt_builder = data_extraction_prompt_builder
        self.response_generation_prompt_builder = response_generation_prompt_builder
        self.transition_evaluator = transition_evaluator
        self.handler_system = handler_system
        self.fsm_resolver = fsm_resolver
        self.field_extraction_prompt_builder = (
            field_extraction_prompt_builder or FieldExtractionPromptBuilder()
        )
        # Content-keyed, bounded; see `_get_classifier`. The lock guards
        # check/construct/evict/insert as one unit (review W2, step 2).
        self._classifier_cache: dict[str, Classifier] = {}
        self._classifier_cache_lock = threading.Lock()
        # Per-pipeline executors by kind name (read-only after this line, so any
        # thread may read it); they override the module table of `engine/kinds/registry.py`
        # for this pipeline only. See `executors.check_executors`.
        self._executors = check_executors(executors)

    def get_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Resolve current State from FSM definition."""
        log = (
            logger.bind(conversation_id=conversation_id) if conversation_id else logger
        )

        fsm_def = self.fsm_resolver(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = (
                f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            )
            log.error(error_msg)
            raise StateNotFoundError(error_msg)

        return fsm_def.states[instance.current_state]

    # ----------------------------------------------------------
    # Handler execution bridge
    # ----------------------------------------------------------

    def execute_handlers(
        self,
        instance: FSMInstance,
        timing: HandlerTiming,
        *,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute handlers at specified timing point.

        Everything after ``timing`` is keyword-only: the method lost its old
        third positional parameter ``conversation_id``, and a positional caller
        must fail loudly instead of binding it to ``current_state``.

        Deep-copies instance context before passing to handlers, then merges
        the delta dict back into the instance. A handler returning a key with
        value ``None`` requests deletion of that key.
        """
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-022
        # Zero handlers at this timing: nothing can read or return a delta, so
        # skip the deep copy (14 -> ~4 copies on a zero-handler advance turn, and
        # a non-copyable context value no longer crashes a handler timing). Do NOT
        # extend this to the D-012 pre-turn snapshots below: rollback needs them.
        # A Mock(spec=HandlerSystem) returns a truthy Mock here and keeps the path.
        #
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-029
        # `handlers_at` is an OPTIONAL fast-path hook, not part of the public
        # `handler_system=` seam: a duck-typed system with only
        # `execute_handlers` must keep working. Do NOT call it unguarded (RB-11).
        handlers_at = getattr(self.handler_system, "handlers_at", None)
        if callable(handlers_at) and not handlers_at(timing):
            return
        context = copy.deepcopy(instance.context.data)

        if error_context:
            context.update(error_context)

        def merge_delta(delta: dict[str, Any]) -> None:
            """Apply a handler delta dict to the instance context (None = delete)."""
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-018
            # An ERROR handler runs after the turn was rolled back; merging its
            # returned dict would write into that rolled-back state (RA-07).
            # update_context is the supported write path. Do NOT re-enable.
            if timing is HandlerTiming.ERROR:
                if delta:
                    logger.debug(
                        f"ERROR handler delta not merged (turn rolled back): "
                        f"{sorted(delta)}"
                    )
                return
            # DECISION plan-2026-09-20T114608-a8e47b88/D-018
            # This is the ONE place a handler-returned `None` deletes a
            # context key -- `ContextCompactor.compact`/`prune` included.
            # Those two only ever see `instance.context.data` (a plain dict,
            # via this method's own `context` variable above); they cannot
            # reach `.metadata` themselves and never delete anything
            # directly, so the fix cannot live inside them without changing
            # their public `context: dict[str, Any]` signature. Every
            # None-delta deletion, from ANY handler, passes through here --
            # do NOT special-case `ContextCompactor` with a second deletion
            # site; that would leave the exact same stale-digest gap for
            # every other handler using this same convention. A deleted
            # key's provenance digest must go with it or a pruned
            # low-entropy value's digest can outlive its plaintext in a
            # persisted session file (findings/residual-findings-verify.md
            # #8). See decisions.md D-018.
            for key, value in delta.items():
                # DECISION plan-2026-09-21T203800-8a03483a/D-020
                # Framework-seeded keys (constants.RESERVED_CONTEXT_KEYS) are
                # never set or deleted by a handler delta. Do NOT widen this to
                # every `has_internal_prefix` key: agents handlers own internal
                # keys (`_replan_count`, `_max_iterations`) that must merge.
                # An unchanged echo (a handler returning its whole context) is
                # skipped silently; only a real overwrite/delete warns (D-035).
                if has_internal_prefix(key) and key in RESERVED_CONTEXT_KEYS:
                    stored = instance.context.data.get(key)
                    if value is None or value != stored:
                        logger.warning(
                            f"Handler delta at {timing.name} tried to "
                            f"{'delete' if value is None else 'overwrite'} "
                            f"reserved context key '{key}'; ignored"
                        )
                    continue
                if value is None:
                    instance.context.data.pop(key, None)
                    prov = instance.context.metadata.get(PROVENANCE_METADATA_KEY)
                    if prov is not None:
                        prov.pop(key, None)
                else:
                    instance.context.data[key] = value

        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys,
            )

            if updated_context:
                merge_delta(updated_context)

        except Exception as e:
            # DECISION plan-2026-07-18T051819-80b0bd4d/D-006 [STALE]: re-raise UNCONDITIONALLY.
            # HandlerSystem.execute_handlers raises only when handler_error_mode == "raise" OR
            # the failing handler is critical=True, so anything escaping it already
            # means "must propagate". Do NOT reintroduce an `if handler_error_mode == "raise"`
            # re-gate here: it swallowed every critical failure under "continue".
            #
            # SCOPE OF THE partial-context MERGE BELOW: it preserves what earlier
            # handlers at this timing point already produced, and that survives for
            # CONTEXT_UPDATE. It does NOT survive at POST_TRANSITION:
            # _execute_state_transition intentionally restores the pre-transition
            # snapshot (see the rollback below) taken BEFORE these handlers ran, so
            # the rollback overwrites this merge. That override is deliberate and
            # correct — a half-applied transition is worse than a lost partial
            # delta. Pinned by tests/test_fsm_llm/test_pipeline_handler_contract.py::
            # TestPostTransitionHandlerFailure.  It also does not survive for a key
            # that the pre-transition CONTEXT_UPDATE rollback owns (D-005 shape (c));
            # every other key at that timing point is preserved as stated.
            #
            # UPDATE (D-002, plan-2026-09-12T065608-089d0ec7): for PRE_PROCESSING
            # and POST_PROCESSING specifically, this merge no longer survives a
            # raise that escapes to MessagePipeline.process()/process_stream():
            # those two call sites are now each individually wrapped in a
            # restore-on-exception block that reverts the whole pre-turn snapshot
            # (turn-atomicity), which runs immediately after this merge and undoes
            # it. The merge below still happens (so a caller catching the
            # exception at THIS layer, below process(), still sees the partial
            # delta applied to `instance.context.data` for one statement), but
            # process()/process_stream() wipe it before the exception reaches
            # their caller. See D-002 in decisions.md and
            # test_pipeline_handler_contract.py::TestPartialHandlerResultsPreserved
            # (updated by the same step) for the narrowed contract.
            merge_delta(getattr(e, "partial_context", None) or {})
            logger.error(f"Handler execution error at {timing.name}: {e!s}")
            raise
