"""The one Pass-1 seam of ``MessagePipeline`` (``Pass1Mixin``): extraction through the state's
executor, then transition evaluation and execution. Shared by the sync and the stream turn.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

from fsm_llm.context import clean_context_keys, commit_with_key_rollback
from fsm_llm.core.definitions import DataExtractionResponse, FSMInstance
from fsm_llm.engine.kinds import executor_for
from fsm_llm.engine.kinds.records import _is_agent_managed, _record_provenance
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .events import record_executor_result, record_executor_start
from .records import _TurnRecord
from .surface import _PipelineSurface


class Pass1Mixin(_PipelineSurface):
    """The Pass-1 seam: extraction through the executor, then the transition."""

    # ----------------------------------------------------------
    # Pass 1: Data extraction + transition
    # ----------------------------------------------------------

    def _execute_extraction_and_transition_pass(
        self,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        *,
        turn: _TurnRecord | None = None,
    ) -> tuple[DataExtractionResponse, bool, str | None]:
        """Execute Pass 1: Data Extraction + Transition Evaluation + Execution.

        ``turn``, when given, receives the transition evaluation outcome.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction and transition pass")

        # Step 1: the state's own Pass-1 extraction. The state's kind picks the
        # executor (the one dispatch seam of both turn paths): a conversation
        # state runs the unified field-based extraction (converts
        # required_context_keys and merges with explicit field_extractions), a
        # completion state its one completion call (engine/kinds/completion.py).
        source_state = self.get_state(instance, conversation_id)
        executor = executor_for(source_state, self._executors)
        record_executor_start(source_state)
        extraction_response = executor.extract(
            self, instance, user_message, conversation_id, source_state
        )
        record_executor_result(source_state, extraction_response)

        # Step 2: Update context with extracted data
        if extraction_response.extracted_data:
            # DECISION plan-2026-10-01T093600-944e2692/D-021: a completion
            # state's data is its one typed result (validated public key,
            # JSON-native CompletionResponse dump) and is committed as is. Do
            # NOT run it through clean_context_keys: that drops None values
            # and internal-prefixed keys at every depth, so `text: None` and a
            # tool argument such as `{"_id": 1}` or `{"x": None}` would be
            # silently removed from the calls the consumer runs. Every other
            # state's model-extracted data is still cleaned. See D-021.
            if executor.cleans_extracted:
                extraction_response.extracted_data = clean_context_keys(
                    extraction_response.extracted_data, conversation_id
                )

            if extraction_response.extracted_data:
                committed = extraction_response.extracted_data

                # DECISION plan-2026-07-19T191147-4b664252/D-005 [STALE]
                # SHAPE (c) of three DIFFERENT partial-commit contracts in the turn pipeline.
                # Roll back ONLY the keys this statement commits, then re-raise:
                #   (a) _execute_state_transition — FULL context snapshot restore, because
                #       a half-applied transition is worse than a lost handler delta.
                #   (b) post-transition CONTEXT_UPDATE (below) — NO rollback; the
                #       transition is already committed and cannot be undone here.
                #   (c) HERE — scoped rollback. Do NOT "align" this with (a) by
                #       deep-copying all of context.data: that would also discard the
                #       deltas of CONTEXT_UPDATE handlers that already SUCCEEDED, which
                #       D-006 guarantees survive at this timing point (pinned by
                #       test_pipeline_handler_contract.py::TestPartialHandlerResultsPreserved).
                # Without this, a permanently-failing critical handler wrote the field
                # once and then blocked every later turn, while the user turn was popped
                # from history — get_data() and history disagreed, and the extraction
                # skip-guard below meant a retry never re-ran. Precedence note: if a
                # handler delta touched one of `committed`'s keys, the rollback wins.
                # The snapshot is SHALLOW on purpose: see `commit_with_key_rollback`
                # (context/commit.py), which holds the BOUNDARY note and does the commit.
                commit_with_key_rollback(
                    instance.context,
                    committed,
                    lambda: self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        current_state=instance.current_state,
                        updated_keys=set(committed.keys()),
                    ),
                    lambda handler_err: log.warning(
                        f"CONTEXT_UPDATE handler failed "
                        f"({type(handler_err).__name__}: {handler_err}), rolling back "
                        f"extracted keys {sorted(committed)}"
                    ),
                )
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-015: recorded only
                # after the handlers succeeded, so a rollback leaves no stale
                # provenance; a handler edit of a key fails the digest compare.
                _record_provenance(instance, committed)

        # Step 3: Transition Evaluation and Execution
        transition_occurred, previous_state = (
            self._execute_transition_evaluation_and_execution(
                instance, user_message, conversation_id, turn=turn
            )
        )

        # Step 4: Post-transition extraction — re-extract in the new state
        # if a transition occurred.  This handles the common case where
        # the user provides data relevant to the *next* state in the same
        # message (e.g., providing email+age when the FSM just collected
        # the name).  Skipped for agent-managed FSMs (detected by
        # "agent_trace" context key) to avoid extra LLM calls.
        is_agent_fsm = _is_agent_managed(instance.context.data)
        if transition_occurred and not is_agent_fsm:
            new_state = self.get_state(instance, conversation_id)
            executor_for(new_state, self._executors).after_transition(
                self,
                instance,
                user_message,
                conversation_id,
                extraction_response,
                previous_state,
                new_state,
            )

        log.debug("Data extraction and transition pass completed")
        return extraction_response, transition_occurred, previous_state
