from __future__ import annotations

"""
MessagePipeline: The 2-pass message processing engine.

Encapsulates all LLM-driven processing logic extracted from FSMManager:
- Pass 1: Data extraction + transition evaluation + state transition
- Pass 2: Response generation from final state
- Handler execution bridge (deep-copy context, merge deltas)

FSMManager delegates to this class for all message processing.
The pipeline does not own instances or locks — those remain in FSMManager.
"""

import copy
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.ast import Term
    from ..runtime.oracle import LiteLLMOracle

from ..constants import (
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
)
from ..context import clean_context_keys
from ..handlers import (
    CONTEXT_DATA_VAR,
    CURRENT_STATE_VAR,
    HANDLER_RUNNER_VAR_NAME,
    TARGET_STATE_VAR,
    UPDATED_KEYS_VAR,
    HandlerSystem,
    HandlerTiming,
    make_handler_runner,
    required_env_bindings,
)
from ..logging import logger
from ..runtime._litellm import LLMInterface
from .._models import (
    ClassificationError,
    DataExtractionResponse,
    FieldExtractionResponse,
    InvalidTransitionError,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    StateNotFoundError,
    TransitionEvaluationResult,
)
from .classification import Classifier
from .definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    FieldExtractionConfig,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
    TransitionEvaluation,
    TransitionOption,
)
from .extraction import ExtractionEngine
from .prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator


@dataclass
class _TurnState:
    """Per-turn shared state threaded through compiled-path callback closures.

    # DECISION D-S8b-01 — per-turn mutable env state
    # One instance is allocated per `process_compiled` call and captured by
    # reference in every `_make_cb_*` closure. `CB_EVAL_TRANSIT` writes
    # `last_evaluation`; `CB_RESOLVE_AMBIG` reads it. `CB_EXTRACT*` writes
    # `extraction_response`; `CB_RESPOND` reads it. Mutable-in-env is the
    # sanctioned pattern for stateful bindings under eager-Let sequencing
    # (LESSONS: "λ-Kernel Host-Callable Escape Hatch"). The cost is a small
    # departure from the paper's "callbacks are self-contained" framing,
    # accepted to preserve byte-for-byte semantic fidelity with the original
    # transition-evaluation-and-execution logic (retired in S11). See
    # plans/plan_2026-04-24_4ec5abc0/decisions.md#D-S8b-01.
    """

    extraction_response: DataExtractionResponse | None = None
    last_evaluation: TransitionEvaluation | None = None
    transition_occurred: bool = False
    previous_state: str | None = None
    # True once the first extraction callback has run the full
    # `_execute_data_extraction` dispatcher for this turn. Subsequent
    # extraction callbacks are no-ops so per-callback bindings do not
    # diverge from legacy dispatcher semantics (which runs bulk/field/
    # class in a coordinated single pass). See _make_cb_extract family.
    extraction_dispatcher_ran: bool = False
    # A.D4(b) — caller's streaming intent for this turn. Set to True by
    # `process_stream_compiled` before env-build (step 5) so that
    # response-position callbacks (D1 synthetic, eventually D3 fallback)
    # can return an Iterator[str] under streaming. D2's CB_APPEND_HISTORY
    # iterator-aware wrap dispatches on the value type — does NOT need to
    # read this flag. False default preserves byte-equivalence for every
    # non-streaming caller. See plan_2026-04-28_ca542489 step 3.
    stream: bool = False


class MessagePipeline:
    """2-pass message processing pipeline.

    Handles data extraction, transition evaluation, state transitions,
    response generation, and handler execution. Stateless with respect
    to conversation instances — all state is passed as parameters.

    Entry points:

    - :meth:`process_compiled` — compiled λ-term dispatch for the 2-pass
      flow. Primary production path post-S11.
    - :meth:`process_stream_compiled` — Pass 1 synchronous, Pass 2 streams
      via the compiled λ-term. See D-S8-00..03, D-S9-00, D-S10-00,
      D-S11-00 for scope and design rationale.
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
        compiled_term_resolver: Callable[[str], Term] | None = None,
        oracle: LiteLLMOracle | None = None,
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
        # S8b: resolver for compiled λ-terms. When None (default),
        # `process_compiled` falls back to the S8-probe inline compile path
        # (for backward-compat with tests constructing MessagePipeline
        # directly). When supplied (by FSMManager), hits the S7 LRU cache.
        self.compiled_term_resolver = compiled_term_resolver
        # M4 (merge spec §3 I1+I2) — Program owns exactly one Oracle.
        # When supplied by FSMManager (via API, via Program), we field-read
        # ``self._oracle`` instead of constructing ``LiteLLMOracle(self.llm_interface)``
        # at each call site. When None (back-compat for tests that
        # construct MessagePipeline directly with only ``llm_interface``),
        # we lazily wrap the llm_interface so the field is always live.
        # The 7 prior LiteLLMOracle(...) construction sites collapse to
        # ``self._oracle`` field-reads — see test_oracle_ownership.py.
        if oracle is None:
            from ..runtime.oracle import LiteLLMOracle

            oracle = LiteLLMOracle(llm_interface)
        self._oracle = oracle
        # Phase C (0.8.0) — Pass-1 extraction cluster delegated to a
        # dedicated engine. The engine reads ``self._oracle``,
        # ``self.llm_interface``, prompt builders, and ``fsm_resolver``
        # via property pass-throughs so the M4 single-Oracle invariant
        # holds and runtime mutations on the pipeline propagate.
        self._extraction = ExtractionEngine(self)

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
        conversation_id: str,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute handlers at specified timing point.

        Deep-copies instance context before passing to handlers, then merges
        the delta dict back into the instance. A handler returning a key with
        value ``None`` requests deletion of that key.
        """
        context = copy.deepcopy(instance.context.data)

        if error_context:
            context.update(error_context)

        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys,
            )

            if updated_context:
                for key, value in updated_context.items():
                    if value is None:
                        instance.context.data.pop(key, None)
                    else:
                        instance.context.data[key] = value

        except Exception as e:
            logger.error(f"Handler execution error at {timing.name}: {e!s}")
            if self.handler_system.error_mode == "raise":
                raise

    # ----------------------------------------------------------
    # R5 step 4 — handler-runner env extension for spliced terms
    # ----------------------------------------------------------

    def _build_handler_env_extension(
        self,
        instance: FSMInstance,
        *,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build the env bindings required by a handler-spliced compiled term.

        Plan_43d56276 step 4 (R5 narrow, D-STEP-04-RESOLUTION). The
        composed term emitted by ``handlers.compose`` references seven
        env-bound names per splice point:

        * :data:`HANDLER_RUNNER_VAR_NAME` — the host-callable that
          dispatches to :class:`HandlerSystem` (per-turn variant: merges
          deltas back into ``instance.context.data`` and obeys
          ``error_mode``, mirroring the pre-R5
          :meth:`MessagePipeline.execute_handlers` semantics).
        * :data:`CURRENT_STATE_VAR`, :data:`TARGET_STATE_VAR` — the
          state ids passed to handlers' ``should_execute``.
        * :data:`CONTEXT_DATA_VAR` — the live context dict (the runner
          deep-copies internally; merge-back happens after).
        * :data:`UPDATED_KEYS_VAR` — only meaningful at CONTEXT_UPDATE
          (kept host-side per D-STEP-04-RESOLUTION); ``None`` here.
        * The 8 ``_handler_timing_<value>`` constants returned by
          :func:`required_env_bindings` so the splicer can reference
          timing strings via Vars rather than literal embedding.

        Only PRE/POST_PROCESSING actually invoke the runner under R5
        narrow; the other 6 splices are identity. The bindings are
        emitted unconditionally so the env contract holds for any
        future term shape (forward-compat).
        """

        base_runner = make_handler_runner(self.handler_system)

        def per_turn_runner(
            timing_str: str,
            current_state_arg: str,
            target_state_arg: str | None,
            context_arg: dict[str, Any],
            updated_keys_arg: set[str] | None = None,
        ) -> dict[str, Any]:
            # Deep-copy mirrors pre-R5 :meth:`execute_handlers` semantics
            # (line 231 above): handlers see an isolated dict so failed
            # handlers cannot corrupt the live instance.
            ctx_copy = copy.deepcopy(context_arg)
            try:
                updated_context = base_runner(
                    timing_str,
                    current_state_arg,
                    target_state_arg,
                    ctx_copy,
                    updated_keys_arg,
                )
            except Exception as exc:
                logger.error(f"Handler execution error at {timing_str}: {exc!s}")
                if self.handler_system.error_mode == "raise":
                    raise
                return {}

            # Merge back into the live instance. None values request key
            # deletion (mirrors pre-R5 :meth:`execute_handlers` behavior).
            if updated_context:
                for key, value in updated_context.items():
                    if value is None:
                        instance.context.data.pop(key, None)
                    else:
                        instance.context.data[key] = value
            return updated_context or {}

        env: dict[str, Any] = {
            HANDLER_RUNNER_VAR_NAME: per_turn_runner,
            CURRENT_STATE_VAR: current_state or instance.current_state,
            TARGET_STATE_VAR: target_state,
            CONTEXT_DATA_VAR: instance.context.data,
            UPDATED_KEYS_VAR: updated_keys,
        }
        env.update(required_env_bindings())
        return env

    # ----------------------------------------------------------
    # Compiled-term dispatch (2-pass processing, S8+)
    # ----------------------------------------------------------

    def process_compiled(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        *,
        tier: int | None = None,
    ) -> str:
        # DECISION D-S8-01 / D-S11-00 — compiled-path dispatch (tier-widened).
        # S8-probe originally routed a narrow response-only cohort through
        # the compiled λ-term as a correctness probe. S8b parameterized by
        # `tier` and widened cohort support. S9/S10 flipped the default;
        # S11 deleted the legacy `process` / `process_stream` methods — this
        # is now the only 2-pass dispatch path. Callbacks above the current
        # tier bind to a sentinel that raises NotImplementedError
        # (D-S8-03 / D-S8b-02). When `compiled_term_resolver` is wired,
        # default tier is 3 (full cohort); otherwise default tier is 0
        # (probe-only) for back-compat with tests constructing
        # MessagePipeline directly.
        from ..runtime.executor import Executor
        from .compile_fsm import compile_fsm

        if tier is None:
            tier = 3 if self.compiled_term_resolver is not None else 0

        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
            fsm_def = self.fsm_resolver(instance.fsm_id)
            self._check_compiled_cohort(fsm_def, tier=tier)

            # PRE_PROCESSING / POST_PROCESSING are now spliced into the
            # compiled term (R5 step 4, D-STEP-04-RESOLUTION). The host-
            # side `execute_handlers` calls that previously bracketed the
            # `Executor().run(...)` invocation are deleted. The composed
            # term emitted by `dialog/fsm.py::get_composed_term` carries
            # the splice; here we extend `env` with the runner + state
            # bindings so the spliced host_call nodes evaluate.

            # Build env + unwrap 4 Abs layers to reach the inner Case
            # (F3 — compiled term expects pre-bound inputs in env).
            # compile_fsm guarantees the outer shape is
            # Abs→Abs→Abs→Abs→Case; narrow with assertions to satisfy
            # the tagged-union type checker. S8b: prefer the resolver
            # (LRU-cached) when FSMManager wired it in; fall back to
            # inline compile for direct-construction callers (tests).
            from ..runtime.ast import Abs as _Abs

            if self.compiled_term_resolver is not None:
                # Resolver supplies the (handler-)composed term per
                # `dialog/fsm.py::get_composed_term`.
                term = self.compiled_term_resolver(instance.fsm_id)
            else:
                # Direct-construction path (tests). Compose inline so the
                # handler-spliced shape is exercised even when no
                # FSMManager is wiring `get_composed_term`.
                from ..handlers import compose as _compose_handlers

                base_term = compile_fsm(fsm_def)
                term = _compose_handlers(base_term, list(self.handler_system.handlers))
            assert isinstance(term, _Abs)
            inner1 = term.body
            assert isinstance(inner1, _Abs)
            inner2 = inner1.body
            assert isinstance(inner2, _Abs)
            inner3 = inner2.body
            assert isinstance(inner3, _Abs)
            case_body = inner3.body

            turn_state = _TurnState()
            env = self._build_compiled_env(
                instance, message, conversation_id, turn_state, tier=tier
            )
            # R5 step 4: env extension for handler-spliced terms. When no
            # handlers are registered, `compose` is identity; the extra
            # env bindings are unused and cheap. When PRE/POST_PROCESSING
            # handlers are present, the term invokes the runner via the
            # HOST_CALL Combinator nodes.
            env.update(self._build_handler_env_extension(instance))

            # CB_RESPOND returns str; the Case evaluates to CB_RESPOND's
            # return value (all 4 branches call it). Cast through Any
            # since Executor.run's signature is untyped.
            #
            # R6.2 — cohort states emit a Leaf that requires an Oracle. We
            # wrap self.llm_interface in a LiteLLMOracle so cohort Leaf calls
            # are routed to the same litellm call path as legacy CB_RESPOND.
            # Non-cohort states do not invoke the oracle (their branch is
            # App(CB_RESPOND, instance), a host-callable that calls the LLM
            # directly via self.llm_interface).
            # M4 (merge spec §3 I1+I2) — collapse to Program-owned Oracle field.
            response_any: Any = Executor(oracle=self._oracle).run(case_body, env)
            response: str = response_any

            # Post-transition re-extract now runs INSIDE `_make_cb_respond`
            # before response generation (D-S9-06). Outer wrap removed to
            # avoid double-firing.

            return response

    def process_stream_compiled(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        *,
        tier: int | None = None,
    ) -> Iterator[str]:
        # DECISION D-S10-00 / D-S11-00 / A.M3d-narrowed — compiled-
        # streaming analog of ``process_compiled``. Same cohort guard,
        # same env builder, same executor. Post-A.M3d-narrowed
        # (plan_2026-04-29_0f87b9c4) the rebind of CB_RESPOND to a
        # streaming sibling is dropped; streaming flows through the
        # executor's stream-mode branch on D2 ``Leaf(streaming=True)``
        # via ``StreamingOracle.invoke_stream``, and any residual
        # ``App(CB_RESPOND, instance)`` (D3 terminal fallback /
        # explicit-False) returns a single ``str`` that the entry-point
        # ``iter([result])`` normalisation below wraps to a
        # single-chunk Iterator[str]. S11 retired the legacy
        # ``process_stream`` wrapper — this is now the only streaming
        # dispatch path. See plans/plan_2026-04-24_aedc6d3c/plan.md
        # (S10/S11) and plans/plan_2026-04-29_0f87b9c4/ (A.M3d-narrowed).
        from ..runtime.executor import Executor
        from .compile_fsm import compile_fsm

        if tier is None:
            tier = 3 if self.compiled_term_resolver is not None else 0

        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
            fsm_def = self.fsm_resolver(instance.fsm_id)
            self._check_compiled_cohort(fsm_def, tier=tier)

            # R5 step 4 (D-STEP-04-RESOLUTION) — PRE/POST_PROCESSING are
            # spliced into the composed term. For streaming, the
            # POST_PROCESSING splice is BYPASSED here because the
            # response Leaf returns an Iterator and the spliced
            # POST_PROCESSING host_call would fire BEFORE iterator
            # exhaustion — wrong lifecycle. We strip the splice for
            # streaming (call `case_body` directly via the inner branch)
            # and instead invoke POST_PROCESSING in `finally` after
            # `yield from` completes. PRE_PROCESSING is fired here once
            # before iterator construction. See D-S10-02 for prior
            # streaming-lifecycle reasoning.
            #
            # Streaming therefore stays on the host-driven PRE/POST
            # bracket pattern, but the underlying handler dispatch goes
            # through `make_handler_runner` for execution-path
            # uniformity (per Option gamma).

            from ..runtime.ast import Abs as _Abs
            from ..runtime.ast import Let as _Let

            if self.compiled_term_resolver is not None:
                term = self.compiled_term_resolver(instance.fsm_id)
            else:
                # Direct-construction path — compose inline (see
                # `process_compiled` for rationale).
                from ..handlers import compose as _compose_handlers

                base_term = compile_fsm(fsm_def)
                term = _compose_handlers(base_term, list(self.handler_system.handlers))
            assert isinstance(term, _Abs)
            inner1 = term.body
            assert isinstance(inner1, _Abs)
            inner2 = inner1.body
            assert isinstance(inner2, _Abs)
            inner3 = inner2.body
            assert isinstance(inner3, _Abs)
            case_body = inner3.body

            # If handlers are present, the composed term wraps the Case
            # in the PRE/POST_PROCESSING shape:
            #   Let(post_result, Let(pre_h, host_call, Case), Let(post_h, host_call, Var(post_result)))
            # For streaming we want the inner Case directly. We unwrap
            # by walking the structure until we find the innermost Case
            # while preserving the PRE_PROCESSING side-effect (host_call
            # before Case evaluation). This is structurally the value
            # of the inner Let — its body is the Case.
            from ..runtime.ast import Case as _Case

            # Pre-fire PRE_PROCESSING via the runner directly; then
            # extract the inner Case for streaming evaluation.
            handler_env = self._build_handler_env_extension(instance)
            runner = handler_env[HANDLER_RUNNER_VAR_NAME]
            stream_inner_case: Any = case_body
            if isinstance(case_body, _Let):
                # Composed shape — fire PRE_PROCESSING manually, drop
                # the wrappers for streaming.
                runner(
                    HandlerTiming.PRE_PROCESSING.value,
                    instance.current_state,
                    None,
                    instance.context.data,
                    None,
                )
                # Walk to the innermost Case: the structure is
                # Let(post_r, Let(pre_h, _, Case), Let(post_h, _, Var)).
                outer = case_body
                if isinstance(outer.value, _Let):
                    inner_let = outer.value
                    if isinstance(inner_let.body, _Case):
                        stream_inner_case = inner_let.body

            # A.D4(b) (plan_ca542489 step 5) — declare streaming intent in
            # the per-turn state BEFORE env-build. D1's synthetic-response
            # callback reads turn_state.stream at call time and returns
            # iter([sentinel]) instead of the raw string when True.
            turn_state = _TurnState(stream=True)
            env = self._build_compiled_env(
                instance, message, conversation_id, turn_state, tier=tier
            )
            # Streaming does not use the spliced POST_PROCESSING (see
            # block comment above); env bindings are still present for
            # any HOST_CALL nodes the inner Case may reference (none in
            # the current shape, but forward-compat).
            env.update(handler_env)
            # A.M3d-narrowed (plan_2026-04-29_0f87b9c4 step 8): the
            # CB_RESPOND env-rebind to the streaming sibling is dropped.
            # Post-A.M3c the default response path is the D2
            # ``Leaf(streaming=True)`` chain (handled by the executor's
            # stream-mode branch via ``StreamingOracle.invoke_stream``);
            # the only states still emitting ``App(CB_RESPOND, instance)``
            # are the D3 terminal-non-cohort fallback (when
            # ``output_schema_ref`` is unset) and the explicit-False
            # regression-coverage tests. Both yield a single ``str``
            # which the entry-point ``iter([result])`` normalisation
            # below wraps into a single-chunk Iterator[str] — chunked
            # streaming is unavailable for those paths but functional
            # parity is preserved.

            # A.D4(b) — Executor receives oracle (so D2's streaming-flagged
            # Leaf can route through `oracle.invoke_stream`) AND stream=True
            # (so the streaming-branch fires for `Leaf.streaming=True`).
            # Non-streaming Leaves (cohort, extraction) still call invoke
            # regardless. The mixed-mode contract is per-Leaf streaming
            # capability + per-call streaming intent.
            stream_any: Any = Executor(oracle=self._oracle).run(
                stream_inner_case, env, stream=True
            )
            # Normalise return shape: cohort terminal Leaves emit
            # `streaming=False` per D-005 mutual exclusion (mid-stream
            # schema enforcement is unreliable), so they return a string
            # from `oracle.invoke`. Wrap in a single-chunk iterator so the
            # caller's `yield from` always sees Iterator[str].
            stream_iter: Iterator[str] = (
                iter([stream_any]) if isinstance(stream_any, str) else stream_any
            )

            # DECISION D-S10-02 — POST_PROCESSING in `finally` so it fires
            # after iterator exhaustion OR caller-break (GeneratorExit).
            # Differs from process_compiled where Executor.run returns a
            # str synchronously; stream lifecycle must track the iterator.
            try:
                yield from stream_iter
            finally:
                runner(
                    HandlerTiming.POST_PROCESSING.value,
                    instance.current_state,
                    None,
                    instance.context.data,
                    None,
                )

    # ----------------------------------------------------------
    # S8b: compiled-path env builder + callback factories
    # ----------------------------------------------------------

    def _build_compiled_env(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
        *,
        tier: int,
    ) -> dict[str, Any]:
        """Build the λ-executor env for `process_compiled` at given tier.

        Each tier wires progressively more callbacks. Slots above tier
        bind to `_not_in_cohort` (D-S8b-02: fail-loud sentinel).
        """
        from .compile_fsm import (
            CB_APPEND_HISTORY,
            CB_CLASS_EXTRACT,
            CB_EVAL_TRANSIT,
            CB_EXTRACT,
            CB_FIELD_EXTRACT,
            CB_RENDER_RESPONSE_PROMPT,
            CB_RESOLVE_AMBIG,
            CB_RESPOND,
            CB_RESPOND_SYNTHETIC,
            CB_TRANSIT,
            COHORT_RESPONSE_PROMPT_VAR,
            VAR_CONV_ID,
            VAR_INSTANCE,
            VAR_MESSAGE,
            VAR_STATE_ID,
            _is_cohort_state,
        )

        def _not_in_cohort(_inst: Any) -> Any:
            raise NotImplementedError(
                f"process_compiled: callback invoked outside tier={tier} "
                f"cohort — compiler emitted a Let that this tier does not "
                f"wire. This is the D-S8b-02 fail-loud sentinel."
            )

        env: dict[str, Any] = {
            VAR_STATE_ID: instance.current_state,
            VAR_MESSAGE: message,
            VAR_CONV_ID: conversation_id,
            VAR_INSTANCE: instance,
            CB_RESPOND: self._make_cb_respond(
                instance, message, conversation_id, turn_state
            ),
            # M3b — non-cohort response Leaf rendering callback.
            # Bound unconditionally so any opt-in
            # (`_emit_response_leaf_for_non_cohort=True`) state in any tier
            # finds the callable in env. The callable runs at Let-time —
            # i.e. AFTER upstream extraction / transition / ambig
            # callbacks have mutated `instance.current_state` and
            # `instance.context.data` — so the rendered prompt reflects
            # the post-transition state. Returns the rendered prompt
            # string for the Leaf to substitute via str.format.
            #
            # Field=False (default at M3b) → compiler emits no Let+App
            # for this name, the binding is harmlessly unused.
            # Field=True → exactly one Leaf-call per response = strict
            # Theorem-2 equality.
            CB_RENDER_RESPONSE_PROMPT: self._make_cb_render_response_prompt(
                instance, message, conversation_id, turn_state
            ),
            # D1 (plan_f1003066) — empty-`response_instructions` synthetic
            # callback. Bound unconditionally so the compile-time gate in
            # `_compile_state` (only fires under M3a opt-in flag) finds the
            # callable. Issues 0 oracle calls; returns the synthetic
            # f"[{state.id}]" and appends to conversation history.
            CB_RESPOND_SYNTHETIC: self._make_cb_respond_synthetic(
                instance, message, conversation_id, turn_state
            ),
            # D2 (plan_f1003066) — post-Leaf history-append callback.
            # Curried 2-arg: App(App(CB_APPEND_HISTORY, instance), value).
            # Called only under the M3a opt-in flag (the standard non-cohort
            # path); harmlessly unused at default-False.
            CB_APPEND_HISTORY: self._make_cb_append_history(
                instance, message, conversation_id, turn_state
            ),
            # CB_TRANSIT is reserved but never emitted by the compiler
            # (D-S5-01). Always sentinel — even at tier=3.
            CB_TRANSIT: _not_in_cohort,
        }

        # Tier 1+: extractions wired.
        if tier >= 1:
            env[CB_EXTRACT] = self._make_cb_extract(
                instance, message, conversation_id, turn_state
            )
            env[CB_FIELD_EXTRACT] = self._make_cb_extract(
                instance, message, conversation_id, turn_state
            )
            env[CB_CLASS_EXTRACT] = self._make_cb_extract(
                instance, message, conversation_id, turn_state
            )
        else:
            env[CB_EXTRACT] = _not_in_cohort
            env[CB_FIELD_EXTRACT] = _not_in_cohort
            env[CB_CLASS_EXTRACT] = _not_in_cohort

        # Tier 2+: transition evaluation wired (step 3).
        if tier >= 2:
            env[CB_EVAL_TRANSIT] = self._make_cb_eval_transit(
                instance, message, conversation_id, turn_state
            )
        else:
            env[CB_EVAL_TRANSIT] = _not_in_cohort

        # Tier 3: curried resolve-ambig wired (step 4).
        if tier >= 3:
            env[CB_RESOLVE_AMBIG] = self._make_cb_resolve_ambig(
                instance, conversation_id, turn_state
            )
        else:
            env[CB_RESOLVE_AMBIG] = _not_in_cohort

        # R6.2 — cohort Leaf env binding (D-S1-03). For cohort states (terminal
        # response-only, _is_cohort_state == True), the compiled term emits a
        # Leaf("{response_prompt_rendered}", input_vars=("response_prompt_rendered",))
        # that the executor substitutes via str.format. We pre-render the full
        # response prompt here at env-build time using the same renderer the
        # legacy CB_RESPOND closure uses (build_response_prompt) — preserves
        # byte-parity with the host path. For non-cohort states the Leaf is
        # not in the dispatched branch, so the binding is harmlessly unused.
        try:
            fsm_def = self.fsm_resolver(instance.fsm_id)
            current_state_obj = fsm_def.states.get(instance.current_state)
            if current_state_obj is not None and _is_cohort_state(
                current_state_obj, fsm_def
            ):
                # R9a fix — apply context_scope before rendering so cohort path
                # honours the same scoping the legacy CB_RESPOND path applies.
                # build_response_prompt reads instance.context.data directly;
                # swap-and-restore keeps the rest of the instance intact.
                original_ctx_data = instance.context.data
                scoped_ctx = self._apply_context_scope(
                    original_ctx_data,
                    current_state_obj,
                    conversation_id=instance.fsm_id,
                )
                if scoped_ctx is not original_ctx_data:
                    instance.context.data = scoped_ctx
                try:
                    env[COHORT_RESPONSE_PROMPT_VAR] = (
                        self.response_generation_prompt_builder.build_response_prompt(
                            instance,
                            current_state_obj,
                            fsm_def,
                            user_message=message,
                        )
                    )
                finally:
                    if scoped_ctx is not original_ctx_data:
                        instance.context.data = original_ctx_data
            else:
                # Forward-compat: bind a sentinel so any leaked Leaf evaluation
                # fails loud rather than KeyError.
                env[COHORT_RESPONSE_PROMPT_VAR] = (
                    f"<COHORT_PROMPT_NOT_RESOLVED for state {instance.current_state!r}>"
                )
        except Exception:
            # If fsm resolution fails, fall back without the cohort binding.
            # The Leaf would never fire for non-cohort states anyway; cohort
            # states with a missing fsm_def are pathological — let the
            # downstream KeyError surface from executor._eval_leaf.
            pass

        return env

    def _make_cb_respond(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], str]:
        """`CB_RESPOND` binding. Reads turn_state for extraction_response,
        transition_occurred, previous_state. Falls back to empty / False
        for tiers that haven't populated them.
        """

        def _respond(inst: FSMInstance) -> str:
            # DECISION D-S9-06 — post-transition re-extract must run BEFORE
            # response generation. S8b originally placed it as an outer wrap
            # (after Executor.run), which meant CB_RESPOND built the Pass-2
            # prompt with stale `extracted_data` and `context` (user_name
            # not yet extracted during the first turn that transitions on a
            # field-required state). Prompt-string smoke SC6 caught this.
            # Ensuring turn_state.extraction_response is non-None so post-tx
            # updates land (`_execute_data_extraction` never returns None).
            if turn_state.extraction_response is None:
                turn_state.extraction_response = DataExtractionResponse(
                    extracted_data={}, confidence=1.0
                )
            if turn_state.transition_occurred and "agent_trace" not in (
                instance.context.data
            ):
                self._post_transition_reextract(
                    instance, message, turn_state, conversation_id
                )
            extraction = turn_state.extraction_response
            return self._execute_response_generation_pass(
                inst,
                message,
                extraction,
                turn_state.transition_occurred,
                turn_state.previous_state,
                conversation_id,
            )

        return _respond

    def _make_cb_render_response_prompt(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], str]:
        """M3b — `CB_RENDER_RESPONSE_PROMPT` binding.

        Returns a closure ``(FSMInstance) -> str`` that the executor
        invokes from the Let-bound App at the response position
        (compile_fsm `_compile_state` non-cohort Leaf branch). Renders
        the response prompt using the SAME ``build_response_prompt``
        renderer that the legacy ``_cb_respond`` path uses, with the
        SAME turn_state-derived arguments (``extracted_data``,
        ``transition_occurred``, ``previous_state``, ``user_message``).

        Mirrors the post-transition re-extract ordering from
        ``_make_cb_respond`` (D-S9-06): if a transition occurred this
        turn, re-run extraction so the prompt reflects post-transition
        ``context.data``.

        The Leaf that follows this Let in the AST then ships exactly
        one ``oracle.invoke(rendered_prompt)`` call — Theorem-2 strict
        equality holds (1 Leaf = 1 oracle call per non-cohort
        response).
        """

        def _render(inst: FSMInstance) -> str:
            # Mirror D-S9-06 ordering from `_make_cb_respond`.
            if turn_state.extraction_response is None:
                turn_state.extraction_response = DataExtractionResponse(
                    extracted_data={}, confidence=1.0
                )
            if turn_state.transition_occurred and "agent_trace" not in (
                inst.context.data
            ):
                self._post_transition_reextract(
                    inst, message, turn_state, conversation_id
                )
            extraction = turn_state.extraction_response
            fsm_def = self.fsm_resolver(inst.fsm_id)
            current_state = fsm_def.states[inst.current_state]
            return self.response_generation_prompt_builder.build_response_prompt(
                instance=inst,
                state=current_state,
                fsm_definition=fsm_def,
                extracted_data=extraction.extracted_data if extraction else None,
                transition_occurred=turn_state.transition_occurred,
                previous_state=turn_state.previous_state,
                user_message=message,
            )

        return _render

    def _make_cb_respond_synthetic(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Any]:
        """D1 (plan_f1003066, merge spec §6b A.D1) — synthetic-response
        callback for non-cohort opt-in states with empty
        ``response_instructions`` (NOT None — the legacy predicate at
        ``turn.py:2244-2247`` matches `is not None and not <str>`).

        Mirrors the legacy fast-path at ``_make_cb_respond`` — returns
        the synthetic ``f"[{state.id}]"`` and appends it to
        ``instance.context.conversation`` — but **issues 0 oracle calls**
        instead of legacy's 1 sentinel ``oracle.invoke(".")``. The
        sentinel call was a wire-level wart; under the M3a opt-in path
        we drop it (the LiteLLM short-circuit was the only reason it was
        cheap, and no user-visible behaviour relied on it). Theorem-2
        strict equality holds: 0 oracle calls for an opt-in state with
        empty response_instructions.

        **A.D4(b) extension** (plan_ca542489 step 3): when
        ``turn_state.stream is True`` (set by ``process_stream_compiled``),
        the callable returns ``iter([synthetic])`` — a single-chunk
        iterator that mirrors the legacy streaming I4 fast-path at
        ``turn.py:1334-1342``. History append still happens once. Under
        non-streaming (``stream=False``) the callable returns the synthetic
        string verbatim, preserving pre-A.D4 byte-equivalence.

        Resolves D1 of the four divergences blocking the M3c default
        flip (see ``plans/plan_2026-04-28_6597e394/decisions.md`` D-009).
        """

        def _synthetic(inst: FSMInstance) -> Any:
            fsm_def = self.fsm_resolver(inst.fsm_id)
            current_state = fsm_def.states[inst.current_state]
            synthetic = f"[{current_state.id}]"
            inst.context.conversation.add_system_message(synthetic)
            logger.debug(
                "D1 synthetic response (opt-in non-cohort, empty "
                "response_instructions): state={} synthetic={} stream={}",
                current_state.id,
                synthetic,
                turn_state.stream,
            )
            if turn_state.stream:
                # Single-chunk iterator — preserves the streaming I4
                # contract from streaming-surface findings §3 (legacy
                # streaming path yields the synthetic value as one chunk).
                return iter([synthetic])
            return synthetic

        return _synthetic

    def _make_cb_append_history(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Callable[[Any], Any]]:
        """D2 (plan_f1003066, merge spec §6b A.D2) — post-Leaf
        history-append callback. **A.D4(b) extension** (plan_ca542489
        step 2): iterator-aware so the same outer Let wrap handles both
        non-streaming string Leaves and streaming-flagged Iterator
        Leaves under one compiled term.

        Curried 2-arg: the executor reduces
        ``App(App(CB_APPEND_HISTORY, instance), response_value)`` to
        ``factory(instance)(response_value)``.

        Two value-shape branches:

        - ``response_value: str`` (non-streaming) — append the string to
          ``inst.context.conversation``, set ``last_response_generation``
          for debug, return the string unchanged.

        - ``response_value: Iterator[str]`` (streaming) — return a
          tee-on-exhaustion generator that yields each chunk to the
          consumer while accumulating; on exhaustion (or GeneratorExit),
          ``"".join(chunks)`` is appended to history. (The retired
          streaming sibling carried this same finally block; A.M3d-narrowed
          consolidated it here as the single accumulation site.)
          ``last_response_generation`` is NOT
          set on the streaming path — preserves §3 I3 from
          ``plan_2026-04-28_ca542489/findings/streaming-surface.md`` (the
          legacy streaming path also skips it).

        The host App does NOT count toward ``oracle_calls`` (Executor
        only increments on Leaf evaluation). This means D2's wrap stays
        Theorem-2-strict equality preserving for both streaming and
        non-streaming Leaves.
        """

        def _outer(inst: FSMInstance) -> Callable[[Any], Any]:
            def _append_and_return(response_value: Any) -> Any:
                # Non-streaming path: response_value is a plain string
                # (or stringifiable scalar). Preserves pre-A.D4 behaviour.
                if isinstance(response_value, str):
                    response_str = response_value
                    inst.context.conversation.add_system_message(response_str)
                    inst.last_response_generation = ResponseGenerationResponse(
                        message=response_str
                    )
                    return response_str

                # Streaming path: response_value is Iterator[str] (typically
                # the generator returned by oracle.invoke_stream via the
                # Executor's A.D4(b) streaming branch). Tee-on-exhaustion.
                def _tee_and_append() -> Iterator[str]:
                    chunks: list[str] = []
                    try:
                        for chunk in response_value:
                            chunks.append(chunk)
                            yield chunk
                    finally:
                        full = "".join(chunks)
                        inst.context.conversation.add_system_message(full)
                        # I3 preserved: last_response_generation not set
                        # for streaming, mirroring the (now-retired) legacy
                        # streaming sibling (A.M3d-narrowed retired the
                        # standalone method; this path is the canonical
                        # streaming accumulation site).

                return _tee_and_append()

            return _append_and_return

        return _outer

    def _make_cb_extract(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Any]:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return self._extraction._make_cb_extract(
            instance, message, conversation_id, turn_state
        )

    def _make_cb_eval_transit(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], str]:
        """`CB_EVAL_TRANSIT` binding (tier 2+). Step 3 implements."""

        def _eval_transit(_inst: FSMInstance) -> str:
            extraction = turn_state.extraction_response or DataExtractionResponse(
                extracted_data={}, confidence=1.0
            )
            discriminant, evaluation = self._evaluate_and_apply_deterministic(
                instance, message, extraction, conversation_id, turn_state
            )
            turn_state.last_evaluation = evaluation
            return discriminant

        return _eval_transit

    def _make_cb_resolve_ambig(
        self,
        instance: FSMInstance,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Callable[[str], Any]]:
        """Curried `CB_RESOLVE_AMBIG` binding (tier 3, step 4).

        The AST `App(App(CB_RESOLVE_AMBIG, inst), msg)` reduces via
        `Executor._apply` to `callable(inst)(msg)`. Outer returns the
        per-message closure.
        """

        def _curried(_inst: FSMInstance) -> Callable[[str], Any]:
            def _inner(msg: str) -> Any:
                if turn_state.last_evaluation is None:
                    raise LLMResponseError(
                        "CB_RESOLVE_AMBIG fired without CB_EVAL_TRANSIT "
                        "producing an evaluation — compiler contract "
                        "violation"
                    )
                extraction = turn_state.extraction_response or DataExtractionResponse(
                    extracted_data={}, confidence=1.0
                )
                target = self._resolve_ambiguous_transition(
                    turn_state.last_evaluation,
                    msg,
                    extraction,
                    instance,
                    conversation_id,
                )
                if target and target != instance.current_state:
                    previous = instance.current_state
                    self._execute_state_transition(instance, target, conversation_id)
                    turn_state.transition_occurred = True
                    turn_state.previous_state = previous
                return None

            return _inner

        return _curried

    def _evaluate_and_apply_deterministic(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> tuple[str, TransitionEvaluation | None]:
        """Evaluate transitions; apply ONLY on DETERMINISTIC.

        Defers AMBIGUOUS apply to `CB_RESOLVE_AMBIG` to satisfy the S5/S6
        compiled contract.

        # DECISION D-S8b-03 / D-S11-00 — the pre-compiled eval+execute unit
        # applied the AMBIGUOUS transition inline; this method splits the
        # eval+apply unit so AMBIGUOUS apply is deferred to
        # `CB_RESOLVE_AMBIG`. The pre-compiled method was retired in S11.
        # See plans/plan_2026-04-24_4ec5abc0/decisions.md#D-S8b-03.

        Returns `(discriminant, evaluation | None)` where discriminant ∈
        {"advanced", "blocked", "ambiguous"}. On "ambiguous", evaluation
        is the TransitionEvaluation object; otherwise None.
        """
        log = logger.bind(conversation_id=conversation_id)
        current_state = self.get_state(instance, conversation_id)

        if not current_state.transitions:
            # Terminal — never emitted by compiler, but defensive.
            return "blocked", None

        previous_state_id = instance.current_state
        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state, instance.context, extraction_response.extracted_data
        )

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")
            if target_state:
                self._execute_state_transition(instance, target_state, conversation_id)
                turn_state.transition_occurred = True
                turn_state.previous_state = previous_state_id
            return "advanced", None

        if evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            log.info("Ambiguous transition — deferring apply to CB_RESOLVE_AMBIG")
            return "ambiguous", evaluation

        if evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            return "blocked", None

        # Unreachable: the enum has 3 values.
        return "blocked", None

    def _post_transition_reextract(
        self,
        instance: FSMInstance,
        user_message: str,
        turn_state: _TurnState,
        conversation_id: str,
    ) -> None:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        self._extraction._post_transition_reextract(
            instance, user_message, turn_state, conversation_id
        )

    @staticmethod
    def _state_may_be_ambiguous(state: State) -> bool:
        """Static over-approximation: state may produce AMBIGUOUS transition.

        True iff the state has ≥2 transitions AND ≥2 of them are
        unconditional (no `conditions`). These compete on priority only;
        `TransitionEvaluator` returns AMBIGUOUS when priorities tie and
        cannot statically split them. All-guarded transitions are trusted
        (JsonLogic guards are assumed mutually exclusive — heuristic, not
        proven). Used by the tier<3 cohort gate (D-S9-07).
        """
        if len(state.transitions) < 2:
            return False
        unconditional_count = sum(1 for t in state.transitions if not t.conditions)
        return unconditional_count >= 2

    def _check_probe_cohort(self, fsm_def: FSMDefinition) -> None:
        """Reject FSMs outside the S8-probe cohort (D-S8-01).

        Cohort: every state has no transitions, no extractions.
        Thin back-compat wrapper over :meth:`_check_compiled_cohort` at
        tier=0.
        """
        self._check_compiled_cohort(fsm_def, tier=0)

    def _check_compiled_cohort(self, fsm_def: FSMDefinition, *, tier: int = 3) -> None:
        """Reject FSMs outside the `tier` compiled-path cohort (S8b).

        Tiers (widening):

        - **tier 0** (S8-probe): no transitions, no extractions.
        - **tier 1** (extractions-only): no transitions; any extractions OK.
        - **tier 2** (deterministic transitions): any transitions; any
          extractions. Ambiguity is NOT statically rejected — if it fires
          at runtime, `CB_RESOLVE_AMBIG` (still sentinel at tier<3) raises.
          See D-S8b-02.
        - **tier 3** (full): any FSM. All 6 real callbacks wired.

        # DECISION D-S8b-02 — sentinel-at-tier<max fail-loud policy
        # Callbacks above the current tier raise NotImplementedError at
        # runtime. No silent fallback path exists post-S11. See
        # plans/plan_2026-04-24_4ec5abc0/decisions.md#D-S8b-02.
        """
        if tier not in (0, 1, 2, 3):
            raise ValueError(
                f"process_compiled: invalid cohort tier={tier!r}; must be 0, 1, 2, or 3"
            )
        if tier >= 3:
            # Full cohort — nothing to reject.
            return
        for state_id, state in fsm_def.states.items():
            # DECISION D-S9-07 (D-S8b-02 revisit) — graduate tier<3 from
            # runtime-sentinel-on-AMBIGUOUS to static rejection of
            # structurally ambiguous-prone states. Safe over-approximation:
            # if ≥2 transitions are unconditional (no guard), the evaluator
            # returns AMBIGUOUS on every input (see `_make_ambiguous_fsm`
            # fixture). States with single transitions or all-guarded
            # transitions pass the static check; runtime sentinel still
            # catches any edge case the heuristic misses (tier<3 keeps its
            # fail-loud contract). Applies to every tier below 3 since
            # tier<2 already rejects transitions wholesale (redundant there;
            # kept symmetric for clarity). Tier 3 admits all FSMs.
            if self._state_may_be_ambiguous(state):
                raise ValueError(
                    f"process_compiled: tier={tier} cohort violation — "
                    f"state {state_id!r} has {sum(1 for t in state.transitions if not t.conditions)} "
                    f"unconditional transitions; may resolve to AMBIGUOUS "
                    f"at runtime (use tier=3 or add guards)"
                )
            if tier < 2 and state.transitions:
                raise ValueError(
                    f"process_compiled: tier={tier} cohort violation — state "
                    f"{state_id!r} has transitions (raise tier to 2 or 3)"
                )
            if tier < 1:
                if state.extraction_instructions:
                    raise ValueError(
                        f"process_compiled: tier={tier} cohort violation — "
                        f"state {state_id!r} has extraction_instructions"
                    )
                if state.field_extractions:
                    raise ValueError(
                        f"process_compiled: tier={tier} cohort violation — "
                        f"state {state_id!r} has field_extractions"
                    )
                if state.classification_extractions:
                    raise ValueError(
                        f"process_compiled: tier={tier} cohort violation — "
                        f"state {state_id!r} has classification_extractions"
                    )
                # S9: mirror widened CB_EXTRACT emission predicate in
                # fsm_compile.py. Legacy auto-synthesizes extraction for
                # required_context_keys — so at tier<1 the compiled path
                # would emit CB_EXTRACT against a sentinel. Reject here.
                if state.required_context_keys:
                    raise ValueError(
                        f"process_compiled: tier={tier} cohort violation — "
                        f"state {state_id!r} has required_context_keys"
                    )

    # ----------------------------------------------------------
    # Initial response generation
    # ----------------------------------------------------------

    def generate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """Generate initial response for conversation start (no extraction/transition)."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data={},
            transition_occurred=False,
            previous_state=None,
            user_message="",
        )

        # DECISION D-R10-7.3 (step 8 finalised): always route initial
        # response through oracle.invoke. Wire-level byte-equivalence
        # verified (only system_prompt + user_message reach litellm; both
        # paths send identical values since user_message="" here). The
        # auxiliary request fields (extracted_data, context, etc.) are
        # not consumed by LiteLLMInterface.generate_response on the wire
        # side. Per-callback flag dropped; ResponseGenerationRequest
        # construction retired since the oracle path doesn't use it.
        # M4 — Program-owned Oracle field-read.
        oracle = self._oracle
        message_str = oracle.invoke(system_prompt)
        response = ResponseGenerationResponse(message=str(message_str))
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    # ----------------------------------------------------------
    # Pass 1: Field-based extraction (replaces bulk extract_data)
    # ----------------------------------------------------------

    def _bulk_extract_from_instructions(
        self,
        instance: FSMInstance,
        user_message: str,
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return self._extraction._bulk_extract_from_instructions(
            instance, user_message, state, conversation_id
        )

    @staticmethod
    def _build_field_configs_from_state(state: State) -> list[FieldExtractionConfig]:
        """Auto-convert legacy state fields to FieldExtractionConfig list.

        Translates ``required_context_keys`` + ``extraction_instructions``
        into per-field configs so the pipeline can use the unified
        ``extract_field`` primitive for all extraction.  Explicit
        ``field_extractions`` on the state are appended after the
        auto-generated ones.
        """
        configs: list[FieldExtractionConfig] = []

        # Collect all required keys: from state-level AND from transition
        # conditions' requires_context_keys.  This ensures fields that
        # transitions depend on are extracted even if the state doesn't
        # list them in its own required_context_keys.
        all_required_keys: list[str] = list(state.required_context_keys or [])
        if state.transitions:
            for transition in state.transitions:
                if transition.conditions:
                    for condition in transition.conditions:
                        if condition.requires_context_keys:
                            for key in condition.requires_context_keys:
                                if key not in all_required_keys:
                                    all_required_keys.append(key)

        # Auto-convert required keys → one config per key
        if all_required_keys:
            instructions = (
                state.extraction_instructions
                or "Extract the value of this field from the user's input."
            )
            for key in all_required_keys:
                configs.append(
                    FieldExtractionConfig(
                        field_name=key,
                        field_type="any",
                        extraction_instructions=(
                            f"Extract the '{key}' field. {instructions}"
                        ),
                        context_keys=None,  # all context
                        required=True,
                        confidence_threshold=state.extraction_confidence_threshold,
                    )
                )

        # Append explicit field_extractions (user-defined, take priority)
        if state.field_extractions:
            # Avoid duplicates: explicit configs override auto-generated ones
            explicit_names = {fc.field_name for fc in state.field_extractions}
            configs = [c for c in configs if c.field_name not in explicit_names]
            configs.extend(state.field_extractions)

        return configs

    def _execute_data_extraction(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> DataExtractionResponse:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return self._extraction._execute_data_extraction(
            instance, user_message, conversation_id
        )

    def _execute_field_extractions(
        self,
        instance: FSMInstance,
        user_message: str,
        field_configs: list[FieldExtractionConfig],
        conversation_id: str,
    ) -> list[FieldExtractionResponse]:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return self._extraction._execute_field_extractions(
            instance, user_message, field_configs, conversation_id
        )

    @staticmethod
    def _validate_field_extraction(
        response: FieldExtractionResponse,
        config: FieldExtractionConfig,
    ) -> FieldExtractionResponse:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return ExtractionEngine._validate_field_extraction(response, config)

    # ----------------------------------------------------------
    # Classification-based extraction
    # ----------------------------------------------------------

    def _execute_classification_extractions(
        self,
        current_state: State,
        user_message: str,
        instance: FSMInstance,
        conversation_id: str,
        *,
        configs_override: list[ClassificationExtractionConfig] | None = None,
    ) -> dict[str, Any]:
        """Delegate to :class:`ExtractionEngine` (Phase C, 0.8.0)."""
        return self._extraction._execute_classification_extractions(
            current_state,
            user_message,
            instance,
            conversation_id,
            configs_override=configs_override,
        )

    def _resolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str,
        extraction_response: DataExtractionResponse,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str:
        """Resolve ambiguous transition using classification.

        Classification is always-on for ambiguous transitions. Builds a
        ClassificationSchema from available transition options and uses
        the Classifier to make a structured, confidence-scored decision.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug(
            f"Resolving ambiguous transition with {len(evaluation.available_options)} options"
        )

        current_state = self.get_state(instance, conversation_id)

        schema = self._build_transition_classification_schema(
            current_state,
            evaluation.available_options,
        )

        model = getattr(self.llm_interface, "model", None)
        if model is None:
            raise InvalidTransitionError(
                "Cannot determine LLM model for classification-based "
                "transition resolution"
            )

        classifier = Classifier(
            schema=schema,
            model=model,
        )

        try:
            result: ClassificationResult = classifier.classify(user_message)
        except (ClassificationError, Exception) as e:
            log.warning(
                f"Classification failed during ambiguous transition resolution: {e}"
            )
            log.warning("Falling back to current state (no transition)")
            instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT] = {
                "error": str(e),
                "fallback": True,
            }
            return instance.current_state

        log.debug(
            f"Classification result: intent={result.intent}, "
            f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
        )

        # Store classification result in context for debugging
        instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT] = {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "entities": result.entities,
        }

        # Store as transition decision for debugging
        instance.last_transition_decision = result

        # Handle fallback intent (low confidence or unknown) — stay in current state
        if result.intent == TRANSITION_CLASSIFICATION_FALLBACK_INTENT:
            log.info(
                "Classification returned fallback intent — staying in current state"
            )
            return instance.current_state

        # Validate the classified intent is a valid target state
        valid_targets = {opt.target_state for opt in evaluation.available_options}
        if result.intent not in valid_targets:
            raise InvalidTransitionError(
                f"Classification returned unknown target '{result.intent}'. "
                f"Valid options: {sorted(valid_targets)}"
            )

        log.info(
            f"Classification-based transition selected: {result.intent} "
            f"(confidence={result.confidence:.2f})"
        )
        return result.intent

    # ----------------------------------------------------------
    # Classification schema builder
    # ----------------------------------------------------------

    @staticmethod
    def _build_transition_classification_schema(
        state: State,
        options: list[TransitionOption],
    ) -> ClassificationSchema:
        """Build a ClassificationSchema from transition options.

        If the state has a custom ``transition_classification`` dict config,
        merges user-provided descriptions and thresholds. Otherwise
        auto-generates intents from transition descriptions.
        """
        config = state.transition_classification

        if isinstance(config, dict):
            # Manual mode: user provides intent descriptions
            intents = []
            for opt in options:
                custom = config.get(opt.target_state, {})
                description = (
                    custom.get("description")
                    or opt.description
                    or f"Transition to {opt.target_state}"
                )
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = config.get(
                "confidence_threshold",
                DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
            )
        else:
            # Auto mode: generate from transition option descriptions
            intents = []
            for opt in options:
                description = opt.description or f"Transition to {opt.target_state}"
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE

        # Add fallback intent for low-confidence cases
        intents.append(
            IntentDefinition(
                name=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
                description="None of the above options clearly match the user's intent",
            )
        )

        return ClassificationSchema(
            intents=intents,
            fallback_intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
            confidence_threshold=confidence_threshold,
        )

    def _execute_state_transition(
        self, instance: FSMInstance, target_state: str, conversation_id: str
    ) -> None:
        """Execute state transition with PRE/POST handler integration and rollback."""
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        self.execute_handlers(
            instance,
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state,
        )

        # Deep-copy full context for rollback if POST_TRANSITION handlers fail
        old_context_snapshot = copy.deepcopy(instance.context.data)

        instance.current_state = target_state
        instance.context.data.update(
            {
                "_previous_state": old_state,
                "_current_state": target_state,
                "_transition_timestamp": time.time(),
            }
        )

        try:
            self.execute_handlers(
                instance,
                HandlerTiming.POST_TRANSITION,
                conversation_id,
                current_state=target_state,
                target_state=target_state,
            )
        except Exception as handler_err:
            log.warning(
                f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}"
            )
            instance.current_state = old_state
            if old_context_snapshot is not None:
                instance.context.data.clear()
                instance.context.data.update(old_context_snapshot)
            else:
                log.error("Rollback snapshot was None, cannot safely restore context")
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")

    # ----------------------------------------------------------
    # Pass 2: Response generation
    # ----------------------------------------------------------

    def _execute_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> str:
        """Execute Pass 2: Response Generation based on final state."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        current_state = self.get_state(instance, conversation_id)

        # Fast-path for states with empty response_instructions (e.g. agent
        # intermediate states).  We build a minimal prompt and let the LLM
        # interface decide whether to skip the API call (LiteLLMInterface
        # returns a synthetic response for short system prompts).
        if (
            current_state.response_instructions is not None
            and not current_state.response_instructions
        ):
            # DECISION D-R10-7.4 (step 8 finalised): empty-response_instructions
            # fast-path routes through oracle.invoke with the "." sentinel.
            # The sentinel short-circuits the LLM call regardless of which
            # path is taken (wire-equivalent at the boundary). Per-callback
            # flag dropped; ResponseGenerationRequest construction retired.
            # M4 — Program-owned Oracle field-read.
            oracle = self._oracle
            msg_str = oracle.invoke(".")
            response = ResponseGenerationResponse(message=str(msg_str))
            synthetic = f"[{current_state.id}]"
            instance.context.conversation.add_system_message(synthetic)
            log.debug("Skipped response generation (empty response_instructions)")
            return synthetic

        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message,
        )

        # Apply context scoping if the state defines read_keys
        context_for_llm = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            current_state,
            conversation_id,
        )

        # Only enforce structured output format on terminal states
        output_response_format = None
        if not current_state.transitions:
            output_response_format = instance.context.data.get(
                "_output_response_format"
            )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=context_for_llm,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            response_format=output_response_format,
        )

        # DECISION D-PIVOT-1-CALLSITE (step 11, plan_2026-04-27_32652286):
        # canonical Pass-2 main response site rewired through `oracle.invoke`
        # using the new `user_message=` and `response_format=` kwargs added
        # in step 10 (D-PIVOT-1-ORACLE). The 3 wire-relevant fields
        # (system_prompt + user_message + response_format) are preserved
        # byte-equivalently. Auxiliary fields (extracted_data / context /
        # transition_occurred / previous_state) are dropped — they don't
        # reach the litellm wire (verified per runtime/_litellm.py:254-264).
        # Same trade-off as D-STEP-7.3. Replaces the deferred-site marker
        # at D-R10-7.5.
        # M4 — Program-owned Oracle field-read.
        _oracle = self._oracle
        message_str = _oracle.invoke(
            request.system_prompt,
            user_message=request.user_message,
            response_format=request.response_format,
        )
        # The legacy path's ResponseGenerationResponse envelope (message,
        # message_type, reasoning) is reduced to just the message string
        # at the wire boundary; reconstruct a minimal envelope so the
        # downstream `instance.last_response_generation` and conversation
        # history continue to work. Matches the precedent set by D-R10-7.3
        # (initial response site) where the oracle returns only the
        # message body and we wrap it back into a ResponseGenerationResponse.
        response = ResponseGenerationResponse(
            message=message_str,
            message_type="response",
            reasoning=None,
        )
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.debug("Response generation pass completed")
        return response.message

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    @staticmethod
    def _apply_context_scope(
        context: dict[str, Any],
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Filter context by state's context_scope if defined.

        If the state has ``context_scope`` with ``read_keys``, returns
        only the keys listed. Missing keys are silently skipped (states
        may be entered before all keys are populated).

        If ``context_scope`` is ``None``, returns the full context
        unchanged (backward-compatible default).
        """
        if state.context_scope is None:
            return context

        read_keys = state.context_scope.get("read_keys")
        if not read_keys:
            return context

        scoped = {k: v for k, v in context.items() if k in read_keys}
        missing = [k for k in read_keys if k not in context]
        if missing:
            log = logger.bind(conversation_id=conversation_id)
            log.debug(
                f"Context scope: state '{state.id}' requested keys "
                f"{missing} but they are not in context"
            )
        return scoped

    @staticmethod
    def _clean_empty_context_keys(
        data: dict[str, Any], conversation_id: str, remove_none_values: bool = True
    ) -> dict[str, Any]:
        """Clean invalid keys from context data. Delegates to context module."""
        return clean_context_keys(data, conversation_id, remove_none_values)
