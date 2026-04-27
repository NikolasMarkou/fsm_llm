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
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.ast import Term

from ..constants import (
    CLASSIFICATION_EXTRACTION_RESULT_SUFFIX,
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
from ..llm import LLMInterface
from ..logging import logger
from .classification import Classifier
from .definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    InvalidTransitionError,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    State,
    StateNotFoundError,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from .prompts import (
    ClassificationPromptConfig,
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator

# --- Type coercion dispatch for field extraction validation ---


def _coerce_int(v: Any) -> int:
    return v if isinstance(v, int) else int(v)


def _coerce_float(v: Any) -> float:
    return v if isinstance(v, float) else float(v)


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def _coerce_str(v: Any) -> str:
    return v if isinstance(v, str) else str(v)


def _coerce_list(v: Any) -> Any:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise TypeError("not a list")
        return parsed
    return v


def _coerce_dict(v: Any) -> Any:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, dict):
            raise TypeError("not a dict")
        return parsed
    return v


_TYPE_COERCERS: dict[str, Callable[[Any], Any]] = {
    "int": _coerce_int,
    "float": _coerce_float,
    "bool": _coerce_bool,
    "str": _coerce_str,
    "list": _coerce_list,
    "dict": _coerce_dict,
    # "any" — no coercion, not in dispatch dict
}


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
            from ..runtime.oracle import LiteLLMOracle as _LiteLLMOracle

            response_any: Any = Executor(oracle=_LiteLLMOracle(self.llm_interface)).run(
                case_body, env
            )
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
        # DECISION D-S10-00 / D-S11-00 — compiled-streaming analog of
        # `process_compiled`. Same cohort guard, same env builder, same
        # executor; only CB_RESPOND is rebound to its streaming sibling
        # (_make_cb_respond_stream) so the Case branch returns an
        # Iterator[str] instead of str. S11 retired the legacy
        # `process_stream` wrapper — this is now the only streaming
        # dispatch path. See plans/plan_2026-04-24_aedc6d3c/plan.md.
        from ..runtime.executor import Executor
        from .compile_fsm import CB_RESPOND, compile_fsm

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

            turn_state = _TurnState()
            env = self._build_compiled_env(
                instance, message, conversation_id, turn_state, tier=tier
            )
            # Streaming does not use the spliced POST_PROCESSING (see
            # block comment above); env bindings are still present for
            # any HOST_CALL nodes the inner Case may reference (none in
            # the current shape, but forward-compat).
            env.update(handler_env)
            # Rebind CB_RESPOND with the streaming variant for this turn.
            env[CB_RESPOND] = self._make_cb_respond_stream(
                instance, message, conversation_id, turn_state
            )

            stream_any: Any = Executor().run(stream_inner_case, env)
            stream_iter: Iterator[str] = stream_any

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
            CB_CLASS_EXTRACT,
            CB_EVAL_TRANSIT,
            CB_EXTRACT,
            CB_FIELD_EXTRACT,
            CB_RESOLVE_AMBIG,
            CB_RESPOND,
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

    def _make_cb_respond_stream(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Iterator[str]]:
        # DECISION D-S10-01 — streaming sibling of `_make_cb_respond`. Returns
        # Iterator[str] so the Case branch yields streamable chunks via the
        # executor. Post-transition re-extract ordering (D-S9-06) is
        # preserved by running it BEFORE delegating to
        # `_stream_response_generation_pass`.

        def _respond_stream(inst: FSMInstance) -> Iterator[str]:
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
            yield from self._stream_response_generation_pass(
                inst,
                message,
                extraction,
                turn_state.transition_occurred,
                turn_state.previous_state,
                conversation_id,
            )

        return _respond_stream

    def _make_cb_extract(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Any]:
        """`CB_EXTRACT` / `CB_FIELD_EXTRACT` / `CB_CLASS_EXTRACT` binding.

        All three slots share this implementation — the FIRST extraction
        callback to fire delegates to `_execute_data_extraction` and fires
        CONTEXT_UPDATE; subsequent calls within the same turn are no-ops.

        Why: the compiler emits separate Lets for bulk / field / class
        based on state configuration, but `_execute_data_extraction`
        coordinates them in a single pass (with cross-stage behaviors like
        skip-if-in-context and multi-pass retry). Per-callback primitives
        would diverge semantically — assumption A3 in plan.md. Using a
        single dispatched entry point guarded by
        `extraction_dispatcher_ran` preserves byte-for-byte equivalence
        with the pre-compiled 2-pass flow (retired in S11) at the cost of
        the λ-kernel's per-callback granularity (acceptable — the tiered
        cohort test suite is the compliance gate, not formal
        single-responsibility per slot).
        """

        def _extract(_inst: FSMInstance) -> Any:
            if turn_state.extraction_dispatcher_ran:
                return None
            turn_state.extraction_dispatcher_ran = True

            extraction_response = self._execute_data_extraction(
                instance, message, conversation_id
            )
            turn_state.extraction_response = extraction_response

            # Mirror the context-update + CONTEXT_UPDATE handler fire used
            # by the pre-compiled 2-pass flow (retired in S11).
            if extraction_response.extracted_data:
                extraction_response.extracted_data = self._clean_empty_context_keys(
                    data=extraction_response.extracted_data,
                    conversation_id=conversation_id,
                )
                if extraction_response.extracted_data:
                    instance.context.update(extraction_response.extracted_data)
                    self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        current_state=instance.current_state,
                        updated_keys=set(extraction_response.extracted_data.keys()),
                    )
            return None

        return _extract

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
        """Post-transition re-extraction outer wrap (S8b step 3).

        Factored from the pre-compiled 2-pass flow (retired in S11).
        Preserves exception-swallow-with-warning and the `missing_configs`
        filter. Caller guards on `turn_state.transition_occurred` and the
        `agent_trace` check.
        """
        log = logger.bind(conversation_id=conversation_id)
        new_state = self.get_state(instance, conversation_id)
        new_configs = self._build_field_configs_from_state(new_state)
        missing_configs = [
            c
            for c in new_configs
            if c.field_name not in instance.context.data
            or instance.context.data.get(c.field_name) is None
        ]
        if not missing_configs:
            return

        log.debug(
            f"Post-transition extraction in "
            f"'{instance.current_state}' for "
            f"{[c.field_name for c in missing_configs]}"
        )
        try:
            post_results = self._execute_field_extractions(
                instance, user_message, missing_configs, conversation_id
            )
            post_data: dict[str, Any] = {}
            for result in post_results:
                if result.is_valid and result.value is not None:
                    post_data[result.field_name] = result.value

            if post_data:
                post_data = self._clean_empty_context_keys(
                    data=post_data, conversation_id=conversation_id
                )
                if post_data:
                    instance.context.update(post_data)
                    if turn_state.extraction_response is not None:
                        turn_state.extraction_response.extracted_data.update(post_data)
                    self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        current_state=instance.current_state,
                        updated_keys=set(post_data.keys()),
                    )
        except Exception as e:
            log.warning(f"Post-transition extraction failed (non-fatal): {e}")

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

    def _stream_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> Iterator[str]:
        """Stream Pass 2: yield response tokens as they arrive."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)

        # Fast-path for empty response_instructions
        if (
            current_state.response_instructions is not None
            and not current_state.response_instructions
        ):
            synthetic = f"[{current_state.id}]"
            instance.context.conversation.add_system_message(synthetic)
            yield synthetic
            return

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

        context_for_llm = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            current_state,
            conversation_id,
        )

        # Only enforce structured output format on terminal states (no
        # outgoing transitions).  Applying it on intermediate states forces
        # the model to produce JSON when the prompt asks for free-form text,
        # which can cause small models to hang or produce garbage.
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

        # Accumulate chunks to store in conversation history
        chunks: list[str] = []
        try:
            for chunk in self.llm_interface.generate_response_stream(request):
                chunks.append(chunk)
                yield chunk
        finally:
            # Store accumulated response even if generator is interrupted
            if chunks:
                full_message = "".join(chunks)
                instance.context.conversation.add_system_message(full_message)
                log.debug("Streaming response generation completed")

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

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message="",
            extracted_data={},
            context=instance.context.get_user_visible_data(),
            transition_occurred=False,
            previous_state=None,
        )

        # DECISION D-R10-7.3: route through oracle.invoke when
        # FSM_LLM_ORACLE_RESPONSE=1; default OFF preserves M1 byte-equivalence
        # of the recorded request shape (extracted_data/context fields
        # collapse on oracle path). At the litellm wire level the two
        # paths are byte-equivalent (only system_prompt + user_message
        # are sent), but the request.model_dump() differs because oracle
        # path constructs a fresh ResponseGenerationRequest.
        if os.environ.get("FSM_LLM_ORACLE_RESPONSE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            from ..runtime.oracle import LiteLLMOracle

            oracle = LiteLLMOracle(self.llm_interface)
            message_str = oracle.invoke(system_prompt)
            response = ResponseGenerationResponse(message=str(message_str))
        else:
            response = self.llm_interface.generate_response(request)
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
        """Bulk-extract data when a state has extraction_instructions but no
        explicit required_context_keys or field_extractions.

        Uses a single LLM call with a simple prompt to extract whatever
        data the instructions describe.  Returns a dict of extracted
        key-value pairs (may be empty).
        """
        log = logger.bind(conversation_id=conversation_id)

        prompt = (
            f"Extract information from the user's message.\n\n"
            f"Instructions: {state.extraction_instructions}\n\n"
            f"User message: {user_message}\n\n"
            f'Respond with JSON: {{"extracted_data": {{"key": "value", ...}}, '
            f'"confidence": 0.95, "reasoning": "..."}}\n\n'
            f"Only include keys for information actually present in the "
            f"user's message. Use descriptive snake_case key names."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            # DECISION D-R10-7.1: route through oracle.invoke when
            # FSM_LLM_ORACLE_EXTRACT=1; default OFF preserves M1 byte-equivalence.
            # Wire-level non-equivalence acknowledged (E8): legacy path sends
            # [{system}, {user}] message array via _make_llm_call; oracle path
            # sends ResponseGenerationRequest(system_prompt=prompt, user_message="")
            # — different message shape. Step-7 parity test expected to flag this;
            # flag stays default-OFF accordingly. Bundle-C step 8 prunes only
            # green-gated sites.
            if os.environ.get("FSM_LLM_ORACLE_EXTRACT", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                from ..runtime.oracle import LiteLLMOracle

                oracle = LiteLLMOracle(self.llm_interface)
                # Oracle returns the raw model string; reuse the same content
                # cleaning pipeline below by wrapping in a minimal shim with
                # the choices[0].message.content shape the legacy path expects.
                raw_str = oracle.invoke(prompt + f"\n\n(user said: {user_message})")

                class _ContentShim:
                    def __init__(self, content_str: str) -> None:
                        self.choices = [
                            type(
                                "_C",
                                (),
                                {
                                    "message": type(
                                        "_M", (), {"content": content_str}
                                    )()
                                },
                            )()
                        ]

                response = _ContentShim(raw_str)
            else:
                response = self.llm_interface._make_llm_call(
                    messages, "data_extraction"
                )
            content = response.choices[0].message.content
            if isinstance(content, str):
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                content = re.sub(
                    r"^```(?:json)?\s*\n?", "", content, flags=re.MULTILINE
                )
                content = re.sub(r"\n?```\s*$", "", content).strip()

            if isinstance(content, str):
                import json as json_mod

                data = json_mod.loads(content)
            elif isinstance(content, dict):
                data = content
            else:
                return {}

            extracted = data.get("extracted_data", data)
            if isinstance(extracted, dict):
                # Filter out None/empty values
                return {
                    k: v
                    for k, v in extracted.items()
                    if v is not None and v != "" and v != {}
                }
        except Exception as e:
            log.warning(f"Bulk extraction fallback failed: {e}")

        return {}

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
        """Execute data extraction via per-field ``extract_field`` calls
        and classification extractions.

        Builds a unified list of ``FieldExtractionConfig`` from both
        legacy ``required_context_keys`` and explicit ``field_extractions``,
        then extracts each field individually.  Also runs any
        ``classification_extractions`` declared on the state.  Supports
        multi-pass retry for missing required fields (up to ``extraction_retries``).
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing field-based data extraction")

        current_state = self.get_state(instance, conversation_id)

        # Build unified field configs
        all_configs = self._build_field_configs_from_state(current_state)

        has_field_configs = bool(all_configs)
        has_classification_configs = bool(current_state.classification_extractions)

        has_extraction_instructions = bool(current_state.extraction_instructions)

        if not has_field_configs and not has_classification_configs:
            if has_extraction_instructions:
                # Fallback: bulk extraction for states with instructions
                # but no explicit field configs.  Uses a single LLM call
                # to extract any relevant data the instructions describe.
                log.debug(
                    "No field configs but extraction_instructions present; "
                    "using bulk extraction fallback"
                )
                bulk_data = self._bulk_extract_from_instructions(
                    instance, user_message, current_state, conversation_id
                )
                # Don't overwrite values already set in context (e.g. by
                # handlers) — bulk extraction is best-effort for NEW data.
                if bulk_data:
                    existing = instance.context.data
                    bulk_data = {
                        k: v for k, v in bulk_data.items() if existing.get(k) is None
                    }
                if bulk_data:
                    response = DataExtractionResponse(
                        extracted_data=bulk_data,
                        confidence=0.8,
                    )
                    instance.last_extraction_response = response
                    return response

            log.debug("No fields or classifications to extract for this state")
            response = DataExtractionResponse(extracted_data={}, confidence=1.0)
            instance.last_extraction_response = response
            return response

        extracted_data: dict[str, Any] = {}
        confidences: list[float] = []

        # --- Field extractions ---
        if has_field_configs:
            # Skip fields already set in context (e.g. by handlers)
            existing = instance.context.data
            all_configs = [c for c in all_configs if existing.get(c.field_name) is None]
            results = self._execute_field_extractions(
                instance, user_message, all_configs, conversation_id
            )
            for result in results:
                if result.is_valid and result.value is not None:
                    extracted_data[result.field_name] = result.value
                    confidences.append(result.confidence)

            log.debug(
                f"Field extraction pass 1: "
                f"{list(extracted_data.keys()) or 'no data'}, "
                f"min_confidence={min(confidences) if confidences else 0.0:.2f}"
            )

        # --- Classification extractions ---
        if has_classification_configs:
            classification_data = self._execute_classification_extractions(
                current_state, user_message, instance, conversation_id
            )
            extracted_data.update(classification_data)

        # --- Multi-pass retry for missing required fields ---
        max_retries = current_state.extraction_retries
        if max_retries > 0:
            for retry_num in range(1, max_retries + 1):
                existing_context = instance.context.data

                # Find missing required field configs
                missing_field_configs = (
                    [
                        cfg
                        for cfg in all_configs
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_field_configs
                    else []
                )

                # Find missing required classification configs
                missing_class_configs = (
                    [
                        cfg
                        for cfg in (current_state.classification_extractions or [])
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_classification_configs
                    else []
                )

                if not missing_field_configs and not missing_class_configs:
                    break

                missing_names = [c.field_name for c in missing_field_configs] + [
                    c.field_name for c in missing_class_configs
                ]
                log.info(
                    f"Extraction retry {retry_num}/{max_retries}: "
                    f"missing={missing_names}"
                )

                if missing_field_configs:
                    retry_results = self._execute_field_extractions(
                        instance, user_message, missing_field_configs, conversation_id
                    )
                    for result in retry_results:
                        if result.is_valid and result.value is not None:
                            extracted_data[result.field_name] = result.value
                            confidences.append(result.confidence)

                if missing_class_configs:
                    retry_class_data = self._execute_classification_extractions(
                        current_state,
                        user_message,
                        instance,
                        conversation_id,
                        configs_override=missing_class_configs,
                    )
                    extracted_data.update(retry_class_data)

        # Build final response — check all sources for missing required fields
        all_required_names: list[str] = []
        if has_field_configs:
            all_required_names.extend(
                cfg.field_name for cfg in all_configs if cfg.required
            )
        if has_classification_configs:
            all_required_names.extend(
                cfg.field_name
                for cfg in (current_state.classification_extractions or [])
                if cfg.required
            )

        min_confidence = min(confidences) if confidences else 0.0
        response = DataExtractionResponse(
            extracted_data=extracted_data,
            confidence=min_confidence,
            additional_info_needed=any(
                name not in extracted_data and name not in instance.context.data
                for name in all_required_names
            ),
        )
        instance.last_extraction_response = response

        log.debug(
            f"Data extraction complete: "
            f"{list(extracted_data.keys())}, "
            f"confidence={min_confidence:.2f}"
        )
        return response

    def _execute_field_extractions(
        self,
        instance: FSMInstance,
        user_message: str,
        field_configs: list[FieldExtractionConfig],
        conversation_id: str,
    ) -> list[FieldExtractionResponse]:
        """Execute targeted field extractions for a list of configs.

        Runs one LLM call per field.  Each config specifies its own
        instructions, dynamic context selection, and validation rules.

        Previously extracted values are added to the dynamic context
        for subsequent extractions, enabling dependent field extraction
        (e.g., tool_input can see that tool_name was already extracted).
        """
        log = logger.bind(conversation_id=conversation_id)
        results: list[FieldExtractionResponse] = []
        # Accumulate extracted values so later fields can see earlier ones
        extracted_so_far: dict[str, Any] = {}

        for field_config in field_configs:
            log.debug(
                f"Extracting field '{field_config.field_name}' "
                f"(type={field_config.field_type})"
            )

            # Build dynamic context from config.context_keys
            if field_config.context_keys is not None:
                dynamic_context = {
                    k: v
                    for k, v in instance.context.data.items()
                    if k in field_config.context_keys
                }
            else:
                # Apply state-level context_scope as default filter
                current_state = self.get_state(instance, conversation_id)
                dynamic_context = self._apply_context_scope(
                    instance.context.get_user_visible_data(),
                    current_state,
                    conversation_id,
                )

            # Include previously extracted fields so the LLM can use them
            if extracted_so_far:
                dynamic_context.update(extracted_so_far)

            # Build prompt
            system_prompt = (
                self.field_extraction_prompt_builder.build_field_extraction_prompt(
                    instance=instance,
                    field_config=field_config,
                    user_message=user_message,
                    dynamic_context=dynamic_context,
                )
            )

            # Build request
            request = FieldExtractionRequest(
                system_prompt=system_prompt,
                user_message=user_message,
                field_name=field_config.field_name,
                field_type=field_config.field_type,
                context=dynamic_context,
                validation_rules=field_config.validation_rules,
            )

            # Call LLM
            try:
                # DECISION D-R10-7.2: route through oracle.invoke when
                # FSM_LLM_ORACLE_FIELD_EXTRACT=1; default OFF preserves M1
                # byte-equivalence. Wire-level non-equivalence acknowledged:
                # legacy extract_field -> _make_llm_call wraps the field
                # schema in an outer {field_name, value, confidence, ...}
                # JSON envelope that small Ollama models (qwen3.5:4b) parse
                # incorrectly (D-008 in oracle.py predates this); oracle
                # path uses _invoke_structured which bypasses that wrapper
                # and routes through direct litellm.completion. Different
                # wire shape; flag stays OFF until a future PR aligns the
                # extract_field outer-envelope vs direct-schema paths.
                if os.environ.get(
                    "FSM_LLM_ORACLE_FIELD_EXTRACT", ""
                ).lower() in ("1", "true", "yes", "on"):
                    from ..runtime.oracle import LiteLLMOracle

                    oracle = LiteLLMOracle(self.llm_interface)
                    # Build a minimal Pydantic schema mirroring the field
                    # extraction response payload shape so oracle's
                    # structured path returns a dict we can re-wrap.
                    from pydantic import BaseModel, Field

                    class _FieldOut(BaseModel):
                        value: Any = Field(default=None)
                        confidence: float = 0.0
                        reasoning: str = ""

                    result_dict = oracle.invoke(
                        request.system_prompt
                        + f"\n\n(user said: {request.user_message})",
                        schema=_FieldOut,
                    )
                    response = FieldExtractionResponse(
                        field_name=request.field_name,
                        field_type=request.field_type,
                        value=result_dict.get("value")
                        if isinstance(result_dict, dict)
                        else None,
                        confidence=float(
                            result_dict.get("confidence", 0.0)
                            if isinstance(result_dict, dict)
                            else 0.0
                        ),
                        reasoning=str(
                            result_dict.get("reasoning", "")
                            if isinstance(result_dict, dict)
                            else ""
                        ),
                    )
                else:
                    response = self.llm_interface.extract_field(request)
            except Exception as e:
                log.warning(
                    f"Field extraction failed for '{field_config.field_name}': {e}"
                )
                response = FieldExtractionResponse(
                    field_name=field_config.field_name,
                    value=None,
                    confidence=0.0,
                    is_valid=False,
                    validation_error=f"LLM call failed: {e}",
                )

            # Validate and coerce
            response = self._validate_field_extraction(response, field_config)

            log.debug(
                f"Field '{field_config.field_name}': "
                f"value={response.value!r}, confidence={response.confidence:.2f}, "
                f"valid={response.is_valid}"
            )
            results.append(response)

            # Feed successful extractions into context for subsequent fields
            if response.is_valid and response.value is not None:
                extracted_so_far[field_config.field_name] = response.value

        return results

    @staticmethod
    def _validate_field_extraction(
        response: FieldExtractionResponse,
        config: FieldExtractionConfig,
    ) -> FieldExtractionResponse:
        """Validate and type-coerce a field extraction response."""
        # Skip validation if already failed
        if not response.is_valid or response.value is None:
            return response

        # Reject values that are obviously the field name echoed back —
        # small models sometimes confuse the JSON template keys with values.
        if isinstance(response.value, str) and response.value.strip().lower() in (
            config.field_name.lower(),
            "field_name",
            "value",
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=None,
                confidence=0.0,
                reasoning="Model echoed field name instead of extracting a value",
                is_valid=False,
                validation_error="Extracted value matches field name (model confusion)",
            )

        # Confidence threshold check
        if (
            config.confidence_threshold > 0.0
            and response.confidence < config.confidence_threshold
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(
                    f"Confidence {response.confidence:.2f} below threshold "
                    f"{config.confidence_threshold:.2f}"
                ),
            )

        # Type coercion via dispatch
        value = response.value
        try:
            coercer = _TYPE_COERCERS.get(config.field_type)
            if coercer is not None:
                value = coercer(value)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(f"Type coercion to {config.field_type} failed: {e}"),
            )

        # Validation rules
        rules = config.validation_rules or {}
        if "allowed_values" in rules:
            if value not in rules["allowed_values"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value {value!r} not in allowed values: "
                        f"{rules['allowed_values']}"
                    ),
                )

        if "min_length" in rules and isinstance(value, str):
            if len(value) < rules["min_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} below minimum {rules['min_length']}"
                    ),
                )

        if "max_length" in rules and isinstance(value, str):
            if len(value) > rules["max_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} exceeds maximum "
                        f"{rules['max_length']}"
                    ),
                )

        if "pattern" in rules and isinstance(value, str):
            import re

            try:
                if not re.match(rules["pattern"], value):
                    return FieldExtractionResponse(
                        field_name=response.field_name,
                        value=value,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        is_valid=False,
                        validation_error=(
                            f"Value does not match pattern: {rules['pattern']}"
                        ),
                    )
            except re.error as e:
                logger.error(f"Invalid regex pattern {rules['pattern']!r}: {e}")
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=0.0,
                    reasoning=f"Invalid regex pattern: {e}",
                    is_valid=False,
                    validation_error=f"Invalid regex pattern: {e}",
                )

        # All checks passed — return with coerced value
        return FieldExtractionResponse(
            field_name=response.field_name,
            value=value,
            confidence=response.confidence,
            reasoning=response.reasoning,
            is_valid=True,
        )

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
        """Run classification extractions and return extracted data.

        For each :class:`ClassificationExtractionConfig`, builds a
        :class:`ClassificationSchema`, creates a :class:`Classifier`,
        and stores the result in two context keys:

        - ``field_name`` → intent string (simple, JsonLogic-friendly)
        - ``_{field_name}_classification`` → full result dict (debugging)

        Args:
            current_state: Current state (for config lookup).
            user_message: User input to classify.
            instance: FSM instance (for model fallback).
            conversation_id: Logging context.
            configs_override: If provided, run only these configs
                (used during retry).

        Returns:
            Dict of extracted key-value pairs to merge into context.
        """
        log = logger.bind(conversation_id=conversation_id)
        configs = configs_override or current_state.classification_extractions or []
        if not configs:
            return {}

        model = getattr(self.llm_interface, "model", None)
        extracted: dict[str, Any] = {}

        for config in configs:
            effective_model = config.model or model
            if not effective_model:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}': "
                        "no LLM model available"
                    )
                log.warning(
                    f"Classification extraction '{config.field_name}': "
                    "no LLM model available, skipping"
                )
                continue

            try:
                schema = ClassificationSchema(
                    intents=config.intents,
                    fallback_intent=config.fallback_intent,
                    confidence_threshold=config.confidence_threshold,
                )

                prompt_config = None
                if config.prompt_config:
                    prompt_config = ClassificationPromptConfig(**config.prompt_config)

                classifier = Classifier(
                    schema=schema,
                    model=effective_model,
                    config=prompt_config,
                )

                result: ClassificationResult = classifier.classify(user_message)

                log.debug(
                    f"Classification extraction '{config.field_name}': "
                    f"intent={result.intent}, confidence={result.confidence:.2f}"
                )

                # Always store fallback intent so the context key exists
                # for downstream JsonLogic conditions
                if result.intent == config.fallback_intent:
                    extracted[config.field_name] = result.intent
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"fallback intent '{result.intent}' stored"
                    )
                    continue

                # Skip low confidence
                if result.confidence < config.confidence_threshold:
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"confidence {result.confidence:.2f} below threshold "
                        f"{config.confidence_threshold}, skipping"
                    )
                    continue

                # Store simple value (user-visible, works with JsonLogic)
                extracted[config.field_name] = result.intent

                # Store full result (internal key, debugging)
                suffix = CLASSIFICATION_EXTRACTION_RESULT_SUFFIX
                full_key = f"_{config.field_name}{suffix}"
                full_result: dict[str, Any] = {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "entities": result.entities,
                }
                # Include context snapshot if configured
                if config.context_keys:
                    full_result["context_snapshot"] = {
                        k: instance.context.data.get(k)
                        for k in config.context_keys
                        if k in instance.context.data
                    }
                extracted[full_key] = full_result

                log.info(
                    f"Classification extraction '{config.field_name}' = "
                    f"'{result.intent}' (confidence={result.confidence:.2f})"
                )

            except (
                ClassificationError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                OSError,
            ) as e:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}' "
                        f"failed: {e}"
                    ) from e
                log.warning(
                    f"Classification extraction '{config.field_name}' failed: {e}"
                )
                continue

        return extracted

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
            request = ResponseGenerationRequest(
                system_prompt=".",
                user_message=user_message,
                extracted_data=extraction_response.extracted_data,
                context={},
                transition_occurred=transition_occurred,
                previous_state=previous_state,
            )
            # DECISION D-R10-7.4: route through oracle.invoke when
            # FSM_LLM_ORACLE_CLASSIFIER=1; default OFF preserves M1
            # byte-equivalence. Wire-level parity (system_prompt="."
            # sentinel + user_message). Note: name "CLASSIFIER" is
            # historical from plan v1 step 7.4 — this is actually the
            # fast-path "skip LLM" sentinel for empty response_instructions
            # states, NOT the classifier proper.
            if os.environ.get("FSM_LLM_ORACLE_CLASSIFIER", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                from ..runtime.oracle import LiteLLMOracle

                oracle = LiteLLMOracle(self.llm_interface)
                # The "." sentinel + user_message go to the wire as-is;
                # we replicate by passing system_prompt="." through invoke
                # and letting the underlying generate_response detect the
                # sentinel and short-circuit. invoke does NOT know about
                # the sentinel, so we route through generate_response
                # directly via _invoke_unstructured (no env, no schema).
                msg_str = oracle.invoke(".")
                response = ResponseGenerationResponse(message=str(msg_str))
            else:
                response = self.llm_interface.generate_response(request)
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

        # DECISION D-R10-7.5: route through oracle.invoke when
        # FSM_LLM_ORACLE_CLASSIFIER_RESP=1; default OFF preserves M1
        # byte-equivalence. Wire-level parity (system_prompt + user_message
        # both reach litellm verbatim). Note: name "CLASSIFIER_RESP" is
        # historical from plan v1 step 7.5 — this is actually the canonical
        # Pass-2 main response generation site.
        if os.environ.get("FSM_LLM_ORACLE_CLASSIFIER_RESP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            from ..runtime.oracle import LiteLLMOracle

            oracle = LiteLLMOracle(self.llm_interface)
            # Pass system_prompt as the oracle prompt; user_message is
            # dropped since LiteLLMOracle._invoke_unstructured pins
            # user_message="". At the wire this differs from the legacy
            # path which sends the user_message in the user role.
            # Wire-level non-equivalence on this site: STOP-IF raised.
            msg_str = oracle.invoke(system_prompt)
            response = ResponseGenerationResponse(message=str(msg_str))
        else:
            response = self.llm_interface.generate_response(request)
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
