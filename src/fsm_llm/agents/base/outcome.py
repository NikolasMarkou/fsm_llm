"""The answer, the trace and the verdict of a finished run (``OutcomeMixin``).

Answer extraction (the overridable ``_extract_answer`` and the ``_answer_source_keys`` that mirrors it), the
cut-off verdict, the forced-stop reason, the execution-evidence rule, the trace, the context filter and the
structured-output parse. ``_answer_source_keys`` stays beside ``_extract_answer``: a pattern that overrides one
overrides the other (D-027). ``_run_outcome``, the one success rule, is in ``agent.py`` because it names
``BaseAgent`` itself. The method bodies are the old ones (plan d454fd62 step 87).
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel

from fsm_llm.agents.constants import ContextKeys, ErrorMessages, StopReason
from fsm_llm.agents.definitions import AgentTrace
from fsm_llm.agents.handlers import recorded_forced_reason
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.definitions import CONTEXT_KEY_CUT_OFF
from fsm_llm.llm.json_extraction import extract_json_from_text
from fsm_llm.logging import logger
from fsm_llm.tools import ToolCall

from .helpers import artifact_text
from .surface import _AgentSurface


class OutcomeMixin(_AgentSurface):
    """The answer, trace and verdict of a finished run."""

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract answer with a fallback chain.

        1. Try each key in *extra_keys* (e.g. ``DRAFT_OUTPUT``)
        2. Try responses in reverse order
        3. Return default message
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-046: no ``final_answer``
        # read here or in a pattern override. No pattern writes that key; the
        # only writer was a model bulk reply (React `think` under
        # use_classification=True), which then replaced the conclude reply and
        # its refused-action report. Do NOT add a context key the model can
        # fill ahead of the speaking state's reply. See decisions.md D-046.
        # Primary: pattern-specific answer keys (artifacts written by a
        # typed field of the pattern's own states)
        for key in extra_keys or []:
            val = artifact_text(final_context.get(key)).strip()
            if val:
                return val

        # Secondary: last non-empty response
        for response in reversed(responses):
            if response and response.strip():
                return response.strip()

        return ErrorMessages.NO_ANSWER

    @staticmethod
    def _has_execution_evidence(
        final_context: dict[str, Any],
        evidence_keys: list[str],
    ) -> bool:
        """True if any ``evidence_keys`` context value proves real execution.

        Generic over the three planner evidence shapes:
        - a non-empty dict (e.g. ReWOO ``evidence`` mapping) → real;
        - a list whose dict entries all carry a ``success`` flag (e.g.
          Orchestrator ``worker_results``) → real only if ≥1 entry succeeded
          (placeholder/failed-worker entries are NOT evidence);
        - any other non-empty list (e.g. plan_execute ``step_results``, which
          have no per-entry success flag) → real.
        """
        for key in evidence_keys:
            value = final_context.get(key)
            if not value:
                continue
            if isinstance(value, dict):
                return True
            if isinstance(value, list):
                dict_entries = [e for e in value if isinstance(e, dict)]
                if dict_entries and all("success" in e for e in dict_entries):
                    if any(e.get("success") for e in dict_entries):
                        return True
                    # all entries failed/placeholder → not evidence; keep looking
                else:
                    return True
        return False

    def _answer_source_keys(
        self, final_context: dict[str, Any], extra_answer_keys: list[str] | None
    ) -> list[str] | None:
        """The context keys the run's answer text is read from, ``None`` for a reply.

        Interface contract (caller: ``_answer_was_cut_off``): the default
        mirrors ``_extract_answer``: the first of *extra_answer_keys* with
        non-empty text is the answer's one source (``[key]``); none has text
        and the answer is the last reply (``None``). A pattern that overrides
        ``_extract_answer`` overrides this too: a list names the keys (``[]``
        when nothing in this run's own turns is the answer: ADaPT's
        decomposition, whose sub-runs report their own cut-offs). Never raises.
        """
        for key in extra_answer_keys or []:
            if artifact_text(final_context.get(key)).strip():
                return [key]
        return None

    def _answer_was_cut_off(
        self, final_context: dict[str, Any], extra_answer_keys: list[str] | None
    ) -> bool:
        """True when the text that ships as the answer was cut off at ``max_tokens``.

        Interface contract (callers: ``_standard_run``, ``ADaPTAgent.run``;
        the one place that turns core's cut-off record into a verdict):
        *final_context* carries ``CONTEXT_KEY_CUT_OFF`` when the last turn was
        cut (``_run_conversation_loop``). Answer drawn from the replies
        (``_answer_source_keys`` is ``None``): any cut of the last turn counts
        (``reply``, or a field it extracted). Answer drawn from context keys:
        only a cut of one of THOSE fields counts; an unused reply that was cut
        beside a whole answer does not. Never raises.
        """
        # DECISION plan-2026-11-06T093029-0068b055/D-027: decide truncation
        # where the answer source is known. Do NOT flag "the last reply was
        # cut" for a pattern whose answer is a context key (EvalOpt,
        # MakerChecker, ADaPT, PromptChain): the whole answer then read as
        # success=False in 12.28. Do NOT copy this rule into a pattern: it
        # overrides `_answer_source_keys` only. See decisions.md D-027.
        record = final_context.get(CONTEXT_KEY_CUT_OFF)
        if not isinstance(record, dict) or not record:
            return False
        fields = record.get("fields") or []
        source = self._answer_source_keys(final_context, extra_answer_keys)
        if source is None:
            return bool(record.get("reply") or fields)
        return any(key in fields for key in source)

    @staticmethod
    def _completion_is_real(
        final_context: dict[str, Any],
        trace: AgentTrace,
        extra_answer_keys: list[str] | None,
    ) -> bool:
        """True if the run produced a genuine result.

        A real completion has either a designated answer key (a
        pattern-specific ``extra_answer_key``) or at least one executed tool
        call. When BOTH are absent the ``answer``
        can only have come from the prose-fallback in ``_extract_answer``
        (a planner state's Pass-2 text leaking as the result) — that is a
        degenerate completion, not a success. Planner patterns are judged by
        execution evidence instead (``_run_outcome``).
        """
        has_answer_key = any(
            artifact_text(final_context.get(k)).strip()
            for k in (extra_answer_keys or [])
        )
        tools_executed = len(trace.tool_calls) > 0
        return has_answer_key or tools_executed

    @staticmethod
    def _forced_stop_reason(
        final_context: dict[str, Any], *, judged: bool = False
    ) -> StopReason | None:
        """The ``StopReason`` of a forced stop, or ``None`` if none was forced.

        Interface contract (callers: ``_run_outcome``, ``ADaPTAgent.run``):
        a run was forced when ``max_iterations_reached is True`` (limiter,
        stall) or a forcing handler recorded a reason under
        ``ContextKeys.FORCED_STOP_REASON`` (forced pass at a revision limit).
        Returns that recorded reason when it is one of ``FORCED_STOP_REASONS``,
        else ``StopReason.MAX_ITERATIONS`` for a bare forced flag. With
        ``judged`` (EvalOpt, MakerChecker: a judge handler rules on every
        output that ships) only a recorded reason counts; the bare flag says
        the budget ran out, not that the verdict was overridden. Never raises.
        """
        recorded = recorded_forced_reason(final_context)
        if recorded is not None:
            return recorded
        if not judged and final_context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True:
            return StopReason.MAX_ITERATIONS
        return None

    def _build_trace(
        self,
        final_context: dict[str, Any],
        iteration: int,
    ) -> AgentTrace:
        """Build an AgentTrace from the AGENT_TRACE context entries.

        Agents that don't use tools get an empty trace (just iteration count).
        """
        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )

        for step in trace_data:
            if isinstance(step, dict) and "action" in step:
                # A pattern that names the tool itself (REWOO) wins over the
                # `name(args)` label, which cuts a name holding "(".
                tool_name = (
                    step.get("tool_name") or step.get("action", "").split("(")[0]
                )
                if tool_name and tool_name != ContextKeys.NO_TOOL:
                    tool_input = step.get("tool_input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {"input": tool_input}
                    trace.tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            parameters=tool_input,
                            reasoning=step.get("thought", ""),
                        )
                    )

        return trace

    # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
    # This filter's output feeds `AgentResult.final_context`, which is returned
    # straight to the agent's caller. Do NOT re-inline `k.startswith("_")` here:
    # that check is case-SENSITIVE and only sees the literal `_` prefix, so
    # `system_password`, `internal_token` and `__dunder` all leaked through it
    # (F-13, measured). `has_internal_prefix` is the single canonical predicate
    # over INTERNAL_KEY_PREFIXES and case-folds. See decisions.md D-003.
    @staticmethod
    def _filter_context(context: dict[str, Any]) -> dict[str, Any]:
        """Remove internal-prefixed keys from context.

        Top-level only: nested dict values are not recursed into (that is a
        separate contract, see `fsm_llm.context.clean_context_keys`).
        """
        return {k: v for k, v in context.items() if not has_internal_prefix(k)}

    def _try_parse_structured_output(
        self, answer: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Validate the run's output against ``config.output_schema`` if set.

        Interface contract: returns a Pydantic model instance, or ``None``
        when no schema is configured or no source validates. Sources, in
        order: the JSON in *answer* (the model's own output), context keys
        named like schema fields, JSON in the last tool observations.
        """
        # `output_schema` is declared `type | None` but `AgentConfig.validate_output_schema`
        # (definitions.py) guarantees any non-None value is a Pydantic BaseModel subclass,
        # so `.model_fields` is always present at runtime. Cast narrows for mypy (683,693).
        # Annotation-only — no runtime change.
        schema = cast("type[BaseModel] | None", self.config.output_schema)
        if schema is None:
            return None

        # 1. The answer's own JSON.
        # DECISION plan-2026-11-05T185903-5980fbb7/D-004: this comes BEFORE the
        # context keys. A key that shares a schema field's name (a caller's
        # initial_context hint, `task`, a handler key) used to build a valid
        # object from defaults, so the model's JSON was never read and the hint
        # shipped as structured_output under success=True. Do NOT put the
        # context-key step first again. See decisions.md D-004.
        failure: str | None = None
        try:
            data = extract_json_from_text(answer)
            if data is None:
                data = json.loads(answer)
            if isinstance(data, dict):
                return schema(**data)
            failure = f"expected dict, got {type(data).__name__}"
        except Exception as e:
            failure = str(e)

        # 2. Context keys named like schema fields.
        # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: emit a diagnostic instead of
        # silently swallowing the validation error — a structured_output of None
        # otherwise gives no clue which fields were missing/invalid.
        if context:
            fields = {
                k: context[k]
                for k in schema.model_fields
                if k in context and context[k] is not None
            }
            if fields:
                try:
                    return schema(**fields)
                except Exception as e:
                    logger.debug(
                        f"Structured output: context-key construction of "
                        f"{schema.__name__} failed ({e}); "
                        f"present keys={sorted(fields)}, "
                        f"schema keys={sorted(schema.model_fields)}. "
                        f"Falling back to the observations."
                    )

            # 3. Scan tool observations for JSON matching the schema
            observations = context.get("observations", [])
            if isinstance(observations, list):
                for obs in reversed(observations):
                    if not isinstance(obs, str):
                        continue
                    # Entries are formatted as "[Step n] Tool: ... | Result: ...";
                    # scan only the trailing result segment so embedded JSON is found.
                    segment = obs.rsplit("Result:", 1)[-1] if "Result:" in obs else obs
                    data = extract_json_from_text(segment)
                    if isinstance(data, dict):
                        try:
                            return schema(**data)
                        except Exception:
                            continue

        logger.warning(f"Structured output validation failed: {failure}")
        return None
