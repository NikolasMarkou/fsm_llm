"""
The synthesizer: a strong model compiles a goal into a gated FSM.

``FSMSynthesizer.compile`` asks the strong model for one FSM v4.1 JSON object,
runs the gates, and on a failure feeds the failures back, up to
``policy.max_repair_rounds`` times. It does not touch the library or approval.

# DECISION plan-2026-11-10T194139-2cefeb61/D-005
# A reply cut off at ``max_tokens`` is a failed round and is NEVER parsed (a cut
# object can still be valid JSON and mean less than the model wrote). A provider
# outage is NOT a failed round: it is raised, chained, with no retry (a retry
# spends the strong model's budget on a dead provider). Do NOT repair a reply
# the gates did not see, and do NOT put model text in an error message.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fsm_llm.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    CompletionRequest,
    CompletionResponse,
    FSMDefinition,
    LLMResponseError,
    LLMUsage,
    SynthesizeConfig,
)
from fsm_llm.llm import (
    LiteLLMInterface,
    LLMInterface,
    implements_complete,
    interface_model,
)
from fsm_llm.okf import capped_line
from fsm_llm.utilities import extract_json_from_text

from .exceptions import SynthesisError
from .gates import GateFinding, GateReport, run_gates
from .policy import SynthesisPolicy
from .prompts import REPLY_FAILURES, compile_messages, repair_message

_NO_REPLY = "(no reply)"  # the assistant turn echoed for a reply with no text


@dataclass(frozen=True)
class CompileResult:
    """A compile that passed the gates.

    ``definition`` is the loaded abort-injected procedure and ``data`` the same
    as a plain dict (what the library stores); ``report`` is the gate verdict,
    ``rounds`` the strong-model calls made, ``usage`` a snapshot of the
    interface's call counters (its totals so far, not only this compile) or
    ``None`` when the interface keeps none.
    """

    definition: FSMDefinition
    data: dict[str, Any]
    report: GateReport
    rounds: int
    usage: LLMUsage | None


def _checked(
    goal: object, inputs: Any, return_keys: Any
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(inputs, return_keys)`` as checked tuples; ``ValueError`` for a bad goal
    or name.

    The synthesize field's own key rule decides (the config is built only to
    run it); a name must also be an identifier, so it can never spell a marker.
    """
    if not isinstance(goal, str) or not capped_line(goal, 1):
        raise ValueError("goal must be a non-blank string")
    config = SynthesizeConfig(
        goal="x", inputs=inputs, return_keys=return_keys, on_failure="x"
    )
    names = config.inputs + config.return_keys
    if not all(re.fullmatch(ASCII_IDENTIFIER_PATTERN, n) for n in names):
        raise ValueError("key names must be identifiers (letters, digits, _)")
    return config.inputs, config.return_keys


class FSMSynthesizer:
    """Compiles goals into gated procedures with a strong model.

    Interface contract (callers: ``obtain`` of step 10, the tools, tests):
        Exactly one of ``llm`` (an ``LLMInterface`` that implements
        ``complete``) or ``model`` (a name; a ``LiteLLMInterface`` is built)
        or ``ValueError``: the strong model is never an accidental default.
        ``policy`` (default ``SynthesisPolicy()``) bounds the compile; ``clock`` is the deadline's time source.
        ``compile`` returns a ``CompileResult`` or raises ``SynthesisError``
        whose ``details["rounds"]`` lists the failure codes of every round (no
        model text); a provider outage raises it chained from the
        ``LLMResponseError``. Bad arguments raise ``ValueError`` before any
        call. Caller data is never mutated.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        *,
        model: str | None = None,
        policy: SynthesisPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if llm is not None and model is None:
            if not implements_complete(llm):
                raise ValueError("llm must be an LLMInterface that implements complete")
        elif llm is None and model is not None:
            llm = LiteLLMInterface(model=model)
        else:
            raise ValueError("pass exactly one of llm or model")
        self.llm: LLMInterface = llm
        self.model = model or interface_model(llm)
        self.policy = policy or SynthesisPolicy()
        self._clock = clock

    def compile(
        self,
        goal: str,
        *,
        inputs: Sequence[str] = (),
        return_keys: Sequence[str],
    ) -> CompileResult:
        """Compile ``goal`` into a gate-passed, abort-injected FSM.

        ``inputs`` are the keys the driver provides, ``return_keys`` (at least
        one) the keys the procedure must hand back; both are identifier-only
        public names. At most ``1 + policy.max_repair_rounds`` calls.
        """
        ins, keys = _checked(goal, inputs, return_keys)
        policy = self.policy
        messages = compile_messages(goal, ins, keys, policy)
        deadline = self._clock() + policy.max_synthesis_seconds
        failed: list[tuple[str, ...]] = []
        reason = "no round passed the gates"
        for attempt in range(1 + policy.max_repair_rounds):
            if attempt and self._clock() >= deadline:
                reason = "the compile deadline passed"
                break
            request = CompletionRequest(
                messages=messages, max_tokens=policy.max_compile_tokens
            )
            try:
                reply = self.llm.complete(request)
            except LLMResponseError as e:
                raise SynthesisError(
                    "the strong model call failed", {"rounds": failed}
                ) from e
            report = _judge(reply, policy, ins, keys)
            if report.definition is not None and report.injected is not None:
                usage = getattr(self.llm, "usage", None)
                return CompileResult(
                    report.definition,
                    report.injected,
                    report,
                    attempt + 1,
                    usage() if callable(usage) else None,
                )
            failed.append(tuple(dict.fromkeys(f.code for f in report.findings)))
            messages = [
                *messages,
                {"role": "assistant", "content": reply.text or _NO_REPLY},
                {"role": "user", "content": repair_message(report)},
            ]
        last = ", ".join(failed[-1]) if failed else "none"
        raise SynthesisError(
            f"{reason} after {len(failed)} failed round(s); last codes: {last}",
            {"rounds": failed},
        )


def _judge(
    reply: CompletionResponse,
    policy: SynthesisPolicy,
    inputs: tuple[str, ...],
    return_keys: tuple[str, ...],
) -> GateReport:
    """One reply to one report: a reply that is not a readable JSON object
    fails as one G1 finding (truncated first, never parsed), else the gates run."""
    code = (
        "reply-truncated"
        if reply.truncated
        else "reply-malformed"
        if reply.kind != "final"
        else ""
    )
    data = None if code else extract_json_from_text(reply.text or "")
    if data is None:
        code = code or "reply-not-json-object"
        return GateReport((GateFinding("G1", code, "", REPLY_FAILURES[code]),))
    return run_gates(data, policy, inputs=inputs, return_keys=return_keys)
