"""
Small-model battle-hardening for the harness protocol.

This module is why the harness can drive a multi-step protocol with a 4B model
(``ollama_chat/qwen3.5:4b``) without the run silently derailing.  It has three
jobs and it does all three by *composing* existing ``fsm_llm`` primitives --
it writes no second JSON parser, no second brace scanner and no second Ollama
thinking-suppressor (all three are named Complexity-Budget BREACH conditions).

1. **Noise removal + parsing.**  :func:`strip_model_noise` unconditionally
   removes ``<think>`` / ``<thinking>`` tags and markdown fences for EVERY call
   type, routing around the known M6 asymmetry in core (``llm.py``'s
   field-extraction parser strips both at ``llm.py:879-888``; its
   response-generation parser at ``llm.py:728-763`` does not).
   :func:`parse_json_payload` then delegates to the canonical 4-strategy
   recovery ladder ``fsm_llm.utilities.extract_json_from_text``
   (``utilities.py:186-423``), whose Strategy 3 owns the repo's only
   string/escape-aware brace scanner (``_match_brace_partners``,
   ``utilities.py:119-183``).

2. **Fail-closed exact-type coercion.**  :func:`type_matches`, :func:`as_int`
   and :func:`coerce_worker_output` are the *load-bearing* half of invariant I8
   (see decisions.md D-025 and D-059).  The JsonLogic gate behind them uses
   soft comparison -- ``"3" >= 3``, ``3.0 >= 3`` and even ``True >= 3`` all
   evaluate True in ``fsm_llm.expressions`` -- so a garbled worker reply must
   be rejected *here*, before it ever reaches context.  These three are the
   package's ONLY exact-type predicates: ``harness.py`` imports them rather
   than keeping private twins (D-059 closed that duplication).

3. **Harness-level retry.**  :func:`retry` exists because
   ``LiteLLMInterface(retries=N)`` is a **measured no-op** for ``ollama_chat/*``
   and ``ollama/*`` (``llm.py:254-258``).  Provider-level retry is simply not
   available for the harness's own default model, so retry lives here.

**Ollama call preparation is deliberately absent -- see decisions.md D-030.**
``LiteLLMInterface._make_llm_call`` already applies every helper in
``fsm_llm/ollama.py`` on the harness's behalf; adding a second layer here would
be redundant duplication, not hardening.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fsm_llm.definitions import LLMResponseError
from fsm_llm.logging import logger
from fsm_llm.utilities import extract_json_from_text

from .constants import Defaults

__all__ = [
    "RETRYABLE_EXCEPTIONS",
    "RoleOutput",
    "as_int",
    "coerce_worker_output",
    "parse_json_payload",
    "parse_role_output",
    "retry",
    "strip_model_noise",
    "type_matches",
]

T = TypeVar("T")

# DECISION plan-2026-07-21T125237-191b2eb2/D-030
# There is deliberately NO ollama call-preparation function in this module, and
# adding one would be duplication rather than hardening. Measured against the
# real call path: `LiteLLMInterface._make_llm_call` already
#   - calls `apply_ollama_params` for EVERY call type via
#     `_apply_model_specific_params` (llm.py:599,633-642) -- so
#     `reasoning_effort="none"` is always set, and `temperature=0` is set for
#     the structured (extraction) call types;
#   - calls `prepare_ollama_messages` for EVERY call type (llm.py:602-606) --
#     so the `/nothink` prefix and the schema-in-prompt append always happen;
#   - calls `build_ollama_response_format` for the extraction call types
#     (llm.py:575-580), and forwards an explicit caller `response_format`
#     otherwise (llm.py:584-597).
# Every LLM call the harness makes -- worker agents via `create_agent`, the
# driver's own Pass-1/Pass-2 turns -- goes through that method. A second layer
# here would be a second Ollama thinking-suppressor, a named Complexity-Budget
# BREACH. If this ever stops being true, fix it in `fsm_llm/ollama.py`'s
# consumers, not by growing a shadow copy here.
# See decisions.md D-030.


# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

#: Balanced ``<think>...</think>`` / ``<thinking>...</thinking>`` blocks.
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.I)

#: Any REMAINING lone opening or closing think tag (a truncated 4B reply).
_THINK_TAG_RE = re.compile(r"</?\s*think(?:ing)?\b[^>]*>", re.I)

#: A markdown fence marker with an optional language hint (```json, ```, ...).
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*")


def strip_model_noise(text: str) -> str:
    """Remove reasoning tags and markdown fences from a small model's reply.

    Contract:
        - Parameter: any object; a non-``str`` returns ``""``.
        - Returns the input with ``<think>``/``<thinking>`` blocks, orphaned
          think tags and markdown fence markers removed, then stripped.
        - Never raises, for any input.
        - Idempotent: ``strip_model_noise(strip_model_noise(x)) == \
strip_model_noise(x)``.

    This is a PRE-FILTER only.  It deliberately does not attempt to locate the
    JSON payload inside surrounding prose -- that is
    :func:`parse_json_payload`'s job, delegated to ``extract_json_from_text``.
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-027
    # The two-step tag removal below is NOT redundant, and the obvious
    # simplifications both destroy real payloads:
    #   1. Do NOT collapse this to `re.sub(r"<think>.*", "", text, re.DOTALL)`
    #      ("just drop everything from the tag onwards"). A truncated 4B reply
    #      routinely arrives as `<think>I should answer\n{"findings_count": 3}`
    #      -- the JSON sits AFTER an unclosed tag, and that simplification
    #      deletes it. Balanced blocks are removed WITH their content (they are
    #      reasoning); an UNMATCHED tag has only its markup removed, so the
    #      payload survives on either side of it.
    #   2. Do NOT add prose trimming here ("strip everything before the first
    #      `{`"). Locating a JSON span in prose is brace scanning, and this
    #      package is forbidden from owning a second brace scanner -- the only
    #      string/escape-aware one in the repo is `_match_brace_partners`
    #      (utilities.py:119), reached via `extract_json_from_text` Strategy 3.
    #      A naive first-`{`/last-`}` trim mis-handles a `}` inside a quoted
    #      string, which is exactly the open M5 defect in llm.py:704-720.
    # See decisions.md D-027.
    if not isinstance(text, str):
        return ""
    cleaned = _THINK_BLOCK_RE.sub(" ", text)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    cleaned = _FENCE_RE.sub(" ", cleaned)
    return cleaned.strip()


def parse_json_payload(text: Any) -> dict[str, Any] | None:
    """Recover a JSON object from a small model's noisy reply.

    Contract:
        - Parameter: the model's raw reply (any type; non-``str`` yields
          ``None``).
        - Returns the parsed ``dict``, or ``None`` when nothing recoverable is
          present.  A JSON array or scalar at top level is NOT a payload and
          yields ``None`` -- the protocol only ever exchanges objects.
        - Never raises, for any input.  Parse failure is a value, not an
          exception, because every caller treats it as "record success=False
          and leave the gate BLOCKED".
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = strip_model_noise(text)
    for candidate in (cleaned, text) if cleaned != text else (cleaned,):
        if not candidate.strip():
            continue
        try:
            parsed = extract_json_from_text(candidate)
        except Exception as exc:  # pragma: no cover - ladder never raises today
            logger.debug(f"extract_json_from_text raised on harness input: {exc}")
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Typed role output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleOutput:
    """The typed result of parsing one worker's raw reply.

    Part of the worker-seam data group (``RoleRequest`` in, ``RoleOutput``
    out); see decisions.md D-028 for its budget accounting.

    Attributes:
        payload: The parsed object, or ``{}`` when nothing was recoverable.
        success: ``True`` only when a JSON object was recovered AND every
            requested key is present.  This is the flag the driver records as
            ``success=False`` when a 4B model emits garbage.
        failure_reason: ``None`` on success; otherwise one of
            ``"empty-reply"``, ``"unparseable"`` or
            ``"missing-keys:<a>,<b>"``.
        missing_keys: The requested keys absent from ``payload``.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)
    success: bool = False
    failure_reason: str | None = None
    missing_keys: tuple[str, ...] = ()


def parse_role_output(
    raw: Any,
    *,
    expected_keys: Iterable[str] = (),
) -> RoleOutput:
    """Turn a worker's raw reply into a :class:`RoleOutput`.

    Contract:
        - ``raw`` may be a ``str`` (the usual case -- ``AgentResult.answer``),
          a ``Mapping`` (an already-structured worker result), ``None``, or
          anything else (stringified as a last resort).
        - ``expected_keys`` names the keys the role's schema requires; a
          payload missing any of them is a FAILURE, not a partial success --
          fail closed.
        - Never raises, for any input.
    """
    if raw is None:
        return RoleOutput(failure_reason="empty-reply")

    if isinstance(raw, Mapping):
        payload: dict[str, Any] | None = dict(raw)
    else:
        text = raw if isinstance(raw, str) else str(raw)
        if not text.strip():
            return RoleOutput(failure_reason="empty-reply")
        payload = parse_json_payload(text)

    if payload is None:
        return RoleOutput(failure_reason="unparseable")

    missing = tuple(key for key in expected_keys if key not in payload)
    if missing:
        return RoleOutput(
            payload=payload,
            success=False,
            failure_reason="missing-keys:" + ",".join(missing),
            missing_keys=missing,
        )
    return RoleOutput(payload=payload, success=True)


# ---------------------------------------------------------------------------
# Fail-closed exact-type coercion (invariant I8, primary layer)
# ---------------------------------------------------------------------------

# DECISION plan-2026-07-21T125237-191b2eb2/D-028
# These coercers are EXACT-TYPE ON PURPOSE and must never be made "helpful".
# The gate they feed is soft-comparing JsonLogic (see decisions.md D-025): in
# `fsm_llm.expressions`, `less()` runs both operands through `float()` and
# `greater_or_equal(a, b)` is `greater(a, b) or soft_equals(a, b)`. Measured
# consequence: `"3" >= 3`, `3.0 >= 3` and `True >= 3` are ALL True. So a
# coercer that "helpfully" accepted the string "3", the float 3.0 or the bool
# True would hand a 4B model's garbled reply straight through a HARD gate.
#
# Two specific traps a future simplification will walk into:
#   1. `isinstance(True, int)` is True in Python. A bare `isinstance(v, int)`
#      admits `True` as the integer 1. Every int path below rejects `bool`
#      FIRST, explicitly.
#   2. `int("3")` / `int(3.0)` succeed. Do NOT reach for `int(value)` in a
#      try/except -- that is precisely the leniency this layer exists to deny.
#
# A miss returns the caller's default, or drops the key. It never guesses.
#
# There is exactly ONE exact-type predicate in this package -- `type_matches`.
# Do NOT add per-type wrappers back (`coerce_int`/`coerce_bool`/`coerce_str`
# existed from step 5 to step 7e with zero call sites between them; deleted by
# D-059). `type_matches(v, int)` already says everything they said, in one
# call, and a second spelling of the same rule is exactly how the harness.py
# twins that D-059 removed came to exist in the first place.
# See decisions.md D-028 and D-059.


def type_matches(value: Any, expected: type) -> bool:
    """Exact-runtime-type predicate used by the worker-reply allowlist (I8).

    Interface contract (shared helper; 2 call sites, both in this module:
    :func:`as_int` and :func:`coerce_worker_output`.  It is exported because it
    is the package's single spelling of "exactly this type" -- ``harness.py``
    reaches it through :func:`coerce_worker_output` rather than keeping the
    private twin it carried until step 7e, see decisions.md D-059):
        - Parameters: any ``value``, and the ``type`` the protocol expects.
        - Returns ``True`` only when ``value``'s runtime type is exactly
          right.  ``bool`` never satisfies ``int``; ``int`` never satisfies
          ``bool``; subclasses of other types are accepted as usual.
        - Never raises unless ``expected`` is not a class (``isinstance``'s own
          ``TypeError``, which is a programmer error, not model noise).
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def as_int(value: Any, default: int) -> int:
    """Return *value* when it is exactly an ``int``, else *default*.

    Interface contract (shared helper; ~19 call sites in ``harness.py`` for
    the protocol counters, which must always produce a number):
        - ``bool`` is rejected, numeric strings are rejected, ``float`` is
          rejected (even ``3.0``), ``None`` is rejected -- every one of them
          yields *default*.  See the D-028 block above for why leniency here
          would hand a garbled reply through a HARD gate.
        - Never raises, for any input.
    """
    return value if type_matches(value, int) else default


def coerce_worker_output(
    payload: Mapping[str, Any],
    allowlist: Mapping[str, type],
    *,
    where: str = "worker",
) -> dict[str, Any]:
    """Filter a worker's payload down to allowed keys of the exact right type.

    Interface contract (shared helper, 2 call sites: ``roles.py``'s default
    worker factory and ``harness.py``'s ``_apply_role_result`` -- BOTH live,
    verified by grep at step 7e; see decisions.md D-059):
        - ``payload``: whatever the worker returned, already parsed.
        - ``allowlist``: ``{context_key: expected_type}`` for the dispatching
          state.  The TABLE is owned by the caller (``harness.py``'s
          ``_WORKER_WRITABLE``); this function owns only the ALGORITHM, so the
          two can never diverge into copies.
        - ``where``: a label used only in the drop warning.
        - Returns a new ``dict`` containing exactly the allowlisted keys whose
          values pass :func:`type_matches`.  Every other key -- unknown,
          absent, ``None``, or the wrong runtime type -- is DROPPED, which
          leaves the corresponding gate BLOCKED.
        - Never raises for a ``Mapping`` payload; a non-``Mapping`` yields
          ``{}``.
    """
    if not isinstance(payload, Mapping) or not allowlist:
        return {}

    accepted: dict[str, Any] = {}
    for key, expected in allowlist.items():
        if key not in payload:
            continue
        value = payload[key]
        if not type_matches(value, expected):
            logger.warning(
                f"{where} returned {key}={value!r} ({type(value).__name__}); "
                f"expected {expected.__name__}. Dropping it -- "
                f"the gate stays closed."
            )
            continue
        accepted[key] = value
    return accepted


# ---------------------------------------------------------------------------
# Harness-level retry
# ---------------------------------------------------------------------------

#: Exceptions worth retrying: transient failures at the LLM/transport boundary.
#:
#: ``LLMResponseError`` is core's system-boundary wrapper (``llm.py:354,481,
#: 515``) and covers provider timeouts, empty replies and transport faults.
#: ``TimeoutError`` and ``ConnectionError`` cover a raw socket/HTTP failure
#: reaching the harness unwrapped.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    LLMResponseError,
    TimeoutError,
    ConnectionError,
)


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = Defaults.RETRY_ATTEMPTS,
    base_delay: float = Defaults.RETRY_BASE_DELAY,
    max_delay: float = Defaults.RETRY_MAX_DELAY,
    backoff_factor: float = Defaults.RETRY_BACKOFF_FACTOR,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "call",
) -> T:
    """Call *fn* with bounded exponential backoff, retrying only transient faults.

    Interface contract (shared helper, 2+ call sites: ``roles.py``'s default
    worker factory from step 6 and the live spike from step 7):
        - ``fn``: a zero-argument callable.  Bind arguments with
          ``functools.partial`` or a lambda.
        - ``attempts``: total calls, not extra calls.  ``attempts=1`` disables
          retrying.  Must be >= 1.
        - Delay before attempt *n* (1-based) is
          ``min(base_delay * backoff_factor ** (n - 1), max_delay)``.
        - ``retry_on``: a strict ALLOWLIST.  Anything not listed propagates
          from the first attempt, uncaught and unslept.
        - ``sleep``: injected so tests run instantly.
        - Returns ``fn()``'s value; re-raises the LAST retryable exception
          when every attempt fails.
        - Raises ``ValueError`` for ``attempts < 1`` (a programmer error, not
          model noise).
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-029
    # Two things here look like over-engineering and are not:
    #   1. This function EXISTS because `LiteLLMInterface(retries=N)` is a
    #      MEASURED no-op for `ollama_chat/*` and `ollama/*` (llm.py:254-258):
    #      exactly one provider request is made regardless of N. Do NOT delete
    #      this and "just pass retries=" -- the harness's own default model is
    #      `ollama_chat/qwen3.5:4b`, so that path retries nothing at all.
    #   2. `retry_on` is a strict ALLOWLIST, not "except Exception". A
    #      deterministic failure -- a pydantic ValidationError, a schema
    #      mismatch, a HarnessError, a 400 -- cannot succeed on attempt 2, so
    #      retrying it three times only burns 3x the wall clock, and on a 4B
    #      model each attempt is tens of seconds. Core made the same call for
    #      the same reason at llm.py:553-563 (`max_retries`, not
    #      `num_retries`, precisely to avoid retrying 400/401).
    # See decisions.md D-029.
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    # Every attempt EXCEPT the last, each followed by a backoff sleep.
    for attempt in range(1, attempts):
        try:
            return fn()
        except retry_on as exc:
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            logger.warning(
                f"{description} failed on attempt {attempt}/{attempts} "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s"
            )
            sleep(delay)

    # The last attempt is made outside the loop so its failure propagates
    # unchanged -- the caller sees the real exception, not a wrapper, and there
    # is no unreachable "we never ran" branch for a reader to puzzle over.
    try:
        return fn()
    except retry_on as exc:
        logger.error(f"{description} failed after {attempts} attempts: {exc}")
        raise


# DECISION plan-2026-07-21T125237-191b2eb2/D-059
# There is deliberately NO `build_response_format` helper in this module, and
# re-adding one would be duplication rather than hardening. It existed from
# step 5 to step 7e with ZERO call sites while its own docstring asserted two
# ("`roles.py` from step 6 ... and `test_hardening.py` from step 13") -- the
# grep is unambiguous, `roles.py` never called it. The reason it has no callers
# is structural, not "not wired yet": every worker this package dispatches goes
# through `create_agent`/`BaseAgent` (roles.py:568,676-682), and setting
# `AgentConfig.output_schema` makes `BaseAgent._init_context` (base.py:172-184)
# build the identical `{"type": "json_schema", ...}` envelope itself. A second
# builder here would be a shadow copy of core's, kept in lockstep by hand.
# If a future step really does need a direct `API`/`LiteLLMInterface` call that
# bypasses `BaseAgent`, pass `AgentConfig.output_schema` instead; only if that
# is impossible should this come back, WITH its call site in the same commit.
# See decisions.md D-059.
