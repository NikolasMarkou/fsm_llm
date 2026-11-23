"""``CallMixin``: the non-streaming request of ``LiteLLMInterface`` (``_make_llm_call`` over its
one body ``_amake_llm_call``) and the reasoning-trace readers it uses.

Layer ``llm`` (plan 51fde95b, step 3): split out of ``providers/litellm.py`` to keep that
module under the size guard; the bodies moved unchanged and step 4.4 made ``_make_llm_call`` the
``run_sync`` wrapper of the async body, and step 4.7 added ``warn_bypassed_twins``, the subclass
guard of D-009. Nothing here looks up a provider binding
(``completion`` and the like stay in ``litellm.py``, where a test patches them), so this module
imports neither litellm nor ``providers/litellm.py``.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable
from types import FrameType
from typing import TYPE_CHECKING, Any

from fsm_llm.aio import acall, async_name, run_sync
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.llm.parsing import ParseMixin
from fsm_llm.llm.text import resolve_reasoning_trace
from fsm_llm.logging import logger

# The sync names of ``LiteLLMInterface`` whose async twin a subclass must override with them.
_SYNC_TWINS = (
    "generate_response",
    "generate_response_stream",
    "extract_field",
    "extract_bulk_data",
    "complete",
    "_send",
    "_make_llm_call",
)


def _is_machinery(frame: FrameType) -> bool:
    """A frame of this package, of ``abc`` or of an ``__init_subclass__`` hook: never the
    class statement a user wrote."""
    name = frame.f_globals.get("__name__", "")
    return (
        name == "fsm_llm"
        or name.startswith("fsm_llm.")
        or name == "abc"
        or frame.f_code.co_name == "__init_subclass__"
    )


def warn_bypassed_twins(cls: type, twins: Iterable[str] = _SYNC_TWINS) -> None:
    """Warn once per sync twin that ``cls`` defines without its async twin.

    Interface contract: ``cls`` is a class being created (call it from ``__init_subclass__``:
    ``LiteLLMInterface`` with the default ``twins``, ``FSMSynthesizer`` with its own three names,
    plan ace7a8c8 D-068); ``twins`` are the sync names to check. A name of ``twins`` defined in the body of ``cls``
    and not its ``async_name`` gets one ``RuntimeWarning`` attributed to the first frame
    outside this package, ``abc`` and any ``__init_subclass__`` (the class statement, also
    through an intermediate class with its own hook); nothing else is read or changed.
    """
    # DECISION plan-2026-11-19T025425-51fde95b/D-009: this only WARNS and changes no dispatch.
    # An async caller reaches the inherited native twin, so an override of the sync name alone
    # is bypassed; do NOT make `acall` MRO-aware to fix that here (a second rule inside the
    # hot helper that the engine part of the async core would have to carry and test).
    # The reverse (an `a` twin overridden alone) is NOT warned (D-045): it is the normal way to
    # wrap a twin with `await super()`, and a body that really suspends is a documented rule.
    # See decisions.md D-009.
    for name in twins:
        twin = async_name(name)
        if name in vars(cls) and twin not in vars(cls):
            frame: FrameType | None = sys._getframe()
            level = 1
            while frame is not None and _is_machinery(frame):
                frame, level = frame.f_back, level + 1
            warnings.warn(
                f"{cls.__name__} overrides {name} but not {twin}: async callers reach the "
                "inherited native twin and bypass the override",
                RuntimeWarning,
                stacklevel=level,
            )


class CallMixin:
    """``_make_llm_call`` and the thinking-trace helpers of ``LiteLLMInterface``.

    Mixed into ``LiteLLMInterface``; ``self`` is that interface.
    """

    if TYPE_CHECKING:
        # Provided by ``RequestMixin`` and ``LiteLLMInterface`` (declared for mypy only).
        def _build_call_params(
            self,
            messages: list[dict[str, Any]],
            call_type: str,
            *,
            response_format: dict[str, Any] | None = None,
            stream: bool = False,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> dict[str, Any]: ...

    def _make_llm_call(
        self,
        messages: list[dict[str, str | None]],
        call_type: str,
        response_format: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
    ) -> Any:
        """
        Make LLM API call with appropriate configuration.

        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).
            temperature: per-call temperature (a Pass-2 request's
                ``temperature``); ``None`` keeps the interface's.

        Returns:
            Raw LLM response
        """
        return run_sync(
            self._amake_llm_call(
                messages, call_type, response_format, temperature=temperature
            )
        )

    async def _amake_llm_call(
        self,
        messages: list[dict[str, str | None]],
        call_type: str,
        response_format: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
    ) -> Any:
        """The async twin of ``_make_llm_call`` and its one body: build, send, validate."""
        call_params = self._build_call_params(
            messages,
            call_type,
            response_format=response_format,
            temperature=temperature,
        )

        # Make the API call
        response = await acall(self, "_send", call_params, call_type=call_type)

        # Validate response structure. D-022: `getattr(response, "choices", None)`
        # replaces `hasattr(response, "choices") and response.choices` -- see the
        # D-022 comment of ``LiteLLMInterface.agenerate_response_stream`` for why this
        # "not falsy" shape is safe to convert.
        choices = getattr(response, "choices", None) if response else None
        if not choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = choices[0]
        # D-022: `getattr(choice, "message", None) is not None` replaces
        # `hasattr(choice, "message")` -- safe, since a `message` attribute
        # present-but-`None` falls through to the SAME raise via the `content`
        # check below either way (`hasattr(None, "content")` is `False`).
        #
        # The SECOND check is deliberately LEFT as `hasattr()`, not converted.
        # DECISION plan-2026-09-20T114608-a8e47b88/D-022
        # `hasattr(choice.message, "content")` checks only that the `content`
        # ATTRIBUTE EXISTS, regardless of its value -- and a real litellm
        # `Message` legitimately carries `content=None` for an Ollama
        # reasoning-only reply (H3, `test_make_llm_call_recovers_from_none_content`).
        # `getattr(choice.message, "content", None) is not None` CANNOT
        # replicate this: it cannot distinguish "attribute absent" from
        # "attribute present, value None" (both collapse to the same `None`),
        # so it would raise "Response missing message content" for that
        # legitimate None-content case, before `_extract_content_from_thinking`
        # ever runs -- a real regression, not a style nit. Confirmed empirically
        # (not just reasoned): temporarily forcing this conversion during EXECUTE
        # made `test_make_llm_call_recovers_from_none_content` fail. This is the
        # `hasattr()` call D-003 already reserved: "falls back to a documented
        # exception... only if that test cannot be made to pass." See decisions.md
        # D-022 (this call site) and D-003 (the original judgment call).
        message = getattr(choice, "message", None)
        if message is None or not hasattr(message, "content"):
            raise LLMResponseError("Response missing message content")

        content = message.content
        if not content:
            # H3: `content` is falsy — empty string OR None. Ollama reasoning-
            # only replies arrive as content=None (not ""), so this must fire
            # on None too, else C2 recovery never runs in production.
            content = self._extract_content_from_thinking(choice.message)
            if content is not None:
                choice.message.content = content

        if not content:
            raise LLMResponseError("LLM returned empty content")

        return response

    @staticmethod
    def _extract_content_from_thinking(message) -> str | None:
        """Extract structured content from a model's reasoning/thinking trace.

        Some models (e.g. Qwen 3.5 via Ollama) place the actual answer in a
        reasoning field and leave ``content`` empty or ``None``. This helper
        tries to recover the last JSON object from that trace, falling back to
        the last non-empty line.

        Returns the extracted content string, or ``None`` if no reasoning
        trace is present.
        """
        # DECISION plan-2026-07-21T045419-9925aa3a/D-002
        # The reasoning-trace field-name resolution (reasoning_content ->
        # legacy thinking -> thinking_blocks) now lives in the SINGLE shared
        # `resolve_reasoning_trace` (llm/text.py) so this reader and
        # engine/classification.py::_extract_response can never re-diverge. Do NOT
        # revert to a `hasattr(message, "thinking")`/`.thinking`-only read:
        # installed litellm RENAMES `thinking` to `reasoning_content` and
        # DELETES `thinking` before building the Message/Delta object, making a
        # `.thinking`-only read dead code for the project's own
        # former default model. Rationale + anti-simplification note live at the
        # resolver's D-002 anchor. See decisions.md D-001/D-002.
        trace = resolve_reasoning_trace(message)
        if not trace:
            return None
        logger.debug("Content empty but reasoning trace present, extracting from it")
        return ParseMixin._recover_content_from_trace(trace)

    @staticmethod
    def _find_nested_key(data: dict, key: str, max_depth: int = 3) -> Any:
        """Search nested dicts for a key, returning first match."""
        if max_depth <= 0:
            return None
        for v in data.values():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
                found = CallMixin._find_nested_key(v, key, max_depth - 1)
                if found is not None:
                    return found
        return None
