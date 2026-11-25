"""``SendMixin``: the three helpers of the send family of ``LiteLLMInterface``.

Layer ``llm`` (plan 3f4530d3, step 1a): split out of ``providers/litellm.py`` to keep that module
off the size guard; the bodies moved unchanged. ``_send`` and ``_asend`` stay in
``providers/litellm.py``: they hold the module-level ``completion`` and ``acompletion`` lookups
that a test patches, and ``observer.is_observed`` reads ``vars(c)`` of the class that defines
``_send``, ``_asend`` and the ``observes_requests`` marker. Nothing here looks up a provider
binding, so this module imports neither litellm nor ``providers/litellm.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsm_llm.core.definitions import FINISH_REASON_LENGTH
from fsm_llm.llm.observer import (
    SendHandle,
    abegin_send,
    aend_send,
    begin_send,
    end_send,
    fail_send,
)
from fsm_llm.llm.parsing import _finish_reason
from fsm_llm.llm.usage import _meter_of, _UsageMeter, usage_kind
from fsm_llm.logging import logger


class SendMixin:
    """``_begin_send``, ``_failed_send`` and ``_finished_send``: what comes before, after a
    failure of, and after a success of the one provider call of ``_send`` and ``_asend``."""

    if TYPE_CHECKING:
        # Provided by ``LiteLLMInterface`` (declared for mypy only).
        model: str

    def _begin_send(
        self, call_params: dict[str, Any], call_type: str
    ) -> tuple[str, _UsageMeter, SendHandle | None]:
        """What comes BEFORE the provider call: the usage kind, the meter, the observer token.

        One of the three helpers of the send family (``_begin_send``, ``_failed_send``,
        ``_finished_send``); ``begin_send`` fails closed, so a refusing observer stops the
        call before it is made.
        """
        kind, meter = self._kind_and_meter(call_params, call_type)
        # DECISION plan-2026-11-17T100529-21e6f00d/D-002: the observer gets an
        # allow-listed copy and is not a counter; begin_send fails closed. See observer.py.
        token = begin_send(call_params, call_type=call_type)
        return kind, meter, token

    async def _abegin_send(
        self, call_params: dict[str, Any], call_type: str
    ) -> tuple[str, _UsageMeter, SendHandle | None]:
        """The async twin of ``_begin_send``."""
        kind, meter = self._kind_and_meter(call_params, call_type)
        token = await abegin_send(call_params, call_type=call_type)
        return kind, meter, token

    def _kind_and_meter(
        self, call_params: dict[str, Any], call_type: str
    ) -> tuple[str, _UsageMeter]:
        # DECISION plan-2026-10-01T093600-944e2692/D-004 [SUPERSEDED BY
        # plan-2026-11-18T071726-5a6a1f80/D-014 on resets, and BY
        # plan-2026-11-19T025425-51fde95b/D-007 on "here": the send family of `_send`
        # and `_asend`]: the meter is per
        # instance and fed here only; a reset moves a marker, see llm/usage.py.
        # Do NOT add a process-global counter, do NOT count in the public
        # methods, do NOT read usage off a stream (its chunks carry none).
        # Readers that want counts own (inject) the interface. See D-004, D-014.
        return usage_kind(call_type, bool(call_params.get("stream"))), _meter_of(self)

    def _failed_send(
        self,
        meter: _UsageMeter,
        kind: str,
        token: SendHandle | None,
        exc: BaseException,
    ) -> None:
        """A request that raised: one call and one error on the meter, then the observer."""
        meter.record_error(kind)
        fail_send(token, exc)

    def _finished_send(
        self,
        meter: _UsageMeter,
        kind: str,
        token: SendHandle | None,
        call_params: dict[str, Any],
        call_type: str,
        response: Any,
    ) -> Any:
        """A request that returned: the meter, the observer, the cut-off warning; the reply."""
        meter.record(kind, None if call_params.get("stream") else response)
        end_send(token, response)
        return self._warn_if_cut(call_params, call_type, response)

    async def _afinished_send(
        self,
        meter: _UsageMeter,
        kind: str,
        token: SendHandle | None,
        call_params: dict[str, Any],
        call_type: str,
        response: Any,
    ) -> Any:
        """The async twin of ``_finished_send``."""
        meter.record(kind, None if call_params.get("stream") else response)
        await aend_send(token, response)
        return self._warn_if_cut(call_params, call_type, response)

    def _warn_if_cut(
        self, call_params: dict[str, Any], call_type: str, response: Any
    ) -> Any:
        # DECISION plan-2026-11-06T093029-0068b055/D-024: finish_reason
        # "length" is read HERE, once per non-streamed reply (a streamed one
        # is not read, as for usage). A cut reply is warned about but never
        # raised on: it may be prose that is fine as far as it goes. Do NOT
        # add a per-method check beside this one (a second site drifts); the
        # parse paths read the same reason through `_finish_reason` to name
        # the cause in their own errors. See decisions.md D-024.
        if (
            not call_params.get("stream")
            and _finish_reason(response) == FINISH_REASON_LENGTH
        ):
            logger.warning(
                f"Reply from '{call_params.get('model', self.model)}' "
                f"({call_type}) was cut off at "
                f"max_tokens={call_params.get('max_tokens')}; raise max_tokens "
                "(or LLM_MAX_TOKENS) if the full reply is needed"
            )
        return response
