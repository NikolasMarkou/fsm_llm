"""What every route module shares: the error mapping, the bounded await, the JSON
body reader and the API-key gate dependency."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, NoReturn, TypeVar

from fastapi import Depends, HTTPException
from starlette.requests import Request

from fsm_llm.core.definitions import ConversationBusyError
from fsm_llm.logging import logger

from ..exceptions import ConversationEndingError, MonitorCapacityError
from .application import _require_api_key

_T = TypeVar("_T")

# Default timeout for synchronous FSM/LLM operations wrapped in asyncio.to_thread
_LLM_OPERATION_TIMEOUT = 120.0


def _raise_http(e: Exception, action: str) -> NoReturn:
    """Map a manager exception to an HTTP error (one mapping for all routes).

    KeyError -> 404, TypeError (wrong instance type) / ValueError -> 400,
    ConversationEndingError / ConversationBusyError -> 409, MonitorCapacityError -> 429,
    NotImplementedError -> 501, anything else -> 500
    with a generic detail (the exception text is logged, never returned).
    """
    if isinstance(e, HTTPException):
        raise e
    if isinstance(e, KeyError):
        raise HTTPException(status_code=404, detail="not found") from e
    if isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=404, detail="not found") from e
    if isinstance(e, ConversationEndingError):  # before its base class below
        raise HTTPException(
            status_code=409, detail="conversation is already being ended"
        ) from e
    if isinstance(e, ConversationBusyError):
        raise HTTPException(
            status_code=409, detail="conversation is busy with another turn"
        ) from e
    if isinstance(e, MonitorCapacityError):
        raise HTTPException(status_code=429, detail=str(e)) from e
    if isinstance(e, NotImplementedError):
        raise HTTPException(status_code=501, detail=str(e) or "not available") from e
    if isinstance(e, TypeError | ValueError):
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.error(f"Failed to {action}: {e}")
    raise HTTPException(status_code=500, detail="Internal server error") from e


async def _bounded(
    awaitable: Awaitable[_T],
    action: str,
    timeout_detail: str,
    timeout: float = _LLM_OPERATION_TIMEOUT,
) -> _T:
    """Await *awaitable* for at most *timeout* seconds: a timeout is a 504 with
    *timeout_detail*, any other failure goes through ``_raise_http``.

    A worker thread behind ``asyncio.to_thread`` cannot be stopped on timeout;
    what it started (a conversation) may still show up afterwards.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        raise HTTPException(status_code=504, detail=timeout_detail) from None
    except Exception as e:
        _raise_http(e, action)


async def _json_body(request: Request) -> Any:
    """The request's JSON body, or a 400."""
    try:
        return await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid JSON") from e


_GATED = [Depends(_require_api_key)]
