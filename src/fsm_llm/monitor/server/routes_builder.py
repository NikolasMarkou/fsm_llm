"""Routes: meta-agent builder sessions, and the session store they share with the
dashboard push (``websocket.py`` sweeps stale sessions)."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import HTTPException

from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.llm import resolve_model
from fsm_llm.logging import logger

from ..constants import MAX_BUILDER_SESSIONS
from ..definitions import BuilderSendRequest, BuilderStartRequest
from .application import app
from .support import _GATED, _LLM_OPERATION_TIMEOUT, _raise_http

# Builder builds run a multi-step LLM loop — needs more time
_BUILDER_OPERATION_TIMEOUT = 300.0

# --- Builder (Meta-Agent) ---

# Session store: session_id -> (agent, last_used_timestamp)
_builder_sessions: dict[str, tuple[Any, float]] = {}
_BUILDER_SESSION_TTL = 3600.0  # 1 hour since last use
# Session IDs with an in-flight `send` — guards against concurrent sends to the
# same MetaBuilderAgent corrupting its internal state (the agent is not
# re-entrant). Mutated only under _get_builder_lock().
_builder_busy: set[str] = set()
# Strong references to fire-and-forget tasks (the loop keeps only weak ones).
_background_tasks: set[asyncio.Future[Any]] = set()
# Lock protecting _builder_sessions. Initialized lazily (not at module import
# time) to avoid creating an asyncio.Lock before an event loop exists (Python
# deprecated that pattern in 3.10; the floor is 3.11).
_builder_sessions_lock: asyncio.Lock | None = None


def _get_builder_lock() -> asyncio.Lock:
    """Return the asyncio.Lock for _builder_sessions, creating it lazily."""
    global _builder_sessions_lock
    if _builder_sessions_lock is None:
        _builder_sessions_lock = asyncio.Lock()
    return _builder_sessions_lock


def _get_builder_internal_state(agent: Any) -> dict[str, Any]:
    """Safely extract internal state from a builder agent using public API."""
    try:
        result: dict[str, Any] = agent.get_internal_state()
        return result
    except Exception as e:
        logger.debug(f"Builder internal state unavailable: {e}")
        return {"phase": "unknown", "turn_count": 0}


async def _cleanup_stale_builder_sessions() -> None:
    """Remove builder sessions unused for longer than the TTL (never one with
    a send in flight)."""
    now = time.time()
    async with _get_builder_lock():
        stale = [
            sid
            for sid, (_, ts) in _builder_sessions.items()
            if now - ts > _BUILDER_SESSION_TTL and sid not in _builder_busy
        ]
        for sid in stale:
            _builder_sessions.pop(sid, None)
            logger.debug(f"Cleaned up stale builder session: {sid}")


@app.post("/api/builder/start", dependencies=_GATED)
async def api_builder_start(req: BuilderStartRequest) -> dict[str, Any]:
    """Start a new builder session using the meta-agent."""
    await _cleanup_stale_builder_sessions()
    async with _get_builder_lock():
        if len(_builder_sessions) >= MAX_BUILDER_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"too many builder sessions (max {MAX_BUILDER_SESSIONS})",
            )
    try:
        from fsm_llm.agents.meta_builder import (
            MetaBuilderAgent as MetaAgent,
        )
        from fsm_llm.agents.meta_builder import (
            MetaBuilderConfig as MetaAgentConfig,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="fsm_llm.agents meta-builder not available",
        ) from None

    try:
        model = resolve_model(req.model)  # unset: ModelNotConfiguredError -> 400
    except ModelNotConfiguredError as e:
        _raise_http(e, "start builder session")

    config_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    config = MetaAgentConfig(**config_kwargs)
    agent = MetaAgent(config=config)

    try:
        initial_message = req.artifact_type or ""
        response = await asyncio.wait_for(
            asyncio.to_thread(agent.start, initial_message),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Builder start failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    session_id = f"builder-{uuid.uuid4().hex[:8]}"
    async with _get_builder_lock():
        _builder_sessions[session_id] = (agent, time.time())

    result: dict[str, Any] = {
        "session_id": session_id,
        "response": response,
        "is_complete": agent.is_complete(),
    }

    result["internal_state"] = _get_builder_internal_state(agent)

    return result


@app.post("/api/builder/send", dependencies=_GATED)
async def api_builder_send(req: BuilderSendRequest) -> dict[str, Any]:
    """Send a message to an existing builder session.

    Concurrent sends to the same session are rejected (409) — the underlying
    MetaBuilderAgent is not re-entrant and a second in-flight send would corrupt
    its state.
    """
    async with _get_builder_lock():
        entry = _builder_sessions.get(req.session_id)
        busy = entry is not None and req.session_id in _builder_busy
        if entry is not None and not busy:
            _builder_busy.add(req.session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Builder session not found")
    if busy:
        raise HTTPException(
            status_code=409, detail="Builder session is processing another message"
        )
    agent = entry[0]

    # The worker thread cannot be stopped on timeout; the session stays busy
    # until it really finishes, so a second send can never run concurrently
    # on the non-re-entrant agent.
    work = asyncio.ensure_future(asyncio.to_thread(agent.send, req.message))

    async def _release() -> None:
        async with _get_builder_lock():
            _builder_busy.discard(req.session_id)
            if req.session_id in _builder_sessions:
                _builder_sessions[req.session_id] = (agent, time.time())

    def _release_when_done(_fut: asyncio.Future[Any]) -> None:
        task = asyncio.ensure_future(_release())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    try:
        response = await asyncio.wait_for(
            asyncio.shield(work), timeout=_BUILDER_OPERATION_TIMEOUT
        )
    except TimeoutError:
        work.add_done_callback(_release_when_done)
        raise HTTPException(
            status_code=504, detail="Builder operation timed out"
        ) from None
    except Exception as e:
        await _release()
        logger.error(f"Builder send failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    except BaseException:
        # Cancelled: the worker thread keeps running, so the session is
        # released when it ends (as for a timeout), never left busy for good.
        work.add_done_callback(_release_when_done)
        raise

    try:
        result: dict[str, Any] = {
            "response": response,
            "is_complete": agent.is_complete(),
        }

        result["internal_state"] = _get_builder_internal_state(agent)

        if agent.is_complete():
            result.update(_builder_result_fields(agent))
            # Clean up completed session
            async with _get_builder_lock():
                _builder_sessions.pop(req.session_id, None)

        return result
    except Exception as e:
        logger.error(f"Builder send failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        await _release()


def _builder_result_fields(agent: Any) -> dict[str, Any]:
    """The built artifact of a completed builder agent (no exception text)."""
    try:
        build_result = agent.get_result()
        return {
            "artifact": build_result.artifact,
            "artifact_json": build_result.artifact_json,
            "artifact_type": build_result.artifact_type.value,
            "is_valid": build_result.is_valid,
            "validation_errors": build_result.validation_errors,
        }
    except Exception as e:
        logger.error(f"Builder result extraction failed: {e}")
        return {
            "artifact": {},
            "artifact_json": "{}",
            "artifact_type": "unknown",
            "is_valid": False,
            "validation_errors": ["Result extraction failed"],
            "error": "Result extraction failed",
        }


@app.get("/api/builder/result/{session_id}", dependencies=_GATED)
async def api_builder_result(session_id: str) -> dict[str, Any]:
    """Get the current state of a builder session.

    While a send is in flight the agent is being mutated by a worker thread,
    so only ``{"busy": true}`` is reported.
    """
    async with _get_builder_lock():
        entry = _builder_sessions.get(session_id)
        busy = session_id in _builder_busy
    if entry is None:
        raise HTTPException(status_code=404, detail="Builder session not found")
    if busy:
        return {"session_id": session_id, "busy": True}
    agent = entry[0]

    result: dict[str, Any] = {
        "session_id": session_id,
        "busy": False,
        "is_complete": agent.is_complete(),
    }

    # Always include internal state for live monitoring
    result["internal_state"] = _get_builder_internal_state(agent)

    if agent.is_complete():
        result.update(_builder_result_fields(agent))

    return result


@app.delete("/api/builder/{session_id}", dependencies=_GATED)
async def api_builder_delete(session_id: str) -> dict[str, Any]:
    """Delete a builder session (refused with 409 while a send is in flight)."""
    async with _get_builder_lock():
        if session_id in _builder_busy:
            raise HTTPException(
                status_code=409, detail="Builder session is processing a message"
            )
        removed = _builder_sessions.pop(session_id, None)
    return {"deleted": removed is not None}
