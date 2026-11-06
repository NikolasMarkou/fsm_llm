"""
A2A (Agent-to-Agent) Protocol for Remote Agents.

AgentServer wraps any FSM-LLM agent as an HTTP endpoint.
RemoteAgentTool wraps a remote agent URL as a local tool.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fsm_llm.logging import logger
from fsm_llm.security import api_key_matches, checked_api_key, presented_api_key

from .base import strip_caller_context
from .constants import Defaults
from .definitions import ToolDefinition
from .exceptions import ToolExecutionError

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

#: Default ``AgentServer`` input bound, in characters: ``len(task) +
#: len(json.dumps(context, ensure_ascii=False))``.
DEFAULT_MAX_INPUT_CHARS = 100_000

#: Client-facing text for an agent run that raised. The exception itself is
#: logged server-side under the error id sent alongside it.
_AGENT_FAILED_MESSAGE = "Agent execution failed"


if _HAS_FASTAPI:
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-006: keep these request/
    # response models at MODULE level. Do NOT move them back inside
    # `_create_app`: with `from __future__ import annotations` every handler
    # annotation is a string, and FastAPI resolves it against the handler's
    # module globals only. A function-local `InvokeRequest` does not resolve,
    # so FastAPI silently treated `request` as a required QUERY parameter and
    # every `/invoke` and `/stream` call with a valid JSON body got 422.
    class _InvokeRequest(PydanticBaseModel):
        task: str
        context: dict[str, Any] | None = None

    class _InvokeResponse(PydanticBaseModel):
        answer: str
        success: bool
        stop_reason: str | None = None
        iterations: int = 0
        tools_used: list[str] = []


def _require_httpx() -> None:
    if not _HAS_HTTPX:
        raise ImportError(
            "Remote agent client requires 'httpx'. "
            'From a clone: pip install -e ".[a2a]" or pip install httpx'
        )


def _server_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Remote request context as the agent may see it.

    ``AgentServer`` is a trust boundary: besides the run-owned keys every
    agent drops, a remote client may not set ANY internal-prefix key (driver
    grants, policy inputs, pattern counters). Dropped keys are logged, not
    rejected (D-002 of plan 06a5ec0a).
    """
    return strip_caller_context(
        context, source="AgentServer request context", drop_internal=True
    )


def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ImportError(
            "AgentServer requires 'fastapi'. "
            'From a clone: pip install -e ".[monitor]" or pip install fastapi'
        )


class AgentServer:
    """Wraps an FSM-LLM agent as an HTTP endpoint.

    Exposes ``/invoke`` for the full result and ``/stream`` for a single
    deferred SSE event, plus open ``/health`` and ``/info`` routes.

    Security (opt-in):
        ``api_key``: when set, ``/invoke`` and ``/stream`` require
        ``Authorization: Bearer <key>`` or ``X-API-Key: <key>`` and return 401
        otherwise. ``None`` (the default) leaves the server UNAUTHENTICATED;
        bind it to localhost or put it behind an authenticating proxy. An
        empty or whitespace-only key raises ``ValueError``.
        ``max_input_chars``: requests whose ``len(task) + len(json.dumps(
        context or {}, ensure_ascii=False))`` exceeds it get 413 before the agent
        runs. Default 100,000; ``None`` disables the check. The HTTP body is
        still parsed first, so a transport-level body cap belongs to a
        reverse proxy.
        ``max_concurrent``: agent runs allowed at once (default
        ``Defaults.SERVER_MAX_CONCURRENT``). A request arriving while all
        slots are taken gets 503. A slot is held until the agent's thread
        finishes, including a run that already timed out (504). There is no
        per-client rate limiting.

    A run that raises returns 500 (``/invoke``) or an SSE ``error`` event
    (``/stream``) with a generic message and an ``error_id``; the exception
    is logged with that id and never sent to the client.

    Example::

        from fsm_llm.agents.remote import AgentServer

        server = AgentServer(agent=my_react_agent, api_key="change-me")
        server.run()  # Starts uvicorn
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8500,
        name: str | None = None,
        timeout: float = Defaults.TIMEOUT_SECONDS,
        api_key: str | None = None,
        max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
        max_concurrent: int = Defaults.SERVER_MAX_CONCURRENT,
    ) -> None:
        _require_fastapi()
        if isinstance(max_concurrent, bool) or not (
            isinstance(max_concurrent, int) and max_concurrent >= 1
        ):
            raise ValueError(
                f"max_concurrent must be an int >= 1, got {max_concurrent!r}"
            )
        self._agent = agent
        self._host = host
        self._port = port
        self._name = name or getattr(agent, "__class__", type(agent)).__name__
        self._timeout = timeout
        self._api_key = checked_api_key(api_key)
        self._max_input_chars = max_input_chars
        self._max_concurrent = max_concurrent
        self._slots = asyncio.Semaphore(max_concurrent)
        self._app = self._create_app()

    def _require_api_key(self, request: Request) -> None:
        """FastAPI dependency: 401 unless the request carries ``self._api_key``.

        No-op when ``api_key`` is ``None``. Guards ``/invoke`` and ``/stream``.
        """
        if self._api_key is None:
            return
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-006
        # The compare is `security.api_key_matches` (hmac.compare_digest on
        # surrogateescape bytes), shared with the monitor server through core so
        # agents carry no monitor dependency. Do NOT compare with `!=` or hand
        # `str` to compare_digest here, and do NOT copy the body back.
        if not api_key_matches(presented_api_key(request.headers), self._api_key):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    def _check_input_size(self, request: _InvokeRequest) -> None:
        """Raise 413 when the request exceeds ``max_input_chars``.

        Called first in BOTH ``/invoke`` and ``/stream`` so the agent (and its
        LLM calls) never starts on an oversize input.
        """
        if self._max_input_chars is None:
            return
        context = json.dumps(request.context or {}, ensure_ascii=False)
        size = len(request.task) + len(context)
        if size > self._max_input_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"input is {size} characters; the limit is {self._max_input_chars}"
                ),
            )

    async def _start_run(self, request: _InvokeRequest) -> asyncio.Future[Any]:
        """Take a run slot and start the agent in a worker thread.

        Raises 503 when every slot is taken. The returned future is the run
        itself; await it through :meth:`_await_run`.
        """
        context = _server_context(request.context)
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-024: the slot is released
        # by the RUN FUTURE's done callback, not by `async with` around the
        # request. Do NOT scope the slot to the handler: on a 504 `wait_for`
        # returns while the worker thread keeps running (threads cannot be
        # cancelled), so a handler-scoped slot would let timed-out runs pile
        # up without bound. Do NOT wait for a slot either: saturation is 503.
        # The `locked()` check and `acquire()` are atomic on the event loop
        # (an unlocked semaphore's acquire never suspends).
        if self._slots.locked():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"server busy: {self._max_concurrent} agent run(s) "
                    "already in flight; retry later"
                ),
            )
        await self._slots.acquire()
        try:
            run = asyncio.ensure_future(
                asyncio.to_thread(
                    self._agent.run, request.task, initial_context=context
                )
            )
        except BaseException:
            self._slots.release()
            raise
        run.add_done_callback(self._release_slot)
        return run

    def _release_slot(self, run: asyncio.Future[Any]) -> None:
        self._slots.release()
        # Mark an orphaned (timed-out) run's exception as retrieved so asyncio
        # does not log "exception was never retrieved" for it.
        if not run.cancelled():
            run.exception()

    async def _await_run(self, run: asyncio.Future[Any]) -> Any:
        """Await *run* up to the server timeout; the run survives a timeout."""
        return await asyncio.wait_for(asyncio.shield(run), timeout=self._timeout)

    @staticmethod
    def _log_failure(route: str, exc: BaseException) -> str:
        """Log *exc* with a fresh error id and return that id."""
        error_id = uuid.uuid4().hex[:12]
        logger.opt(exception=exc).error(
            f"AgentServer {route} run failed (error_id={error_id}): {exc!r}"
        )
        return error_id

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with /invoke and /stream endpoints."""
        app = FastAPI(title=f"FSM-LLM Agent: {self._name}")
        guarded = [Depends(self._require_api_key)]

        @app.post("/invoke", response_model=_InvokeResponse, dependencies=guarded)
        async def invoke(request: _InvokeRequest):
            """Invoke the agent with a task and return the full result."""
            self._check_input_size(request)
            run = await self._start_run(request)
            try:
                # F10 (plan-2026-09-12T065608-089d0ec7/D-014): `wait_for` only
                # detaches from the run future on timeout; the worker thread
                # keeps running `self._agent.run` (threads cannot be
                # cancelled), wastes budget harmlessly and its result is
                # discarded. Accepted trade-off: every agent class builds a
                # call-local `AgentHandlers` per run (D-014, plan
                # 45a654de/D-012), so an orphaned run cannot corrupt another
                # request's counters. Cooperative cancellation is out of scope.
                result = await self._await_run(run)
                return _InvokeResponse(
                    answer=result.answer,
                    success=result.success,
                    stop_reason=getattr(result, "stop_reason", None),
                    iterations=result.trace.total_iterations,
                    tools_used=result.trace.tools_used,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Agent execution timed out ({self._timeout}s)",
                ) from None
            except Exception as e:
                error_id = self._log_failure("/invoke", e)
                raise HTTPException(
                    status_code=500,
                    detail=f"{_AGENT_FAILED_MESSAGE} (error_id={error_id})",
                ) from None

        @app.post("/stream", dependencies=guarded)
        async def stream(request: _InvokeRequest):
            """Run the agent and emit a single deferred terminal SSE event.

            This is NOT incremental/token streaming: the agent runs to
            completion first, then one ``data:`` event with the final result
            is sent. The SSE framing exists for client compatibility only.
            """
            # Before the StreamingResponse exists: a 413 or 503 raised inside
            # the generator would arrive after a 200 status line. The run
            # starts here too, so its slot is released by the run itself even
            # if the generator is never iterated.
            self._check_input_size(request)
            run = await self._start_run(request)

            async def event_generator():
                try:
                    result = await self._await_run(run)
                    # Send the single terminal result as one SSE event
                    data = json.dumps(
                        {
                            "answer": result.answer,
                            "success": result.success,
                            "stop_reason": getattr(result, "stop_reason", None),
                            "iterations": result.trace.total_iterations,
                        }
                    )
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': f'Agent execution timed out ({self._timeout}s)'})}\n\n"
                except Exception as e:
                    error_id = self._log_failure("/stream", e)
                    payload = {"error": _AGENT_FAILED_MESSAGE, "error_id": error_id}
                    yield f"data: {json.dumps(payload)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        @app.get("/health")
        async def health():
            return {"status": "ok", "agent": self._name}

        @app.get("/info")
        async def info():
            return {
                "name": self._name,
                "agent_type": type(self._agent).__name__,
            }

        return app

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI app for custom mounting or testing."""
        return self._app

    def run(self, **kwargs: Any) -> None:
        """Start the server with uvicorn."""
        import uvicorn

        uvicorn.run(
            self._app,
            host=self._host,
            port=self._port,
            **kwargs,
        )


class RemoteAgentTool:
    """Wraps a remote agent URL as a local ToolDefinition.

    The tool sends tasks to a remote AgentServer's /invoke endpoint
    and returns the result as a string. ``api_key``, when set, is sent as
    ``Authorization: Bearer <key>`` on every invoke; an empty or
    whitespace-only key raises ``ValueError``.

    Example::

        from fsm_llm.agents.remote import RemoteAgentTool

        tool = RemoteAgentTool(
            url="http://localhost:8500",
            name="billing_agent",
            description="Handle billing queries",
        )
        registry.register(tool.to_tool_definition())
    """

    def __init__(
        self,
        url: str,
        name: str,
        description: str,
        timeout: float = Defaults.REMOTE_TOOL_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> None:
        _require_httpx()
        self._url = url.rstrip("/")
        self._name = name
        self._description = description
        self._timeout = timeout
        self._api_key = checked_api_key(api_key)

    def _answer(self, data: Any) -> str:
        """The answer in an ``/invoke`` response body.

        Raises ``ToolExecutionError`` when the server reports
        ``success: false``; a body without ``success`` counts as successful.
        """
        if not isinstance(data, dict):
            return str(data)
        answer = str(data.get("answer", data))
        if data.get("success", True) is False:
            raise ToolExecutionError(
                f"Remote agent '{self._name}' did not succeed: {answer}",
                tool_name=self._name,
            )
        return answer

    def _post_args(self, task: str, context: dict[str, Any] | None) -> dict[str, Any]:
        """The ``post`` arguments shared by ``invoke`` and ``ainvoke``."""
        payload: dict[str, Any] = {"task": task}
        if context:
            payload["context"] = context
        headers = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return {"json": payload, "headers": headers}

    def invoke(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Invoke the remote agent synchronously.

        Args:
            task: The task to send to the remote agent.
            context: Optional context dict.

        Returns:
            The agent's answer as a string.

        Raises:
            ToolExecutionError: the server reported ``success: false``.
        """
        _require_httpx()
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._url}/invoke", **self._post_args(task, context)
            )
            response.raise_for_status()
            return self._answer(response.json())

    async def ainvoke(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Invoke the remote agent asynchronously (same contract as ``invoke``)."""
        _require_httpx()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._url}/invoke", **self._post_args(task, context)
            )
            response.raise_for_status()
            return self._answer(response.json())

    def to_tool_definition(self) -> ToolDefinition:
        """Create a ToolDefinition that calls this remote agent.

        The tool accepts a single ``task`` parameter.
        """

        def execute(task: str) -> str:
            return self.invoke(task)

        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameter_schema={
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to send to the remote agent",
                    },
                },
                "required": ["task"],
            },
            execute_fn=execute,
        )

    @property
    def url(self) -> str:
        """Return the remote agent URL."""
        return self._url

    def health_check(self) -> bool:
        """Check if the remote agent is healthy."""
        _require_httpx()
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._url}/health")
                return response.status_code == 200
        except Exception:
            return False
