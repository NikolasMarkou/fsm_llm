"""
A2A (Agent-to-Agent) Protocol for Remote Agents.

AgentServer wraps any FSM-LLM agent as an HTTP endpoint.
RemoteAgentTool wraps a remote agent URL as a local tool.
"""

from __future__ import annotations

import hmac
import json
from typing import Any, cast

from .definitions import ToolDefinition

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

#: Default ``AgentServer`` input bound: ``len(task) + len(json.dumps(context))``.
DEFAULT_MAX_INPUT_CHARS = 100_000


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
        iterations: int = 0
        tools_used: list[str] = []


def _require_httpx() -> None:
    if not _HAS_HTTPX:
        raise ImportError(
            "Remote agent client requires 'httpx'. "
            "Install with: pip install fsm-llm[a2a] or pip install httpx"
        )


def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ImportError(
            "AgentServer requires 'fastapi'. "
            "Install with: pip install fsm-llm[monitor] or pip install fastapi"
        )


class AgentServer:
    """Wraps an FSM-LLM agent as an HTTP endpoint.

    Exposes ``/invoke`` for the full result and ``/stream`` for a single
    deferred SSE event, plus open ``/health`` and ``/info`` routes.

    Security (opt-in):
        ``api_key``: when set, ``/invoke`` and ``/stream`` require
        ``Authorization: Bearer <key>`` or ``X-API-Key: <key>`` and return 401
        otherwise. ``None`` (the default) leaves the server UNAUTHENTICATED;
        bind it to localhost or put it behind an authenticating proxy.
        ``max_input_chars``: requests whose ``len(task) +
        len(json.dumps(context or {}))`` exceeds it get 413 before the agent
        runs. Default 100,000; ``None`` disables the check. The HTTP body is
        still parsed first, so a transport-level body cap belongs to a
        reverse proxy. There is no rate limiting.

    Example::

        from fsm_llm_agents.remote import AgentServer

        server = AgentServer(agent=my_react_agent, api_key="change-me")
        server.run()  # Starts uvicorn
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8500,
        name: str | None = None,
        timeout: float = 300.0,
        api_key: str | None = None,
        max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        _require_fastapi()
        self._agent = agent
        self._host = host
        self._port = port
        self._name = name or getattr(agent, "__class__", type(agent)).__name__
        self._timeout = timeout
        self._api_key = api_key
        self._max_input_chars = max_input_chars
        self._app = self._create_app()

    def _require_api_key(self, request: Request) -> None:
        """FastAPI dependency: 401 unless the request carries ``self._api_key``.

        No-op when ``api_key`` is ``None``. Guards ``/invoke`` and ``/stream``.
        """
        if self._api_key is None:
            return
        auth_header = request.headers.get("authorization", "")
        token: str | None = None
        if auth_header.lower().startswith("bearer "):
            token = auth_header[len("bearer ") :].strip()
        if token is None:
            token = request.headers.get("x-api-key")
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-006: a byte-for-byte copy
        # of the monitor's `_require_api_key` compare (fsm_llm_monitor/
        # server.py, its D-016/D-020 anchors). Do NOT compare with `!=` (timing
        # side channel) and do NOT pass `str` to `compare_digest`: its str
        # overload raises TypeError on non-ASCII input, turning an
        # attacker-sent header byte into a 500. Encode both sides with
        # `surrogateescape` first. Not imported from the monitor so agents
        # carries no monitor dependency; review both copies together.
        if token is None or not hmac.compare_digest(
            token.encode("utf-8", "surrogateescape"),
            self._api_key.encode("utf-8", "surrogateescape"),
        ):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    def _check_input_size(self, request: _InvokeRequest) -> None:
        """Raise 413 when the request exceeds ``max_input_chars``.

        Called first in BOTH ``/invoke`` and ``/stream`` so the agent (and its
        LLM calls) never starts on an oversize input.
        """
        if self._max_input_chars is None:
            return
        size = len(request.task) + len(json.dumps(request.context or {}))
        if size > self._max_input_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"input is {size} characters; the limit is {self._max_input_chars}"
                ),
            )

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with /invoke and /stream endpoints."""
        app = FastAPI(title=f"FSM-LLM Agent: {self._name}")
        guarded = [Depends(self._require_api_key)]

        @app.post("/invoke", response_model=_InvokeResponse, dependencies=guarded)
        async def invoke(request: _InvokeRequest):
            """Invoke the agent with a task and return the full result."""
            import asyncio

            self._check_input_size(request)
            try:
                # F10 (plan-2026-09-12T065608-089d0ec7/D-014, supersedes the
                # D-004 note previously here): asyncio.wait_for only detaches
                # from this awaited future on timeout — the asyncio.to_thread
                # executor thread keeps running self._agent.run to completion
                # in the background, since Python threads cannot be forcibly
                # cancelled. This is an accepted trade-off, not a bug — but
                # ONLY as of D-014's real per-call-local AgentHandlers fix
                # (react.py/parallel_react.py). D-004's earlier fix (reusing
                # a reassigned `self._handlers` attribute) did NOT actually
                # achieve isolation between calls (see decisions.md D-013/
                # D-014), so this comment's claim was false until D-014
                # landed. Now that AgentHandlers is a true call-local object
                # never round-tripped through `self`, an orphaned thread
                # cannot corrupt a DIFFERENT request's counters — it just
                # wastes CPU/LLM-call budget harmlessly until it finishes on
                # its own, and its result is discarded. Cooperative
                # cancellation (a deadline check inside the ReAct loop) would
                # close this fully but is a materially larger change, out of
                # scope for this plan.
                #
                # SCOPE: `AgentServer` wraps `agent: Any`. The isolation
                # above originally held ONLY for `ReactAgent`/
                # `ParallelReactAgent` (the two classes D-014 fixed here in
                # this package). `ReflexionAgent`, `PlanExecuteAgent`, and
                # `ReasoningReactAgent` used to keep a single `self._handlers`
                # reused across calls (`.reset()`, not rebuilt per call) --
                # that residual has since been CLOSED: plan
                # plan-2026-09-12T135914-45a654de's D-012 applied the same
                # call-local `AgentHandlers` pattern (threaded per-call,
                # never round-tripped through `self`) to all three of those
                # classes. `AgentServer` may now be pointed at any of the 5
                # agent classes above without reopening the F9-shaped
                # cross-request corruption -- see decisions.md D-012/D-014.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._agent.run,
                        request.task,
                        initial_context=request.context,
                    ),
                    timeout=self._timeout,
                )
                return _InvokeResponse(
                    answer=result.answer,
                    success=result.success,
                    iterations=result.trace.total_iterations,
                    tools_used=result.trace.tools_used,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Agent execution timed out ({self._timeout}s)",
                ) from None
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post("/stream", dependencies=guarded)
        async def stream(request: _InvokeRequest):
            """Run the agent and emit a single deferred terminal SSE event.

            This is NOT incremental/token streaming: the agent runs to
            completion first, then one ``data:`` event with the final result
            is sent. The SSE framing exists for client compatibility only.
            """
            import asyncio

            # Before the StreamingResponse exists: a 413 raised inside the
            # generator would arrive after a 200 status line.
            self._check_input_size(request)

            async def event_generator():
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._agent.run,
                            request.task,
                            initial_context=request.context,
                        ),
                        timeout=self._timeout,
                    )
                    # Send the single terminal result as one SSE event
                    data = json.dumps(
                        {
                            "answer": result.answer,
                            "success": result.success,
                            "iterations": result.trace.total_iterations,
                        }
                    )
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': f'Agent execution timed out ({self._timeout}s)'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
    ``Authorization: Bearer <key>`` on every invoke.

    Example::

        from fsm_llm_agents.remote import RemoteAgentTool

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
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        _require_httpx()
        self._url = url.rstrip("/")
        self._name = name
        self._description = description
        self._timeout = timeout
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        """Auth headers shared by ``invoke`` and ``ainvoke``."""
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def invoke(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Invoke the remote agent synchronously.

        Args:
            task: The task to send to the remote agent.
            context: Optional context dict.

        Returns:
            The agent's answer as a string.
        """
        _require_httpx()
        payload: dict[str, Any] = {"task": task}
        if context:
            payload["context"] = context

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._url}/invoke", json=payload, headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()
            return cast(str, data.get("answer", str(data)))

    async def ainvoke(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Invoke the remote agent asynchronously."""
        _require_httpx()
        payload: dict[str, Any] = {"task": task}
        if context:
            payload["context"] = context

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._url}/invoke", json=payload, headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()
            return cast(str, data.get("answer", str(data)))

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
