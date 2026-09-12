from __future__ import annotations

"""
A2A (Agent-to-Agent) Protocol for Remote Agents.

AgentServer wraps any FSM-LLM agent as an HTTP endpoint.
RemoteAgentTool wraps a remote agent URL as a local tool.
"""

import json
from typing import Any, cast

from .definitions import ToolDefinition

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


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

    Exposes ``/invoke`` for full results and ``/stream`` for token streaming.

    Example::

        from fsm_llm_agents.remote import AgentServer

        server = AgentServer(agent=my_react_agent, host="0.0.0.0", port=8500)
        server.run()  # Starts uvicorn
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8500,
        name: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        _require_fastapi()
        self._agent = agent
        self._host = host
        self._port = port
        self._name = name or getattr(agent, "__class__", type(agent)).__name__
        self._timeout = timeout
        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with /invoke and /stream endpoints."""
        app = FastAPI(title=f"FSM-LLM Agent: {self._name}")

        class InvokeRequest(PydanticBaseModel):
            task: str
            context: dict[str, Any] | None = None

        class InvokeResponse(PydanticBaseModel):
            answer: str
            success: bool
            iterations: int = 0
            tools_used: list[str] = []

        @app.post("/invoke", response_model=InvokeResponse)
        async def invoke(request: InvokeRequest):
            """Invoke the agent with a task and return the full result."""
            import asyncio

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
                # SCOPE: `AgentServer` wraps `agent: Any` -- the isolation
                # above holds ONLY when `self._agent` is a `ReactAgent`/
                # `ParallelReactAgent` (the two classes D-014 actually fixed).
                # `ReflexionAgent`, `PlanExecuteAgent`, and
                # `ReasoningReactAgent` still keep a single `self._handlers`
                # reused across calls (`.reset()`, not rebuilt per call) --
                # if `AgentServer` is ever pointed at one of those, this
                # comment's isolation claim does NOT hold for it, and the
                # original F9-shaped cross-request corruption (see
                # decisions.md D-013/D-014) remains live. Named, deliberately
                # out-of-scope residual; not silently fixed by D-014.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._agent.run,
                        request.task,
                        initial_context=request.context,
                    ),
                    timeout=self._timeout,
                )
                return InvokeResponse(
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

        @app.post("/stream")
        async def stream(request: InvokeRequest):
            """Run the agent and emit a single deferred terminal SSE event.

            This is NOT incremental/token streaming: the agent runs to
            completion first, then one ``data:`` event with the final result
            is sent. The SSE framing exists for client compatibility only.
            """
            import asyncio

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
    and returns the result as a string.

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
    ) -> None:
        _require_httpx()
        self._url = url.rstrip("/")
        self._name = name
        self._description = description
        self._timeout = timeout

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
            response = client.post(f"{self._url}/invoke", json=payload)
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
            response = await client.post(f"{self._url}/invoke", json=payload)
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
