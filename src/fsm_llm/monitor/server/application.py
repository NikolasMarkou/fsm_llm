"""The FastAPI application of the monitor and the state it shares.

This module owns the ``app`` object, every piece of process-wide state that
``configure()`` rebinds (``_manager``, ``_api_key``, ``_flows``,
``_CORS_ORIGIN_REGEX``, ``_requests_processed``) and the functions that read
them (the Host and Origin checks, the middleware, the API-key dependency, the
WebSocket authorisation). A name that ``configure()`` rebinds is read only
here: a ``from .application import _api_key`` elsewhere would bind a stale copy.

The three routes that read that state or the template directory (``/``,
``/health``, ``/api/auth``) are registered here, first, as they were in the old
single module. The other routes live in the ``routes_*`` modules and
``websocket.py``; the package ``__init__`` imports them in the old order, which
is the order FastAPI matches them in.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from fsm_llm.logging import logger
from fsm_llm.security import api_key_matches, checked_api_key, presented_api_key

from ..__version__ import __version__
from ..constants import DEFAULT_TRUSTED_HOSTS, MAX_REQUEST_BODY_BYTES
from ..instance_manager import InstanceManager, read_json_file

STATIC_DIR = Path(__file__).parent.parent / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

# Counter tracking how many HTTP requests have been processed since server start.
# Used by configure() to warn when CORS mutation would be a no-op (Starlette
# builds its middleware stack lazily on the first request).
_requests_processed: int = 0


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage server startup/shutdown lifecycle."""
    yield
    # Shutdown: remove loguru sink to prevent resource leaks
    if _manager is not None:
        _manager.global_collector.cleanup()


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


# CORS origins -- the dashboard is served by this same server, so it needs no
# CORS; these only allow OTHER origins to read the API. Defaults to the
# default port on localhost. Override via configure(cors_origins=[...]) or the
# FSM_LLM_MONITOR_CORS_ORIGINS env var (comma-separated).
_CORS_ORIGINS: list[str] = _env_list("FSM_LLM_MONITOR_CORS_ORIGINS") or [
    "http://localhost:8420",
    "http://127.0.0.1:8420",
]
# DECISION plan-2026-09-28T090000-3c9e41d2/D-001
# No default origin regex: `https?://(localhost|127.0.0.1)(:\d+)?` trusted
# every local dev server on any port. Same-origin requests need no entry.
_CORS_ORIGIN_REGEX: str | None = None

# Host header allow-list (DNS-rebinding defence). "*" disables the check.
# "testserver" is the Host Starlette's TestClient sends.
_TRUSTED_HOSTS: list[str] = _env_list("FSM_LLM_MONITOR_TRUSTED_HOSTS") or [
    *DEFAULT_TRUSTED_HOSTS,
    "testserver",
]

app = FastAPI(
    title="FSM-LLM Monitor",
    version=__version__,
    docs_url="/api/docs",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self' ws: wss:; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _refusal(status_code: int, detail: str) -> JSONResponse:
    """A guard refusal: it never reaches ``call_next``, so it gets the security
    headers here."""
    return JSONResponse(
        {"detail": detail}, status_code=status_code, headers=_SECURITY_HEADERS
    )


def _host_only(host_header: str) -> str:
    """The host part of a Host header value (port removed, IPv6 unbracketed)."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host[1:]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _host_allowed(host_header: str | None) -> bool:
    if "*" in _TRUSTED_HOSTS:
        return True
    if not host_header:
        return False
    host = _host_only(host_header)
    return any(host == _host_only(h) for h in _TRUSTED_HOSTS)


def _origin_allowed(origin: str | None, host_header: str | None) -> bool:
    """A browser request may act on the monitor only from the monitor's own
    origin or a configured CORS origin. No Origin header (curl, scripts,
    same-origin GETs in some browsers) is allowed; auth still applies."""
    if not origin:
        return True
    if origin == "null":
        return False
    if "*" in _CORS_ORIGINS or origin in _CORS_ORIGINS:
        return True
    if _CORS_ORIGIN_REGEX and re.fullmatch(_CORS_ORIGIN_REGEX, origin):
        return True
    if not host_header:
        return False
    return urlsplit(origin).netloc.lower() == host_header.strip().lower()


@app.middleware("http")
async def _security_guard(request: Request, call_next):
    """Host allow-list, same-origin check for state-changing requests, body
    size cap, and security headers on every response.

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-001
    # A cross-site page could POST to the monitor (a no-cors fetch with no
    # Content-Type is parsed as JSON) and a rebinding DNS name could read it.
    # Do NOT drop the Origin check on unsafe methods or the Host check: CORS
    # only governs READING responses, not sending requests.
    """
    host = request.headers.get("host")
    if not _host_allowed(host):
        return _refusal(400, "invalid host header")
    if request.method in _UNSAFE_METHODS and not _origin_allowed(
        request.headers.get("origin"), host
    ):
        return _refusal(403, "cross-origin request refused")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            too_big = int(length) > MAX_REQUEST_BODY_BYTES
        except ValueError:
            too_big = True
        if too_big:
            return _refusal(413, "request body too large")
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


@app.middleware("http")
async def _count_requests(request: Request, call_next):
    """Increment _requests_processed on each HTTP request so configure() can
    warn when CORS mutation is called too late (after the first request)."""
    global _requests_processed
    _requests_processed += 1
    return await call_next(request)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Prevent browser caching of static JS/CSS files during development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


class _BodyCapMiddleware:
    """Pure ASGI: refuse a request body past ``MAX_REQUEST_BODY_BYTES`` that
    declares no ``Content-Length`` (a chunked body), before any route or the
    API-key dependency reads it.

    Contract: an http scope with a ``content-length`` header, and any other
    scope type, passes through (``_security_guard`` judges the declared
    length). Otherwise the body messages are buffered up to the cap plus one
    chunk; past the cap the 413 refusal (with the security headers) is sent
    and the rest is never read; within it, the buffered messages are replayed
    to the app and the live ``receive`` takes over. Never raises for a body.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or any(
            name == b"content-length" for name, _ in scope["headers"]
        ):
            await self.app(scope, receive, send)
            return
        buffered: list[Any] = []
        size = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break  # a disconnect: the app sees it replayed
            size += len(message.get("body", b""))
            if size > MAX_REQUEST_BODY_BYTES:
                await _refusal(413, "request body too large")(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Any:
            return buffered.pop(0) if buffered else await receive()

        await self.app(scope, replay, send)


# Added last, so it is the OUTERMOST layer: the body is judged before the Host
# check, the API-key dependency or any route reads a byte of it.
app.add_middleware(_BodyCapMiddleware)

# Global instance manager — set via configure()
_manager: InstanceManager | None = None

# DECISION plan-2026-09-12T065608-089d0ec7/D-008: optional API-key gate for
# mutating monitor routes (F2). `_api_key` is re-derived on EVERY configure()
# call (api_key param, else FSM_LLM_MONITOR_API_KEY env var, else None) —
# unlike `_CORS_ORIGINS` (only mutated when `cors_origins is not None`, so a
# repeat configure() call intentionally leaves a prior CORS override in
# place). Do NOT copy the CORS "only overwrite if not None" pattern here: it
# would let a previously-set `_api_key` leak across independent configure()
# calls (e.g. between tests), which is exactly the isolation `_api_key` needs.
# See `_require_api_key` below for the per-request timing note (D-008).
# Initialised from the env var at import so `uvicorn fsm_llm.monitor.server:app`
# (which never calls configure()) still honours FSM_LLM_MONITOR_API_KEY.
_api_key: str | None = os.environ.get("FSM_LLM_MONITOR_API_KEY") or None

# Flow data loaded from static/flows.json
_flows: dict[str, Any] = {}


def _load_flows() -> dict[str, Any]:
    """Load agent/workflow flow definitions from the JSON data file."""
    flows_path = STATIC_DIR / "flows.json"
    if flows_path.exists():
        result: dict[str, Any] = read_json_file(flows_path)
        return result
    return {"agents": {}, "workflows": {}}


def configure(
    *,
    manager: InstanceManager | None = None,
    cors_origins: list[str] | None = None,
    api_key: str | None = None,
    trusted_hosts: list[str] | None = None,
) -> None:
    """Configure the global instance manager for the web server.

    All arguments are keyword-only.

    :param manager: The ``InstanceManager`` the server reads. Defaults to a new
        one. To show conversations of an ``API`` you created yourself, call
        ``manager.attach_api(api)`` and pass that manager here.
    :param cors_origins: List of allowed CORS origins. Defaults to localhost only.
        Pass ``["*"]`` to allow all origins (not recommended for production).
    :param api_key: Optional API key required (via an ``Authorization: Bearer
        <key>`` or ``X-API-Key`` header) to reach mutating REST routes. Falls
        back to the ``FSM_LLM_MONITOR_API_KEY`` env var when not given
        explicitly. Defaults to ``None`` (no auth — byte-identical to prior
        behavior). Unlike ``cors_origins``, this is re-derived on EVERY call
        (see the D-008 comment on the ``_api_key`` global) — **calling
        ``configure()`` again later (e.g. only to change ``cors_origins``)
        without also re-passing ``api_key`` silently clears any previously
        configured key** (falling through to the env var, or to ``None`` if
        the env var is also unset). If you configure a key once, you must
        re-pass it (or rely on the env var) on every subsequent ``configure()``
        call in the same process, or auth will silently disable itself. A
        WARNING is logged (D-016) when this specific case — a previously-set
        key being cleared by a call that did not itself supply ``api_key`` —
        is detected. An empty or whitespace-only ``api_key`` raises
        ``ValueError``; an empty env var still means no key.
    :param trusted_hosts: Host header values the server answers to (DNS
        rebinding defence). Defaults to localhost; ``["*"]`` disables the
        check (needed when serving on a LAN address). Like ``cors_origins``,
        only changed when given.
    """
    global _manager, _flows, _CORS_ORIGINS, _CORS_ORIGIN_REGEX, _api_key
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-006: raise on an empty or
    # whitespace-only api_key, before any global changes (`checked_api_key`).
    # Do NOT store it and do NOT map it to None (a caller passing
    # os.getenv("KEY", "") would silently get an open monitor). The env var ""
    # still means unset.
    checked_api_key(api_key)
    new_api_key = (
        api_key
        if api_key is not None
        else (os.environ.get("FSM_LLM_MONITOR_API_KEY") or None)
    )
    # DECISION plan-2026-09-12T065608-089d0ec7/D-016: warn (rather than stay
    # silent) when a previously-configured key is about to be cleared by a
    # re-configure() call that did not itself pass api_key= — mirroring the
    # existing CORS-mutation warning below. Guarded so it never fires on the
    # first configure() call (nothing to clear yet) or when the caller
    # explicitly re-supplies the same/a new key.
    if api_key is None and _api_key is not None and new_api_key is None:
        logger.warning(
            "configure() called without api_key= and FSM_LLM_MONITOR_API_KEY "
            "is unset; a previously configured API key is being cleared — "
            "mutating routes will no longer require authentication."
        )
    _api_key = new_api_key
    if _requests_processed > 0:
        logger.warning(
            f"configure() called after server has processed {_requests_processed} "
            "request(s); CORS changes will not take effect until the next server "
            "restart."
        )
    if trusted_hosts is not None:
        _TRUSTED_HOSTS[:] = trusted_hosts
    if cors_origins is not None:
        _CORS_ORIGINS[:] = cors_origins
        _CORS_ORIGIN_REGEX = None  # Disable regex when explicit origins are provided
        # NOTE: This mutates FastAPI's internal middleware kwargs directly.
        # This only works when called BEFORE the first request, because Starlette
        # builds its middleware stack lazily on first request. After that, these
        # kwargs are no longer consulted. Do not call configure() after startup.
        for middleware in app.user_middleware:
            if hasattr(middleware, "kwargs"):
                if "allow_origins" in middleware.kwargs:
                    middleware.kwargs["allow_origins"] = cors_origins
                if "allow_origin_regex" in middleware.kwargs:
                    middleware.kwargs["allow_origin_regex"] = None
    _flows = _load_flows()

    # Detach the old manager (log sink, handlers on its APIs) -- but never the
    # manager being re-installed: configure(manager=current, api_key=...) is
    # how a caller rotates the key, and cleaning it up killed its log feed.
    if _manager is not None and _manager is not manager:
        _manager.close()

    if manager is not None:
        _manager = manager
    else:
        _manager = InstanceManager()


def get_manager() -> InstanceManager:
    global _manager
    if _manager is None:
        _manager = InstanceManager()
    return _manager


def get_flows() -> dict[str, Any]:
    """The agent and workflow pattern flows, loaded on first use (an embedder
    running ``uvicorn ...server:app`` never calls ``configure()``)."""
    global _flows
    if not _flows:
        _flows = _load_flows()
    return _flows


def _require_api_key(request: Request) -> None:
    """FastAPI dependency gating mutating routes behind an optional API key.

    DECISION plan-2026-09-12T065608-089d0ec7/D-008: no-op when `_api_key` is
    `None` (the default) — byte-identical to pre-F2 behavior for every
    existing caller who never sets `FSM_LLM_MONITOR_API_KEY`/`api_key`. When
    `_api_key` is set, requires a matching `Authorization: Bearer <key>` OR
    `X-API-Key` header, else raises 401.

    Timing note (do NOT assume this mirrors the CORS gotcha in `configure()`
    above): `configure()`'s CORS mutation only takes effect if called BEFORE
    the first request, because Starlette builds its middleware stack lazily
    on first request. `_require_api_key` has NO such restriction — FastAPI
    resolves `Depends(_require_api_key)` fresh on every request, reading the
    module-level `_api_key` global at THAT time, so `configure(api_key=...)`
    can be (and in tests, is) called again after the server has already
    processed requests, and the new key takes effect immediately.
    """
    if _api_key is None:
        return
    # The compare itself (hmac.compare_digest on surrogateescape bytes, D-016,
    # D-020) is `security.api_key_matches`, shared with `AgentServer`.
    if not api_key_matches(presented_api_key(request.headers), _api_key):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def _api_key_matches(token: str | None) -> bool:
    return _api_key is None or api_key_matches(token, _api_key)


# --- WebSocket authorisation (used by websocket.py) ---

_WS_AUTH_TIMEOUT = 5.0
_WS_CLOSE_UNAUTHORIZED = 4401
_WS_CLOSE_FORBIDDEN = 4403


async def _ws_authorized(websocket: WebSocket) -> bool:
    """Origin and API-key checks for ``/ws`` (browsers apply no CORS to
    WebSockets, so the Origin must be checked here).

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-001
    # With a key configured the client must send ``{"type": "auth",
    # "api_key": "..."}`` as its first message within 5 s. Do NOT move the
    # key into the URL (it would land in access logs and browser history).
    """
    host = websocket.headers.get("host")
    if not _host_allowed(host) or not _origin_allowed(
        websocket.headers.get("origin"), host
    ):
        await websocket.close(code=_WS_CLOSE_FORBIDDEN)
        return False
    if _api_key is None:
        return True
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), _WS_AUTH_TIMEOUT)
        message = json.loads(raw)
        token = message.get("api_key") if isinstance(message, dict) else None
    except Exception:
        token = None
    if not isinstance(token, str) or not _api_key_matches(token):
        await websocket.close(code=_WS_CLOSE_UNAUTHORIZED)
        return False
    return True


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/auth")
async def api_auth() -> dict[str, bool]:
    """Whether an API key is required (never gated, never reveals the key)."""
    return {"auth_required": _api_key is not None}
