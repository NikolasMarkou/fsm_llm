"""
FastAPI web server for fsm_llm.monitor.

Serves the Grafana-inspired dark dashboard UI and provides REST + WebSocket APIs
for real-time monitoring, launching, and controlling FSM conversations, agents,
and workflows.

Package layout (the old single module ``monitor/server.py``, same dotted path):

- ``application``: the ``app`` object, its middleware, ``configure``, the
  process-wide state ``configure`` rebinds and the code that reads it, the
  ``STATIC_DIR`` and ``TEMPLATE_DIR`` reads, and the routes ``/``, ``/health``
  and ``/api/auth``.
- ``support``: the error mapping, the bounded await, the JSON body reader and
  the API-key gate dependency every route module uses.
- ``routes_monitoring``, ``routes_fsm``, ``routes_workflow``, ``routes_agent``,
  ``routes_visualize``, ``routes_builder`` and ``websocket``: the other routes,
  each module registering on the one ``app``.

The route modules register on ``app`` when they are imported, and FastAPI matches
routes in registration order, so they are imported below in the order the old
module declared them. ``# isort: off`` keeps a formatter from sorting them.
"""

from __future__ import annotations

from .application import (
    STATIC_DIR,
    TEMPLATE_DIR,
    app,
    configure,
    get_flows,
    get_manager,
)

# isort: off
from . import routes_monitoring  # noqa: F401
from . import routes_fsm  # noqa: F401
from . import routes_workflow  # noqa: F401
from . import routes_agent  # noqa: F401
from . import routes_visualize  # noqa: F401
from . import routes_builder  # noqa: F401
from . import websocket  # noqa: F401

# isort: on

__all__ = [
    "STATIC_DIR",
    "TEMPLATE_DIR",
    "app",
    "configure",
    "get_flows",
    "get_manager",
]
