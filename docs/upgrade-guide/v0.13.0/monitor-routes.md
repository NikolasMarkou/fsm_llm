# Monitor route functions

Covers FSM-LLM 0.12.0 to 0.13.0. The monitor server (`fsm_llm.monitor.server`) is now a package of route modules. The REST and WebSocket interface did not change: paths, methods, fields and status codes are the same, and the route functions keep their bodies. What changed is where a Python caller finds the functions. This page matters only if your code, a test or a patch imports a route function directly. A client of the HTTP API needs nothing.

## The 49 route functions moved out of fsm_llm.monitor.server

The 49 route functions that 0.12.0 defined at module level in `fsm_llm.monitor.server` no longer import from it. `from fsm_llm.monitor.server import api_metrics` raised nothing in 0.12.0 and raises `ImportError` now. Import the function from the module that defines it (the table below), or call the REST route instead.

```text
fsm_llm.monitor.server.application        index, health, api_auth, no_cache_static
fsm_llm.monitor.server.routes_monitoring  api_metrics, api_conversations, api_conversation,
                                          api_activity, api_events, api_logs, api_config_get,
                                          api_config_set, api_dashboard_config_get,
                                          api_dashboard_config_set, api_dashboard_config_delete,
                                          api_info, api_capabilities, api_instances,
                                          api_instance_detail, api_instance_events,
                                          api_instance_destroy
fsm_llm.monitor.server.routes_fsm         api_fsm_launch, api_fsm_start_conversation,
                                          api_fsm_converse, api_fsm_end_conversation,
                                          api_fsm_conversations
fsm_llm.monitor.server.routes_workflow    api_workflow_presets, api_workflow_launch,
                                          api_workflow_advance, api_workflow_cancel,
                                          api_workflow_event, api_workflow_status,
                                          api_workflow_instances
fsm_llm.monitor.server.routes_agent       api_agent_launch, api_agent_status,
                                          api_agent_result, api_agent_cancel
fsm_llm.monitor.server.routes_visualize   api_fsm_load, api_fsm_visualize,
                                          api_fsm_visualize_preset, api_presets, api_preset_fsm,
                                          api_agent_visualize, api_workflow_visualize
fsm_llm.monitor.server.routes_builder     api_builder_start, api_builder_send,
                                          api_builder_result, api_builder_delete
fsm_llm.monitor.server.websocket          websocket_endpoint
```

The module attribute `templates` (the `Jinja2Templates` of 0.12.0) is now `fsm_llm.monitor.server.application.templates`. Unchanged and still importable from `fsm_llm.monitor.server`: `app`, `configure`, `get_manager`, `STATIC_DIR` and `TEMPLATE_DIR` (`get_flows` is new in this release).

A patch aims at the module that performs the lookup: a test that patched `fsm_llm.monitor.server.<name>` for a moved name must now name the route module that holds the function, and a private name (a leading underscore) is an attribute of the module that defines it. The full old-to-new table, private names included, is `tests/fixtures/module_path_map.json` under the key `fsm_llm.monitor.server`.

**Before**

```python
from fsm_llm.monitor.server import api_metrics, api_fsm_launch, websocket_endpoint
```

**After**

```python
from fsm_llm.monitor.server import STATIC_DIR, TEMPLATE_DIR, app, configure, get_manager
from fsm_llm.monitor.server.application import (
    api_auth,
    health,
    index,
    no_cache_static,
    templates,
)
from fsm_llm.monitor.server.routes_agent import (
    api_agent_cancel,
    api_agent_launch,
    api_agent_result,
    api_agent_status,
)
from fsm_llm.monitor.server.routes_builder import (
    api_builder_delete,
    api_builder_result,
    api_builder_send,
    api_builder_start,
)
from fsm_llm.monitor.server.routes_fsm import (
    api_fsm_conversations,
    api_fsm_converse,
    api_fsm_end_conversation,
    api_fsm_launch,
    api_fsm_start_conversation,
)
from fsm_llm.monitor.server.routes_monitoring import (
    api_activity,
    api_capabilities,
    api_config_get,
    api_config_set,
    api_conversation,
    api_conversations,
    api_dashboard_config_delete,
    api_dashboard_config_get,
    api_dashboard_config_set,
    api_events,
    api_info,
    api_instance_destroy,
    api_instance_detail,
    api_instance_events,
    api_instances,
    api_logs,
    api_metrics,
)
from fsm_llm.monitor.server.routes_visualize import (
    api_agent_visualize,
    api_fsm_load,
    api_fsm_visualize,
    api_fsm_visualize_preset,
    api_preset_fsm,
    api_presets,
    api_workflow_visualize,
)
from fsm_llm.monitor.server.routes_workflow import (
    api_workflow_advance,
    api_workflow_cancel,
    api_workflow_event,
    api_workflow_instances,
    api_workflow_launch,
    api_workflow_presets,
    api_workflow_status,
)
from fsm_llm.monitor.server.websocket import websocket_endpoint

import fsm_llm.monitor.server as server

assert not hasattr(server, "api_metrics")  # moved, not re-exported
assert server.app is app
```
