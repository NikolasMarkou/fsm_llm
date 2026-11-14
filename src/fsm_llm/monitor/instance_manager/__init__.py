"""
Instance manager for fsm_llm.monitor.

Manages multiple concurrent FSM, workflow, and agent instances with
per-instance event collection and lifecycle management.

Package layout (plan d454fd62 step 93; the old ``monitor/instance_manager.py``,
same dotted path): ``manager`` (``InstanceManager``, the shell with the
constructor), the mixins ``snapshots``, ``fsm``, ``workflow``, ``agent`` and
``lifecycle``, ``managed`` (``ManagedFSM``, ``ManagedWorkflow``,
``ManagedAgent``), ``handlers`` (the observer handlers), ``workflow_presets``,
``examples`` (the example FSM presets and the ``examples/`` directory read) and
``surface`` (declarations for mypy only). A private name is imported from its
defining module; a patch aims at the module that performs the lookup (for
example ``fsm_llm.monitor.instance_manager.lifecycle.logger``), never the
package.
"""

from .examples import (
    find_examples_dir,
    load_preset_json,
    read_json_file,
    validate_preset_id,
)
from .handlers import (
    monitor_handlers_on,
    register_monitor_handlers,
    unregister_monitor_handlers,
)
from .managed import ManagedAgent, ManagedFSM, ManagedInstance, ManagedWorkflow
from .manager import InstanceManager
from .snapshots import snapshot_from_api

__all__ = [
    "InstanceManager",
    "ManagedAgent",
    "ManagedFSM",
    "ManagedInstance",
    "ManagedWorkflow",
    "find_examples_dir",
    "load_preset_json",
    "monitor_handlers_on",
    "read_json_file",
    "register_monitor_handlers",
    "snapshot_from_api",
    "unregister_monitor_handlers",
    "validate_preset_id",
]
