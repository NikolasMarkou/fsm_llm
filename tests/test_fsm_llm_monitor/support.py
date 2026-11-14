"""Helpers shared by several test files of this suite. Not collected (no test_ prefix)."""

from __future__ import annotations

from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.instance_manager import InstanceManager


def _manager(**config) -> InstanceManager:
    mgr = InstanceManager(config=MonitorConfig(**config))
    mgr.global_collector.cleanup()
    return mgr


def server_package_modules() -> list:
    """The imported module objects of the ``fsm_llm.monitor.server`` package: the
    facade and every module in it."""
    import importlib

    import fsm_llm.monitor.server as facade

    return [facade] + [
        importlib.import_module(f"{facade.__name__}.{name}")
        for name in server_modules()
        if name != "__init__"
    ]


def server_modules() -> dict[str, str]:
    """The source text of every module of the ``fsm_llm.monitor.server`` package,
    by module name (``__init__`` included). The server is a package since step
    94 of plan d454fd62: a test that reads "the server's source" reads all of it,
    never the facade alone."""
    import inspect
    from pathlib import Path

    import fsm_llm.monitor.server as facade

    package = Path(inspect.getsourcefile(facade)).parent  # type: ignore[arg-type]
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
    }
