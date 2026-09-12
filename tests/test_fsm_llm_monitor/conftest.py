"""Shared pytest fixtures for tests/test_fsm_llm_monitor/."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_monitor_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ambient FSM_LLM_MONITOR_API_KEY from leaking into tests.

    Bug (plan-2026-09-12T065608-089d0ec7/D-020): `configure()`
    (`fsm_llm_monitor/server.py`) falls back to
    ``os.environ["FSM_LLM_MONITOR_API_KEY"]`` whenever a test/fixture calls
    it without an explicit ``api_key=`` argument — which almost every
    `setup_method`/test in this package does. If a developer's or CI's
    shell happens to export that variable, `configure()` silently turns on
    the auth gate, and every OTHER test in this package that assumes its
    mutating routes are open starts failing with 401s unrelated to what it
    is actually testing (reproduced: 22 failures in `test_app.py` alone with
    the var set). Tests must not be sensitive to the ambient environment, so
    this autouse fixture removes the var for the duration of every test in
    this package, regardless of what the invoking shell has exported.
    `TestApiKeyGate`'s own tests are unaffected — they always pass
    `api_key=` explicitly (or rely on this cleared-env default), never the
    ambient var.
    """
    monkeypatch.delenv("FSM_LLM_MONITOR_API_KEY", raising=False)
