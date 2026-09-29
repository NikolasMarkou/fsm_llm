"""Agents suite fixtures: every test runs offline unless marked live."""

from __future__ import annotations

import pytest

from tests.conftest import block_network


@pytest.fixture(autouse=True)
def _offline_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Refuse IPv4/IPv6 connects (``tests.conftest.block_network``)."""
    block_network(monkeypatch, request.node)
