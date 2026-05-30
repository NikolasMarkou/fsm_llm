"""Tests for opt-in parallel sampling in SelfConsistencyAgent.

Parallel sampling must be deterministic and identical to serial: results are
assembled in sample-index order regardless of completion order.
"""

from __future__ import annotations

import threading

import pytest

from fsm_llm_agents import AgentConfig, SelfConsistencyAgent
from fsm_llm_agents.exceptions import AgentError


def _make_agent(num_samples, max_workers):
    return SelfConsistencyAgent(
        config=AgentConfig(model="mock/model"),
        num_samples=num_samples,
        max_workers=max_workers,
    )


def _patch_generate(agent, monkeypatch):
    """Make _generate_single deterministic: answer encodes the temperature."""

    def fake(fsm_def, task, temperature, initial_context):
        return (f"t{temperature:.4f}", 0.9)

    monkeypatch.setattr(agent, "_generate_single", fake)


class TestParallelSelfConsistency:
    def test_max_workers_validation(self):
        with pytest.raises(AgentError):
            SelfConsistencyAgent(num_samples=3, max_workers=0)

    def test_default_is_serial(self):
        agent = _make_agent(3, 1)
        assert agent.max_workers == 1

    def test_parallel_matches_serial(self, monkeypatch):
        serial = _make_agent(5, 1)
        parallel = _make_agent(5, 4)
        _patch_generate(serial, monkeypatch)
        _patch_generate(parallel, monkeypatch)

        r_serial = serial.run("q")
        r_parallel = parallel.run("q")

        # Samples assembled in identical (index) order → identical aggregation.
        assert r_serial.final_context["samples"] == r_parallel.final_context["samples"]
        assert r_serial.answer == r_parallel.answer

    def test_parallel_runs_all_samples(self, monkeypatch):
        agent = _make_agent(6, 3)
        _patch_generate(agent, monkeypatch)
        result = agent.run("q")
        assert len(result.final_context["samples"]) == 6

    def test_parallel_actually_concurrent(self, monkeypatch):
        """At least 2 samples overlap in time when max_workers > 1."""
        agent = _make_agent(4, 4)
        active = {"now": 0, "max": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=5)

        def fake(fsm_def, task, temperature, initial_context):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            try:
                barrier.wait()  # forces at least 2 to be in-flight together
            except threading.BrokenBarrierError:
                pass
            with lock:
                active["now"] -= 1
            return ("ans", 0.5)

        monkeypatch.setattr(agent, "_generate_single", fake)
        agent.run("q")
        assert active["max"] >= 2

    def test_parallel_drops_failed_samples(self, monkeypatch):
        agent = _make_agent(5, 5)

        def fake(fsm_def, task, temperature, initial_context):
            if temperature < 0.5:
                raise RuntimeError("boom")
            return ("ok", 0.7)

        monkeypatch.setattr(agent, "_generate_single", fake)
        result = agent.run("q")
        # Some samples dropped, but survivors aggregate fine.
        assert result.success
        assert all(s == "ok" for s in result.final_context["samples"])
