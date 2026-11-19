"""Plan 51fde95b step 12.1: the confirmed remainders of the plan 52ce5464 monitor audit
(a closed manager is not served again; the workflow run list has a cap)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fsm_llm.monitor import constants
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.server import app, application, configure
from tests.test_fsm_llm_monitor.support import _manager


class TestClosedManagerIsNotServedAgain:
    def test_the_flag_follows_close(self):
        mgr = _manager()
        assert mgr.closed is False
        mgr.close()
        assert mgr.closed is True

    def test_a_second_run_of_the_app_serves_a_fresh_manager(self):
        mgr = _manager()
        configure(manager=mgr)
        with TestClient(app):
            assert application.get_manager() is mgr
        assert mgr.closed

        with TestClient(app) as client:
            fresh = application.get_manager()
            try:
                assert fresh is not mgr and not fresh.closed
                assert client.get("/api/capabilities").status_code == 200
            finally:
                fresh.close()

    def test_a_live_manager_stays_installed(self):
        mgr = _manager()
        configure(manager=mgr)
        with TestClient(app):
            assert application.get_manager() is mgr
            assert not mgr.closed


class TestWorkflowRunListHasACap:
    def _listed(self, count: int) -> tuple[list[str], list[str]]:
        mgr: InstanceManager = _manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")
        ids = [f"run-{i:04d}" for i in range(count)]
        managed.run_ids[:] = ids
        mgr.get_workflow_status = lambda _iid, wid: {  # type: ignore[method-assign]
            "workflow_instance_id": wid
        }
        listed = mgr.get_workflow_instances(managed.instance_id)
        return ids, [r["workflow_instance_id"] for r in listed]

    def test_the_newest_runs_are_listed_in_start_order(self):
        ids, listed = self._listed(constants.MAX_WORKFLOW_RUNS_LISTED + 5)
        assert listed == ids[-constants.MAX_WORKFLOW_RUNS_LISTED :]

    def test_a_list_at_the_cap_is_complete(self):
        ids, listed = self._listed(constants.MAX_WORKFLOW_RUNS_LISTED)
        assert listed == ids
