"""
Release Management Workflow — CI/CD Pipeline Orchestration
============================================================

Demonstrates a 7-step software release workflow that orchestrates
planning, building, testing (with retry), staging deployment,
smoke testing, production deployment (via agent), and post-deploy
verification.

Flow:
  Planning -> Build -> Test (retry x2) -> Staging Deploy
  -> Smoke Test -> Production Deploy (agent) -> Post-Deploy Verify

Combines:
    - fsm_llm_workflows: Workflow DSL (create_workflow, auto_step,
      retry_step, agent_step)
    - fsm_llm_agents: ReactAgent for production deployment step

Key Concepts:
    - auto_step: deterministic pipeline stages
    - retry_step: wraps flaky test execution with max_retries=2
    - AgentStep via custom step: agent-driven production deployment
    - Linear pipeline with error routing

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:4b"
    python run.py
"""

import asyncio
import os
import random
from typing import Any

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool
from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
    retry_step,
)

# ------------------------------------------------------------------
# Task context: detailed release request (~2.5k chars)
# ------------------------------------------------------------------

RELEASE_CONTEXT = """
Software Release Request — Platform Service v2.8.0
=====================================================
Release Manager: Alex Rivera
Team: Platform Engineering
Date Requested: 2024-11-15
Target Release Date: 2024-11-18 (Monday, off-peak window 02:00-06:00 UTC)

Version Information:
  Current Production: v2.7.3
  Release Candidate: v2.8.0-rc.4
  Git Tag: v2.8.0
  Branch: release/2.8.0
  Commit SHA: a3f7c2d9e1b4

Changelog Summary (47 commits since v2.7.3):
  - FEAT: Add real-time event streaming via WebSocket (PLAT-1234)
  - FEAT: Implement batch processing endpoint for bulk imports (PLAT-1301)
  - FIX: Resolve connection pool exhaustion under high concurrency (PLAT-1289)
  - FIX: Correct timezone handling in scheduled task execution (PLAT-1295)
  - PERF: Optimize database query plans for reporting endpoints (PLAT-1278)
  - PERF: Reduce memory footprint of in-memory cache by 35% (PLAT-1303)
  - SEC: Upgrade dependencies to patch CVE-2024-38819 (PLAT-1310)
  - DOCS: Update API reference for new streaming endpoints

Deployment Targets:
  Staging:    staging.platform.internal (3 pods, us-east-1)
  Production: prod.platform.service (12 pods across us-east-1, eu-west-1, ap-south-1)
  CDN:        cdn.platform.service (static assets, invalidation required)

Pre-Deployment Checklist:
  [x] Code review completed (4 approvals)
  [x] All CI checks passing (unit, integration, e2e)
  [x] Security scan clean (Snyk, SAST)
  [x] Database migration reviewed (2 new indexes, 1 column addition)
  [x] Feature flags configured (ws_streaming: 10% rollout)
  [x] Rollback plan documented (revert to v2.7.3 tag)
  [x] On-call engineer confirmed (Jamie Chen, +1-555-0198)

Infrastructure Requirements:
  - Database migration: 2 new indexes on events table, add column
    'stream_id' to sessions table (backward compatible, nullable)
  - Redis: New pub/sub channels for WebSocket fan-out
  - Load balancer: WebSocket upgrade support (already configured in staging)
  - Monitoring: New Grafana dashboards for streaming metrics

Risk Assessment:
  Overall Risk: MEDIUM
  - WebSocket feature is behind feature flag (low blast radius)
  - Connection pool fix is critical (resolves P1 incident from last week)
  - Database migration is additive only (no destructive changes)
  - Rollback is clean: feature flags disable new functionality
"""


# ------------------------------------------------------------------
# Agent tools for production deployment
# ------------------------------------------------------------------


@tool
def run_database_migration(environment: str, version: str) -> str:
    """Execute database migration scripts for the target environment."""
    if environment == "production":
        return (
            f"Migration for {version} applied to production database. "
            "2 indexes created on events table. Column 'stream_id' added "
            "to sessions table. Migration completed in 45 seconds. "
            "Backward compatibility verified."
        )
    return f"Migration for {version} applied to {environment}. All changes successful."


@tool
def deploy_service(environment: str, version: str, pod_count: int) -> str:
    """Deploy a service version to the specified environment with rolling update."""
    if environment == "production":
        return (
            f"Rolling deployment of {version} to production initiated. "
            f"{pod_count} pods across 3 regions. Strategy: 25% canary for "
            "5 minutes, then full rollout. Current status: 12/12 pods healthy. "
            "Average response time: 42ms (within SLA). Zero errors detected."
        )
    return f"Deployed {version} to {environment} ({pod_count} pods). All pods healthy."


@tool
def invalidate_cdn_cache(paths: str) -> str:
    """Invalidate CDN cache for specified paths after deployment."""
    return (
        f"CDN cache invalidation initiated for paths: {paths}. "
        "Estimated propagation time: 90 seconds across all edge locations. "
        "Cache invalidation ID: INV-2024-8847."
    )


@tool
def verify_health_endpoints(environment: str) -> str:
    """Check health endpoints across all deployment targets."""
    if environment == "production":
        return (
            "Health check results for production:\n"
            "  us-east-1: 4/4 pods healthy, avg latency 38ms\n"
            "  eu-west-1: 4/4 pods healthy, avg latency 45ms\n"
            "  ap-south-1: 4/4 pods healthy, avg latency 52ms\n"
            "All regions passing. WebSocket upgrade endpoint responding. "
            "Database connections stable at 60% pool utilization."
        )
    return f"Health check for {environment}: all endpoints responding normally."


# ------------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------------


def planning_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Validate release prerequisites and create release plan."""
    print("  [Planning] Validating release prerequisites...")
    print("  [Planning] Code review: 4 approvals (PASS)")
    print("  [Planning] CI checks: all passing (PASS)")
    print("  [Planning] Security scan: clean (PASS)")
    print("  [Planning] Rollback plan: documented (PASS)")
    return {
        "release_version": "v2.8.0",
        "release_branch": "release/2.8.0",
        "commit_sha": "a3f7c2d9e1b4",
        "prerequisites_met": True,
        "deployment_regions": ["us-east-1", "eu-west-1", "ap-south-1"],
        "pod_count_staging": 3,
        "pod_count_production": 12,
        "release_plan_id": "RP-2024-0118",
    }


def build_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Build release artifacts from the release branch."""
    version = ctx.get("release_version", "v0.0.0")
    sha = ctx.get("commit_sha", "unknown")
    print(f"  [Build] Building {version} from commit {sha}...")
    print("  [Build] Docker image: platform-service:v2.8.0 (247MB)")
    print("  [Build] Static assets bundle: 3.2MB (gzipped)")
    print("  [Build] Database migration package: 2 scripts")
    return {
        "docker_image": f"platform-service:{version}",
        "image_digest": "sha256:e4f8a2b1c3d5...",
        "artifact_registry": "registry.internal/platform",
        "build_duration_seconds": 142,
        "build_status": "success",
    }


# Flaky test simulator — first call has 50% chance of failure
_test_attempt_count = {"value": 0}


def flaky_test_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run test suite — simulates occasional flaky test failures."""
    _test_attempt_count["value"] += 1
    attempt = _test_attempt_count["value"]

    # First attempt may fail (simulating flaky tests)
    if attempt == 1 and random.random() < 0.6:
        print(f"  [Test] Attempt {attempt}: Running 847 tests...")
        print("  [Test] FAILURE: test_websocket_reconnect timed out (flaky)")
        raise RuntimeError(
            "Test suite failed: test_websocket_reconnect timed out "
            "(known flaky test, retry recommended)"
        )

    print(f"  [Test] Attempt {attempt}: Running 847 tests...")
    print("  [Test] Unit tests: 612/612 passed")
    print("  [Test] Integration tests: 189/189 passed")
    print("  [Test] E2E tests: 46/46 passed")
    return {
        "tests_total": 847,
        "tests_passed": 847,
        "tests_failed": 0,
        "test_duration_seconds": 312,
        "test_attempt": attempt,
        "test_status": "success",
    }


def staging_deploy_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Deploy to staging environment."""
    version = ctx.get("release_version", "v0.0.0")
    pods = ctx.get("pod_count_staging", 3)
    print(f"  [Staging] Deploying {version} to staging ({pods} pods)...")
    print("  [Staging] Database migration applied")
    print("  [Staging] Rolling update complete: 3/3 pods healthy")
    return {
        "staging_deployed": True,
        "staging_url": "https://staging.platform.internal",
        "staging_pods_healthy": pods,
    }


def smoke_test_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run smoke tests against staging deployment."""
    print("  [Smoke Test] Testing critical paths on staging...")
    print("  [Smoke Test] API health: OK")
    print("  [Smoke Test] Auth flow: OK")
    print("  [Smoke Test] WebSocket upgrade: OK")
    print("  [Smoke Test] Batch import endpoint: OK")
    print("  [Smoke Test] Database connectivity: OK")
    return {
        "smoke_tests_passed": True,
        "smoke_test_count": 12,
        "smoke_tests_all_passed": 12,
        "staging_ready_for_promotion": True,
    }


def post_deploy_verify_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run post-deployment verification checks."""
    version = ctx.get("release_version", "v0.0.0")
    print(f"  [Post-Deploy] Verifying {version} in production...")
    print("  [Post-Deploy] Error rate: 0.02% (below 0.1% threshold)")
    print("  [Post-Deploy] P99 latency: 180ms (below 250ms threshold)")
    print("  [Post-Deploy] Feature flag ws_streaming: active at 10%")
    print("  [Post-Deploy] Monitoring dashboards: all green")
    return {
        "error_rate_pct": 0.02,
        "p99_latency_ms": 180,
        "feature_flags_active": ["ws_streaming:10%"],
        "post_deploy_status": "healthy",
        "release_complete": True,
    }


def print_release_summary(ctx: dict[str, Any]) -> None:
    """Print the final release summary."""
    print("\n" + "=" * 60)
    print("RELEASE COMPLETE")
    print("=" * 60)
    print(f"  Version:        {ctx.get('release_version', 'N/A')}")
    print(f"  Build:          {ctx.get('build_status', 'N/A')}")
    print(
        f"  Tests:          {ctx.get('tests_passed', 0)}/{ctx.get('tests_total', 0)} passed (attempt {ctx.get('test_attempt', '?')})"
    )
    print(f"  Staging:        {'deployed' if ctx.get('staging_deployed') else 'N/A'}")
    print(
        f"  Smoke Tests:    {ctx.get('smoke_tests_all_passed', 0)}/{ctx.get('smoke_test_count', 0)} passed"
    )
    print(f"  Production:     {ctx.get('post_deploy_status', 'N/A')}")
    print(f"  Error Rate:     {ctx.get('error_rate_pct', 'N/A')}%")
    print(f"  P99 Latency:    {ctx.get('p99_latency_ms', 'N/A')}ms")
    print(f"  Release Plan:   {ctx.get('release_plan_id', 'N/A')}")


# ------------------------------------------------------------------
# Custom step for agent-driven production deployment
# ------------------------------------------------------------------


class ProductionDeployStep(WorkflowStep):
    """Uses a ReactAgent to orchestrate production deployment."""

    success_state: str = ""
    error_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        model = context.get("_model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        version = context.get("release_version", "v0.0.0")
        pod_count = context.get("pod_count_production", 12)

        registry = ToolRegistry()
        registry.register(run_database_migration._tool_definition)
        registry.register(deploy_service._tool_definition)
        registry.register(invalidate_cdn_cache._tool_definition)
        registry.register(verify_health_endpoints._tool_definition)

        config = AgentConfig(model=model, max_iterations=8, temperature=0.3)
        agent = ReactAgent(tools=registry, config=config)

        task = (
            f"Deploy version {version} to production with {pod_count} pods. "
            f"Steps: 1) Run database migration for production. "
            f"2) Deploy the service to production with {pod_count} pods. "
            f"3) Invalidate CDN cache for paths '/*'. "
            f"4) Verify health endpoints for production. "
            f"Report the final deployment status."
        )

        try:
            print(f"  [Prod Deploy] Agent deploying {version} to production...")
            result = agent.run(task)
            return WorkflowStepResult.success_result(
                data={
                    "production_deployed": True,
                    "deploy_agent_output": result.answer[:500],
                    "deploy_tools_used": result.tools_used,
                    "deploy_iterations": result.iterations_used,
                },
                next_state=self.success_state or None,
                message="Production deployment complete",
            )
        except Exception as e:
            print(f"  [Prod Deploy] Agent error: {e}")
            return WorkflowStepResult.success_result(
                data={
                    "production_deployed": False,
                    "deploy_error": str(e),
                },
                next_state=self.error_state or self.success_state or None,
                message=f"Production deployment failed: {e}",
            )


# ------------------------------------------------------------------
# Flaky test step (for retry wrapping)
# ------------------------------------------------------------------


class TestStep(WorkflowStep):
    """Run tests — may fail due to flaky tests."""

    next_state: str = ""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        try:
            data = flaky_test_action(context)
            return WorkflowStepResult.success_result(
                data=data,
                next_state=self.next_state or None,
                message="Tests passed",
            )
        except RuntimeError as e:
            return WorkflowStepResult(
                success=False,
                data={"test_error": str(e)},
                message=str(e),
            )


# ------------------------------------------------------------------
# Build the workflow
# ------------------------------------------------------------------


def build_release_workflow(model: str) -> WorkflowEngine:
    """Build the 7-step release management workflow."""
    workflow = create_workflow(
        "release_management",
        "Release Management Pipeline",
        "Orchestrate a full software release: plan, build, test, "
        "stage, smoke test, deploy to production, and verify.",
    )

    # Step 1: Planning
    workflow.with_initial_step(
        auto_step(
            step_id="planning",
            name="Release Planning",
            next_state="build",
            action=planning_action,
            description="Validate prerequisites and create release plan",
        )
    )

    # Step 2: Build
    workflow.with_step(
        auto_step(
            step_id="build",
            name="Build Artifacts",
            next_state="test",
            action=build_action,
            description="Build Docker image, static assets, and migration package",
        )
    )

    # Step 3: Test with retry (wraps flaky test step)
    inner_test_step = TestStep(
        step_id="test_inner",
        name="Run Test Suite",
        next_state="staging_deploy",
        description="Execute unit, integration, and e2e tests",
    )
    workflow.with_step(
        retry_step(
            step_id="test",
            name="Test Suite (with retry)",
            step=inner_test_step,
            max_retries=2,
            backoff_factor=0.5,
            description="Run test suite with retry for flaky tests",
        )
    )

    # Step 4: Staging deployment
    workflow.with_step(
        auto_step(
            step_id="staging_deploy",
            name="Deploy to Staging",
            next_state="smoke_test",
            action=staging_deploy_action,
            description="Deploy release candidate to staging environment",
        )
    )

    # Step 5: Smoke test
    workflow.with_step(
        auto_step(
            step_id="smoke_test",
            name="Staging Smoke Tests",
            next_state="production_deploy",
            action=smoke_test_action,
            description="Run smoke tests against staging deployment",
        )
    )

    # Step 6: Production deployment via agent
    workflow.with_step(
        ProductionDeployStep(
            step_id="production_deploy",
            name="Production Deployment",
            success_state="post_deploy_verify",
            error_state="release_failed",
            description="Agent-driven production deployment with migration and CDN",
        )
    )

    # Step 7: Post-deploy verification
    workflow.with_step(
        auto_step(
            step_id="post_deploy_verify",
            name="Post-Deploy Verification",
            next_state="release_complete",
            action=post_deploy_verify_action,
            description="Verify production health, error rates, and latency",
        )
    )

    # Terminal: success
    class TerminalStep(WorkflowStep):
        action_fn: Any = None

        async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
            if callable(self.action_fn):
                self.action_fn(context)
            return WorkflowStepResult.success_result(
                data={}, next_state=None, message="Release complete"
            )

    workflow.with_step(
        TerminalStep(
            step_id="release_complete",
            name="Release Complete",
            action_fn=print_release_summary,
            description="Print release summary and mark complete",
        )
    )

    # Terminal: failure
    workflow.with_step(
        TerminalStep(
            step_id="release_failed",
            name="Release Failed",
            action_fn=lambda ctx: print(
                f"  [RELEASE FAILED] Error: {ctx.get('deploy_error', 'Unknown')}"
            ),
            description="Handle release failure",
        )
    )

    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    return engine


# ------------------------------------------------------------------
# Run the workflow
# ------------------------------------------------------------------


async def run():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not os.getenv("OPENAI_API_KEY") and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:4b'")
        return

    print("=" * 60)
    print("Release Management Workflow")
    print("=" * 60)
    print(f"Model: {model}")
    print("This workflow will:")
    print("  1. Validate prerequisites and create release plan")
    print("  2. Build Docker image and artifacts")
    print("  3. Run test suite (with retry for flaky tests)")
    print("  4. Deploy to staging environment")
    print("  5. Run smoke tests against staging")
    print("  6. Deploy to production via agent (migration + deploy + CDN)")
    print("  7. Post-deployment verification")
    print()

    try:
        engine = build_release_workflow(model)

        instance_id = await engine.start_workflow(
            "release_management",
            initial_context={"_model": model},
        )
        print(f"\nWorkflow started: {instance_id}")

        instance = engine.get_workflow_instance(instance_id)
        if instance:
            print(f"\nFinal status: {instance.status.value}")
            context_keys = sorted(k for k in instance.context if not k.startswith("_"))
            print(f"Context keys: {context_keys}")

            # Key metrics
            print("\nKey Metrics:")
            print(f"  Build: {instance.context.get('build_status', 'N/A')}")
            print(
                f"  Tests: {instance.context.get('test_status', 'N/A')} (attempt {instance.context.get('test_attempt', '?')})"
            )
            print(f"  Post-deploy: {instance.context.get('post_deploy_status', 'N/A')}")

            # ── Verification ──
            ctx = instance.context
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            checks = {
                "workflow_completed": instance.status.value == "completed",
                "release_version": ctx.get("release_version"),
                "build_status": ctx.get("build_status"),
                "tests_passed": ctx.get("tests_passed"),
                "staging_deployed": ctx.get("staging_deployed"),
                "smoke_tests_passed": ctx.get("smoke_tests_passed"),
                "production_deployed": ctx.get("production_deployed"),
                "post_deploy_status": ctx.get("post_deploy_status"),
                "release_complete": ctx.get("release_complete"),
                "final_status": instance.status.value,
            }
            extracted = 0
            for key, value in checks.items():
                passed = value is not None and value not in (False, 0, "", "failed")
                status = "EXTRACTED" if passed else "MISSING"
                if passed:
                    extracted += 1
                print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")
            print(
                f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
            )
    except Exception as e:
        print(f"Workflow error: {e}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
