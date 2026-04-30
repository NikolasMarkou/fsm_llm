"""
Architecture Review with Structured Output
============================================

Demonstrates EvaluatorOptimizer combined with Pydantic structured output
for system architecture review. The agent generates a structured architecture
report, the evaluator validates completeness and scoring criteria, and the
optimizer refines until all quality gates are met.

This pattern is useful for any task requiring structured analysis with
quantitative scoring and completeness validation.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/architecture_review/run.py
"""

import os

from pydantic import BaseModel, Field

from fsm_llm.stdlib.agents import AgentConfig, EvaluationResult, EvaluatorOptimizerAgent


class ComponentDetail(BaseModel):
    """A system component with description and technology."""

    name: str = Field(description="Component name (e.g., API Gateway, Auth Service)")
    description: str = Field(description="What this component does")
    technology: str = Field(description="Primary technology or framework used")


class RiskItem(BaseModel):
    """An identified architectural risk."""

    category: str = Field(description="Risk category (e.g., performance, security)")
    description: str = Field(description="Description of the risk")
    severity: str = Field(description="Low, Medium, or High")
    mitigation: str = Field(description="Recommended mitigation strategy")


class Recommendation(BaseModel):
    """An architectural recommendation."""

    area: str = Field(description="Area of improvement")
    suggestion: str = Field(description="Specific actionable suggestion")
    priority: str = Field(description="Low, Medium, or High")
    effort: str = Field(description="Estimated effort: Small, Medium, or Large")


class ArchitectureReport(BaseModel):
    """Structured architecture review report."""

    system_name: str = Field(description="Name of the system being reviewed")
    components: list[ComponentDetail] = Field(
        description="List of system components with details"
    )
    risks: list[RiskItem] = Field(description="Identified architectural risks")
    recommendations: list[Recommendation] = Field(
        description="Actionable recommendations for improvement"
    )
    scalability_score: int = Field(
        description="Scalability rating from 1 to 10", ge=1, le=10
    )
    security_score: int = Field(description="Security rating from 1 to 10", ge=1, le=10)
    maintainability_score: int = Field(
        description="Maintainability rating from 1 to 10", ge=1, le=10
    )


def evaluate_architecture_report(output: str, context: dict) -> EvaluationResult:
    """Evaluate the architecture report for completeness and quality."""
    feedback_parts = []
    checks = {}

    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    checks["has_content"] = len(lines) >= 10
    if not checks["has_content"]:
        feedback_parts.append("Output too short — need a comprehensive report.")

    lower = output.lower()

    # Check for component coverage (at least 3)
    component_keywords = [
        "component",
        "service",
        "gateway",
        "database",
        "cache",
        "queue",
        "api",
        "frontend",
        "backend",
        "load balancer",
    ]
    component_hits = sum(1 for kw in component_keywords if kw in lower)
    checks["has_components"] = component_hits >= 3
    if not checks["has_components"]:
        feedback_parts.append(
            "Need at least 3 distinct components described. Include services, "
            "databases, caches, queues, gateways, or similar infrastructure elements."
        )

    # Check for risk identification (at least 3)
    risk_keywords = [
        "risk",
        "vulnerability",
        "threat",
        "failure",
        "bottleneck",
        "single point",
        "weakness",
        "concern",
        "mitigation",
    ]
    risk_hits = sum(1 for kw in risk_keywords if kw in lower)
    checks["has_risks"] = risk_hits >= 3
    if not checks["has_risks"]:
        feedback_parts.append(
            "Need at least 3 risks with categories, severity, and mitigations. "
            "Consider performance, security, availability, and data integrity risks."
        )

    # Check for recommendations (at least 3)
    rec_keywords = [
        "recommend",
        "suggest",
        "improve",
        "should",
        "consider",
        "priority",
        "implement",
        "adopt",
        "migrate",
    ]
    rec_hits = sum(1 for kw in rec_keywords if kw in lower)
    checks["has_recommendations"] = rec_hits >= 3
    if not checks["has_recommendations"]:
        feedback_parts.append(
            "Need at least 3 actionable recommendations with priority and effort "
            "estimates. Cover scalability, security, and maintainability improvements."
        )

    # Check for scoring
    has_scores = any(
        kw in lower
        for kw in [
            "score",
            "rating",
            "scalability",
            "security",
            "maintainability",
            "/10",
        ]
    )
    checks["has_scores"] = has_scores
    if not has_scores:
        feedback_parts.append(
            "Include scalability_score, security_score, and maintainability_score "
            "as integers from 1 to 10."
        )

    # Check for JSON structure
    has_json = "{" in output and "}" in output
    checks["has_json_structure"] = has_json
    if not has_json:
        feedback_parts.append(
            "Output should be valid JSON matching the ArchitectureReport schema with "
            "fields: system_name, components (list of objects with name/description/"
            "technology), risks (list with category/description/severity/mitigation), "
            "recommendations (list with area/suggestion/priority/effort), "
            "scalability_score, security_score, maintainability_score."
        )

    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0
    criteria_met = [name for name, ok in checks.items() if ok]

    feedback = (
        "Comprehensive architecture report!" if passed else " ".join(feedback_parts)
    )

    return EvaluationResult(
        passed=passed,
        score=score,
        feedback=feedback,
        criteria_met=criteria_met,
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.5,
        output_schema=ArchitectureReport,
    )

    agent = EvaluatorOptimizerAgent(
        evaluation_fn=evaluate_architecture_report,
        config=config,
        max_refinements=2,
    )

    task = (
        "Perform a detailed architecture review of a real-time collaborative document "
        "editing platform called 'DocSync'. The platform allows multiple users to "
        "simultaneously edit documents with live cursor tracking, inline comments, "
        "version history, and conflict resolution. The system handles 50,000 concurrent "
        "users across 3 geographic regions (US-East, EU-West, AP-Southeast) with a "
        "target latency of under 100ms for character-level synchronization. "
        "The current architecture uses a Node.js WebSocket gateway behind an NGINX "
        "load balancer, a Go-based operational transformation (OT) engine running as "
        "a stateful service with sticky sessions, PostgreSQL 15 with logical "
        "replication for document storage, Redis 7 cluster for session state and "
        "pub/sub, and an Elasticsearch 8 cluster for full-text search and audit "
        "logging. Authentication flows through an OAuth2 provider (Keycloak) with "
        "JWT tokens and refresh token rotation, and CloudFront CDN serves static "
        "assets and pre-rendered document previews. A RabbitMQ instance handles "
        "async tasks like PDF export, email notifications, and webhook deliveries. "
        "The engineering team (35 developers across 6 squads) has reported increasing "
        "latency during peak hours (200-400ms vs the 100ms target), occasional data "
        "inconsistencies when users reconnect after network partitions due to OT "
        "vector clock drift, growing operational complexity as the service count has "
        "reached 15 microservices with inconsistent observability, and PostgreSQL "
        "write amplification causing replication lag of 2-5 seconds during bulk "
        "imports. They are considering migrating the OT engine to a CRDT-based "
        "approach (Yjs or Automerge), adding Kafka for event sourcing to replace "
        "the current audit log pipeline, and introducing a service mesh (Istio) for "
        "cross-service communication. The platform must maintain SOC 2 Type II "
        "compliance and support HIPAA requirements for healthcare customers. "
        "Return your answer as JSON matching the ArchitectureReport schema with: "
        "system_name, components (list of objects with name/description/technology), "
        "risks (list with category/description/severity/mitigation), recommendations "
        "(list with area/suggestion/priority/effort), and integer scores 1-10 for "
        "scalability_score, security_score, and maintainability_score."
    )

    print("=" * 60)
    print("Architecture Review with Structured Output")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Output schema: {ArchitectureReport.__name__}")
    print("Max refinements: 2")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nRaw answer:\n{result.answer[:500]}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        refinement_count = result.final_context.get("refinement_count", 0)
        eval_passed = result.final_context.get("evaluation_passed", False)
        eval_score = result.final_context.get("evaluation_score", 0)

        print(f"Refinements: {refinement_count}")
        print(f"Evaluation passed: {eval_passed}")
        print(f"Final score: {eval_score}")

        if result.structured_output:
            report = result.structured_output
            print("\nStructured Output (validated):")
            print(f"  System: {report.system_name}")
            print(f"  Scalability: {report.scalability_score}/10")
            print(f"  Security: {report.security_score}/10")
            print(f"  Maintainability: {report.maintainability_score}/10")
            print(f"  Components ({len(report.components)}):")
            for comp in report.components[:5]:
                print(f"    - {comp.name}: {comp.description[:60]}")
            if len(report.components) > 5:
                print(f"    ... and {len(report.components) - 5} more")
            print(f"  Risks ({len(report.risks)}):")
            for risk in report.risks[:3]:
                print(
                    f"    - [{risk.severity}] {risk.category}: {risk.description[:60]}"
                )
            if len(report.risks) > 3:
                print(f"    ... and {len(report.risks) - 3} more")
            print(f"  Recommendations ({len(report.recommendations)}):")
            for rec in report.recommendations[:3]:
                print(f"    - [{rec.priority}] {rec.area}: {rec.suggestion[:60]}")
            if len(report.recommendations) > 3:
                print(f"    ... and {len(report.recommendations) - 3} more")
        else:
            print("\nStructured output: None (validation failed)")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.iterations_used < config.max_iterations,
        "score_present": result.final_context.get("evaluation_score", 0) > 0,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
