"""
Customer Onboarding Workflow — Compliance-Gated Enrollment
============================================================

Demonstrates a 5-step customer onboarding workflow that combines
automated profile setup, agent-driven compliance verification (with
tools for identity, sanctions, and risk checks), account creation,
welcome communication, and training scheduling.

Flow:
  Profile Setup -> Compliance Verification (ReactAgent) -> Account Creation
  -> Welcome Communication -> Training Schedule

Combines:
    - fsm_llm_workflows: Workflow DSL (create_workflow, auto_step)
    - fsm_llm_agents: ReactAgent with 3 compliance tools

Key Concepts:
    - auto_step: deterministic processing stages
    - AgentStep (custom): ReactAgent with verify_identity, check_sanctions,
      assess_risk tools for compliance verification
    - Multi-tool agent operating within a workflow pipeline
    - Error routing on compliance failure

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:4b"
    python run.py
"""

import asyncio
import os
from typing import Any

from fsm_llm.stdlib.agents import AgentConfig, ReactAgent, ToolRegistry, tool
from fsm_llm.stdlib.workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
)

# ------------------------------------------------------------------
# Task context: detailed customer onboarding request (~2k chars)
# ------------------------------------------------------------------

CUSTOMER_PROFILE = {
    "company_name": "Meridian Analytics Ltd.",
    "company_registration": "UK-2019-87432",
    "incorporation_date": "2019-03-22",
    "jurisdiction": "United Kingdom",
    "industry": "Data Analytics and Business Intelligence",
    "annual_revenue_usd": 4200000,
    "employee_count": 45,
    "primary_contact_name": "David Chen",
    "primary_contact_email": "d.chen@meridian-analytics.co.uk",
    "primary_contact_phone": "+44-20-7946-0123",
    "primary_contact_title": "Chief Operating Officer",
    "registered_address": "14 Canary Wharf, London E14 5AB, United Kingdom",
    "website": "https://meridian-analytics.co.uk",
    "tax_id": "GB-923-4567-89",
    "bank_name": "Barclays Business",
    "bank_account_iban": "GB29BARC20201530093459",
    "beneficial_owner_name": "David Chen",
    "beneficial_owner_nationality": "British",
    "beneficial_owner_dob": "1985-09-14",
    "beneficial_owner_pep_status": False,
    "expected_monthly_transaction_volume": 150,
    "expected_monthly_transaction_value_usd": 350000,
    "service_tier_requested": "enterprise",
    "use_case_description": (
        "Meridian Analytics requires our platform for real-time data pipeline "
        "management, automated reporting, and API integration with their existing "
        "business intelligence stack. They plan to onboard 30 users initially "
        "with growth to 100 users within 12 months."
    ),
}

TASK_CONTEXT = """
Customer Onboarding Request
==============================
Company: Meridian Analytics Ltd. (UK-2019-87432)
Jurisdiction: United Kingdom | Industry: Data Analytics and BI
Annual Revenue: $4.2M USD | Employees: 45
Registered Address: 14 Canary Wharf, London E14 5AB, United Kingdom

Primary Contact: David Chen (COO)
Email: d.chen@meridian-analytics.co.uk | Phone: +44-20-7946-0123

Beneficial Owner: David Chen (British, DOB: 1985-09-14)
PEP Status: No | Tax ID: GB-923-4567-89

Banking: Barclays Business (IBAN: GB29BARC20201530093459)
Expected Volume: 150 transactions/month (~$350K USD/month)

Service Tier: Enterprise
Use Case: Real-time data pipeline management, automated reporting,
and API integration with existing BI stack. Initial 30 users,
scaling to 100 within 12 months.

Compliance Requirements:
  - Identity verification of beneficial owner and company registration
  - Sanctions screening against OFAC, EU, and UK sanctions lists
  - Risk assessment based on jurisdiction, industry, and transaction profile
  - KYB (Know Your Business) documentation collection
  - AML (Anti-Money Laundering) risk scoring

The customer has provided all required documentation upfront and
requests expedited onboarding to meet a project deadline of December 1st.
The enterprise tier includes dedicated support, custom SLA, and
priority API rate limits.
"""


# ------------------------------------------------------------------
# Agent tools for compliance verification
# ------------------------------------------------------------------


@tool
def verify_identity(
    entity_name: str, registration_number: str, jurisdiction: str
) -> str:
    """Verify the identity of a business entity against government registries."""
    return (
        f"Identity verification for '{entity_name}' (Reg: {registration_number}):\n"
        f"  Registry: Companies House ({jurisdiction})\n"
        f"  Status: ACTIVE (incorporated 2019-03-22)\n"
        f"  Registered Address: Confirmed match\n"
        f"  Directors: David Chen (COO), Priya Sharma (CEO)\n"
        f"  Filing Status: Up to date (last annual return: 2024-03-22)\n"
        f"  Verification Result: PASSED\n"
        f"  Verification ID: VER-2024-UK-88341"
    )


@tool
def check_sanctions(entity_name: str, owner_name: str, nationality: str) -> str:
    """Screen entity and beneficial owner against OFAC, EU, and UK sanctions lists."""
    return (
        f"Sanctions screening for '{entity_name}' and '{owner_name}':\n"
        f"  OFAC SDN List: NO MATCH\n"
        f"  EU Consolidated Sanctions: NO MATCH\n"
        f"  UK Sanctions List: NO MATCH\n"
        f"  PEP Database (Politically Exposed Persons): NO MATCH\n"
        f"  Adverse Media Screening: NO MATCH\n"
        f"  Nationality ({nationality}): No restricted jurisdiction\n"
        f"  Screening Result: CLEAR\n"
        f"  Screening ID: SCR-2024-GLOBAL-44218"
    )


@tool
def assess_risk(
    industry: str,
    jurisdiction: str,
    annual_revenue: str,
    monthly_transaction_value: str,
) -> str:
    """Perform AML risk assessment based on business profile and transaction patterns."""
    return (
        f"Risk Assessment Report:\n"
        f"  Industry Risk ({industry}): LOW (standard commercial sector)\n"
        f"  Jurisdiction Risk ({jurisdiction}): LOW (FATF member, strong AML regime)\n"
        f"  Revenue Profile (${annual_revenue}): STANDARD (within expected range)\n"
        f"  Transaction Profile (${monthly_transaction_value}/month): STANDARD\n"
        f"  Overall AML Risk Score: 18/100 (LOW)\n"
        f"  Enhanced Due Diligence Required: NO\n"
        f"  Recommended Review Frequency: Annual\n"
        f"  Risk Assessment ID: RA-2024-ENT-7219\n"
        f"  Approval: AUTO-APPROVED (score below 30 threshold)"
    )


# ------------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------------


def profile_setup_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Set up customer profile from submitted data."""
    company = ctx.get("company_name", "Unknown")
    contact = ctx.get("primary_contact_name", "Unknown")
    tier = ctx.get("service_tier_requested", "standard")

    print(f"  [Profile Setup] Creating profile for {company}")
    print(f"  [Profile Setup] Primary contact: {contact}")
    print(f"  [Profile Setup] Service tier: {tier}")
    print(f"  [Profile Setup] Jurisdiction: {ctx.get('jurisdiction', 'Unknown')}")

    return {
        "profile_id": "PROF-2024-MER-0891",
        "profile_status": "pending_compliance",
        "onboarding_started": True,
    }


def account_creation_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Create the customer account after compliance clearance."""
    company = ctx.get("company_name", "Unknown")
    tier = ctx.get("service_tier_requested", "standard")
    compliance_passed = ctx.get("compliance_passed", False)

    if not compliance_passed:
        print(f"  [Account Creation] BLOCKED: Compliance not cleared for {company}")
        return {
            "account_created": False,
            "account_blocked_reason": "compliance_pending",
        }

    print(f"  [Account Creation] Creating {tier} account for {company}")
    print("  [Account Creation] API keys generated (2 keys: production + staging)")
    print("  [Account Creation] Rate limits configured: 10,000 req/min (enterprise)")
    print("  [Account Creation] Custom SLA activated: 99.95% uptime guarantee")

    return {
        "account_id": "ACC-2024-ENT-MER-0891",
        "account_created": True,
        "api_key_production": "pk_live_****7xQ2",
        "api_key_staging": "pk_test_****9mR4",
        "rate_limit_per_min": 10000,
        "sla_uptime_pct": 99.95,
    }


def welcome_communication_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Send welcome communications to the customer."""
    email = ctx.get("primary_contact_email", "unknown")
    contact_name = ctx.get("primary_contact_name", "Customer")
    account_id = ctx.get("account_id", "N/A")

    print(f"  [Welcome] Sending welcome package to {contact_name} ({email})")
    print(f"  [Welcome] Welcome email with account ID: {account_id}")
    print("  [Welcome] API documentation link sent")
    print("  [Welcome] Dedicated support channel created: #meridian-support")
    print("  [Welcome] Account manager assigned: Rachel Torres")

    return {
        "welcome_email_sent": True,
        "account_manager": "Rachel Torres",
        "support_channel": "#meridian-support",
        "documentation_link": "https://docs.platform.io/enterprise/getting-started",
    }


def training_schedule_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Schedule onboarding training sessions."""
    company = ctx.get("company_name", "Unknown")
    contact = ctx.get("primary_contact_name", "Customer")

    print(f"  [Training] Scheduling onboarding sessions for {company}")
    print("  [Training] Session 1: Platform Overview (Dec 2, 10:00 GMT)")
    print("  [Training] Session 2: API Integration Workshop (Dec 4, 14:00 GMT)")
    print("  [Training] Session 3: Admin & Security Configuration (Dec 6, 10:00 GMT)")
    print(f"  [Training] Calendar invites sent to {contact}")

    return {
        "training_sessions_scheduled": 3,
        "training_dates": ["2024-12-02", "2024-12-04", "2024-12-06"],
        "training_format": "video_conference",
        "onboarding_target_completion": "2024-12-10",
        "onboarding_complete": True,
    }


def print_onboarding_summary(ctx: dict[str, Any]) -> None:
    """Print the final onboarding summary."""
    print("\n" + "=" * 60)
    print("CUSTOMER ONBOARDING COMPLETE")
    print("=" * 60)
    print(f"  Company:          {ctx.get('company_name', 'N/A')}")
    print(f"  Profile ID:       {ctx.get('profile_id', 'N/A')}")
    print(f"  Account ID:       {ctx.get('account_id', 'N/A')}")
    print(f"  Service Tier:     {ctx.get('service_tier_requested', 'N/A')}")
    print(
        f"  Compliance:       {'PASSED' if ctx.get('compliance_passed') else 'FAILED'}"
    )
    print(f"  Risk Score:       {ctx.get('compliance_risk_score', 'N/A')}")
    print(f"  Account Created:  {ctx.get('account_created', False)}")
    print(f"  SLA:              {ctx.get('sla_uptime_pct', 'N/A')}% uptime")
    print(f"  Account Manager:  {ctx.get('account_manager', 'N/A')}")
    print(f"  Training:         {ctx.get('training_sessions_scheduled', 0)} sessions")
    print(f"  Target Complete:  {ctx.get('onboarding_target_completion', 'N/A')}")


# ------------------------------------------------------------------
# Custom compliance agent step
# ------------------------------------------------------------------


class ComplianceAgentStep(WorkflowStep):
    """Runs a ReactAgent with compliance verification tools."""

    success_state: str = ""
    error_state: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        model = context.get("_model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        company = context.get("company_name", "Unknown")
        reg = context.get("company_registration", "Unknown")
        jurisdiction = context.get("jurisdiction", "Unknown")
        owner = context.get("beneficial_owner_name", "Unknown")
        nationality = context.get("beneficial_owner_nationality", "Unknown")
        industry = context.get("industry", "Unknown")
        revenue = str(context.get("annual_revenue_usd", 0))
        monthly_value = str(context.get("expected_monthly_transaction_value_usd", 0))

        registry = ToolRegistry()
        registry.register(verify_identity._tool_definition)
        registry.register(check_sanctions._tool_definition)
        registry.register(assess_risk._tool_definition)

        config = AgentConfig(model=model, max_iterations=8, temperature=0.3)
        agent = ReactAgent(tools=registry, config=config)

        task = (
            f"Perform full compliance verification for customer onboarding:\n"
            f"Company: {company} (Registration: {reg}, Jurisdiction: {jurisdiction})\n"
            f"Beneficial Owner: {owner} (Nationality: {nationality})\n"
            f"Industry: {industry}, Annual Revenue: ${revenue}\n"
            f"Monthly Transaction Value: ${monthly_value}\n\n"
            f"Steps:\n"
            f"1) Verify identity of the company using verify_identity tool "
            f"with entity_name='{company}', registration_number='{reg}', "
            f"jurisdiction='{jurisdiction}'\n"
            f"2) Check sanctions using check_sanctions tool "
            f"with entity_name='{company}', owner_name='{owner}', "
            f"nationality='{nationality}'\n"
            f"3) Assess risk using assess_risk tool "
            f"with industry='{industry}', jurisdiction='{jurisdiction}', "
            f"annual_revenue='{revenue}', monthly_transaction_value='{monthly_value}'\n"
            f"4) Summarize compliance status: PASSED or FAILED with details."
        )

        try:
            print(f"  [Compliance] Agent verifying {company}...")
            result = agent.run(task)
            output = result.answer.lower()

            # Determine if compliance passed based on agent output
            compliance_passed = any(
                kw in output
                for kw in ["passed", "clear", "approved", "compliant", "no match"]
            )

            print(f"  [Compliance] Tools used: {result.tools_used}")
            print(
                f"  [Compliance] Result: {'PASSED' if compliance_passed else 'REVIEW NEEDED'}"
            )

            return WorkflowStepResult.success_result(
                data={
                    "compliance_passed": compliance_passed,
                    "compliance_summary": result.answer[:500],
                    "compliance_tools_used": result.tools_used,
                    "compliance_iterations": result.iterations_used,
                    "compliance_risk_score": "18/100 (LOW)",
                },
                next_state=self.success_state or None,
                message="Compliance verification complete",
            )
        except Exception as e:
            print(f"  [Compliance] Agent error: {e}")
            return WorkflowStepResult.success_result(
                data={
                    "compliance_passed": False,
                    "compliance_error": str(e),
                },
                next_state=self.error_state or self.success_state or None,
                message=f"Compliance verification failed: {e}",
            )


# ------------------------------------------------------------------
# Terminal step
# ------------------------------------------------------------------


class TerminalStep(WorkflowStep):
    """A step that ends the workflow and optionally runs an action."""

    action: Any = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        if callable(self.action):
            self.action(context)
        return WorkflowStepResult.success_result(
            data={}, next_state=None, message="Workflow complete"
        )


# ------------------------------------------------------------------
# Build the workflow
# ------------------------------------------------------------------


def build_onboarding_workflow(model: str) -> WorkflowEngine:
    """Build the 5-step customer onboarding workflow."""
    workflow = create_workflow(
        "customer_onboarding",
        "Customer Onboarding Pipeline",
        "Onboard a new enterprise customer through profile setup, "
        "compliance verification, account creation, welcome, and training.",
    )

    # Step 1: Profile setup
    workflow.with_initial_step(
        auto_step(
            step_id="profile_setup",
            name="Profile Setup",
            next_state="compliance_verification",
            action=profile_setup_action,
            description="Create customer profile from submitted data",
        )
    )

    # Step 2: Compliance verification via ReactAgent
    workflow.with_step(
        ComplianceAgentStep(
            step_id="compliance_verification",
            name="Compliance Verification",
            success_state="account_creation",
            error_state="onboarding_failed",
            description="Agent-driven KYB, sanctions screening, and risk assessment",
        )
    )

    # Step 3: Account creation
    workflow.with_step(
        auto_step(
            step_id="account_creation",
            name="Account Creation",
            next_state="welcome_communication",
            action=account_creation_action,
            description="Create enterprise account with API keys and SLA",
        )
    )

    # Step 4: Welcome communication
    workflow.with_step(
        auto_step(
            step_id="welcome_communication",
            name="Welcome Communication",
            next_state="training_schedule",
            action=welcome_communication_action,
            description="Send welcome package, docs, and assign account manager",
        )
    )

    # Step 5: Training schedule
    workflow.with_step(
        auto_step(
            step_id="training_schedule",
            name="Training Schedule",
            next_state="onboarding_complete",
            action=training_schedule_action,
            description="Schedule onboarding training sessions",
        )
    )

    # Terminal: success
    workflow.with_step(
        TerminalStep(
            step_id="onboarding_complete",
            name="Onboarding Complete",
            action=lambda ctx: print_onboarding_summary(ctx),
            description="Print summary and mark onboarding complete",
        )
    )

    # Terminal: failure
    workflow.with_step(
        TerminalStep(
            step_id="onboarding_failed",
            name="Onboarding Failed",
            action=lambda ctx: print(
                f"  [ONBOARDING FAILED] Compliance: {ctx.get('compliance_error', 'Unknown error')}"
            ),
            description="Handle onboarding failure",
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
    print("Customer Onboarding Workflow")
    print("=" * 60)
    print(f"Model: {model}")
    print("This workflow will:")
    print("  1. Set up customer profile from submitted data")
    print("  2. Run compliance verification (identity, sanctions, risk)")
    print("  3. Create enterprise account with API keys")
    print("  4. Send welcome package and assign account manager")
    print("  5. Schedule onboarding training sessions")
    print()

    try:
        engine = build_onboarding_workflow(model)

        # Seed context with customer profile data
        initial_context = {
            "_model": model,
            **CUSTOMER_PROFILE,
        }

        instance_id = await engine.start_workflow(
            "customer_onboarding", initial_context=initial_context
        )
        print(f"\nWorkflow started: {instance_id}")

        instance = engine.get_workflow_instance(instance_id)
        if instance:
            print(f"\nFinal status: {instance.status.value}")
            context_keys = sorted(k for k in instance.context if not k.startswith("_"))
            print(f"Context keys: {context_keys}")

            # Key results
            print("\nKey Results:")
            print(
                f"  Compliance: {'PASSED' if instance.context.get('compliance_passed') else 'FAILED'}"
            )
            print(f"  Account: {instance.context.get('account_id', 'N/A')}")
            print(
                f"  Training: {instance.context.get('training_sessions_scheduled', 0)} sessions"
            )
            print(f"  Complete: {instance.context.get('onboarding_complete', False)}")

            # ── Verification ──
            ctx = instance.context
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            checks = {
                "workflow_completed": instance.status.value == "completed",
                "profile_id": ctx.get("profile_id"),
                "compliance_passed": ctx.get("compliance_passed"),
                "account_id": ctx.get("account_id"),
                "account_created": ctx.get("account_created"),
                "welcome_email_sent": ctx.get("welcome_email_sent"),
                "account_manager": ctx.get("account_manager"),
                "training_scheduled": ctx.get("training_sessions_scheduled"),
                "onboarding_complete": ctx.get("onboarding_complete"),
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
