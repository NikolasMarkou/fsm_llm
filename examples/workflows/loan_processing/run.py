"""
Loan Processing Workflow — Multi-Step Approval Pipeline
========================================================

Demonstrates a 6-step loan processing workflow that combines FSM-driven
data intake with automated credit checks, risk assessment, switch-based
approval routing, documentation generation, and notification dispatch.

Flow:
  Intake (FSM) -> Credit Check -> Risk Assessment -> Approval Decision (switch)
    -> [approved]  -> Documentation -> Notification
    -> [review]    -> Documentation -> Notification
    -> [denied]    -> Notification

Combines:
    - fsm_llm_workflows: Workflow DSL (create_workflow, auto_step,
      conversation_step, switch_step)
    - fsm_llm: Core FSM via ConversationStep for applicant intake

Key Concepts:
    - ConversationStep: embed an FSM conversation to collect applicant data
    - auto_step: deterministic processing (credit check, risk, docs)
    - switch_step: N-way routing based on approval decision
    - Context threading across 6 pipeline stages

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

from fsm_llm_workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    conversation_step,
    create_workflow,
    switch_step,
)

# ------------------------------------------------------------------
# Task context: detailed loan application (~2k chars)
# ------------------------------------------------------------------

LOAN_APPLICATION = {
    "applicant_name": "Sarah Mitchell",
    "applicant_email": "sarah.mitchell@email.com",
    "applicant_phone": "+1-555-0142",
    "date_of_birth": "1988-06-15",
    "social_security_last4": "7823",
    "employment_status": "full-time",
    "employer_name": "Cascade Technologies Inc.",
    "annual_income": 92000,
    "years_employed": 4,
    "loan_type": "personal",
    "loan_amount_requested": 35000,
    "loan_purpose": "home renovation",
    "loan_term_months": 60,
    "existing_debts": 12500,
    "monthly_rent_or_mortgage": 1800,
    "checking_account_balance": 14200,
    "savings_account_balance": 28500,
    "credit_score_self_reported": 720,
    "bankruptcy_history": False,
    "previous_loans_with_us": 1,
    "previous_loan_repaid_on_time": True,
    "collateral_offered": "none",
    "co_signer": False,
    "residential_address": "4521 Oakridge Drive, Portland, OR 97201",
    "years_at_address": 3,
    "citizenship": "US citizen",
}

TASK_CONTEXT = """
Loan Application Processing Request
=====================================
Applicant: Sarah Mitchell (DOB: 1988-06-15)
Contact: sarah.mitchell@email.com | +1-555-0142
Address: 4521 Oakridge Drive, Portland, OR 97201 (3 years)

Employment: Full-time at Cascade Technologies Inc. for 4 years
Annual Income: $92,000 | Monthly Rent/Mortgage: $1,800

Loan Details:
  Type: Personal Loan
  Amount Requested: $35,000
  Purpose: Home renovation (kitchen remodel and bathroom upgrades)
  Term: 60 months (5 years)

Financial Profile:
  Self-Reported Credit Score: 720
  Existing Debts: $12,500 (auto loan remaining balance)
  Checking Account: $14,200
  Savings Account: $28,500
  Total Liquid Assets: $42,700

History:
  Previous Loans with Us: 1 (repaid on time)
  Bankruptcy History: None
  Collateral Offered: None
  Co-signer: None

The applicant has a stable employment history and has previously
maintained a good relationship with our institution. The debt-to-income
ratio needs evaluation given the existing auto loan balance. The loan
purpose (home renovation) may increase property value but no collateral
is offered against the personal loan. Standard credit bureau verification
and risk scoring should be performed before routing to the appropriate
approval authority level.
"""


# ------------------------------------------------------------------
# FSM definition for applicant intake (3 states)
# ------------------------------------------------------------------


def build_intake_fsm() -> dict:
    """3-state FSM to collect and verify applicant information."""
    return {
        "name": "LoanIntake",
        "description": "Collect and verify loan applicant details",
        "initial_state": "collect_personal",
        "persona": (
            "A professional loan intake officer who carefully collects "
            "applicant information and verifies key details"
        ),
        "states": {
            "collect_personal": {
                "id": "collect_personal",
                "description": "Collect personal and employment details",
                "purpose": "Gather applicant identity and employment information",
                "extraction_instructions": (
                    "Extract the following fields from user input: "
                    "applicant_name (full name), applicant_email (email address), "
                    "employment_status (full-time, part-time, or self-employed), "
                    "employer_name (company name), annual_income (numeric yearly salary), "
                    "years_employed (how long at current employer). "
                    "Parse income as a number without currency symbols. "
                    "If employment status is not explicitly stated, infer from context."
                ),
                "response_instructions": (
                    "Acknowledge the personal details received and ask about "
                    "the loan specifics: amount, purpose, and desired term."
                ),
                "transitions": [
                    {
                        "target_state": "collect_loan_details",
                        "description": "Move to loan details once personal info is captured",
                        "conditions": [
                            {
                                "description": "Has applicant name and income",
                                "requires_context_keys": [
                                    "applicant_name",
                                    "annual_income",
                                ],
                                "logic": {
                                    "and": [
                                        {
                                            "has_context": "applicant_name",
                                        },
                                        {
                                            "has_context": "annual_income",
                                        },
                                    ]
                                },
                            }
                        ],
                    }
                ],
            },
            "collect_loan_details": {
                "id": "collect_loan_details",
                "description": "Collect loan amount, purpose, and term",
                "purpose": "Gather the specifics of the loan request",
                "extraction_instructions": (
                    "Extract: loan_amount_requested (numeric dollar amount), "
                    "loan_purpose (what the money will be used for), "
                    "loan_term_months (desired repayment period in months), "
                    "existing_debts (total outstanding debt amount, numeric). "
                    "Convert years to months if the user specifies term in years. "
                    "Parse all monetary values as numbers without symbols."
                ),
                "response_instructions": (
                    "Confirm the loan details and ask about existing debts "
                    "and financial assets for the application."
                ),
                "transitions": [
                    {
                        "target_state": "confirm_application",
                        "description": "Proceed to confirmation once loan details captured",
                        "conditions": [
                            {
                                "description": "Has loan amount and purpose",
                                "requires_context_keys": [
                                    "loan_amount_requested",
                                    "loan_purpose",
                                ],
                                "logic": {
                                    "and": [
                                        {
                                            "has_context": "loan_amount_requested",
                                        },
                                        {
                                            "has_context": "loan_purpose",
                                        },
                                    ]
                                },
                            }
                        ],
                    }
                ],
            },
            "confirm_application": {
                "id": "confirm_application",
                "description": "Review and confirm application details",
                "purpose": "Final verification of all collected information",
                "extraction_instructions": (
                    "Extract confirmation (boolean: true if the applicant "
                    "confirms the details are correct, false otherwise). "
                    "Look for affirmative words like yes, correct, confirmed, "
                    "looks good, that is right."
                ),
                "response_instructions": (
                    "Summarize all collected details and confirm the "
                    "application will be submitted for processing."
                ),
                "transitions": [],
            },
        },
    }


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
# Processing functions for each pipeline stage
# ------------------------------------------------------------------


def credit_check_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Simulate credit bureau check and return credit assessment."""
    self_reported = ctx.get("credit_score_self_reported", 700)
    has_bankruptcy = ctx.get("bankruptcy_history", False)
    previous_good = ctx.get("previous_loan_repaid_on_time", False)

    # Simulate bureau score (close to self-reported with slight adjustment)
    bureau_score = min(850, max(300, int(self_reported) + 5))
    if has_bankruptcy:
        bureau_score = max(300, bureau_score - 150)
    if previous_good:
        bureau_score = min(850, bureau_score + 10)

    result = {
        "bureau_credit_score": bureau_score,
        "credit_report_id": "CR-2024-88421",
        "derogatory_marks": 0 if not has_bankruptcy else 2,
        "open_accounts": 4,
        "credit_utilization_pct": 28,
        "credit_check_passed": bureau_score >= 580,
    }
    print(
        f"  [Credit Check] Bureau score: {bureau_score} (self-reported: {self_reported})"
    )
    print(f"  [Credit Check] Report ID: {result['credit_report_id']}")
    print(f"  [Credit Check] Utilization: {result['credit_utilization_pct']}%")
    return result


def risk_assessment_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Calculate risk score and determine approval path."""
    income = float(ctx.get("annual_income", 0))
    loan_amount = float(ctx.get("loan_amount_requested", 0))
    existing_debts = float(ctx.get("existing_debts", 0))
    credit_score = int(ctx.get("bureau_credit_score", 600))
    years_employed = int(ctx.get("years_employed", 0))
    monthly_rent = float(ctx.get("monthly_rent_or_mortgage", 0))

    # Debt-to-income ratio
    monthly_income = income / 12
    monthly_debt_payments = (existing_debts / 36) + monthly_rent  # Estimate
    dti_ratio = (
        (monthly_debt_payments / monthly_income * 100) if monthly_income > 0 else 100
    )

    # Loan-to-income ratio
    lti_ratio = (loan_amount / income * 100) if income > 0 else 100

    # Risk score (0-100, lower is better)
    risk_score = 50  # Base
    risk_score -= min(20, (credit_score - 600) / 10)  # Credit boost
    risk_score += max(0, dti_ratio - 30)  # DTI penalty
    risk_score -= min(10, years_employed * 2)  # Employment stability
    risk_score += max(0, lti_ratio - 40)  # High loan amount penalty
    risk_score = max(0, min(100, risk_score))

    # Determine approval path
    if risk_score <= 30 and credit_score >= 700:
        decision = "approved"
    elif risk_score <= 60:
        decision = "review"
    else:
        decision = "denied"

    result = {
        "risk_score": round(risk_score, 1),
        "dti_ratio": round(dti_ratio, 1),
        "lti_ratio": round(lti_ratio, 1),
        "approval_decision": decision,
        "max_approved_amount": loan_amount if decision != "denied" else 0,
        "suggested_rate_pct": round(5.5 + (risk_score / 20), 2),
    }
    print(f"  [Risk Assessment] DTI ratio: {result['dti_ratio']}%")
    print(f"  [Risk Assessment] Risk score: {result['risk_score']}/100")
    print(f"  [Risk Assessment] Decision: {decision.upper()}")
    print(f"  [Risk Assessment] Suggested rate: {result['suggested_rate_pct']}%")
    return result


def documentation_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Generate loan documentation package."""
    decision = ctx.get("approval_decision", "review")
    amount = ctx.get("max_approved_amount", 0)
    rate = ctx.get("suggested_rate_pct", 0)
    term = ctx.get("loan_term_months", 60)

    monthly_payment = 0
    if amount > 0 and rate > 0 and term > 0:
        monthly_rate = rate / 100 / 12
        monthly_payment = (
            amount
            * (monthly_rate * (1 + monthly_rate) ** term)
            / ((1 + monthly_rate) ** term - 1)
        )

    docs = {
        "doc_package_id": "DOC-2024-LN-3347",
        "documents_generated": [
            "loan_agreement.pdf",
            "truth_in_lending_disclosure.pdf",
            "privacy_notice.pdf",
        ],
        "monthly_payment": round(monthly_payment, 2),
        "total_repayment": round(monthly_payment * term, 2),
    }
    if decision == "review":
        docs["documents_generated"].append("manual_review_request.pdf")

    print(f"  [Documentation] Package: {docs['doc_package_id']}")
    print(f"  [Documentation] Monthly payment: ${docs['monthly_payment']:.2f}")
    print(f"  [Documentation] Documents: {len(docs['documents_generated'])} generated")
    return docs


def notification_action(ctx: dict[str, Any]) -> dict[str, Any]:
    """Send notification to applicant and internal teams."""
    decision = ctx.get("approval_decision", "unknown")
    email = ctx.get("applicant_email", "unknown")
    name = ctx.get("applicant_name", "Applicant")

    print(f"\n  [Notification] Sending {decision} notification to {email}")
    if decision == "approved":
        print(f"  [Notification] Congratulations {name}! Loan approved.")
    elif decision == "review":
        print(f"  [Notification] {name}, your application is under manual review.")
    else:
        print(f"  [Notification] {name}, we regret your application was not approved.")

    return {"notification_sent": True, "notification_channel": "email"}


def print_loan_summary(ctx: dict[str, Any]) -> None:
    """Print the final loan processing summary."""
    print("\n" + "=" * 60)
    print("LOAN PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Applicant:       {ctx.get('applicant_name', 'N/A')}")
    print(f"  Loan Amount:     ${ctx.get('loan_amount_requested', 0):,.2f}")
    print(f"  Loan Purpose:    {ctx.get('loan_purpose', 'N/A')}")
    print(f"  Credit Score:    {ctx.get('bureau_credit_score', 'N/A')}")
    print(f"  Risk Score:      {ctx.get('risk_score', 'N/A')}/100")
    print(f"  DTI Ratio:       {ctx.get('dti_ratio', 'N/A')}%")
    print(f"  Decision:        {ctx.get('approval_decision', 'N/A').upper()}")
    print(f"  Suggested Rate:  {ctx.get('suggested_rate_pct', 'N/A')}%")
    print(f"  Monthly Payment: ${ctx.get('monthly_payment', 0):,.2f}")
    print(f"  Doc Package:     {ctx.get('doc_package_id', 'N/A')}")
    print(f"  Notified:        {ctx.get('notification_sent', False)}")


# ------------------------------------------------------------------
# Build the workflow
# ------------------------------------------------------------------


def build_loan_workflow(model: str) -> WorkflowEngine:
    """Build the 6-step loan processing workflow."""
    intake_fsm = build_intake_fsm()

    workflow = create_workflow(
        "loan_processing",
        "Loan Processing Pipeline",
        "Process a loan application through intake, credit check, risk "
        "assessment, approval routing, documentation, and notification.",
    )

    # Step 1: Intake via FSM conversation
    workflow.with_initial_step(
        conversation_step(
            step_id="intake",
            name="Applicant Intake",
            fsm_definition=intake_fsm,
            model=model,
            auto_messages=[
                (
                    "Hi, I'm Sarah Mitchell, email sarah.mitchell@email.com. "
                    "I work full-time at Cascade Technologies Inc., "
                    "earning $92,000 per year, been there 4 years."
                ),
                (
                    "I'd like a personal loan of $35,000 for home renovation, "
                    "specifically kitchen remodel and bathroom upgrades. "
                    "I'd prefer a 60-month term. I currently have $12,500 "
                    "in existing debt from an auto loan."
                ),
                "Yes, that all looks correct. Please proceed with the application.",
            ],
            context_mapping={
                "applicant_name": "applicant_name",
                "applicant_email": "applicant_email",
                "annual_income": "annual_income",
                "employer_name": "employer_name",
                "loan_amount_requested": "loan_amount_requested",
                "loan_purpose": "loan_purpose",
                "loan_term_months": "loan_term_months",
                "existing_debts": "existing_debts",
            },
            success_state="credit_check",
            error_state="notify",
            description="Collect applicant info via FSM-driven conversation",
        )
    )

    # Step 2: Credit check
    workflow.with_step(
        auto_step(
            step_id="credit_check",
            name="Credit Bureau Check",
            next_state="risk_assessment",
            action=credit_check_action,
            description="Verify credit score with bureau and assess credit history",
        )
    )

    # Step 3: Risk assessment
    workflow.with_step(
        auto_step(
            step_id="risk_assessment",
            name="Risk Assessment",
            next_state="approval_decision",
            action=risk_assessment_action,
            description="Calculate risk score, DTI ratio, and determine approval path",
        )
    )

    # Step 4: Approval routing via switch
    workflow.with_step(
        switch_step(
            step_id="approval_decision",
            name="Approval Decision Router",
            key="approval_decision",
            cases={
                "approved": "documentation",
                "review": "documentation",
                "denied": "notify",
            },
            default_state="notify",
            description="Route to documentation or denial based on risk assessment",
        )
    )

    # Step 5: Documentation generation
    workflow.with_step(
        auto_step(
            step_id="documentation",
            name="Generate Documentation",
            next_state="notify",
            action=documentation_action,
            description="Generate loan agreement and disclosure documents",
        )
    )

    # Step 6: Notification
    workflow.with_step(
        auto_step(
            step_id="notify",
            name="Send Notification",
            next_state="complete",
            action=notification_action,
            description="Notify applicant and internal teams of decision",
        )
    )

    # Terminal step
    workflow.with_step(
        TerminalStep(
            step_id="complete",
            name="Processing Complete",
            action=lambda ctx: print_loan_summary(ctx),
            description="Final summary and workflow completion",
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
    print("Loan Processing Workflow")
    print("=" * 60)
    print(f"Model: {model}")
    print("This workflow will:")
    print("  1. Collect applicant info via FSM conversation (auto-driven)")
    print("  2. Run credit bureau check")
    print("  3. Perform risk assessment and scoring")
    print("  4. Route approval decision (approved / review / denied)")
    print("  5. Generate loan documentation")
    print("  6. Send notification to applicant")
    print()

    try:
        engine = build_loan_workflow(model)

        # Seed context with data the FSM may not extract
        initial_context = {
            "credit_score_self_reported": LOAN_APPLICATION[
                "credit_score_self_reported"
            ],
            "bankruptcy_history": LOAN_APPLICATION["bankruptcy_history"],
            "previous_loan_repaid_on_time": LOAN_APPLICATION[
                "previous_loan_repaid_on_time"
            ],
            "years_employed": LOAN_APPLICATION["years_employed"],
            "monthly_rent_or_mortgage": LOAN_APPLICATION["monthly_rent_or_mortgage"],
            "applicant_email": LOAN_APPLICATION["applicant_email"],
        }

        instance_id = await engine.start_workflow(
            "loan_processing", initial_context=initial_context
        )
        print(f"\nWorkflow started: {instance_id}")

        instance = engine.get_workflow_instance(instance_id)
        if instance:
            print(f"\nFinal status: {instance.status.value}")
            context_keys = sorted(k for k in instance.context if not k.startswith("_"))
            print(f"Context keys: {context_keys}")
    except Exception as e:
        print(f"Workflow error: {e}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
