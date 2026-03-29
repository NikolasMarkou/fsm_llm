"""
Regulatory Compliance with Maker-Checker
==========================================

Demonstrates the MakerChecker pattern for regulatory compliance report
generation and review. The "maker" drafts a compliance assessment covering
multiple regulatory frameworks, while the "checker" reviews the report
against strict completeness and accuracy criteria.

This simulates a real-world compliance workflow where a compliance officer
drafts findings and a senior auditor reviews before submission.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/regulatory_compliance/run.py
"""

import os

from fsm_llm_agents import AgentConfig, MakerCheckerAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=20,
        temperature=0.3,
    )

    agent = MakerCheckerAgent(
        maker_instructions=(
            "Draft a comprehensive regulatory compliance report for a fintech "
            "company operating across the United States and European Union. "
            "The report must cover the following 6 regulatory areas in detail:\n"
            "1. DATA PRIVACY (GDPR & CCPA): Assess data collection practices, "
            "consent mechanisms, data subject rights implementation, cross-border "
            "data transfer safeguards, data retention policies, and breach "
            "notification procedures.\n"
            "2. FINANCIAL REGULATIONS (PCI-DSS & SOX): Evaluate payment card "
            "data handling, encryption standards, access controls, audit trails, "
            "internal controls over financial reporting, and segregation of duties.\n"
            "3. ANTI-MONEY LAUNDERING (AML/KYC): Review customer due diligence "
            "procedures, transaction monitoring systems, suspicious activity "
            "reporting workflows, sanctions screening, and beneficial ownership "
            "verification.\n"
            "4. CONSUMER PROTECTION (CFPB & FCA): Assess fair lending practices, "
            "transparent fee disclosure, complaint handling procedures, vulnerable "
            "customer policies, and marketing compliance.\n"
            "5. OPERATIONAL RESILIENCE (DORA & OCC Guidelines): Evaluate business "
            "continuity plans, third-party risk management, incident response "
            "procedures, recovery time objectives, and cyber resilience testing.\n"
            "6. AI/ALGORITHMIC GOVERNANCE (EU AI Act & SR 11-7): Review model risk "
            "management framework, algorithmic bias testing, explainability "
            "requirements, human oversight mechanisms, and model validation processes.\n\n"
            "For each area, provide: current compliance status (Compliant, Partially "
            "Compliant, Non-Compliant), specific findings with evidence references, "
            "identified gaps, remediation actions with timelines, and risk rating "
            "(Low/Medium/High/Critical)."
        ),
        checker_instructions=(
            "Review the compliance report as a senior regulatory auditor. "
            "Score 0.0-1.0 based on these criteria:\n"
            "- COVERAGE: All 6 regulatory areas addressed in detail (+0.20)\n"
            "- SPECIFICITY: Findings include specific controls, evidence "
            "references, and regulatory citation (+0.15)\n"
            "- RISK RATINGS: Each area has a clear compliance status and risk "
            "rating with justification (+0.15)\n"
            "- REMEDIATION: Actionable remediation steps with realistic timelines "
            "for each finding (+0.15)\n"
            "- CONSISTENCY: No contradictions between sections, consistent "
            "terminology, and logical flow (+0.10)\n"
            "- COMPLETENESS: Cross-border implications addressed, no major "
            "regulatory requirements omitted (+0.15)\n"
            "- PROFESSIONAL TONE: Suitable for submission to board of directors "
            "and regulatory bodies (+0.10)\n"
            "Set checker_passed=true only if quality_score >= 0.7. "
            "Provide specific feedback citing which sections need improvement "
            "and what is missing from each."
        ),
        config=config,
        max_revisions=3,
        quality_threshold=0.7,
    )

    task = (
        "Conduct a full regulatory compliance assessment for PayStream Financial "
        "Technologies, a Series C fintech startup ($85M raised, valued at $420M) that "
        "provides a digital payments platform and AI-powered lending product. PayStream "
        "operates in 12 US states and 5 EU countries (Germany, France, Netherlands, "
        "Ireland, Spain), processing $2.1B in annual transaction volume with 1.8 million "
        "active users and 45,000 merchant partners. The company employs 320 people "
        "including a 12-person compliance team and a 5-person internal audit function. "
        "The technology stack processes 150,000 daily transactions through AWS-hosted "
        "microservices with PCI-DSS Level 1 certification. The company recently expanded "
        "its product line to include: (1) instant peer-to-peer payments via mobile app "
        "with real-time fraud scoring, (2) AI-driven credit scoring for small business "
        "loans up to $250K using a gradient-boosted decision tree model trained on 2.3M "
        "historical applications, (3) a cryptocurrency exchange integration allowing "
        "users in 3 EU markets to buy/sell Bitcoin and Ethereum with fiat on-ramp, and "
        "(4) a buy-now-pay-later (BNPL) feature for e-commerce partners with 4-installment "
        "plans up to $1,500. Recent events requiring immediate attention: a data breach in "
        "Q2 affecting 12,000 EU customers where encrypted PII was exfiltrated through a "
        "misconfigured API gateway (resolved, but the Irish DPC has opened a formal "
        "investigation and GDPR Article 33 notification was filed 48 hours late), a "
        "pending CFPB inquiry into fee transparency for the BNPL product after 340 "
        "consumer complaints about undisclosed late fees, a model validation audit "
        "finding that the AI credit scoring model shows 8% higher rejection rates for "
        "applicants in predominantly minority zip codes (potential fair lending violation "
        "under ECOA), and a third-party vendor risk assessment revealing that 3 of 12 "
        "critical vendors lack SOC 2 Type II certification. The board has requested this "
        "compliance report ahead of a planned Series D fundraise ($150M target) and "
        "potential IPO filing in 18 months. The report must be comprehensive enough to "
        "satisfy due diligence requirements from institutional investors, demonstrate a "
        "credible path to full compliance across all jurisdictions, and provide a "
        "prioritized remediation roadmap with cost estimates and timeline milestones."
    )

    print("=" * 60)
    print("Regulatory Compliance — Maker-Checker Pattern")
    print("=" * 60)
    print(f"Model: {model}")
    print("Max revisions: 3")
    print("Quality threshold: 0.7")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nCompliance Report:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        revision_count = result.final_context.get("revision_count", 0)
        checker_passed = result.final_context.get("checker_passed", False)
        quality_score = result.final_context.get("quality_score", 0)

        print(f"Revisions: {revision_count}")
        print(f"Checker passed: {checker_passed}")
        print(f"Quality score: {quality_score}")

        feedback = result.final_context.get("checker_feedback", "")
        if feedback:
            print(f"Auditor feedback: {str(feedback)[:300]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
