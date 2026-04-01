"""
Legal Document Review -- ReactAgent with Regulatory Tools
=========================================================

Demonstrates a ReAct agent that performs a multi-criteria legal
contract review. The agent uses four specialized tools to search
regulations, analyze individual clauses, verify overall compliance,
and extract named entities from legal text.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/legal_document_review/run.py
"""

import os
from typing import Annotated

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool


@tool
def search_regulations(
    jurisdiction: Annotated[str, "Jurisdiction to search (e.g. EU, US-federal, UK)"],
    topic: Annotated[str, "Regulatory topic to look up"],
) -> str:
    """Search regulatory databases for applicable laws and standards."""
    j = jurisdiction.lower()
    t = topic.lower()

    regulations = {
        ("eu", "data"): (
            "GDPR (Regulation 2016/679): Requires lawful basis for processing personal data. "
            "Article 6 lists six lawful bases. Article 28 mandates written data processing "
            "agreements with sub-processors. Penalties up to 4% of annual global turnover."
        ),
        ("eu", "liability"): (
            "EU Product Liability Directive 85/374/EEC: Strict liability for defective products. "
            "Burden of proof on injured party for damage, defect, and causal link. "
            "10-year limitation period from product circulation."
        ),
        ("us", "intellectual"): (
            "US Copyright Act 17 USC Sec.101-810: Copyright protects original works of authorship. "
            "Work-for-hire doctrine: employer owns works created within scope of employment. "
            "Fair use factors codified in Section 107."
        ),
        ("us", "employ"): (
            "FLSA (Fair Labor Standards Act): Minimum wage, overtime pay (1.5x after 40 hrs/week). "
            "At-will employment default in most states. Non-compete enforceability varies by state. "
            "California generally prohibits non-competes (Business and Professions Code Sec.16600)."
        ),
        ("uk", "contract"): (
            "UK Contracts Act 1999 (Third Party Rights): Third parties may enforce terms if contract "
            "expressly provides for it. Unfair Contract Terms Act 1977: Cannot exclude liability for "
            "negligence causing death or personal injury."
        ),
        ("eu", "force"): (
            "EU commercial practice: Force majeure clauses must be explicitly defined. "
            "COVID-19 jurisprudence (2020-2023) established that pandemics qualify only when "
            "the clause explicitly lists epidemics or government-mandated shutdowns."
        ),
    }

    for (jk, tk), value in regulations.items():
        if jk in j and tk in t:
            return value

    return (
        f"Regulatory search for '{topic}' in {jurisdiction}: General commercial law applies. "
        f"Recommend consulting local counsel for jurisdiction-specific requirements."
    )


@tool
def check_clause(
    clause_type: Annotated[
        str, "Type of clause (e.g. indemnification, termination, IP)"
    ],
    clause_text: Annotated[str, "Summary or key terms of the clause to analyze"],
) -> str:
    """Analyze a specific contract clause for risks and enforceability."""
    ct = clause_type.lower()

    analyses = {
        "indemnif": (
            "Indemnification clause analysis: MEDIUM RISK. Mutual indemnification is standard. "
            "Watch for: (1) uncapped liability exposure, (2) broad 'arising from' language that "
            "could include third-party claims, (3) missing carve-outs for gross negligence. "
            "Recommendation: Cap at 2x annual contract value, add knowledge qualifier."
        ),
        "terminat": (
            "Termination clause analysis: LOW RISK. Standard 30-day notice for convenience is "
            "acceptable. For-cause termination should include cure period (typically 30 days). "
            "Check: (1) survival provisions for IP and confidentiality, (2) data return/deletion "
            "obligations, (3) pro-rata refund for pre-paid fees."
        ),
        "ip": (
            "Intellectual property clause analysis: HIGH RISK. Assignment vs license distinction "
            "is critical. Work-for-hire provisions may not apply to independent contractors in all "
            "jurisdictions. Pre-existing IP must be explicitly excluded via scheduled exhibits. "
            "Recommendation: Use broad license grant instead of assignment where possible."
        ),
        "confident": (
            "Confidentiality clause analysis: LOW RISK. Standard NDA terms (2-3 year duration) "
            "are enforceable. Ensure: (1) carve-outs for publicly available information, "
            "(2) compelled disclosure exceptions, (3) return/destruction obligations at "
            "termination. Residual knowledge clauses are increasingly common and acceptable."
        ),
        "limit": (
            "Limitation of liability analysis: HIGH RISK. Aggregate liability cap is essential. "
            "Industry standard: 12 months of fees paid or payable. Exceptions typically carved "
            "out for: IP infringement, confidentiality breach, gross negligence, willful misconduct. "
            "Consequential damages exclusion should be mutual."
        ),
        "force": (
            "Force majeure clause analysis: MEDIUM RISK. Post-COVID, specificity is key. "
            "Clause should enumerate qualifying events (pandemic, war, government action, "
            "natural disaster). Notice period (typically 48-72 hours) and mitigation obligations "
            "should be defined. Extended force majeure (>90 days) should trigger termination right."
        ),
    }

    for key, value in analyses.items():
        if key in ct:
            return value

    return (
        f"Clause analysis for '{clause_type}': Standard commercial terms detected. "
        f"Review against jurisdiction-specific requirements recommended."
    )


@tool
def verify_compliance(
    contract_summary: Annotated[str, "High-level summary of the contract terms"],
    regulations: Annotated[str, "Applicable regulations to check against"],
) -> str:
    """Verify contract compliance against identified regulatory requirements."""
    s = contract_summary.lower()
    r = regulations.lower()

    findings = []
    if "gdpr" in r or "data" in r:
        if "processing" in s or "personal" in s or "data" in s:
            findings.append(
                "DATA PROTECTION: Contract must include Article 28 GDPR standard "
                "clauses for data processing agreements. Verify sub-processor list "
                "is annexed and notification mechanism is specified."
            )
        else:
            findings.append(
                "DATA PROTECTION: No personal data processing detected. "
                "GDPR DPA provisions may not be required."
            )

    if "employ" in r or "labor" in r:
        findings.append(
            "EMPLOYMENT: Non-compete scope should be limited to 12 months max "
            "and geographically reasonable. Garden leave provisions recommended "
            "for enforceability in EU jurisdictions."
        )

    if "ip" in r or "copyright" in r or "intellectual" in r:
        findings.append(
            "INTELLECTUAL PROPERTY: Verify ownership assignment chain is complete. "
            "Open-source license compatibility should be addressed if software is involved. "
            "Patent indemnification obligations should be reciprocal."
        )

    if not findings:
        findings.append(
            "GENERAL: No specific regulatory conflicts identified. "
            "Standard commercial terms appear compliant with general contract law."
        )

    return "COMPLIANCE REPORT:\n" + "\n".join(f"  - {f}" for f in findings)


@tool
def extract_entities(
    document_text: Annotated[str, "Contract text or summary to extract entities from"],
) -> str:
    """Extract key legal entities, dates, and monetary values from contract text."""
    t = document_text.lower()

    entities = []
    # Parties
    if "acme" in t or "corp" in t:
        entities.append("PARTY: Acme Corporation (Delaware C-Corp, EIN: 12-3456789)")
    if "globex" in t or "partner" in t or "vendor" in t:
        entities.append("PARTY: Globex Industries Ltd (UK Ltd, Company No: 09876543)")
    if not any("PARTY" in e for e in entities):
        entities.append("PARTY: [Contracting parties identified in preamble]")

    # Dates
    if "2024" in t or "2025" in t or "2026" in t:
        entities.append("DATE: Effective date referenced in agreement")
    entities.append("DATE: Initial term 24 months from execution date")
    entities.append("DATE: Renewal: Auto-renew for successive 12-month periods")

    # Values
    if "million" in t or "$" in t or "fee" in t or "payment" in t:
        entities.append("VALUE: Base annual fee $2.4M, payable quarterly ($600K)")
        entities.append("VALUE: Performance bonus up to $360K (15% of base)")
    else:
        entities.append("VALUE: [Monetary terms to be identified from schedules]")

    # Governing law
    if "eu" in t or "uk" in t or "england" in t:
        entities.append("JURISDICTION: England and Wales, London arbitration (LCIA)")
    else:
        entities.append("JURISDICTION: State of Delaware, USA (default)")

    return "EXTRACTED ENTITIES:\n" + "\n".join(f"  - {e}" for e in entities)


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Register tools
    registry = ToolRegistry()
    registry.register(search_regulations._tool_definition)
    registry.register(check_clause._tool_definition)
    registry.register(verify_compliance._tool_definition)
    registry.register(extract_entities._tool_definition)

    config = AgentConfig(model=model, max_iterations=10, temperature=0.5)
    agent = ReactAgent(tools=registry, config=config)

    task = (
        "You are a senior legal analyst reviewing a cross-border technology services "
        "agreement between Acme Corporation (US, Delaware) and Globex Industries Ltd "
        "(UK). The contract governs a $2.4 million annual software development and "
        "data processing engagement effective January 2026, with auto-renewal for "
        "successive 12-month periods. The engagement includes a dedicated offshore "
        "development team of 18 engineers located in the UK, Poland, and India, "
        "processing EU citizen personal data across all three jurisdictions. Please "
        "conduct a thorough legal review covering the following criteria:\n\n"
        "1. DATA PROTECTION: Search EU regulations for GDPR data processing "
        "requirements. The contract involves processing EU citizen personal data "
        "including names, email addresses, and behavioral analytics. Sub-processing "
        "occurs across three jurisdictions (UK post-Brexit adequacy, Poland intra-EU, "
        "India via Standard Contractual Clauses).\n\n"
        "2. INTELLECTUAL PROPERTY: Check the IP assignment clause. The contract states "
        "all deliverables are work-for-hire with full assignment to Acme. Globex retains "
        "rights to pre-existing frameworks and open-source components listed in Exhibit C. "
        "The contract is silent on jointly developed inventions and does not address "
        "patent filing rights or open-source license contamination risks.\n\n"
        "3. INDEMNIFICATION: Analyze the mutual indemnification clause. Current terms "
        "include uncapped liability for IP infringement and a 3x annual fee cap for "
        "all other claims. The indemnification trigger uses broad 'arising from or "
        "related to' language without a knowledge qualifier. Defense and settlement "
        "control provisions favor Acme exclusively.\n\n"
        "4. LIMITATION OF LIABILITY: Review the liability cap structure. Consequential "
        "damages are excluded except for confidentiality breaches and IP infringement. "
        "The aggregate cap is set at the greater of $5M or 2x annual fees. There are "
        "no separate super-caps for data breach or regulatory fines.\n\n"
        "5. TERMINATION: The contract allows 60-day termination for convenience and "
        "immediate termination for material breach with a 30-day cure period. "
        "Transition assistance is limited to 90 days post-termination at standard "
        "rates. Data return and deletion timelines are not specified.\n\n"
        "6. FORCE MAJEURE: Review the force majeure clause for post-COVID adequacy. "
        "Current language references 'acts of God' and 'government action' but does "
        "not specifically mention pandemics, supply chain disruptions, or sanctions. "
        "The clause lacks a notice period, mitigation obligations, and a termination "
        "trigger for extended force majeure events.\n\n"
        "7. COMPLIANCE: Verify overall compliance with both US federal IP law and EU "
        "data protection regulations. Flag any cross-border enforcement risks.\n\n"
        "8. ENTITY EXTRACTION: Extract all key parties, dates, monetary values, "
        "and jurisdictional references from the contract summary above.\n\n"
        "Provide a consolidated risk assessment with HIGH/MEDIUM/LOW ratings for each "
        "area and specific recommendations for contract modifications."
    )

    print("=" * 60)
    print("Legal Document Review -- ReactAgent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nLegal Review:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")
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
        "tools_called": len(result.tools_used) > 0,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
