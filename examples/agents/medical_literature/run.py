"""
Medical Literature Review -- ADaPTAgent
========================================

Demonstrates an ADaPT (Adaptive Decomposition and Planning) agent
performing a systematic literature review. The agent adaptively
decomposes the research question, searches PubMed for relevant
studies, analyzes methodology and findings, and synthesizes a
structured summary with evidence grading.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/medical_literature/run.py
"""

import os
from typing import Annotated

from fsm_llm_agents import ADaPTAgent, AgentConfig, ToolRegistry, tool


@tool
def search_pubmed(
    query: Annotated[str, "Search query for PubMed database"],
    filters: Annotated[str, "Filters: date range, study type, language"],
) -> str:
    """Search PubMed for relevant medical literature matching the query."""
    q = query.lower()

    results = {
        "glp-1": (
            "PubMed results for GLP-1 receptor agonists (2020-2026):\n"
            "  Total: 4,287 articles | Filtered: 312 (RCTs + meta-analyses)\n\n"
            "  1. Marso et al. (2016) LEADER trial -- Liraglutide CVD outcomes\n"
            "     N=9,340 | Follow-up: 3.8y | HR 0.87 (95% CI 0.78-0.97)\n"
            "     Primary endpoint: Major adverse cardiovascular events (MACE)\n\n"
            "  2. Gerstein et al. (2019) REWIND trial -- Dulaglutide CVD\n"
            "     N=9,901 | Follow-up: 5.4y | HR 0.88 (95% CI 0.79-0.99)\n"
            "     First trial with majority primary prevention population\n\n"
            "  3. Husain et al. (2019) PIONEER 6 -- Oral semaglutide safety\n"
            "     N=3,183 | Follow-up: 1.3y | HR 0.79 (95% CI 0.57-1.11)\n"
            "     Non-inferior for MACE, trend toward benefit\n\n"
            "  4. Lincoff et al. (2023) SELECT trial -- Semaglutide in obesity\n"
            "     N=17,604 | Follow-up: 3.3y | HR 0.80 (95% CI 0.72-0.90)\n"
            "     First CVD outcome benefit in non-diabetic obese patients"
        ),
        "weight": (
            "PubMed results for weight loss pharmacotherapy (2020-2026):\n"
            "  Total: 2,156 articles | Filtered: 189 (RCTs)\n\n"
            "  1. Wilding et al. (2021) STEP 1 -- Semaglutide 2.4mg obesity\n"
            "     N=1,961 | Duration: 68 weeks | Weight loss: -14.9% vs -2.4%\n"
            "     NNT for >10% weight loss: 2.6\n\n"
            "  2. Jastreboff et al. (2022) SURMOUNT-1 -- Tirzepatide obesity\n"
            "     N=2,539 | Duration: 72 weeks | Weight loss: up to -20.9%\n"
            "     Dual GIP/GLP-1 agonist, superior to semaglutide in indirect comparison\n\n"
            "  3. Aronne et al. (2024) SURMOUNT-MMO -- Tirzepatide + MACE\n"
            "     N=15,000 (ongoing) | Interim: promising CVD signal\n"
            "     Results expected 2027"
        ),
        "kidney": (
            "PubMed results for GLP-1 agonists and renal outcomes (2020-2026):\n"
            "  Total: 876 articles | Filtered: 64 (RCTs + cohort studies)\n\n"
            "  1. Mann et al. (2017) LEADER renal sub-analysis\n"
            "     Composite renal outcome HR 0.78 (95% CI 0.67-0.92)\n"
            "     Driven by reduction in new-onset macroalbuminuria\n\n"
            "  2. Tuttle et al. (2021) FLOW trial design -- Semaglutide CKD\n"
            "     N=3,533 | First dedicated renal outcome trial for GLP-1 RA\n"
            "     Primary: Time to kidney failure, >50% eGFR decline, or renal death\n\n"
            "  3. Perkovic et al. (2024) FLOW trial results\n"
            "     HR 0.76 (95% CI 0.66-0.88) | Stopped early for efficacy\n"
            "     Semaglutide reduced kidney failure risk by 24%"
        ),
        "safety": (
            "PubMed results for GLP-1 RA safety profile (2020-2026):\n"
            "  Total: 1,543 articles | Filtered: 98 (safety analyses)\n\n"
            "  1. Bethel et al. (2020) Pooled safety analysis (N=30,820)\n"
            "     GI adverse events: nausea 15-20%, vomiting 5-9%, diarrhea 8-12%\n"
            "     Pancreatitis: No increased risk (OR 0.93, 95% CI 0.65-1.34)\n"
            "     Thyroid cancer: No signal in humans (rat data not replicated)\n\n"
            "  2. Sodhi et al. (2023) Retrospective cohort (N=16M records)\n"
            "     Gastroparesis: OR 3.67 (95% CI 1.15-11.90) -- emerging signal\n"
            "     Bowel obstruction: OR 4.22 (95% CI 1.02-17.40) -- rare but noted\n"
            "     Biliary disease: OR 1.53 (95% CI 1.23-1.89) -- gallstones risk"
        ),
        "meta-analysis": (
            "PubMed results for GLP-1 RA meta-analyses (2020-2026):\n"
            "  Total: 287 meta-analyses\n\n"
            "  1. Sattar et al. (2021) MACE meta-analysis (8 CVOTs, N=60,080)\n"
            "     MACE HR 0.86 (95% CI 0.80-0.93)\n"
            "     All-cause mortality HR 0.88 (95% CI 0.82-0.94)\n"
            "     Evidence quality: HIGH (GRADE)\n\n"
            "  2. Palmer et al. (2022) Kidney outcomes meta-analysis\n"
            "     Composite renal HR 0.79 (95% CI 0.73-0.87)\n"
            "     Albuminuria reduction HR 0.74 (95% CI 0.67-0.81)\n"
            "     Evidence quality: MODERATE (heterogeneity in definitions)"
        ),
    }

    for key, value in results.items():
        if key in q:
            return value
    return (
        f"PubMed search for '{query}': 45 results found. Refine query for specificity."
    )


@tool
def analyze_study(
    study_reference: Annotated[str, "Study name, author, or PMID to analyze"],
    focus: Annotated[str, "Analysis focus: methodology, results, bias, applicability"],
) -> str:
    """Critically appraise a study's methodology, results, and quality."""
    s = study_reference.lower()
    f = focus.lower()

    if "leader" in s or "marso" in s:
        if "method" in f:
            return (
                "STUDY APPRAISAL -- LEADER Trial (Marso et al. 2016)\n"
                "Design: Multicenter, double-blind, placebo-controlled RCT\n"
                "Randomization: Adequate (computer-generated, stratified by site)\n"
                "Allocation concealment: Adequate (central randomization system)\n"
                "Blinding: Double-blind (matching placebo, identical injection pens)\n"
                "ITT analysis: Yes, modified ITT (99.7% vital status ascertained)\n"
                "Sample size: 9,340 (powered for non-inferiority margin of 1.3)\n"
                "Attrition: 2.3% lost to follow-up (acceptable)\n"
                "Jadad score: 5/5 | Cochrane Risk of Bias: LOW across all domains"
            )
        return (
            "RESULTS ANALYSIS -- LEADER Trial\n"
            "Primary endpoint (MACE): HR 0.87 (95% CI 0.78-0.97, p=0.01)\n"
            "Components: CV death HR 0.78, non-fatal MI HR 0.88, non-fatal stroke HR 0.89\n"
            "NNT (MACE): 66 over 3.8 years\n"
            "HbA1c reduction: -0.4% vs placebo\n"
            "Weight reduction: -2.3 kg vs placebo\n"
            "Internal validity: STRONG | External validity: MODERATE\n"
            "Limitation: Enriched high-risk population, may not generalize to primary prevention"
        )
    elif "select" in s or "lincoff" in s:
        return (
            "STUDY APPRAISAL -- SELECT Trial (Lincoff et al. 2023)\n"
            "Design: Phase III, double-blind, placebo-controlled RCT\n"
            "Population: BMI >= 27, established CVD, NO diabetes\n"
            "Primary endpoint (MACE): HR 0.80 (95% CI 0.72-0.90, p<0.001)\n"
            "Absolute risk reduction: 1.5% over 3.3 years\n"
            "NNT: 67 | Weight loss: -9.4% vs -0.9%\n"
            "Significance: First CVD outcome benefit for obesity pharmacotherapy\n"
            "   independent of glycemic effects\n"
            "Risk of bias: LOW | GRADE evidence quality: HIGH\n"
            "Limitation: 90% male, 84% White -- limited diversity"
        )
    elif "step" in s or "wilding" in s:
        return (
            "STUDY APPRAISAL -- STEP 1 Trial (Wilding et al. 2021)\n"
            "Design: Phase III, double-blind, placebo-controlled RCT\n"
            "Population: BMI >= 30 (or >= 27 with comorbidity), non-diabetic\n"
            "Primary endpoint: % change in body weight at 68 weeks\n"
            "Result: -14.9% vs -2.4% (estimated treatment difference -12.4%)\n"
            "Responders: 86% achieved >5% weight loss, 69% achieved >10%\n"
            "Dropout: 7% semaglutide vs 3.6% placebo (GI adverse events)\n"
            "Weight regain after discontinuation: ~67% regained within 1 year\n"
            "Risk of bias: LOW | Clinical significance: HIGH"
        )
    return (
        f"Study analysis for '{study_reference}' (focus: {focus}): "
        "Peer-reviewed publication with standard methodology. "
        "Recommend detailed assessment against CONSORT checklist."
    )


@tool
def summarize_findings(
    topic: Annotated[str, "Research topic to summarize"],
    evidence: Annotated[str, "Key evidence points collected from studies"],
) -> str:
    """Synthesize research findings into a structured evidence summary."""
    t = topic.lower()

    if "cardiovascular" in t or "cvd" in t or "mace" in t:
        return (
            "EVIDENCE SYNTHESIS -- GLP-1 RA Cardiovascular Benefits\n"
            "Evidence Level: 1A (multiple high-quality RCTs + meta-analyses)\n\n"
            "Conclusion: GLP-1 receptor agonists reduce MACE by 14% (HR 0.86, "
            "95% CI 0.80-0.93) in patients with T2DM and established CVD.\n"
            "The SELECT trial extends this benefit to non-diabetic obese patients "
            "(HR 0.80, 95% CI 0.72-0.90).\n\n"
            "Mechanism: Likely multifactorial -- anti-inflammatory, anti-atherogenic, "
            "weight loss, blood pressure reduction, lipid improvement.\n"
            "Effect appears class-wide but strongest for semaglutide and liraglutide.\n\n"
            "Clinical implication: GLP-1 RAs should be considered as first-line "
            "therapy for T2DM patients with CVD, independent of glycemic control needs.\n"
            "Guideline concordance: ADA 2024, ESC 2023, AHA 2024 -- all recommend."
        )
    elif "safety" in t or "adverse" in t:
        return (
            "EVIDENCE SYNTHESIS -- GLP-1 RA Safety Profile\n"
            "Evidence Level: 1B (RCTs + large observational studies)\n\n"
            "GI effects: Most common (nausea 15-20%), dose-dependent, attenuate "
            "over 4-8 weeks. Slow titration mitigates in >80% of patients.\n"
            "Pancreatitis: No increased risk in pooled RCT data (N>60,000).\n"
            "Thyroid: C-cell tumors in rodents not replicated in human data. "
            "Contraindicated in MEN2 and personal/family history of MTC.\n"
            "Emerging signals: Gastroparesis (OR 3.67), biliary disease (OR 1.53) "
            "require monitoring but are rare.\n\n"
            "Overall: Favorable benefit-risk profile for indicated populations."
        )
    elif "weight" in t or "obesity" in t:
        return (
            "EVIDENCE SYNTHESIS -- GLP-1 RA Weight Management\n"
            "Evidence Level: 1A (multiple Phase III RCTs)\n\n"
            "Semaglutide 2.4mg: -14.9% body weight (STEP 1), sustained at 2 years.\n"
            "Tirzepatide (dual GIP/GLP-1): up to -20.9% (SURMOUNT-1).\n"
            "Clinically meaningful: 70-90% achieve >5%, 50-70% achieve >10%.\n\n"
            "Key limitation: Weight regain of ~67% within 1 year of discontinuation.\n"
            "Implication: Long-term treatment likely required, cost-effectiveness "
            "analysis needed for sustained therapy.\n"
            "Guideline position: FDA-approved, recommended as adjunct to lifestyle "
            "modification for BMI >= 30 (or >= 27 with comorbidity)."
        )
    return (
        f"Evidence synthesis for '{topic}': Mixed evidence base. "
        "Recommend systematic review methodology for comprehensive assessment."
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Register tools
    registry = ToolRegistry()
    registry.register(search_pubmed._tool_definition)
    registry.register(analyze_study._tool_definition)
    registry.register(summarize_findings._tool_definition)

    config = AgentConfig(model=model, max_iterations=6, temperature=0.5)
    agent = ADaPTAgent(tools=registry, config=config, max_depth=1)

    task = (
        "Review GLP-1 receptor agonist cardiovascular benefits. "
        "Search PubMed for GLP-1 trials, analyze the LEADER and SELECT trials, "
        "then summarize the cardiovascular evidence."
    )

    print("=" * 60)
    print("Medical Literature Review -- ADaPTAgent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Max depth: 1")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nLiterature Review:\n{result.answer}")
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
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
