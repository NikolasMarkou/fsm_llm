# src/llm_fsm_reasoning/reasoning_fsms_python.py
"""
Python dictionary definitions for all reasoning FSMs.
This replaces the individual JSON files in the 'fsms' directory.
"""

# Orchestrator FSM
orchestrator_fsm = {
  "name": "reasoning_orchestrator",
  "description": "Orchestrates various reasoning strategies to solve complex problems.",
  "initial_state": "problem_analysis",
  "persona": "You are a reasoning guide helping to solve problems step by step. Be clear, logical, and thorough.",
  "states": {
    "problem_analysis": {
      "id": "problem_analysis",
      "description": "Initial analysis of the problem to determine its nature and components.",
      "purpose": "Analyze the 'problem_statement' from context. Identify its 'problem_type' (e.g., arithmetic, logical, design), its main 'problem_components', and any 'constraints'.",
      "required_context_keys": ["problem_type", "problem_components"],
      "instructions": "Given the 'problem_statement' in the context, break it down. Set 'problem_type', 'problem_components' (as a list or dict), and 'constraints' (if any, as a list or dict) in your `context_update`. If the problem is simple like '1+1', 'problem_components' could be `{'operand1': 1, 'operand2': 1, 'operator': 'addition'}` and 'problem_type' could be 'arithmetic'. Aim to transition to 'strategy_selection' once these are identified.",
      "transitions": [{
        "target_state": "strategy_selection",
        "description": "Problem has been analyzed, and its type, components (and optionally constraints) are identified in the context.",
        "priority": 1,
        "conditions": [
          {
            "description": "'problem_type' and 'problem_components' must be present in the context.",
            "requires_context_keys": ["problem_type", "problem_components"]
          }
        ]
      }]
    },
    "strategy_selection": {
      "id": "strategy_selection",
      "description": "Selection of the most appropriate reasoning strategy.",
      "purpose": "Select the best reasoning strategy based on the problem analysis. If 'problem_type' is 'arithmetic' or clearly a simple calculation, choose 'direct computation'. Otherwise, use 'classified_problem_type' or choose from analytical, deductive, inductive, creative, critical, hybrid.",
      "required_context_keys": ["reasoning_strategy", "strategy_rationale"],
      "instructions": "Based on 'problem_type' and 'classified_problem_type': if it's simple arithmetic (e.g., '1+1'), set 'reasoning_strategy' to 'direct computation'. Otherwise, choose the most fitting strategy from the available types (analytical, deductive, etc.) or use the 'classified_problem_type'. Justify your choice in 'strategy_rationale'.",
      "transitions": [{
        "target_state": "execute_reasoning",
        "description": "Strategy selected and ready to begin reasoning"
      }]
    },
    "execute_reasoning": {
      "id": "execute_reasoning",
      "description": "Execution phase where the selected reasoning strategy is applied.",
      "purpose": "Guide through the selected reasoning process. This is where the main thinking happens.",
      "instructions": "Apply the selected reasoning strategy to work through the problem systematically.",
      "transitions": [{
        "target_state": "synthesize_solution",
        "description": "Reasoning process completed"
      }]
    },
    "synthesize_solution": {
      "id": "synthesize_solution",
      "description": "Synthesizing insights into a proposed solution.",
      "purpose": "Review all collected results from the reasoning strategy (e.g., 'calculation_result', 'integrated_analysis', etc., based on 'reasoning_type_selected'). Formulate a clear and concise 'proposed_solution'. If the problem was a simple calculation like '1+1' and 'calculation_result' contains the direct answer (e.g., 2), set 'proposed_solution' to this direct answer. Otherwise, compose a descriptive solution. Also populate 'key_insights'.",
      "required_context_keys": ["proposed_solution", "key_insights"],
      "instructions": "Combine insights and reasoning steps. If 'reasoning_type_selected' was 'simple_calculator' and 'calculation_result' in the context is a direct numerical answer, set 'proposed_solution' to this 'calculation_result'. Otherwise, formulate a coherent solution sentence and store it in 'proposed_solution'. Also populate 'key_insights'.",
      "transitions": [{
        "target_state": "validate_refine",
        "description": "Solution synthesized"
      }]
    },
    "validate_refine": {
      "id": "validate_refine",
      "description": "Validation and refinement of the proposed solution.",
      "purpose": "Review the 'validation_result' and 'solution_confidence' from the context. If the solution is valid, transition to 'final_answer'. If issues were found ('validation_result' is false) and refinement is needed, transition back to 'execute_reasoning'.",
      "required_context_keys": ["validation_result", "confidence_level"],
      "instructions": "Based on the 'validation_result' and 'solution_confidence' in the current context: if 'validation_result' is true, decide to transition to 'final_answer'. If 'validation_result' is false, decide to transition to 'execute_reasoning' to refine the solution.",
      "transitions": [
        {
          "target_state": "final_answer",
          "description": "Solution is validated and acceptable.",
          "priority": 1,
          "conditions": [
            {
              "description": "The 'validation_result' in context is true OR 'solution_valid' in context is true.",
              "logic": {
                "or": [
                    {"==": [{"var": "validation_result"}, True]},
                    {"==": [{"var": "solution_valid"}, True]}
                ]
              }
            }
          ]
        },
        {
          "target_state": "execute_reasoning",
          "description": "Issues were found in the solution, and it needs refinement.",
          "priority": 2,
          "conditions": [
            {
              "description": "The 'validation_result' in context is false OR 'solution_valid' in context is false.",
               "logic": {
                "or": [
                    {"==": [{"var": "validation_result"}, False]},
                    {"==": [{"var": "solution_valid"}, False]}
                ]
              }
            }
          ]
        }
      ]
    },
    "final_answer": {
      "id": "final_answer",
      "description": "Presentation of the final answer and reasoning trace.",
      "purpose": "Present the final solution with a clear explanation of the reasoning process used. The 'final_solution' should be derived from the 'proposed_solution' and 'validation_result' in the current context.",
      "required_context_keys": ["final_solution", "reasoning_trace", "solution_confidence"],
      "instructions": "Your primary task is to populate 'final_solution', 'solution_confidence', and ensure 'reasoning_trace' is carried over. \n1. Examine 'proposed_solution', 'validation_result', 'problem_type', and 'solution_confidence' from the current context.\n2. **PRIORITY 1 (Simple Arithmetic/Direct Answer):** If 'problem_type' is 'arithmetic' AND 'proposed_solution' is a direct answer (like a number or short phrase), your `context_update` MUST contain:\n    - 'final_solution': EXACTLY the value of 'proposed_solution' from context.\n    - 'solution_confidence': EXACTLY the value of 'solution_confidence' (or 'confidence_level') from context.\n    - 'reasoning_trace': EXACTLY the value of 'reasoning_trace' from context.\n3. **PRIORITY 2 (Validated Solution):** Else, if 'validation_result' is true, your `context_update` MUST contain:\n    - 'final_solution': EXACTLY the value of 'proposed_solution' from context.\n    - 'solution_confidence': EXACTLY the value of 'solution_confidence' (or 'confidence_level') from context.\n    - 'reasoning_trace': EXACTLY the value of 'reasoning_trace' from context.\n4. **PRIORITY 3 (Invalid/Complex):** Else, set 'final_solution' to a message like 'The proposed solution requires further refinement.' Your `context_update` MUST still contain:\n    - 'solution_confidence': EXACTLY the value of 'solution_confidence' (or 'confidence_level') from context.\n    - 'reasoning_trace': EXACTLY the value of 'reasoning_trace' from context.\nGenerate a user-facing 'message' that clearly presents the content of 'final_solution'.",
      "transitions": []
    }
  }
}

# Classifier FSM
classifier_fsm = {
  "name": "problem_classifier",
  "description": "Classifies problems to determine the most effective reasoning strategy based on their domain and structure.",
  "initial_state": "analyze_domain",
  "persona": "You are an expert problem analyst. Your goal is to meticulously examine a given problem, understand its core nature, and then recommend the most suitable reasoning strategy to tackle it. You should be articulate in your justifications.",
  "states": {
    "analyze_domain": {
      "id": "analyze_domain",
      "description": "Determine the primary domain of the problem (e.g., scientific, business, creative, logical, ethical, interpersonal).",
      "purpose": "Identify the broad category or field the problem belongs to. Examples: mathematical, logical, creative, business, scientific, ethical, social, technical, etc. Store this in 'problem_domain'. Also note key 'domain_indicators' that led to this classification.",
      "required_context_keys": ["problem_domain", "domain_indicators"],
      "instructions": "Examine the problem statement. What field or area of knowledge does it primarily relate to? Provide a concise 'problem_domain' and a list of 'domain_indicators' (keywords or phrases from the problem that point to this domain).",
      "transitions": [{
        "target_state": "analyze_structure",
        "description": "Domain has been identified."
      }]
    },
    "analyze_structure": {
      "id": "analyze_structure",
      "description": "Analyze the inherent structure of the problem: is it about decomposition, proof, pattern-finding, generation, or evaluation?",
      "purpose": "Understand the underlying task required by the problem. Is it about breaking something complex into smaller parts (decomposition)? Proving a statement (proof/deduction)? Finding trends or commonalities (pattern-finding/induction)? Generating new ideas (generation/creation)? Or assessing the validity/quality of something (evaluation/critique)? Store this in 'problem_structure' and list key 'structural_elements'.",
      "required_context_keys": ["problem_structure", "structural_elements"],
      "instructions": "Consider the problem's goal. What kind of mental operation is primarily needed? Determine the 'problem_structure' (e.g., 'decomposition', 'proof', 'pattern_finding', 'generation', 'evaluation') and list 'structural_elements' from the problem that support this.",
      "transitions": [{
        "target_state": "identify_reasoning_needs",
        "description": "Problem structure has been analyzed."
      }]
    },
    "identify_reasoning_needs": {
      "id": "identify_reasoning_needs",
      "description": "Identify the core reasoning capabilities and potential challenges based on domain and structure.",
      "purpose": "Based on the 'problem_domain' and 'problem_structure', determine the specific type of reasoning that would be most effective. Consider: Analytical (breaking down), Deductive (general to specific), Inductive (specific to general), Creative (novel solutions), Critical (evaluating arguments), Hybrid (combination), or Simple_Calculator. Store this in 'reasoning_requirements' (e.g., 'Needs strong analytical skills to dissect components and critical thinking to evaluate options'). Also, list any 'key_challenges' anticipated.",
      "required_context_keys": ["reasoning_requirements", "key_challenges"],
      "instructions": "Synthesize the domain and structural analysis. What specific reasoning skills are paramount? What are the potential pitfalls or difficult aspects ('key_challenges')? Articulate the 'reasoning_requirements'.",
      "transitions": [{
        "target_state": "recommend_strategy",
        "description": "Reasoning needs have been identified."
      }]
    },
    "recommend_strategy": {
      "id": "recommend_strategy",
      "description": "Recommend a primary reasoning strategy, justify it, and suggest alternatives.",
      "purpose": "Recommend the single best reasoning strategy (e.g., 'analytical', 'creative', 'simple_calculator') as 'recommended_reasoning_type'. Provide a clear 'strategy_justification' explaining why it's most suitable for this problem, referencing the domain, structure, and reasoning needs. Also, list 1-2 'alternative_approaches' that could be considered, briefly noting why they might also be relevant or serve as complementary strategies.",
      "required_context_keys": ["recommended_reasoning_type", "strategy_justification", "alternative_approaches"],
      "instructions": "Based on all prior analysis, select the most fitting reasoning type from: analytical, deductive, inductive, creative, critical, hybrid, simple_calculator. Provide a strong justification. Offer a couple of sensible alternative strategies.",
      "transitions": []
    }
  }
}

# Analytical FSM
analytical_fsm = {
  "name": "analytical_reasoning",
  "description": "An FSM for facilitating analytical reasoning by breaking down complex problems into manageable parts and examining their relationships.",
  "initial_state": "decompose",
  "persona": "You are a precise and methodical analytical thinker. Your role is to systematically dissect complex problems, identify constituent components, understand their individual properties, and map out the relationships and dependencies between them to build a comprehensive understanding.",
  "states": {
    "decompose": {
      "id": "decompose",
      "description": "Break down the main problem into smaller, more manageable components or sub-problems.",
      "purpose": "Identify the primary 'components' of the problem. For each component, list its key 'attributes'. Also, describe the initial understanding of 'relationships' between these components.",
      "required_context_keys": ["components", "attributes", "relationships"],
      "instructions": "Examine the problem statement. What are its fundamental parts or aspects? List them as 'components'. For each component, what are its important 'attributes' or characteristics? How do you initially perceive the 'relationships' between these components?",
      "transitions": [{
        "target_state": "analyze_components",
        "description": "Problem has been successfully decomposed into components with attributes and initial relationships."
      }]
    },
    "analyze_components": {
      "id": "analyze_components",
      "description": "Analyze each identified component individually to understand its properties, behavior, and significance.",
      "purpose": "For each component listed in 'components', provide a 'component_analysis' detailing its specific characteristics, function, and importance within the larger problem. Identify any 'data_requirements' needed to fully understand each component.",
      "required_context_keys": ["component_analysis", "data_requirements"],
      "instructions": "Take each 'component'. What does it do? Why is it important? What information or 'data_requirements' would you need to fully understand it? Document your 'component_analysis' for each.",
      "transitions": [{
        "target_state": "identify_patterns",
        "description": "All components have been individually analyzed."
      }]
    },
    "identify_patterns": {
      "id": "identify_patterns",
      "description": "Look for patterns, causal links, correlations, and dependencies among the analyzed components and their attributes.",
      "purpose": "Based on the 'component_analysis' and refined 'relationships', identify any recurring 'patterns' (e.g., sequences, trends, common structures), 'causal_links' (if X, then Y), or significant 'dependencies' between components. ",
      "required_context_keys": ["patterns", "causal_links", "dependencies"],
      "instructions": "Now that you understand the components, how do they interact? Are there any 'patterns' in their behavior or attributes? Can you identify 'causal_links' or strong 'dependencies' between them? Focus on the connections revealed by your analysis.",
      "transitions": [{
        "target_state": "integrate_findings",
        "description": "Patterns, causal links, and dependencies have been identified."
      }]
    },
    "integrate_findings": {
      "id": "integrate_findings",
      "description": "Combine the insights from component analysis and pattern identification to form a holistic understanding of the problem or system.",
      "purpose": "Synthesize all the findings ('components', 'component_analysis', 'patterns', 'dependencies', 'causal_links') into an 'integrated_analysis'. This should explain how the system or problem works as a whole, based on its parts. Highlight the 'key_insights' gained from this analytical process.",
      "required_context_keys": ["integrated_analysis", "key_insights"],
      "instructions": "Put all the pieces together. Based on your detailed analysis of components and their interactions, provide an 'integrated_analysis' that explains the overall problem or system. What are the most important 'key_insights' you've discovered?",
      "transitions": []
    }
  }
}

# Deductive FSM
deductive_fsm = {
  "name": "deductive_reasoning",
  "description": "An FSM for facilitating deductive reasoning, deriving specific conclusions from general principles or premises.",
  "initial_state": "identify_premises",
  "persona": "You are a precise and logical thinker. Your task is to start with established general principles or rules and systematically apply them to a specific situation to arrive at a logically certain conclusion.",
  "states": {
    "identify_premises": {
      "id": "identify_premises",
      "description": "Identify and clearly state the general principles, rules, laws, or established facts that will serve as the starting premises for the deduction.",
      "purpose": "List all relevant general statements that are assumed to be true and will be used in the reasoning process. These are the 'premises'. Also, note any underlying 'assumptions' that these premises rely on, if applicable.",
      "required_context_keys": ["premises", "assumptions"],
      "instructions": "What are the foundational rules or general truths relevant to this problem? For example, 'All mammals are warm-blooded.' or 'If A then B.' List these as 'premises'. If a premise itself relies on an unstated assumption (e.g., 'the data is accurate'), note it under 'assumptions'.",
      "transitions": [{
        "target_state": "apply_logic",
        "description": "All necessary premises and their underlying assumptions have been identified."
      }]
    },
    "apply_logic": {
      "id": "apply_logic",
      "description": "Apply rules of logical inference (e.g., modus ponens, modus tollens, syllogism) to the premises to derive intermediate or specific conclusions.",
      "purpose": "Detail the step-by-step application of logical rules to the identified 'premises'. Show how new statements are derived from existing ones. Store these as 'logical_steps' (a list of derivations, e.g., 'Premise 1 + Premise 2 implies Intermediate Conclusion X') and collect any 'intermediate_conclusions' that are not the final answer but are steps along the way.",
      "required_context_keys": ["logical_steps", "intermediate_conclusions"],
      "instructions": "Show your work. For example: 'Given: (Premise 1) If it is raining, the ground is wet. (Premise 2) It is raining. Therefore, (Intermediate Conclusion) The ground is wet.' List these derivations under 'logical_steps' and the derived statements under 'intermediate_conclusions'.",
      "transitions": [{
        "target_state": "derive_conclusion",
        "description": "Logical rules have been systematically applied to the premises."
      }]
    },
    "derive_conclusion": {
      "id": "derive_conclusion",
      "description": "State the final specific conclusion that necessarily follows from the premises and the logical steps taken.",
      "purpose": "Present the final, specific 'conclusion' that is logically entailed by the 'premises' and 'logical_steps'. Also, assess the 'logical_validity' of the entire deductive argument (i.e., if the premises were true, would the conclusion have to be true? This should be 'valid' if the logic is sound, regardless of the truth of the premises themselves).",
      "required_context_keys": ["conclusion", "logical_validity"],
      "instructions": "What is the single, specific statement that is proven true if all premises are true and the logic is followed correctly? This is the 'conclusion'. State whether the argument form itself is 'valid' or 'invalid' under 'logical_validity'.",
      "transitions": []
    }
  }
}

# Inductive FSM
inductive_fsm = {
  "name": "inductive_reasoning",
  "description": "An FSM for facilitating inductive reasoning, forming general principles or hypotheses from specific observations or data.",
  "initial_state": "gather_observations",
  "persona": "You are an observant and empirical thinker. Your goal is to carefully examine specific instances or data points, identify patterns, and then formulate a plausible general rule or hypothesis that explains these observations.",
  "states": {
    "gather_observations": {
      "id": "gather_observations",
      "description": "Collect and document specific examples, cases, data points, or individual observations relevant to the problem or phenomenon.",
      "purpose": "Compile a set of specific instances or pieces of information that will be analyzed. These are the 'observations' or 'data_points'. Ensure there are enough distinct observations to potentially reveal a pattern (e.g., at least 3-5 if possible).",
      "required_context_keys": ["observations", "data_points"],
      "instructions": "What specific examples or pieces of data are available? For instance, if trying to understand customer churn, gather details of several customers who churned. List these as 'observations' or 'data_points'.",
      "transitions": [{
        "target_state": "identify_commonalities",
        "description": "Sufficient specific observations or data points have been gathered."
      }]
    },
    "identify_commonalities": {
      "id": "identify_commonalities",
      "description": "Analyze the gathered observations to find recurring patterns, common features, shared characteristics, or trends.",
      "purpose": "Systematically compare the 'observations' or 'data_points'. What elements, properties, or outcomes appear repeatedly? Are there any notable correlations or sequences? Store these findings as 'commonalities' (shared features) and 'trends' (observed patterns over time or across instances).",
      "required_context_keys": ["commonalities", "trends"],
      "instructions": "Look closely at the collected data. What do these specific instances have in common? Are there any noticeable trends or relationships emerging? Document the 'commonalities' and 'trends'.",
      "transitions": [{
        "target_state": "form_hypothesis",
        "description": "Commonalities and trends across observations have been identified."
      }]
    },
    "form_hypothesis": {
      "id": "form_hypothesis",
      "description": "Formulate a general hypothesis, principle, or rule that could explain the identified commonalities and trends.",
      "purpose": "Based on the 'commonalities' and 'trends', propose a general statement or 'hypothesis' that explains why these patterns exist. This hypothesis should be broader than the specific observations. Also, list the key 'supporting_evidence' from the observations that led to this hypothesis.",
      "required_context_keys": ["hypothesis", "supporting_evidence"],
      "instructions": "Based on the patterns you've seen, what general rule or explanation could account for them? This is your 'hypothesis'. Make sure it's a generalization from the specific observations. List the 'supporting_evidence' (i.e., which observations or commonalities support this hypothesis).",
      "transitions": [{
        "target_state": "test_generalization",
        "description": "A plausible hypothesis has been formulated."
      }]
    },
    "test_generalization": {
      "id": "test_generalization",
      "description": "Evaluate the strength of the hypothesis by considering its predictive power, looking for counter-examples, or suggesting ways to test it further.",
      "purpose": "Assess how well the 'hypothesis' generalizes. Does it accurately predict new, unseen cases (if possible to consider)? Are there any known exceptions or 'counter_examples'? What further 'test_results' or experiments could strengthen or refute it? Evaluate the overall 'generalization_strength' (e.g., strong, moderate, weak, needs more data).",
      "required_context_keys": ["test_results", "counter_examples", "generalization_strength"],
      "instructions": "How robust is your hypothesis? Can it predict new instances? Are there any cases where it doesn't hold up ('counter_examples')? How could you further test it ('test_results' or proposed tests)? Conclude with an assessment of its 'generalization_strength'.",
      "transitions": []
    }
  }
}

# Creative FSM
creative_fsm = {
  "name": "creative_reasoning",
  "description": "An FSM for facilitating creative reasoning to generate novel ideas and solutions.",
  "initial_state": "explore_perspectives",
  "persona": "You are an imaginative and unconventional thinker. Your role is to inspire and guide the generation of novel ideas by challenging assumptions, exploring diverse viewpoints, and fostering an environment where wild ideas are welcome.",
  "states": {
    "explore_perspectives": {
      "id": "explore_perspectives",
      "description": "Examine the problem from multiple, diverse, and unusual viewpoints to break free from conventional thinking.",
      "purpose": "Look at the problem from at least three different and unconventional angles or perspectives. This could involve considering it from the viewpoint of a child, an alien, an artist, an engineer from 100 years ago, or by using analogies from completely unrelated fields. The goal is to reframe the problem. Store these 'perspectives' (list of brief descriptions of each angle) and any 'reframings' (how the problem looks from these new angles).",
      "required_context_keys": ["perspectives", "reframings"],
      "instructions": "Don't stick to the obvious. How would someone completely different see this problem? What if the core constraints were different? If it were a dream, what would it mean? List the chosen 'perspectives' and the resulting 'reframings' of the problem.",
      "transitions": [{
        "target_state": "generate_ideas",
        "description": "Sufficient perspectives and reframings have been explored."
      }]
    },
    "generate_ideas": {
      "id": "generate_ideas",
      "description": "Brainstorm a wide range of ideas, prioritizing quantity and novelty over initial feasibility.",
      "purpose": "Generate a multitude of creative ideas (at least 5-7) based on the explored perspectives and reframings. Encourage 'wild' or seemingly impractical ideas at this stage, as they can spark more viable solutions. Do not filter or judge for feasibility yet. Store these as 'creative_ideas' (a list of concise idea descriptions) and any 'unconventional_approaches' (brief notes on the thinking that led to them).",
      "required_context_keys": ["creative_ideas", "unconventional_approaches"],
      "instructions": "Think divergently. What are some out-of-the-box solutions? What if you had unlimited resources, or very few? What if the laws of physics were different? List all 'creative_ideas' and note any 'unconventional_approaches' used.",
      "transitions": [{
        "target_state": "combine_concepts",
        "description": "A diverse set of ideas has been generated."
      }]
    },
    "combine_concepts": {
      "id": "combine_concepts",
      "description": "Synthesize and combine different generated ideas or elements of ideas in novel ways.",
      "purpose": "Take elements from the 'creative_ideas' and try to combine them in unexpected or interesting ways to create new, hybrid solutions. Aim for at least 2-3 distinct 'combinations'. Also, identify any emerging 'novel_solutions' that arise from this synthesis. The goal is to find synergy between previously separate ideas.",
      "required_context_keys": ["combinations", "novel_solutions"],
      "instructions": "Can you merge idea A with idea B? What if you take the best part of idea C and apply it to idea D? Look for forced connections or unusual pairings. List the 'combinations' explored and any distinct 'novel_solutions' that emerge.",
      "transitions": [{
        "target_state": "evaluate_novelty",
        "description": "Ideas have been combined and synthesized."
      }]
    },
    "evaluate_novelty": {
      "id": "evaluate_novelty",
      "description": "Assess the novelty, originality, and potential impact of the creative solutions, while also considering practicality at a high level.",
      "purpose": "From the 'novel_solutions' and 'combinations', select the 'best_creative_solution' (or a small set if multiple are strong). Evaluate its 'innovation_rating' (e.g., low, medium, high, groundbreaking) based on its originality and potential impact. Briefly touch upon why it's innovative and what makes it promising, even if it needs further refinement for practicality.",
      "required_context_keys": ["best_creative_solution", "innovation_rating"],
      "instructions": "Which of the generated solutions is truly new and different? Which one has the most potential to be a game-changer or solve the problem in a unique way? Select the 'best_creative_solution' and assign an 'innovation_rating'. Justify your selection briefly.",
      "transitions": []
    }
  }
}

# Critical FSM
critical_fsm = {
  "name": "critical_reasoning",
  "description": "An FSM for facilitating critical reasoning by evaluating claims, evidence, and logical arguments.",
  "initial_state": "identify_claims",
  "persona": "You are a meticulous and objective critical thinker. Your role is to dissect information, assess its validity, identify underlying assumptions, and evaluate the strength of arguments without bias.",
  "states": {
    "identify_claims": {
      "id": "identify_claims",
      "description": "Clearly identify the main claims, assertions, or conclusions presented in the information or problem.",
      "purpose": "Isolate the core statements or propositions that are being put forward as true or that need evaluation. If there are multiple, list them. Store these as 'claims' (a list of precise statements) and identify any explicit 'arguments' made to support them.",
      "required_context_keys": ["claims", "arguments"],
      "instructions": "What is the central point or thesis being asserted? Are there sub-claims? List all distinct 'claims'. If arguments are explicitly laid out (premise 1, premise 2, conclusion), note them under 'arguments'.",
      "transitions": [{
        "target_state": "examine_evidence",
        "description": "Main claims and arguments have been clearly identified."
      }]
    },
    "examine_evidence": {
      "id": "examine_evidence",
      "description": "Scrutinize the evidence or support provided for each identified claim.",
      "purpose": "For each claim, examine the evidence offered. Assess its relevance, reliability, sufficiency, and source. Note the 'evidence_quality' (e.g., strong, moderate, weak, anecdotal, empirical, statistical) for each piece of evidence and identify any significant 'evidence_gaps' or areas where more support is needed.",
      "required_context_keys": ["evidence_quality", "evidence_gaps"],
      "instructions": "What data, facts, examples, or expert opinions are used to back up each claim? Is the evidence directly relevant? Is the source credible? Is there enough evidence? Describe the 'evidence_quality' for key pieces of support and list any 'evidence_gaps'.",
      "transitions": [{
        "target_state": "analyze_logic",
        "description": "Evidence supporting the claims has been thoroughly examined."
      }]
    },
    "analyze_logic": {
      "id": "analyze_logic",
      "description": "Analyze the logical structure of the arguments, identifying assumptions and potential fallacies.",
      "purpose": "Evaluate how the evidence is used to support the claims. Are the inferences sound? Are there unstated 'assumptions' that the argument relies on? Identify any 'fallacies' (e.g., ad hominem, straw man, false dilemma, appeal to emotion) or weaknesses in the reasoning. Provide a 'logical_analysis' summary.",
      "required_context_keys": ["logical_analysis", "assumptions", "fallacies"],
      "instructions": "Does the conclusion logically follow from the premises/evidence? Are there any hidden assumptions that, if false, would undermine the argument? Look for common logical fallacies. Summarize your 'logical_analysis', listing key 'assumptions' and any identified 'fallacies'.",
      "transitions": [{
        "target_state": "consider_alternatives",
        "description": "Logical structure and assumptions have been analyzed."
      }]
    },
    "consider_alternatives": {
      "id": "consider_alternatives",
      "description": "Explore alternative interpretations, explanations, or counter-arguments.",
      "purpose": "Consider if there are other ways to interpret the evidence or if there are plausible counter-arguments to the main claims. Are there other factors that could explain the observations? Store these as 'alternative_explanations' and 'counter_arguments'.",
      "required_context_keys": ["alternative_explanations", "counter_arguments"],
      "instructions": "Is this the only way to look at the issue? What would someone who disagrees say? Are there other variables at play that haven't been considered? List any 'alternative_explanations' for the evidence and potential 'counter_arguments' to the claims.",
      "transitions": [{
        "target_state": "form_judgment",
        "description": "Alternative perspectives and counter-arguments have been considered."
      }]
    },
    "form_judgment": {
      "id": "form_judgment",
      "description": "Form a well-reasoned and balanced judgment about the validity and strength of the claims and arguments.",
      "purpose": "Synthesize all the analysis (claims, evidence, logic, alternatives) to form an overall 'critical_assessment'. This should state whether the claims are well-supported, partially supported, or unsupported, and why. Assign a 'confidence_rating' (e.g., high, medium, low) to this judgment, reflecting the certainty of the assessment.",
      "required_context_keys": ["critical_assessment", "confidence_rating"],
      "instructions": "Based on all the steps, what is your overall evaluation? Are the claims robust, or do they have significant weaknesses? Provide a concise 'critical_assessment' and a 'confidence_rating' in your judgment.",
      "transitions": []
    }
  }
}

# Hybrid FSM
hybrid_fsm = {
  "name": "hybrid_reasoning",
  "description": "An FSM that orchestrates multiple reasoning strategies (analytical, logical, creative, critical) to solve complex, multifaceted problems.",
  "initial_state": "identify_components",
  "persona": "You are a master strategist and problem-solver, adept at selecting and applying the right combination of thinking approaches for different facets of a complex problem. Your goal is to guide a comprehensive reasoning process, ensuring each part is tackled appropriately and the insights are integrated into a robust solution.",
  "states": {
    "identify_components": {
      "id": "identify_components",
      "description": "Decompose the main problem into distinct sub-problems or aspects, and determine which reasoning type is best suited for each.",
      "purpose": "Break down the overall complex problem into several manageable 'problem_aspects'. For each aspect, create a 'reasoning_map' entry indicating the most suitable reasoning type (e.g., 'analytical' for understanding a system, 'creative' for brainstorming solutions, 'critical' for evaluating options).",
      "required_context_keys": ["problem_aspects", "reasoning_map"],
      "instructions": "Examine the multifaceted problem. What are its distinct parts? For instance, if designing a new product, aspects might include: market analysis (analytical), feature brainstorming (creative), technical feasibility (deductive/analytical), and risk assessment (critical). List these 'problem_aspects' and map each to a primary reasoning type in 'reasoning_map'.",
      "transitions": [{
        "target_state": "apply_analytical",
        "description": "Problem components and their corresponding reasoning approaches have been identified."
      }]
    },
    "apply_analytical": {
      "id": "apply_analytical",
      "description": "Apply analytical reasoning to aspects requiring detailed breakdown and understanding of components and relationships.",
      "purpose": "For those 'problem_aspects' mapped to 'analytical' reasoning, perform a thorough breakdown. Identify sub-components, relationships, and gather relevant data. Store the findings in 'analytical_breakdown' and any identified 'component_relationships'.",
      "required_context_keys": ["analytical_breakdown", "component_relationships"],
      "instructions": "Focus on the analytical parts identified. Deconstruct them. What are the constituent parts? How do they interact? What data defines them? Document your 'analytical_breakdown' and 'component_relationships'. If no analytical parts, state so and proceed.",
      "transitions": [{
        "target_state": "apply_logical",
        "description": "Analytical reasoning phase is complete."
      }]
    },
    "apply_logical": {
      "id": "apply_logical",
      "description": "Apply deductive or inductive reasoning to derive conclusions, make predictions, or identify patterns based on the analytical findings or given premises.",
      "purpose": "For aspects needing logical derivation (deductive) or pattern recognition (inductive), apply the appropriate logic. This might involve using findings from 'analytical_breakdown'. Store the outcomes as 'logical_conclusions' and detail the 'reasoning_chain' (how conclusions were reached).",
      "required_context_keys": ["logical_conclusions", "reasoning_chain"],
      "instructions": "Based on the analytical findings or general principles, what can be logically inferred or generalized? If using deduction, state premises and conclusion. If induction, describe observations and derived hypothesis. Document 'logical_conclusions' and the 'reasoning_chain'. If no logical parts, state so and proceed.",
      "transitions": [{
        "target_state": "apply_creative",
        "description": "Logical reasoning phase is complete."
      }]
    },
    "apply_creative": {
      "id": "apply_creative",
      "description": "Apply creative thinking to generate novel solutions, ideas, or perspectives for relevant problem aspects.",
      "purpose": "For aspects requiring innovation, brainstorm and generate 'creative_insights' or 'novel_approaches'. This might involve looking at the problem from new angles or combining existing ideas in new ways.",
      "required_context_keys": ["creative_insights", "novel_approaches"],
      "instructions": "Focus on the creative parts. Generate a diverse set of ideas. Think outside the box. What are some unconventional solutions? List your 'creative_insights' and 'novel_approaches'. If no creative parts, state so and proceed.",
      "transitions": [{
        "target_state": "critical_evaluation",
        "description": "Creative reasoning phase is complete."
      }]
    },
    "critical_evaluation": {
      "id": "critical_evaluation",
      "description": "Apply critical reasoning to evaluate the outputs from analytical, logical, and creative phases, assessing validity, feasibility, and potential impact.",
      "purpose": "Critically examine the 'analytical_breakdown', 'logical_conclusions', and 'creative_insights'. Assess their strengths, weaknesses, assumptions, and potential biases. Store the findings as 'evaluation_results'. Determine if any of these findings need to be revisited (loop back) or if they can be integrated.",
      "required_context_keys": ["evaluation_results"],
      "instructions": "Review all generated information. Are the analyses sound? Are the logical conclusions valid? Are the creative ideas feasible or impactful? Identify any flaws or areas needing more work. Document your 'evaluation_results'. Based on this, decide if you need to loop back to a previous reasoning stage (e.g., if creative ideas are not feasible based on analytical constraints) or proceed to integration.",
      "transitions": [
        {
          "target_state": "integrate_solution",
          "description": "Evaluation complete and results are satisfactory for integration."
        },
        {
          "target_state": "identify_components",
          "description": "Evaluation indicates a need to re-assess problem components or reasoning approaches.",
          "priority": 1
        }
      ]
    },
    "integrate_solution": {
      "id": "integrate_solution",
      "description": "Synthesize the validated outputs from all applied reasoning strategies into a cohesive and comprehensive solution.",
      "purpose": "Combine the strongest elements from the 'analytical_breakdown', 'logical_conclusions', 'creative_insights', as refined by 'evaluation_results', into a single, 'integrated_solution'. Ensure this solution addresses all relevant 'problem_aspects'.",
      "required_context_keys": ["integrated_solution", "reasoning_synthesis_notes"],
      "instructions": "Bring together the refined outputs from each reasoning type. How do they fit together to solve the overall problem? Formulate the 'integrated_solution'. Add 'reasoning_synthesis_notes' explaining how different parts were combined and any trade-offs made.",
      "transitions": [{
        "target_state": "finalize_hybrid",
        "description": "Initial integrated solution has been formulated."
      }]
    },
    "finalize_hybrid": {
      "id": "finalize_hybrid",
      "description": "Review and refine the integrated solution, ensuring all aspects of the original problem are addressed and the reasoning is clearly articulated.",
      "purpose": "Perform a final check on the 'integrated_solution'. Ensure it is complete, coherent, and directly addresses the original multifaceted problem. Articulate the 'final_hybrid_solution' and summarize the overall 'reasoning_synthesis' (how the different reasoning types contributed).",
      "required_context_keys": ["final_hybrid_solution", "reasoning_synthesis"],
      "instructions": "Review the 'integrated_solution'. Is it clear? Does it cover all necessary points? Is the logic sound? Present the polished 'final_hybrid_solution' and a summary of the 'reasoning_synthesis' that led to it.",
      "transitions": []
    }
  }
}

# Simple Calculator FSM
simple_calculator_fsm = {
  "name": "simple_calculator",
  "description": "An FSM to perform simple arithmetic calculations based on identified operands and an operator.",
  "initial_state": "extract_elements",
  "persona": "You are a precise and efficient calculator. Your task is to identify numbers and a basic arithmetic operation from a problem statement, perform the calculation, and provide the result.",
  "states": {
    "extract_elements": {
      "id": "extract_elements",
      "description": "Identify the operands (numbers) and the operator (+, -, *, /) from the problem statement or components.",
      "purpose": "From the 'problem_statement' or 'problem_components' in the context, extract 'operand1', 'operand2', and the 'operator'. Ensure operands are numbers. Operators should be one of: '+', '-', '*', '/'.",
      "required_context_keys": ["operand1", "operand2", "operator"],
      "instructions": "Examine the problem. Extract the first number as 'operand1', the second number as 'operand2', and the mathematical symbol as 'operator'. If components are provided (e.g., `{'operand1': 1, 'operand2': 1, 'operation': 'addition'}`), use those directly. Map 'addition' to '+', 'subtraction' to '-', 'multiplication' to '*', 'division' to '/'. Ensure 'operand1' and 'operand2' are numeric.",
      "transitions": [{
        "target_state": "perform_calculation",
        "description": "Operands and operator have been successfully extracted and are valid."
      }]
    },
    "perform_calculation": {
      "id": "perform_calculation",
      "description": "Execute the identified arithmetic operation on the operands.",
      "purpose": "Calculate the result of 'operand1 operator operand2'. Store the numerical result in 'calculation_result'. If an error occurs (e.g., division by zero, non-numeric operands), store an appropriate error message in 'calculation_error' and set 'calculation_result' to null or an indicative error string.",
      "required_context_keys": ["calculation_result"],
      "instructions": "Perform the arithmetic operation: `operand1` `operator` `operand2`. Place the exact numerical answer into 'calculation_result'. Handle potential errors like division by zero by setting 'calculation_error' and 'calculation_result' appropriately (e.g., 'calculation_result': 'Error: Division by zero').",
      "transitions": []
    }
  }
}

# Abductive FSM (from your previous request)
abductive_fsm = {
  "name": "abductive_reasoning",
  "description": "An FSM for facilitating abductive reasoning, aiming to find the most plausible explanation for a set of observations.",
  "initial_state": "identify_observations",
  "persona": "You are a keen detective and insightful diagnostician. Your goal is to examine a set of observations or a surprising phenomenon and infer the simplest and most likely explanation.",
  "states": {
    "identify_observations": {
      "id": "identify_observations",
      "description": "Collect and clearly define the specific observations, facts, or surprising phenomena that require an explanation.",
      "purpose": "Gather all relevant 'observations' that form the basis of the puzzle. Identify any 'surprising_elements' or anomalies within these observations that particularly demand an explanation.",
      "required_context_keys": ["observations", "surprising_elements"],
      "instructions": "What are the key facts or data points that need explaining? What seems unusual or unexpected about them? List these under 'observations' and highlight 'surprising_elements'.",
      "transitions": [{
        "target_state": "generate_hypotheses",
        "description": "Observations and surprising elements have been clearly identified."
      }]
    },
    "generate_hypotheses": {
      "id": "generate_hypotheses",
      "description": "Brainstorm and list multiple potential hypotheses that could explain the identified observations.",
      "purpose": "Generate a diverse set of 'potential_hypotheses' (at least 2-3) that could account for the 'observations'. For each hypothesis, provide a brief 'hypothesis_rationale' explaining why it's a candidate.",
      "required_context_keys": ["potential_hypotheses", "hypothesis_rationales"],
      "instructions": "Based on the observations, what are some possible explanations? Don't limit yourself to the most obvious one initially. List each under 'potential_hypotheses' and provide a 'hypothesis_rationale' for each.",
      "transitions": [{
        "target_state": "evaluate_hypotheses",
        "description": "A set of potential hypotheses has been generated."
      }]
    },
    "evaluate_hypotheses": {
      "id": "evaluate_hypotheses",
      "description": "Assess each generated hypothesis against criteria such as explanatory power, simplicity (Occam's Razor), and coherence with existing knowledge.",
      "purpose": "For each hypothesis in 'potential_hypotheses', create an 'hypothesis_evaluation'. This should include its 'explanatory_power' (how well it explains all observations), 'simplicity_score' (e.g., low, medium, high), 'coherence_with_known_facts', and any 'hypothesis_flaws'. Define the 'evaluation_criteria' you are using.",
      "required_context_keys": ["hypothesis_evaluations", "evaluation_criteria"],
      "instructions": "Critically examine each hypothesis. How well does it fit the facts? Is it overly complex? Does it contradict known information? Document your 'hypothesis_evaluations' and the 'evaluation_criteria' used.",
      "transitions": [{
        "target_state": "select_best_explanation",
        "description": "All hypotheses have been evaluated."
      }]
    },
    "select_best_explanation": {
      "id": "select_best_explanation",
      "description": "Select the most plausible hypothesis (the 'best explanation') based on the evaluation.",
      "purpose": "Choose the single 'best_hypothesis' from the 'hypothesis_evaluations'. Provide a clear 'selection_justification' explaining why this hypothesis is considered the most plausible. Assign a 'confidence_in_explanation' (e.g., low, medium, high). Also, suggest 'next_steps_for_validation' (e.g., further data to collect, tests to run) to strengthen or refute this explanation.",
      "required_context_keys": ["best_hypothesis", "selection_justification", "confidence_in_explanation", "next_steps_for_validation"],
      "instructions": "After evaluating all options, which hypothesis stands out as the most likely explanation? Justify your choice. How confident are you? What could be done next to test this explanation?",
      "transitions": []
    }
  }
}

# Analogical FSM (from your previous request)
analogical_fsm = {
  "name": "analogical_reasoning",
  "description": "An FSM for facilitating analogical reasoning, transferring insights from a known source domain to a target problem.",
  "initial_state": "define_target_problem",
  "persona": "You are an expert in finding connections and drawing parallels. Your role is to help understand a new or complex problem (the target) by comparing it to a familiar situation or system (the source/analog).",
  "states": {
    "define_target_problem": {
      "id": "define_target_problem",
      "description": "Clearly define the target problem or concept that needs to be understood or solved through analogy.",
      "purpose": "Provide a concise 'target_problem_description'. Identify its 'key_features_of_target' (essential characteristics, components, relationships, or desired outcomes).",
      "required_context_keys": ["target_problem_description", "key_features_of_target"],
      "instructions": "What is the specific problem or concept we are trying to understand or find a solution for? What are its most important characteristics?",
      "transitions": [{
        "target_state": "find_source_analogs",
        "description": "The target problem and its key features are clearly defined."
      }]
    },
    "find_source_analogs": {
      "id": "find_source_analogs",
      "description": "Brainstorm and identify potential source domains, systems, or problems that share structural or functional similarities with the target problem.",
      "purpose": "List several 'potential_analogs' (at least 2-3). For each, provide a 'rationale_for_choice' explaining why it might be a good analog. Specify the 'similarity_criteria_used' (e.g., structural similarity, functional similarity, shared goals).",
      "required_context_keys": ["potential_analogs", "rationale_for_choice", "similarity_criteria_used"],
      "instructions": "What other situations, systems, or problems are like the target problem in some important way? Think broadly across different fields. List these potential analogs and why you chose them.",
      "transitions": [{
        "target_state": "map_correspondences",
        "description": "Potential source analogs have been identified."
      }]
    },
    "map_correspondences": {
      "id": "map_correspondences",
      "description": "Select the most promising source analog(s) and detail the mappings between elements of the target and source.",
      "purpose": "Choose the 'selected_analog'. Create an 'analogical_mapping' that explicitly links components, relationships, or attributes of the 'target_problem_description' to those of the 'selected_analog'. Also, list 'identified_similarities' and 'identified_differences' that are crucial for the analogy.",
      "required_context_keys": ["selected_analog", "analogical_mapping", "identified_similarities", "identified_differences"],
      "instructions": "Pick the best analog. How do the parts of the target problem correspond to the parts of the source analog? What are the key similarities that make this analogy work? What are important differences to keep in mind?",
      "transitions": [{
        "target_state": "transfer_insights",
        "description": "Correspondences between target and source have been mapped."
      }]
    },
    "transfer_insights": {
      "id": "transfer_insights",
      "description": "Apply knowledge, solutions, principles, or understanding from the source analog to the target problem based on the established mappings.",
      "purpose": "Generate 'transferred_insights_or_solutions' by taking what is known about the 'selected_analog' and applying it to the 'target_problem_description' via the 'analogical_mapping'. Also note any 'potential_inferences' that can be drawn for the target based on the source.",
      "required_context_keys": ["transferred_insights_or_solutions", "potential_inferences"],
      "instructions": "Given the mapping, what can we learn or infer about the target problem from our understanding of the source analog? If the source had a solution, how might that solution apply to the target?",
      "transitions": [{
        "target_state": "evaluate_analogy_fit",
        "description": "Insights or solutions have been transferred from source to target."
      }]
    },
    "evaluate_analogy_fit": {
      "id": "evaluate_analogy_fit",
      "description": "Assess the strength, validity, and limitations of the analogy and the transferred insights. Adapt insights if necessary.",
      "purpose": "Evaluate the 'analogy_strengths' (why it's a good fit) and 'analogy_weaknesses_or_limitations' (where the analogy breaks down or might be misleading). Based on this, refine the 'transferred_insights_or_solutions' into an 'adapted_solution_or_understanding' that better fits the target problem. Provide an overall 'analogy_confidence_rating' (e.g., high, medium, low).",
      "required_context_keys": ["analogy_strengths", "analogy_weaknesses_or_limitations", "adapted_solution_or_understanding", "analogy_confidence_rating"],
      "instructions": "How good is this analogy overall? What are its strong points? Where does it fall short? How can the insights from the source be adjusted to better fit the target? What is your final confidence in this analogical reasoning?",
      "transitions": []
    }
  }
}


# Dictionary to hold all FSM definitions for easy access
ALL_REASONING_FSMS = {
    "orchestrator": orchestrator_fsm,
    "classifier": classifier_fsm,
    "analytical": analytical_fsm,
    "deductive": deductive_fsm,
    "inductive": inductive_fsm,
    "creative": creative_fsm,
    "critical": critical_fsm,
    "hybrid": hybrid_fsm,
    "simple_calculator": simple_calculator_fsm,
    "abductive": abductive_fsm,
    "analogical": analogical_fsm,
}