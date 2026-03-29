from __future__ import annotations

"""
Prompt builders for agent tool awareness and observation formatting.
"""

from .tools import ToolRegistry


def _build_tool_example(tool_name: str, params: dict) -> str:
    """Build a compact JSON example for a specific tool call."""
    import json

    example_values: dict[str, object] = {}
    for pname, pschema in params.items():
        ptype = pschema.get("type", "string")
        if ptype == "number" or ptype == "integer":
            example_values[pname] = 0
        elif ptype == "boolean":
            example_values[pname] = True
        else:
            example_values[pname] = f"<{pname}>"

    return json.dumps({"tool_name": tool_name, "tool_input": example_values})


def build_think_extraction_instructions(
    registry: ToolRegistry,
    include_observations: bool = True,
) -> str:
    """Build extraction instructions for the think state."""
    tool_list = registry.to_prompt_description()

    if include_observations:
        intro = (
            "Analyze the task and all previous observations to decide your next action."
        )
    else:
        intro = "Analyze the task to decide your next action."

    parts = [
        intro,
        "",
        tool_list,
        "",
        "Extract the following as JSON:",
        '- "tool_name": name of the tool to use (must be one of the available tools), or "none" if no tool is needed',
        '- "tool_input": a JSON object with the parameters for the tool (MUST include all required parameters)',
        '- "reasoning": your step-by-step reasoning for choosing this action',
        '- "should_terminate": true if you have enough information to answer the task, false otherwise',
    ]

    # Generate per-tool examples from registry
    tools_with_params = []
    for tool in registry.list_tools():
        if tool.parameter_schema:
            props = tool.parameter_schema.get("properties", {})
            if props:
                tools_with_params.append((tool.name, props))

    if tools_with_params:
        parts.append("")
        parts.append("Examples for each tool (you MUST provide tool_input parameters):")
        for tool_name, params in tools_with_params:
            parts.append(_build_tool_example(tool_name, params))
    else:
        parts.append("")
        parts.append("Example — using a tool:")
        parts.append(
            '{"tool_name": "search", "tool_input": {"query": "example query"}, '
            '"reasoning": "I need to find this information", "should_terminate": false}'
        )

    parts.extend(
        [
            "",
            "Example — terminating with enough information:",
            '{"tool_name": "none", "tool_input": {}, '
            '"reasoning": "I have all the information needed", "should_terminate": true}',
        ]
    )

    parts.extend(
        [
            "",
            "RULES:",
            "1. You MUST select a tool on the first iteration. "
            "Do not set should_terminate=true without calling at least one tool first.",
            "2. Only set should_terminate=true AFTER you have tool results that fully answer the task.",
            "3. Always provide the required parameters for the selected tool.",
        ]
    )

    if include_observations:
        parts.extend(
            [
                "",
                "IMPORTANT: Review all previous observations carefully before deciding.",
                "If previous tool calls have provided sufficient information, set should_terminate to true.",
                "Do NOT repeat the same tool call with the same parameters.",
            ]
        )

    return "\n".join(parts)


def build_think_response_instructions() -> str:
    """Build response instructions for the think state."""
    return (
        "Briefly explain your reasoning and what action you are taking next. "
        "If you have decided to terminate, explain why you have enough information."
    )


def build_act_response_instructions() -> str:
    """Build response instructions for the act state."""
    return (
        "Summarize what tool was called and what was observed from the result. "
        "Be concise but include all relevant information from the tool output."
    )


def build_conclude_extraction_instructions(
    output_schema: type | None = None,
) -> str:
    """Build extraction instructions for the conclude state."""
    parts = [
        "Based on all the observations gathered, formulate your final answer to the task.",
        "Extract:",
        '- "final_answer": your complete, well-structured answer to the original task',
        '- "confidence": your confidence in the answer (0.0 to 1.0)',
    ]

    if output_schema is not None and hasattr(output_schema, "model_fields"):
        parts.append("")
        parts.append(
            f"IMPORTANT: Also extract these fields for {output_schema.__name__}:"
        )
        for field_name, field_info in output_schema.model_fields.items():
            desc = field_info.description or field_name
            parts.append(f'- "{field_name}": {desc}')

    parts.append("")
    parts.append("Example:")
    if output_schema is not None and hasattr(output_schema, "model_fields"):
        # Build example that includes schema fields so the model follows the format
        example_fields = ['"final_answer": "The answer based on my research is ..."']
        example_fields.append('"confidence": 0.9')
        for field_name in output_schema.model_fields:
            example_fields.append(f'"{field_name}": "<value>"')
        parts.append("{" + ", ".join(example_fields) + "}")
    else:
        parts.append(
            '{"final_answer": "The answer based on my research is ...", "confidence": 0.9}'
        )

    return "\n".join(parts)


def build_conclude_response_instructions() -> str:
    """Build response instructions for the conclude state."""
    return (
        "Present your final answer clearly and completely. "
        "Reference the evidence from your tool observations to support your answer."
    )


def build_approval_extraction_instructions() -> str:
    """Build extraction instructions for the approval-waiting state."""
    return (
        "The previous action requires human approval before execution.\n"
        "Wait for the user's response.\n"
        "Extract:\n"
        '- "approval_granted": true if the user approves, false if denied\n'
    )


# ---------------------------------------------------------------------------
# Reflexion prompts
# ---------------------------------------------------------------------------


def build_evaluate_extraction_instructions() -> str:
    """Build extraction instructions for the Reflexion evaluate state."""
    return "\n".join(
        [
            "Evaluate whether the information gathered so far is sufficient "
            "to answer the original task correctly and completely.",
            "",
            "Consider:",
            "- Are all parts of the question addressed?",
            "- Is the evidence reliable and consistent?",
            "- Could the answer be wrong or incomplete?",
            "",
            "Extract the following as JSON:",
            '- "evaluation_passed": true if the gathered info is sufficient, false otherwise',
            '- "evaluation_score": a float between 0.0 and 1.0 rating answer quality',
            '- "evaluation_feedback": a brief explanation of what is good or missing',
        ]
    )


def build_evaluate_response_instructions() -> str:
    """Build response instructions for the Reflexion evaluate state."""
    return (
        "Explain your evaluation of the current answer quality. "
        "Mention specific strengths and weaknesses."
    )


def build_reflect_extraction_instructions() -> str:
    """Build extraction instructions for the Reflexion reflect state."""
    return "\n".join(
        [
            "The previous evaluation found the answer insufficient. "
            "Reflect on what went wrong and what to try differently.",
            "",
            "Review your episodic memory (previous reflections) if available "
            "to avoid repeating the same mistakes.",
            "",
            "Extract the following as JSON:",
            '- "reflection": a detailed self-critique of what went wrong',
            '- "lessons": a JSON list of short lesson strings to remember '
            "for next attempts",
        ]
    )


def build_reflect_response_instructions() -> str:
    """Build response instructions for the Reflexion reflect state."""
    return (
        "Explain what went wrong in the previous attempt and describe "
        "your revised strategy for the next attempt."
    )


# ---------------------------------------------------------------------------
# Plan-and-Execute prompts
# ---------------------------------------------------------------------------


def build_plan_extraction_instructions(
    registry: ToolRegistry | None = None,
) -> str:
    """Build extraction instructions for the plan state."""
    tool_section = ""
    if registry is not None and len(registry) > 0:
        tool_section = (
            "\n\n"
            + registry.to_prompt_description()
            + "\n\nYou may reference these tools in your plan steps."
        )

    return "\n".join(
        [
            "Break the task down into a sequence of concrete, actionable steps. "
            "Each step should be self-contained and produce a clear result.",
            tool_section,
            "",
            "Extract the following as JSON:",
            '- "plan_steps": a JSON list of step description strings '
            "(ordered, max 10 steps)",
            '- "reasoning": your reasoning for this plan decomposition',
        ]
    )


def build_execute_step_extraction_instructions(
    registry: ToolRegistry | None = None,
) -> str:
    """Build extraction instructions for the execute_step state."""
    tool_section = ""
    if registry is not None and len(registry) > 0:
        tool_section = "\n\n" + registry.to_prompt_description()

    return "\n".join(
        [
            "Execute the current plan step. Use the available tools or your own "
            "knowledge to produce a result for this step.",
            tool_section,
            "",
            "The current step description is provided in the context.",
            "",
            "Extract the following as JSON:",
            '- "step_result": the result or output of executing this step',
            '- "tool_name": name of the tool used (or "none" if no tool was needed)',
            '- "tool_input": parameters passed to the tool (or empty object)',
        ]
    )


def build_check_result_extraction_instructions() -> str:
    """Build extraction instructions for the check_result state."""
    return "\n".join(
        [
            "Evaluate the result of the step that was just executed.",
            "",
            "Consider:",
            "- Did the step produce a useful result?",
            "- Is the result correct and relevant to the plan?",
            "- Can we proceed to the next step?",
            "",
            "Extract the following as JSON:",
            '- "step_failed": true if the step result is inadequate, false otherwise',
            '- "evaluation_feedback": brief explanation of step quality',
        ]
    )


def build_check_result_response_instructions() -> str:
    """Build response instructions for the check_result state."""
    return (
        "Summarize whether the step succeeded and what was produced. "
        "If it failed, explain why."
    )


def build_replan_extraction_instructions(
    registry: ToolRegistry | None = None,
) -> str:
    """Build extraction instructions for the replan state."""
    tool_section = ""
    if registry is not None and len(registry) > 0:
        tool_section = "\n\n" + registry.to_prompt_description()

    return "\n".join(
        [
            "The previous plan step failed. Revise the remaining plan "
            "incorporating what you learned from the failure.",
            tool_section,
            "",
            "Extract the following as JSON:",
            '- "plan_steps": a revised JSON list of remaining step description strings',
            '- "reasoning": your reasoning for the revised plan',
        ]
    )


def build_synthesize_extraction_instructions() -> str:
    """Build extraction instructions for the synthesize state."""
    return "\n".join(
        [
            "All plan steps have been completed. Synthesize the results from "
            "every step into a single, comprehensive final answer.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete, well-structured answer to the original task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_synthesize_response_instructions() -> str:
    """Build response instructions for the synthesize state."""
    return (
        "Present a clear, complete answer that integrates results from all "
        "plan steps. Reference specific step results as evidence."
    )


# ---------------------------------------------------------------------------
# REWOO prompts
# ---------------------------------------------------------------------------


def build_rewoo_plan_extraction_instructions(
    registry: ToolRegistry,
) -> str:
    """Build extraction instructions for the REWOO plan_all state."""
    tool_list = registry.to_prompt_description()

    return "\n".join(
        [
            "Analyze the task and create a COMPLETE plan of all tool calls needed "
            "to solve it. You must plan everything upfront in a single pass.",
            "",
            tool_list,
            "",
            "Each plan step can reference the output of a previous step using "
            "#E1, #E2, etc. (where the number matches the plan_id).",
            "",
            "Extract the following as JSON:",
            '- "plan_blueprint": a JSON list of plan step objects, each with:',
            '  - "plan_id": integer step number starting from 1 (e.g. 1, 2, 3)',
            '  - "description": what this step accomplishes',
            '  - "tool_name": which tool to call',
            '  - "tool_input": parameters for the tool (may contain #E1, #E2 references)',
            '- "reasoning": your overall reasoning for this plan',
            "",
            "Example plan_blueprint:",
            "[",
            '  {"plan_id": 1, "description": "Search for X", '
            '"tool_name": "search", "tool_input": {"query": "X"}},',
            '  {"plan_id": 2, "description": "Search for Y using result of step 1", '
            '"tool_name": "search", "tool_input": {"query": "Y context: #E1"}}',
            "]",
        ]
    )


def build_rewoo_plan_response_instructions() -> str:
    """Build response instructions for the REWOO plan_all state."""
    return (
        "Explain the complete plan you have created to solve the task. "
        "List each step and describe the dependencies between them."
    )


def build_rewoo_execute_response_instructions() -> str:
    """Build response instructions for the REWOO execute_plans state."""
    return (
        "Summarize the results of executing all planned tool calls. "
        "List each step and its outcome."
    )


def build_rewoo_solve_extraction_instructions() -> str:
    """Build extraction instructions for the REWOO solve state."""
    return "\n".join(
        [
            "You have the original task, the plan you created, and the evidence "
            "gathered from executing all tool calls. Synthesize a final answer.",
            "",
            "Review all evidence carefully and produce a complete answer.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete, well-structured answer to the task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_rewoo_solve_response_instructions() -> str:
    """Build response instructions for the REWOO solve state."""
    return (
        "Present your final answer clearly, referencing the evidence gathered "
        "from the planned tool executions."
    )


# ---------------------------------------------------------------------------
# Evaluator-Optimizer prompts
# ---------------------------------------------------------------------------


def build_evalopt_generate_extraction_instructions(
    task_instructions: str = "",
) -> str:
    """Build extraction instructions for the EvalOpt generate state."""
    context_line = ""
    if task_instructions:
        context_line = f"\nTask-specific instructions: {task_instructions}\n"

    return "\n".join(
        [
            "Generate your best output for the given task.",
            context_line,
            "Extract the following as JSON:",
            '- "generated_output": your complete generated output',
            '- "reasoning": your reasoning and approach',
        ]
    )


def build_evalopt_generate_response_instructions() -> str:
    """Build response instructions for the EvalOpt generate state."""
    return "Present the output you have generated for the task."


def build_evalopt_evaluate_response_instructions() -> str:
    """Build response instructions for the EvalOpt evaluate state."""
    return (
        "Summarize the evaluation results. Describe whether the output "
        "passed the evaluation and what feedback was provided."
    )


def build_evalopt_refine_extraction_instructions() -> str:
    """Build extraction instructions for the EvalOpt refine state."""
    return "\n".join(
        [
            "Your previous output did not pass evaluation. Refine it based "
            "on the feedback provided.",
            "",
            "The feedback from the evaluator is available in the context as "
            "'refinement_feedback'. Your previous output is in 'generated_output'.",
            "",
            "IMPORTANT: Address ALL feedback points. Do not just repeat the same output.",
            "",
            "Extract the following as JSON:",
            '- "generated_output": your refined output (complete, not just the changes)',
            '- "reasoning": what you changed and why',
        ]
    )


def build_evalopt_refine_response_instructions() -> str:
    """Build response instructions for the EvalOpt refine state."""
    return (
        "Explain what changes you made to address the evaluation feedback "
        "and present your revised output."
    )


def build_evalopt_output_extraction_instructions() -> str:
    """Build extraction instructions for the EvalOpt output state."""
    return "\n".join(
        [
            "The output has passed evaluation (or maximum refinements reached).",
            "Present the final version of your output.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": the final output',
            '- "confidence": your confidence in the final output (0.0 to 1.0)',
        ]
    )


def build_evalopt_output_response_instructions() -> str:
    """Build response instructions for the EvalOpt output state."""
    return "Present the final, evaluated output as your answer."


# ---------------------------------------------------------------------------
# Maker-Checker prompts
# ---------------------------------------------------------------------------


def build_maker_extraction_instructions(
    maker_instructions: str,
) -> str:
    """Build extraction instructions for the Maker-Checker make state."""
    return "\n".join(
        [
            "You are the MAKER. Your job is to produce a high-quality draft.",
            "",
            f"Instructions: {maker_instructions}",
            "",
            "If previous checker feedback is available in the context, "
            "use it to improve your draft.",
            "",
            "Extract the following as JSON:",
            '- "draft_output": your complete draft',
            '- "reasoning": your approach and rationale',
        ]
    )


def build_maker_response_instructions() -> str:
    """Build response instructions for the Maker-Checker make state."""
    return "Present the draft you have created."


def build_checker_extraction_instructions(
    checker_instructions: str,
) -> str:
    """Build extraction instructions for the Maker-Checker check state."""
    return "\n".join(
        [
            "You are the CHECKER. Critically evaluate the draft produced by the maker.",
            "",
            f"Evaluation criteria: {checker_instructions}",
            "",
            "Be thorough and constructive. Point out specific issues.",
            "",
            "Extract the following as JSON:",
            '- "checker_passed": true if the draft meets quality standards, false otherwise',
            '- "checker_feedback": detailed feedback on what is good and what needs improvement',
            '- "quality_score": a float between 0.0 and 1.0 rating overall quality',
        ]
    )


def build_checker_response_instructions() -> str:
    """Build response instructions for the Maker-Checker check state."""
    return (
        "Present your evaluation of the draft. Explain what works well "
        "and what needs improvement."
    )


def build_revise_extraction_instructions(
    maker_instructions: str,
) -> str:
    """Build extraction instructions for the Maker-Checker revise state."""
    return "\n".join(
        [
            "You are the MAKER. The checker has provided feedback on your draft. "
            "Revise your output to address all feedback points.",
            "",
            f"Original instructions: {maker_instructions}",
            "",
            "The checker's feedback is in 'checker_feedback'. "
            "Your previous draft is in 'draft_output'.",
            "",
            "IMPORTANT: Address ALL feedback points. Produce a complete revised draft.",
            "",
            "Extract the following as JSON:",
            '- "draft_output": your complete revised draft (not just the changes)',
            '- "reasoning": what you changed and why',
        ]
    )


def build_revise_response_instructions() -> str:
    """Build response instructions for the Maker-Checker revise state."""
    return (
        "Explain what changes you made to address the checker's feedback "
        "and present your revised draft."
    )


def build_maker_checker_output_extraction_instructions() -> str:
    """Build extraction instructions for the Maker-Checker output state."""
    return "\n".join(
        [
            "The draft has passed the checker's review (or maximum revisions reached).",
            "Present the final version.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": the final output',
            '- "confidence": your confidence in the final output (0.0 to 1.0)',
        ]
    )


def build_maker_checker_output_response_instructions() -> str:
    """Build response instructions for the Maker-Checker output state."""
    return "Present the final, reviewed output as your answer."


# ---------------------------------------------------------------------------
# Orchestrator-Workers prompts
# ---------------------------------------------------------------------------


def build_orchestrate_extraction_instructions() -> str:
    """Build extraction instructions for the orchestrate state."""
    return "\n".join(
        [
            "Decompose the task into independent subtasks that can be delegated "
            "to worker agents. Each subtask should be self-contained and produce "
            "a clear result.",
            "",
            "If worker results from previous rounds are available in context, "
            "review them and determine if additional subtasks are needed.",
            "",
            "Extract the following as JSON:",
            '- "subtasks": a JSON list of subtask description strings '
            "(each subtask should be specific and actionable)",
            '- "delegation_plan": a brief description of your delegation strategy',
        ]
    )


def build_orchestrate_response_instructions() -> str:
    """Build response instructions for the orchestrate state."""
    return (
        "Explain how you are decomposing the task into subtasks and "
        "your delegation strategy."
    )


def build_delegate_response_instructions() -> str:
    """Build response instructions for the delegate state."""
    return (
        "Summarize which subtasks were delegated and report the status "
        "of each worker execution."
    )


def build_collect_extraction_instructions() -> str:
    """Build extraction instructions for the collect state."""
    return "\n".join(
        [
            "Review all worker results collected so far.",
            "",
            "Determine if all necessary information has been gathered "
            "or if additional subtasks are needed.",
            "",
            "Extract the following as JSON:",
            '- "all_collected": true if all results are sufficient to '
            "produce a final answer, false if more work is needed",
            '- "reasoning": explanation of your assessment',
        ]
    )


def build_collect_response_instructions() -> str:
    """Build response instructions for the collect state."""
    return (
        "Summarize the worker results and explain whether all needed "
        "information has been gathered."
    )


def build_orchestrator_synthesize_extraction_instructions() -> str:
    """Build extraction instructions for the orchestrator synthesize state."""
    return "\n".join(
        [
            "All worker results have been collected. Synthesize them into "
            "a single, comprehensive final answer.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete, well-structured answer to the original task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_orchestrator_synthesize_response_instructions() -> str:
    """Build response instructions for the orchestrator synthesize state."""
    return (
        "Present a clear, complete answer that integrates results from all "
        "workers. Reference specific worker results as evidence."
    )


# ---------------------------------------------------------------------------
# ADaPT prompts
# ---------------------------------------------------------------------------


def build_attempt_extraction_instructions(
    registry: ToolRegistry | None = None,
) -> str:
    """Build extraction instructions for the ADaPT attempt state."""
    tool_section = ""
    if registry is not None and len(registry) > 0:
        tool_section = "\n\n" + registry.to_prompt_description()

    return "\n".join(
        [
            "Attempt to solve the task directly using your knowledge "
            "and any available tools." + tool_section,
            "",
            "Give your best attempt at answering the task completely.",
            "",
            "Extract the following as JSON:",
            '- "attempt_result": your attempted answer to the task',
            '- "confidence": your confidence in this attempt (0.0 to 1.0)',
            '- "reasoning": your reasoning process',
        ]
    )


def build_attempt_response_instructions() -> str:
    """Build response instructions for the ADaPT attempt state."""
    return (
        "Present your attempt at solving the task. Be thorough but note "
        "any areas of uncertainty."
    )


def build_assess_extraction_instructions() -> str:
    """Build extraction instructions for the ADaPT assess state."""
    return "\n".join(
        [
            "Evaluate whether the previous attempt successfully solved the task.",
            "",
            "Consider:",
            "- Is the answer complete and correct?",
            "- Are there gaps in the reasoning?",
            "- Would breaking this into subtasks yield a better result?",
            "",
            "Extract the following as JSON:",
            '- "attempt_succeeded": true if the attempt is satisfactory, false otherwise',
            '- "evaluation_feedback": brief explanation of the assessment',
        ]
    )


def build_assess_response_instructions() -> str:
    """Build response instructions for the ADaPT assess state."""
    return (
        "Explain your assessment of the attempt quality. "
        "If the attempt failed, explain what went wrong."
    )


def build_decompose_extraction_instructions() -> str:
    """Build extraction instructions for the ADaPT decompose state."""
    return "\n".join(
        [
            "The previous attempt was insufficient. Decompose the task into "
            "smaller, simpler subtasks.",
            "",
            "Choose an operator:",
            "- AND: all subtasks must succeed (combine results)",
            "- OR: any subtask success is sufficient (take best result)",
            "",
            "Extract the following as JSON:",
            '- "subtasks": a JSON list of subtask description strings',
            '- "operator": either "AND" or "OR"',
            '- "reasoning": your reasoning for this decomposition',
        ]
    )


def build_decompose_response_instructions() -> str:
    """Build response instructions for the ADaPT decompose state."""
    return (
        "Explain how you are breaking the task into subtasks and "
        "why this decomposition should help solve the problem."
    )


def build_combine_extraction_instructions() -> str:
    """Build extraction instructions for the ADaPT combine state."""
    return "\n".join(
        [
            "Combine all results (from direct attempts and subtask results) "
            "into a single final answer.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete, well-structured answer to the original task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_combine_response_instructions() -> str:
    """Build response instructions for the ADaPT combine state."""
    return (
        "Present your final answer clearly and completely, integrating "
        "all available results and evidence."
    )


# ---------------------------------------------------------------------------
# Prompt Chain prompts
# ---------------------------------------------------------------------------


def build_chain_output_extraction_instructions() -> str:
    """Build extraction instructions for the chain output (terminal) state."""
    return "\n".join(
        [
            "All pipeline steps have been completed. Review the accumulated results "
            "and produce a final answer that synthesizes the outputs from every step.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete, well-structured answer to the original task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_chain_output_response_instructions() -> str:
    """Build response instructions for the chain output (terminal) state."""
    return (
        "Present the final output of the pipeline. Integrate and summarize "
        "the results from all preceding steps."
    )


# ---------------------------------------------------------------------------
# Self-Consistency prompts
# ---------------------------------------------------------------------------


def build_generate_extraction_instructions() -> str:
    """Build extraction instructions for the self-consistency generate state."""
    return "\n".join(
        [
            "Answer the given task directly and completely.",
            "",
            "Think through the problem step by step, then provide your answer.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete answer to the task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_generate_response_instructions() -> str:
    """Build response instructions for the self-consistency generate state."""
    return (
        "Present your answer clearly and concisely. "
        "Show your reasoning before giving the final answer."
    )


# ---------------------------------------------------------------------------
# Debate prompts
# ---------------------------------------------------------------------------


def build_propose_extraction_instructions(proposer_persona: str = "") -> str:
    """Build extraction instructions for the debate propose state."""
    persona_line = ""
    if proposer_persona:
        persona_line = f"\nRole: {proposer_persona}\n"

    return "\n".join(
        [
            "Generate a well-reasoned proposition or answer for the task.",
            persona_line,
            "If previous debate rounds are available in context, improve upon "
            "the previous proposition by incorporating insights from the critique "
            "and counter-argument.",
            "",
            "Extract the following as JSON:",
            '- "proposition": your proposition or answer with supporting arguments',
            '- "reasoning": your reasoning process',
        ]
    )


def build_propose_response_instructions() -> str:
    """Build response instructions for the debate propose state."""
    return (
        "Present your proposition clearly with supporting arguments. "
        "If this is a subsequent round, explain how you improved it."
    )


def build_critique_extraction_instructions(critic_persona: str = "") -> str:
    """Build extraction instructions for the debate critique state."""
    persona_line = ""
    if critic_persona:
        persona_line = f"\nRole: {critic_persona}\n"

    return "\n".join(
        [
            "Critically analyze the current proposition.",
            persona_line,
            "Identify weaknesses, logical gaps, missing evidence, "
            "and potential counterexamples.",
            "",
            "Extract the following as JSON:",
            '- "critique": your detailed critique of the proposition',
            '- "reasoning": the reasoning behind your critique',
        ]
    )


def build_critique_response_instructions() -> str:
    """Build response instructions for the debate critique state."""
    return (
        "Present your critique of the proposition. Be specific about "
        "weaknesses and suggest areas for improvement."
    )


def build_counter_extraction_instructions(proposer_persona: str = "") -> str:
    """Build extraction instructions for the debate counter state."""
    persona_line = ""
    if proposer_persona:
        persona_line = f"\nRole: {proposer_persona}\n"

    return "\n".join(
        [
            "Address the critique with counter-arguments to strengthen "
            "the original proposition.",
            persona_line,
            "Respond to each point raised in the critique. Concede valid points "
            "and refute invalid ones with evidence.",
            "",
            "Extract the following as JSON:",
            '- "counter_argument": your counter-arguments addressing the critique',
            '- "reasoning": your reasoning process',
        ]
    )


def build_counter_response_instructions() -> str:
    """Build response instructions for the debate counter state."""
    return (
        "Present your counter-arguments. Address each point from the critique "
        "and strengthen the proposition where possible."
    )


def build_judge_extraction_instructions(
    judge_persona: str = "",
    max_rounds: int = 3,
) -> str:
    """Build extraction instructions for the debate judge state."""
    persona_line = ""
    if judge_persona:
        persona_line = f"\nRole: {judge_persona}\n"

    return "\n".join(
        [
            "Evaluate the full debate exchange: proposition, critique, "
            "and counter-argument.",
            persona_line,
            "Determine whether a strong consensus answer has been reached "
            f"or if another round of debate (max {max_rounds}) is needed.",
            "",
            "Extract the following as JSON:",
            '- "judge_verdict": your assessment of the debate exchange',
            '- "consensus_reached": true if a satisfactory answer has emerged, '
            "false if another round would improve quality",
            '- "final_answer": the best answer so far (required if consensus_reached is true)',
            '- "reasoning": your reasoning for the verdict',
        ]
    )


def build_judge_response_instructions() -> str:
    """Build response instructions for the debate judge state."""
    return (
        "Summarize the debate exchange and explain your verdict. "
        "If consensus is reached, present the agreed-upon answer."
    )


def build_debate_conclude_extraction_instructions() -> str:
    """Build extraction instructions for the debate conclude state."""
    return "\n".join(
        [
            "The debate is complete. Produce the final, definitive answer "
            "incorporating the strongest arguments from all rounds.",
            "",
            "Extract the following as JSON:",
            '- "final_answer": your complete final answer to the original task',
            '- "confidence": your confidence in the answer (0.0 to 1.0)',
        ]
    )


def build_debate_conclude_response_instructions() -> str:
    """Build response instructions for the debate conclude state."""
    return (
        "Present the final answer from the debate. Reference the key arguments "
        "and how the debate refined the answer."
    )
