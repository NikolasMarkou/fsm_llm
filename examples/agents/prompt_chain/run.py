"""
Prompt Chain Agent Example — Sequential LLM Pipeline
=====================================================

Demonstrates the Prompt Chain pattern: a linear pipeline of LLM
steps where each step's output feeds into the next. Useful for
decomposing complex generation tasks into focused sub-tasks.

Pipeline:
  1. Research: Gather key points about the topic
  2. Draft: Write a structured outline from the research
  3. Polish: Refine the outline into polished prose

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/prompt_chain/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/prompt_chain/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, ChainStep, PromptChainAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    # Define the chain steps
    chain = [
        ChainStep(
            step_id="research",
            name="Research",
            extraction_instructions=(
                "Extract the following as JSON:\n"
                '- "key_points": list of 3-5 key facts or arguments about the topic\n'
                '- "sources_mentioned": list of any sources or evidence cited\n'
                '- "main_thesis": one-sentence summary of the core argument'
            ),
            response_instructions=(
                "Research the given topic thoroughly. Identify the most important "
                "facts, arguments, and perspectives. Present them clearly as a "
                "research brief."
            ),
        ),
        ChainStep(
            step_id="draft",
            name="Draft",
            extraction_instructions=(
                "Extract the following as JSON:\n"
                '- "outline_sections": list of section titles\n'
                '- "word_count_estimate": estimated word count\n'
                '- "draft_text": the full draft text'
            ),
            response_instructions=(
                "Using the research from the previous step, write a well-structured "
                "draft. Organize the content into clear sections with an introduction, "
                "body paragraphs, and conclusion. Use the key points and thesis "
                "from the research phase."
            ),
        ),
        ChainStep(
            step_id="polish",
            name="Polish",
            extraction_instructions=(
                "Extract the following as JSON:\n"
                '- "final_text": the polished final text\n'
                '- "improvements_made": list of improvements over the draft\n'
                '- "confidence": confidence score 0-1 in the quality'
            ),
            response_instructions=(
                "Polish and refine the draft from the previous step. Improve "
                "clarity, flow, and persuasiveness. Fix any awkward phrasing. "
                "Ensure smooth transitions between sections. Output the final "
                "polished version."
            ),
        ),
    ]

    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.5,
    )

    agent = PromptChainAgent(chain=chain, config=config)

    task = (
        "Write a short essay (200-300 words) arguing why learning to code "
        "is valuable even for people who don't plan to become programmers."
    )
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Steps: {' -> '.join(s.name for s in chain)}")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nFinal output:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        # Show chain results
        chain_results = result.final_context.get("chain_results", [])
        if chain_results:
            print(f"\nChain progression ({len(chain_results)} steps completed):")
            for i, cr in enumerate(chain_results):
                step_name = chain[i].name if i < len(chain) else f"Step {i}"
                preview = str(cr)[:60] + "..." if len(str(cr)) > 60 else str(cr)
                print(f"  {step_name}: {preview}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    chain_results = result.final_context.get("chain_results", [])
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "steps_completed": len(chain_results) >= 1,
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
