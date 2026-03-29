"""
ADaPT Agent with Working Memory
================================

Demonstrates combining the ADaPT (Adaptive Decomposition and Planning)
pattern with WorkingMemory for persistent knowledge accumulation during
complex multi-step research tasks.

The agent adaptively decides whether to tackle a task directly or
decompose it, while storing intermediate findings in working memory
for cross-reference across subtasks.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/adapt_with_memory/run.py
"""

import os

from fsm_llm.memory import WorkingMemory
from fsm_llm_agents import (
    ADaPTAgent,
    AgentConfig,
    ToolRegistry,
    create_memory_tools,
    tool,
)


@tool
def search_papers(query: str) -> str:
    """Search academic papers and research findings."""
    q = query.lower()
    papers = {
        "transformer": (
            "Vaswani et al. (2017) 'Attention Is All You Need': introduced transformer architecture. "
            "Self-attention mechanism replaces recurrence. 65,000+ citations."
        ),
        "bert": (
            "Devlin et al. (2019) 'BERT': bidirectional pre-training for NLP. "
            "Masked language modeling + next sentence prediction. State-of-the-art on 11 benchmarks."
        ),
        "gpt": (
            "Brown et al. (2020) 'GPT-3': 175B parameter autoregressive model. "
            "Few-shot learning without fine-tuning. Emergent abilities at scale."
        ),
        "scaling": (
            "Kaplan et al. (2020) 'Scaling Laws': loss scales as power law with "
            "model size, data, and compute. Chinchilla (2022) showed data scaling "
            "was underappreciated — optimal ratio is ~20 tokens per parameter."
        ),
        "alignment": (
            "Ouyang et al. (2022) 'InstructGPT': RLHF aligns models with human preferences. "
            "Constitutional AI (Bai et al. 2022): self-improvement through principles."
        ),
        "efficiency": (
            "Dao et al. (2022) 'FlashAttention': IO-aware attention, 2-4x speedup. "
            "LoRA (Hu et al. 2021): low-rank adaptation, 10,000x fewer parameters to fine-tune."
        ),
        "multimodal": (
            "Radford et al. (2021) 'CLIP': joint vision-language pre-training. "
            "Ramesh et al. (2022) 'DALL-E 2': text-to-image via diffusion. "
            "GPT-4V (2023): native multimodal understanding."
        ),
        "reasoning": (
            "Wei et al. (2022) 'Chain-of-Thought': step-by-step reasoning improves accuracy. "
            "Yao et al. (2023) 'Tree of Thoughts': systematic exploration of reasoning paths."
        ),
    }
    for key, value in papers.items():
        if key in q:
            return value
    return f"Research on '{query}': active area with ongoing developments."


@tool
def get_metrics(benchmark: str) -> str:
    """Get benchmark results and performance metrics."""
    b = benchmark.lower()
    metrics = {
        "mmlu": "MMLU (Massive Multitask Language Understanding): GPT-4 86.4%, Claude-3 86.8%, Gemini 83.7%.",
        "humaneval": "HumanEval (code): GPT-4 67%, Claude-3 84.9%, Gemini 74.4%.",
        "math": "MATH benchmark: GPT-4 52.9%, Claude-3 60.1%, specialized models reaching 90%+.",
        "cost": "Inference costs (2024): GPT-4 ~$30/M tokens, Claude-3 ~$15/M, open models ~$0.5-2/M via local deployment.",
        "speed": "Inference speed: GPT-4 ~40 tokens/s, Claude-3 ~80 tokens/s, local 7B models ~60 tokens/s on RTX 4090.",
    }
    for key, value in metrics.items():
        if key in b:
            return value
    return f"Benchmark '{benchmark}': results vary across model families."


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Create shared working memory
    memory = WorkingMemory()
    memory_tools = create_memory_tools(memory)

    # Build registry with research + memory tools
    registry = ToolRegistry()
    registry.register(search_papers._tool_definition)
    registry.register(get_metrics._tool_definition)
    for tool_def in memory_tools:
        registry.register(tool_def)

    config = AgentConfig(model=model, max_iterations=12, temperature=0.5)
    agent = ADaPTAgent(tools=registry, config=config, max_depth=3)

    task = (
        "Write a comprehensive research summary on the evolution of Large Language Models. "
        "Cover: (1) key architectural innovations (transformers, scaling), "
        "(2) current capabilities and benchmark performance, "
        "(3) efficiency improvements and cost trends, "
        "(4) alignment and safety research. "
        "Store key findings in memory as you research each area."
    )

    print("=" * 60)
    print("ADaPT Agent with Working Memory")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Max depth: 3")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nResearch Summary:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        # Show accumulated memory
        print(f"\n{'=' * 40}")
        print("Accumulated Working Memory:")
        print("=" * 40)
        for buffer_name in memory.list_buffers():
            data = memory.get_buffer(buffer_name)
            if data:
                print(f"\n  [{buffer_name}]")
                for key, value in data.items():
                    print(f"    {key} = {str(value)[:100]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
