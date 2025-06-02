"""
Example usage of the refactored reasoning engine.
"""
import json
from llm_fsm.logging import logger

from .engine import ReasoningEngine




def main():
    # Initialize the reasoning engine
    engine = ReasoningEngine(model="gpt-4o-mini", temperature=0.7)

    # Example problems to solve
    problems = [
        {
            "type": "analytical",
            "problem": "Design a scalable microservices architecture for an e-commerce platform that needs to handle millions of users.",
            "context": {
                "scale": "10M+ users",
                "requirements": ["high availability", "real-time inventory", "payment processing"]
            }
        },
        {
            "type": "deductive",
            "problem": "Given that all successful tech companies invest heavily in R&D, "
                       "and Company X is a successful tech company, "
                       "what can we conclude about their R&D spending?",
            "context": {}
        },
        {
            "type": "creative",
            "problem": "Propose innovative solutions to reduce plastic waste in urban environments.",
            "context": {
                "constraints": ["cost-effective", "community-driven", "scalable"]
            }
        }
    ]

    # Solve each problem
    for idx, problem_data in enumerate(problems, 1):
        print(f"\n{'=' * 60}")
        print(f"Problem {idx}: {problem_data['type'].upper()} Reasoning")
        print(f"{'=' * 60}")
        print(f"Problem: {problem_data['problem'][:100]}...")

        try:
            # Solve the problem
            solution, trace_info = engine.solve_problem(
                problem_data['problem'],
                initial_context=problem_data['context']
            )

            # Display results
            print(f"\nSolution:\n{solution[:500]}...")
            print(f"\nReasoning Details:")
            print(f"- Total steps: {trace_info['reasoning_trace']['total_steps']}")
            print(f"- Reasoning types used: {', '.join(trace_info['reasoning_trace']['reasoning_types_used'])}")
            print(f"- Final confidence: {trace_info['reasoning_trace']['final_confidence']:.2f}")

            # Save trace to file (optional)
            with open(f"reasoning_trace_{idx}.json", "w") as f:
                json.dump(trace_info['reasoning_trace'], f, indent=2)

        except Exception as e:
            logger.error(f"Error solving problem {idx}: {e}")
            print(f"Error: {e}")

    # Example of using a specific reasoning type directly
    print(f"\n{'=' * 60}")
    print("Custom Problem with Initial Context")
    print(f"{'=' * 60}")

    custom_problem = "Analyze the potential impact of quantum computing on current encryption methods."
    custom_context = {
        "domain": "cybersecurity",
        "timeframe": "5-10 years",
        "focus_areas": ["RSA", "AES", "blockchain"]
    }

    solution, trace_info = engine.solve_problem(custom_problem, custom_context)
    print(f"\nProblem: {custom_problem}")
    print(f"Context: {custom_context}")
    print(f"\nSolution Preview:\n{solution[:300]}...")

    # Demonstrate reasoning trace analysis
    print(f"\n{'=' * 60}")
    print("Reasoning Trace Analysis")
    print(f"{'=' * 60}")

    if trace_info['reasoning_trace']['steps']:
        print("State transitions:")
        for step in trace_info['reasoning_trace']['steps'][:5]:  # First 5 steps
            print(f"  {step['from']} → {step['to']}")
            if 'context_snapshot' in step:
                snapshot_keys = list(step['context_snapshot'].keys())[:3]
                print(f"    Context keys: {', '.join(snapshot_keys)}...")


if __name__ == "__main__":
    main()