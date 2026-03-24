#!/bin/bash
# Run all examples with ollama qwen3.5:4b and report results
set -o pipefail

export LLM_MODEL="ollama_chat/qwen3.5:4b"
export OPENAI_API_KEY="not-needed"

PASS=0
FAIL=0
ERRORS=()
TIMEOUT=120  # seconds per example

run_example() {
    local name="$1"
    local script="$2"
    local input="$3"
    local timeout="${4:-$TIMEOUT}"

    printf "%-50s " "$name"

    local output
    if [ -n "$input" ]; then
        output=$(echo "$input" | timeout "$timeout" python "$script" 2>&1)
    else
        output=$(timeout "$timeout" python "$script" 2>&1)
    fi
    local rc=$?

    if [ $rc -eq 124 ]; then
        echo "TIMEOUT (${timeout}s)"
        ERRORS+=("$name: TIMEOUT after ${timeout}s")
        ((FAIL++))
        return
    fi

    # Check for common error patterns
    if echo "$output" | grep -qiE "Traceback|Error:|error:|FAILED|exception"; then
        # Filter out expected error handling in output
        if echo "$output" | grep -qiE "Traceback \(most recent call last\)"; then
            echo "FAIL (traceback)"
            ERRORS+=("$name: $(echo "$output" | grep -A1 'Error:' | tail -1)")
            ((FAIL++))
            return
        fi
    fi

    if [ $rc -ne 0 ]; then
        echo "FAIL (exit=$rc)"
        ERRORS+=("$name: exit code $rc")
        ((FAIL++))
        return
    fi

    echo "PASS"
    ((PASS++))
}

echo "============================================================"
echo "Running all examples with: $LLM_MODEL"
echo "Timeout per example: ${TIMEOUT}s"
echo "============================================================"
echo ""

# --- Basic ---
echo "--- Basic Examples ---"
run_example "basic/simple_greeting" \
    "examples/basic/simple_greeting/run.py" \
    "Hello, my name is Alice
How are you?
goodbye"

run_example "basic/form_filling" \
    "examples/basic/form_filling/run.py" \
    "John Smith
john@example.com
I need help with my account
exit"

run_example "basic/story_time" \
    "examples/basic/story_time/run.py" \
    "Tell me a story about a dragon
The dragon should be friendly
What happens next?
goodbye"

echo ""
echo "--- Intermediate Examples ---"
run_example "intermediate/book_recommendation" \
    "examples/intermediate/book_recommendation/run.py" \
    "I like science fiction
Something by Isaac Asimov
Yes that sounds great
goodbye"

run_example "intermediate/product_recommendation" \
    "examples/intermediate/product_recommendation/run.py" \
    "I need a laptop for programming
Budget is around 1500 dollars
Yes I'd prefer lightweight
Thanks that's helpful
goodbye"

run_example "intermediate/adaptive_quiz" \
    "examples/intermediate/adaptive_quiz/run.py" \
    "Python
A
B
C
exit"

echo ""
echo "--- Advanced Examples ---"
run_example "advanced/yoga_instructions" \
    "examples/advanced/yoga_instructions/run.py" \
    "beginner
I want to improve flexibility
yes
exit"

run_example "advanced/e_commerce" \
    "examples/advanced/e_commerce/run.py" \
    "I want to buy a laptop
yes add to cart
checkout
John Smith
123 Main St
credit card
confirm
exit"

run_example "advanced/support_pipeline" \
    "examples/advanced/support_pipeline/run.py" \
    "I can't log into my account
I've tried resetting my password
My email is john@example.com
exit"

echo ""
echo "--- Classification Examples ---"
run_example "classification/intent_routing" \
    "examples/classification/intent_routing/run.py" \
    "I need help with billing
exit"

run_example "classification/smart_helpdesk" \
    "examples/classification/smart_helpdesk/run.py" \
    "How do I reset my password?
exit"

echo ""
echo "--- Reasoning Examples ---"
run_example "reasoning/math_tutor" \
    "examples/reasoning/math_tutor/run.py" \
    "What is 15 * 23?
exit" 180

echo ""
echo "--- Workflow Examples ---"
run_example "workflows/order_processing" \
    "examples/workflows/order_processing/run.py" \
    "" 180

echo ""
echo "--- Agent Examples ---"
run_example "agents/react_search" \
    "examples/agents/react_search/run.py" \
    "What is the population of France?
quit"

run_example "agents/hitl_approval" \
    "examples/agents/hitl_approval/run.py" \
    "y
quit" 180

run_example "agents/react_hitl_combined" \
    "examples/agents/react_hitl_combined/run.py" \
    "What is the capital of Japan?
y
quit"

run_example "agents/plan_execute" \
    "examples/agents/plan_execute/run.py" \
    "" 180

run_example "agents/reflexion" \
    "examples/agents/reflexion/run.py" \
    "" 180

run_example "agents/debate" \
    "examples/agents/debate/run.py" \
    "" 180

run_example "agents/self_consistency" \
    "examples/agents/self_consistency/run.py" \
    "" 180

run_example "agents/rewoo" \
    "examples/agents/rewoo/run.py" \
    "" 180

run_example "agents/prompt_chain" \
    "examples/agents/prompt_chain/run.py" \
    "" 180

run_example "agents/evaluator_optimizer" \
    "examples/agents/evaluator_optimizer/run.py" \
    "" 180

run_example "agents/maker_checker" \
    "examples/agents/maker_checker/run.py" \
    "" 180

run_example "agents/classified_dispatch" \
    "examples/agents/classified_dispatch/run.py" \
    "" 180

run_example "agents/classified_tools" \
    "examples/agents/classified_tools/run.py" \
    "" 180

run_example "agents/full_pipeline" \
    "examples/agents/full_pipeline/run.py" \
    "" 180

run_example "agents/hierarchical_tools" \
    "examples/agents/hierarchical_tools/run.py" \
    "" 180

run_example "agents/reasoning_stacking" \
    "examples/agents/reasoning_stacking/run.py" \
    "" 180

run_example "agents/reasoning_tool" \
    "examples/agents/reasoning_tool/run.py" \
    "" 180

run_example "agents/workflow_agent" \
    "examples/agents/workflow_agent/run.py" \
    "" 180

echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed out of $((PASS + FAIL))"
echo "============================================================"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for err in "${ERRORS[@]}"; do
        echo "  - $err"
    done
fi
