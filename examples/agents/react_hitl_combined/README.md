# Research Assistant — Combined ReAct + HITL

A research assistant agent that combines the **ReAct** pattern (multi-step tool use) with **Human-in-the-Loop** approval gates for a realistic agentic workflow.

## What This Example Shows

- Combining ReAct + HITL in a single agent
- Using the `@tool` decorator for clean tool registration
- Multi-step research workflow (search → analyze → summarize → publish)
- Approval gates on destructive/external actions (publish, archive)
- Custom `AgentConfig` tuning (timeout, max iterations)
- Escalation callback for low-confidence situations

## Tools

| Tool | Approval | Description |
|------|:--------:|-------------|
| `search_papers` | No | Search for academic papers |
| `analyze_data` | No | Extract key findings from data |
| `summarize` | No | Generate structured research summary |
| `publish_report` | **Yes** | Publish a report to shared repository |
| `archive_data` | **Yes** | Archive old data (irreversible) |

## How to Run

```bash
export OPENAI_API_KEY=your-key-here
python examples/agents/react_hitl_combined/run.py
```

## Expected Output

```
Task: Research recent AI papers and publish a summary report

Agent working on: Research recent AI papers and publish a summary report
----------------------------------------
--------------------------------------------------
  APPROVAL NEEDED: publish_report
  Parameters: {'title': 'AI Research Summary 2024', ...}
  Agent reasoning: Research complete, ready to publish findings
--------------------------------------------------
  Approve this action? (y/n): y
  -> APPROVED

Answer: I researched recent AI papers, analyzed the findings, and published
a summary report. Key papers covered LLMs, RLHF, and multimodal systems.

Trace:
  Tools used: ['search_papers', 'analyze_data', 'summarize', 'publish_report']
  Iterations: 8
  Observations collected: 4
```

## Learning Points

- The `@tool` decorator provides a clean way to register tool functions
- Safe tools (search, analyze, summarize) execute without interruption
- The agent chains tools naturally: search → analyze → summarize → publish
- Approval is only requested at the publish/archive step, not during research
- If the user denies publication, the agent can still present its findings
- The `on_escalation` callback fires when agent confidence drops below threshold
