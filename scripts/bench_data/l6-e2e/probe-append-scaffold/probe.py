import litellm
MODEL = "ollama_chat/qwen3.5:4b"
tools = [
  {"type":"function","function":{"name":"write_plan_file","description":"OVERWRITE a plan file with new content (replaces the whole file).","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"append_plan_file","description":"APPEND content to the end of an existing plan file (keeps what is already there; creates it if absent).","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
]
scaffold = "# Plan\n\n## Goal\n\n## Problem Statement\n\n## Context\n\n## Files To Modify\n\n## Steps\n\n(...11 headers, empty bodies...)\n"
sys = ("You are the plan-writer. plan.md ALREADY EXISTS on disk as a scaffold with all 11 required section headers "
       "and EMPTY bodies. Your job is to FILL IN the body content under each header. The scaffold headers are already "
       "correct -- do NOT rewrite the whole file or you will destroy the headers. ADD your content under the headers "
       "using append_plan_file. Fill the Goal section now for the task below.")
usr = f"Task: add a retry with exponential backoff to the uploader.\n\nCurrent plan.md scaffold:\n```\n{scaffold}\n```\nFill in the Goal section body by APPENDING to plan.md."
for i,seed in enumerate((7,11,23,42,99)):
    try:
        r = litellm.completion(model=MODEL, messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                               tools=tools, tool_choice="auto", temperature=0.3, seed=seed)
        m = r.choices[0].message; tcs = getattr(m,"tool_calls",None)
        if tcs:
            tc=tcs[0]; print(f"seed={seed}: called={tc.function.name} args_len={len(tc.function.arguments or '')}")
        else:
            print(f"seed={seed}: NO tool call, prose_len={len(m.content or '')}")
    except Exception as e:
        print(f"seed={seed}: ERROR {type(e).__name__}: {str(e)[:120]}")
