# Live probe: does 4b append (not overwrite) a seeded plan.md scaffold?

## Summary
The scaffold+append design's #1 risk (F1 finding 9 + F4): `write_plan_file`
OVERWRITES the seeded scaffold; the model must instead pick `append_plan_file`.
A direct `litellm.completion` probe (5 seeds, `ollama_chat/qwen3.5:4b`, temp 0.3,
both tools offered, `tool_choice="auto"`) shows 4b reliably picks the append
tool and authors real content when told the scaffold already exists.

## The probe (scratch, not a pre-registered bench)
System prompt: "plan.md ALREADY EXISTS as a scaffold with all 11 headers and
EMPTY bodies. FILL IN the bodies. The headers are correct — do NOT rewrite the
whole file or you'll destroy them. ADD content under the headers using
append_plan_file." Both `write_plan_file` (overwrite) and `append_plan_file`
(append) offered.

| seed | tool called | args_len |
|---|---|---|
| 7  | **append_plan_file** | 396 |
| 11 | **append_plan_file** | 421 |
| 23 | **append_plan_file** | 351 |
| 42 | **append_plan_file** | 411 |
| 99 | **append_plan_file** | 379 |

**5/5 append_plan_file, 0/5 write_plan_file**, every call carrying substantive
content (351-421 chars).

## What this establishes
1. **The core scaffold+append assumption HOLDS**: with a clear "scaffold exists,
   append under each header" instruction, 4b chooses the append tool over the
   overwrite tool every time — so a seeded scaffold survives the plan-writer's
   write. This clears the tool-seam risk F1/F4 flagged as needing a probe.
2. **The model authors real content** (not a no-op) — consistent with the ethos
   requirement (the MODEL fills the sections; the driver only seeded headers).

## Caveats / limits
- n=5, one section (Goal), single-turn — a smoke test of the TOOL CHOICE, NOT a
  measurement of whether 4b fills ALL 11 sections substantively across a full
  14-turn plan-writer dispatch. That is what L6 B3 measures.
- Does not test the honest-approval interaction (a placeholder plan being
  denied) — that is unit-tested + measured in B3.
- The append instruction must actually reach the model in the real dispatch (the
  step-2 prompt/tool-scope change) — the probe used an explicit hand-written
  instruction; the build must render an equivalent one.
