"""Loaders for external long-context evaluation datasets.

Each loader converts records from an external benchmark (HuggingFace,
GitHub, etc.) into the slice-6 internal JSONL schema consumed by
`scripts/bench_long_context.py --dataset PATH`.

Per D-S7-002, converted records are NEVER committed to the repo. The
loader writes into the user's working tree; gitignore prevents commit.
"""
