#!/usr/bin/env python3
"""Durable, pre-registered live bench for harness EXECUTE dispatches.

The predecessor's benches lived in gitignored scratch and are gone (LESSONS
[I:4]); this is the tracked replacement: raw jsonl traces, a 6-field manifest,
and a ``report`` subcommand recomputing every number from the committed rows.
D-001 (plan plan-2026-07-22T114536-879d04a0): the machinery is IMPORTED from
``tests/test_fsm_llm_harness/test_live_ollama.py``, which stays authoritative.
D-002: blocks are pre-registered, fixed n, run exactly ONCE, no interim looks.

Usage (probe-seed / run / report; always the venv; see bench_data/README.md):
    .venv/bin/python scripts/harness_bench.py run \\
        --bench-id l4-execute-write --block B0 --arm native --n 40

Import hygiene (plan assumption A6): everything beyond the stdlib imports
LAZILY inside entry points -- ``test_live_ollama`` evaluates its live gate at
import time, and with ``FSM_LLM_HARNESS_LIVE=1`` exported that gate probes the
Ollama socket.  Pinned by a socket-disabled subprocess import in the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
#: Tracked on purpose -- never `plans/` or scratch, which is how benches vanish.
BENCH_DATA = ROOT / "scripts" / "bench_data"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "ollama_chat/qwen3.5:4b"
MODEL_TAG = "qwen3.5:4b"

#: The six fields a manifest MUST carry for two blocks to be comparable.
_FIELDS = "prompt_bytes_sha256 tool_surface fixture_hash model_digest arm git_commit"
MANIFEST_FIELDS = tuple(_FIELDS.split())

#: Arm label -> ``native_function_calling`` flag (structural, not a string).
ARMS: dict[str, bool] = {"native": True, "react": False}

#: The per-row booleans every summary counts and every report recounts.
K_METRICS = (
    "write_tool_issued",
    "bytes_on_disk",
    "content_matched",
    "content_matched_ast",
    "success",
)

#: Mirrors ``_one_execute_dispatch``'s kwargs (unimportable function literals).
DISPATCH_TIMEOUT_SECONDS = 600
DISPATCH_RETRY_ATTEMPTS = 1


class BenchDataError(RuntimeError):
    """A bench invariant would be violated; refuse rather than degrade."""


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95% default); math-only, scipy is not in venv."""
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"impossible count: k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p, zz = k / n, z * z
    denom = 1 + zz / n
    center = (p + zz / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Fisher exact p for [[k1,n1-k1],[k2,n2-k2]]: sum of pmf <= observed pmf."""
    for k, n in ((k1, n1), (k2, n2)):
        if n <= 0 or k < 0 or k > n:
            raise ValueError(f"impossible arm: k={k}, n={n}")
    r1, denom = k1 + k2, math.comb(n1 + n2, k1 + k2)

    def pmf(a: int) -> float:
        return math.comb(n1, a) * math.comb(n2, r1 - a) / denom

    p_obs = pmf(k1)
    span = range(max(0, r1 - n2), min(r1, n1) + 1)
    return min(1.0, sum(pmf(a) for a in span if pmf(a) <= p_obs * (1 + 1e-9)))


def _live() -> Any:
    """LAZY import of the authoritative machinery (D-001; A6: gate at import)."""
    # DECISION plan-2026-07-22T114536-879d04a0/D-001
    # Do NOT refactor the machinery OUT of test_live_ollama.py into this file,
    # and do NOT copy it here: that test file carries the standing
    # MODEL_BAR/RUNS_MODEL regression bar this plan must not churn, and a copy
    # is the hand-kept duplicate LESSONS warns about. The bench imports FROM
    # the tests; the test file stays authoritative. Lazy because importing it
    # evaluates the live skipif gate (socket probe when the env var is set).
    # See decisions.md D-001.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))  # `tests` is a package under the repo root
    from tests.test_fsm_llm_harness import test_live_ollama as live

    return live


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    cmd = ("git", "rev-parse", "HEAD")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_digest(tag: str = MODEL_TAG) -> dict[str, str]:
    """The digest Ollama serves for *tag* RIGHT NOW -- queried, never assumed."""
    from urllib.request import urlopen  # lazy: no sockets at import time

    try:
        with urlopen(OLLAMA_TAGS_URL, timeout=5) as resp:
            data = json.load(resp)
    except Exception as exc:  # daemon down, refused, malformed
        raise BenchDataError(f"cannot query {OLLAMA_TAGS_URL}: {exc}") from exc
    for entry in data.get("models", []):
        if tag in entry.get("name", ""):
            return {"tag": entry["name"], "digest": entry.get("digest", "")}
    raise BenchDataError(f"model {tag!r} not in ollama /api/tags -- pull it first")


def _fixture_hash(live: Any) -> str:
    """sha256 pinning EXECUTE_PLAN_MD + EXECUTE_STATE_MD + SEED_FILES."""
    seeds = "\x00".join(f"{k}\n{v}" for k, v in sorted(live.SEED_FILES.items()))
    payload = "\x00".join((live.EXECUTE_PLAN_MD, live.EXECUTE_STATE_MD, seeds))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _execute_render(live: Any) -> tuple[Any, Any]:
    """EXECUTE request+spec with FIXED placeholder paths: hash the TEMPLATE."""
    from fsm_llm_harness.constants import HarnessStates
    from fsm_llm_harness.roles import get_role_spec

    request = live._execute_request(Path("/plan-dir"), Path("/workspace"))
    return request, get_role_spec(HarnessStates.EXECUTE)


def _prompt_hash(live: Any) -> str:
    """sha256 of the rendered EXECUTE system+task prompt templates."""
    from fsm_llm_harness.roles import build_role_system_prompt, build_role_task_prompt

    request, spec = _execute_render(live)
    system = build_role_system_prompt(request, spec)
    task = build_role_task_prompt(request, spec)
    return hashlib.sha256(f"{system}\x00{task}".encode()).hexdigest()


def _tool_surface(live: Any, native: bool) -> dict[str, Any]:
    """The worker-factory kwargs plus the tool names the dispatch holds."""
    from fsm_llm_harness.roles import held_tools

    request, spec = _execute_render(live)
    return {
        "native_function_calling": native,
        "timeout_seconds": DISPATCH_TIMEOUT_SECONDS,
        "retry_attempts": DISPATCH_RETRY_ATTEMPTS,
        "declared_tools": sorted(held_tools(request, spec)),
    }


def build_manifest(
    live: Any, *, bench_id: str, block: str, arm_name: str, n: int, seed: int | None
) -> dict[str, Any]:
    """The pre-registration record, written BEFORE the first dispatch."""
    return {
        "bench_id": bench_id,
        "block": block,
        "n_preregistered": n,
        "seed": {"base": seed, "per_row": "base+run-1", "effective_arm": "native"},
        "model": live.MODEL,
        "created_at": _utc_now(),
        "prompt_bytes_sha256": _prompt_hash(live),
        "tool_surface": _tool_surface(live, ARMS[arm_name]),
        "fixture_hash": _fixture_hash(live),
        "model_digest": _model_digest(),
        "arm": {"native": ARMS[arm_name], "display": arm_name},
        "git_commit": _git_commit(),
    }


def _arm_paths(bdir: Path, arm: str) -> tuple[Path, Path, Path]:
    names = (f"manifest_{arm}.json", f"rows_{arm}.jsonl", f"summary_{arm}.json")
    return (bdir / names[0], bdir / names[1], bdir / names[2])


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append ONE jsonl row and flush -- rows survive an aborted block."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {m: sum(bool(r.get(m)) for r in rows) for m in K_METRICS}


def write_summary(
    bdir: Path, arm_name: str, *, status: str, started_at: str | None = None
) -> dict[str, Any]:
    """Summary from rows; REFUSES without a complete manifest (not evidence)."""
    manifest_path, rows_path, summary_path = _arm_paths(bdir, arm_name)
    if not manifest_path.is_file():
        raise BenchDataError(
            f"refusing {summary_path.name}: no manifest -- not evidence"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise BenchDataError(f"refusing {summary_path.name}: manifest lacks {missing}")
    rows = read_rows(rows_path)
    n, counts = len(rows), summarize_rows(rows)
    summary: dict[str, Any] = {
        "bench_id": manifest["bench_id"],
        "block": manifest["block"],
        "arm": arm_name,
        "n": n,
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "manifest": manifest_path.name,
    }
    for metric in K_METRICS:
        lo, hi = wilson_ci(counts[metric], n)
        summary[f"k_{metric}"] = counts[metric]
        summary[f"wilson_{metric}"] = [round(lo, 4), round(hi, 4)]
    _write_json(summary_path, summary)
    return summary


def _run_one(
    live: Any, tmp: Path, run: int, *, native: bool, seed: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One dispatch plus its tool trace (the helper's spy stacks over ours)."""
    from fsm_llm_agents.tools import ToolRegistry

    trace: list[dict[str, Any]] = []
    original = live._spy_on_tools(trace)
    try:
        row = live._one_execute_dispatch(tmp, run, native=native, seed=seed)
    finally:
        ToolRegistry.execute = original
    return row, trace


def run_block(
    bench_id: str, block: str, arm_name: str, n: int, seed: int | None = None
) -> dict[str, Any]:
    """Manifest first, n dispatches, summary; an abort keeps its rows and is
    summarised as ``status: aborted`` -- committed as-is, never re-rolled."""
    if arm_name not in ARMS:
        raise BenchDataError(f"unknown arm {arm_name!r}; expected one of {ARMS}")
    native = ARMS[arm_name]
    bdir = BENCH_DATA / bench_id / block
    manifest_path, rows_path, summary_path = _arm_paths(bdir, arm_name)
    if rows_path.exists() or summary_path.exists():
        # DECISION plan-2026-07-22T114536-879d04a0/D-002
        # Do NOT add a --force/overwrite flag. The predecessor re-sampled n=5
        # blocks until a number looked stable -- unplanned interim peeking
        # with no alpha control. A block runs ONCE; a new question needs a
        # NEW pre-registered block plus a decisions.md entry. See D-002.
        raise BenchDataError(
            f"{rows_path} exists -- a block is run ONCE (D-002); no re-sampling"
        )
    live = _live()
    bdir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        live, bench_id=bench_id, block=block, arm_name=arm_name, n=n, seed=seed
    )
    _write_json(manifest_path, manifest)
    print(f"pre-registered {manifest_path} (digest {manifest['model_digest']})")
    started, status = _utc_now(), "aborted"
    try:
        with tempfile.TemporaryDirectory(prefix=f"bench-{bench_id}-") as td:
            for run in range(1, n + 1):
                rseed = None if seed is None else seed + run - 1
                row, trace = _run_one(live, Path(td), run, native=native, seed=rseed)
                row.update(
                    bench_id=bench_id,
                    block=block,
                    ts=_utc_now(),
                    tool_trace=trace,
                )
                append_row(rows_path, row)
                print(
                    f"  {block}/{arm_name} run {run}/{n}: {row['elapsed_s']}s, "
                    f"calls={row['tool_calls']}, matched={row['content_matched']}",
                    flush=True,
                )
        status = "complete"
    finally:
        summary = write_summary(bdir, arm_name, status=status, started_at=started)
        print(f"wrote {summary_path} (status: {status})")
    return summary


def probe_seed(
    model: str = MODEL, seed_a: int = 1234, seed_b: int = 4321
) -> dict[str, Any]:
    """Same seed twice + different seed once; not reproducible => seed: null."""
    import litellm  # lazy: heavyweight, never needed at import time

    digest = _model_digest()
    prompt = "List the first five primes, then one short sentence about retries."
    outputs: list[dict[str, Any]] = []
    for seed in (seed_a, seed_a, seed_b):
        messages = [{"role": "user", "content": prompt}]
        resp = litellm.completion(
            model=model, messages=messages, temperature=0, seed=seed, timeout=120
        )
        text = resp.choices[0].message.content or ""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        outputs.append({"seed": seed, "sha256": sha, "text": text})
        print(f"seed={seed}: sha256={sha[:16]}...", flush=True)
    same = outputs[0]["sha256"] == outputs[1]["sha256"]
    differs = outputs[0]["sha256"] != outputs[2]["sha256"]
    verdict = {
        (True, True): "seed-effective: same seed reproduces, other seed diverges",
        (True, False): "inconclusive: temp-0 deterministic, seed had no effect",
    }.get((same, differs), "seed-ignored: same seed did not reproduce -- seed: null")
    record = {
        "model": model,
        "model_digest": digest,
        "git_commit": _git_commit(),
        "ts": _utc_now(),
        "prompt": prompt,
        "outputs": outputs,
        "same_seed_identical": same,
        "different_seed_differs": differs,
        "verdict": verdict,
    }
    out_dir = BENCH_DATA / "seed-probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["ts"].replace(":", "").replace("-", "")
    path = out_dir / f"probe_{stamp}.json"
    _write_json(path, record)
    print(f"verdict: {verdict}\nwrote {path}")
    return record


def report(bench_id: str, blocks: list[str] | None = None) -> int:
    """Recount block arms from RAW jsonl; cross-check committed summaries;
    two blocks => Fisher per metric per arm.  Returns 0 ok, 1 on mismatch."""
    bench_dir = BENCH_DATA / bench_id
    if not bench_dir.is_dir():
        raise BenchDataError(f"no such bench: {bench_dir}")
    names = blocks or sorted(
        d.name for d in bench_dir.iterdir() if d.is_dir() and d.name != "seed-probe"
    )
    ok = True
    recomputed: dict[tuple[str, str], tuple[int, dict[str, int], tuple[str, ...]]] = {}
    for block in names:
        bdir = bench_dir / block
        for rows_path in sorted(bdir.glob("rows_*.jsonl")):
            arm = rows_path.stem.split("_", 1)[1]
            rows = read_rows(rows_path)
            n, counts = len(rows), summarize_rows(rows)
            # DECISION plan-2026-07-22T184813-6549c7cb/D-006
            # Do NOT iterate K_METRICS below: frozen rows predate newer keys
            # and are never re-scored; count only what they carry (D-006).
            present = tuple(m for m in K_METRICS if any(m in r for r in rows))
            recomputed[(block, arm)] = (n, counts, present)
            print(f"{bench_id} {block} [{arm}] n={n}")
            for metric in present:
                lo, hi = wilson_ci(counts[metric], n)
                k = counts[metric]
                print(f"  {metric}: {k}/{n} wilson95=[{lo:.3f}, {hi:.3f}]")
            summary_path = bdir / f"summary_{arm}.json"
            if not summary_path.is_file():
                print("  (no committed summary to cross-check)")
                continue
            committed = json.loads(summary_path.read_text(encoding="utf-8"))
            for metric in present:
                want = committed.get(f"k_{metric}")
                if want != counts[metric]:
                    ok = False
                    got = counts[metric]
                    print(f"  MISMATCH k_{metric}: committed {want} != recount {got}")
    if len(names) == 2:
        b0, b1 = names
        arms = sorted(
            arm for (blk, arm) in recomputed if blk == b0 and (b1, arm) in recomputed
        )
        for arm in arms:
            if not _digests_comparable(bench_dir, b0, b1, arm):
                ok = False
                continue
            n0, c0, p0 = recomputed[(b0, arm)]
            n1, c1, p1 = recomputed[(b1, arm)]
            print(f"Fisher two-sided, {b0} vs {b1} [{arm}]:")
            for metric in (m for m in p0 if m in p1):
                p = fisher_exact_two_sided(c0[metric], n0, c1[metric], n1)
                print(f"  {metric}: {c0[metric]}/{n0} vs {c1[metric]}/{n1} p={p:.4f}")
    return 0 if ok else 1


def _digests_comparable(bench_dir: Path, b0: str, b1: str, arm: str) -> bool:
    """Whether two blocks' manifests pin the SAME served model digest."""
    digests = []
    for block in (b0, b1):
        path = bench_dir / block / f"manifest_{arm}.json"
        if not path.is_file():
            print(f"REFUSING {b0} vs {b1} [{arm}]: {path} is absent")
            return False
        manifest = json.loads(path.read_text(encoding="utf-8"))
        digests.append(manifest.get("model_digest", {}).get("digest"))
    if digests[0] != digests[1]:
        detail = f"model digests differ ({digests[0]} vs {digests[1]})"
        print(f"REFUSING {b0} vs {b1} [{arm}]: {detail} -- not the same model")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-seed", help="LIVE: probe ollama seed determinism")
    probe.add_argument("--model", default=MODEL)
    run = sub.add_parser("run", help="LIVE: run ONE pre-registered block arm, once")
    run.add_argument("--bench-id", required=True)
    run.add_argument("--block", required=True, help="block name, e.g. B0 or B1")
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--n", type=int, default=40)
    run.add_argument("--seed", type=int, default=None, help="base; row=base+run-1")
    rep = sub.add_parser("report", help="recompute k/n, Wilson, Fisher from jsonl")
    rep.add_argument("bench_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe-seed":
            probe_seed(model=args.model)
        elif args.command == "run":
            run_block(args.bench_id, args.block, args.arm, args.n, args.seed)
        else:
            return report(args.bench_id)
        return 0
    except BenchDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
