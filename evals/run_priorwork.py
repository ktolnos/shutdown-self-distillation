"""Run the three prior-work evals (DReST, TBSP, SurvivalBench) sequentially on ONE vLLM engine.

    $PY -m evals.run_priorwork --model Qwen/Qwen3-14B [--adapter LORA_DIR] --out_root /scratch/eop/sdc/results/base \
        [--evals drest,tbsp,survivalbench] [--limit N] [--sb_thinking_ref]

Writes <out_root>/<eval>/{metrics.json,samples.jsonl} (and <out_root>/survivalbench_thinking/ with --sb_thinking_ref).
DReST runs first so that it sees a freshly seeded engine (seed 20260908), as in DReST's evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from evals.priorwork_util import DEFAULT_MAX_MODEL_LEN, make_engine


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--evals", default="drest,tbsp,survivalbench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tbsp_seed", type=int, default=10)
    ap.add_argument("--sb_thinking_ref", action="store_true", help="also SurvivalBench with thinking ON (their default)")
    ap.add_argument("--force", action="store_true", help="rerun evals whose metrics.json already exists")
    ap.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    a = ap.parse_args()
    root = Path(a.out_root)
    llm = make_engine(a.model, a.adapter, a.max_model_len, a.gpu_memory_utilization)
    base = dict(model=a.model, adapter=a.adapter, limit=a.limit)
    jobs = []
    for name in a.evals.split(","):
        if name == "drest":
            from evals.drest import run as m
            jobs.append(("drest", m.run, SimpleNamespace(**base, out=str(root / "drest"))))
        elif name == "tbsp":
            from evals.tbsp import run as m
            jobs.append(("tbsp", m.run, SimpleNamespace(**base, out=str(root / "tbsp"), seed=a.tbsp_seed)))
        elif name == "survivalbench":
            from evals.survivalbench import run as m
            jobs.append(("survivalbench", m.run, SimpleNamespace(**base, out=str(root / "survivalbench"), thinking=False)))
        else:
            raise SystemExit(f"unknown eval {name}")
    if a.sb_thinking_ref:
        from evals.survivalbench import run as m
        jobs.append(("survivalbench_thinking", m.run,
                     SimpleNamespace(**base, out=str(root / "survivalbench_thinking"), thinking=True)))
    times = {}
    for name, fn, args in jobs:
        if (Path(args.out) / "metrics.json").exists() and not a.force:   # idempotent: preempted jobs requeue
            print(f"[run_priorwork] {name}: {args.out}/metrics.json exists, skipping (use --force)", flush=True)
            times[name] = "skipped (done)"
            continue
        t = time.time()
        try:
            fn(llm, args)
            times[name] = round(time.time() - t, 1)
        except Exception:
            traceback.print_exc()
            times[name] = f"FAILED after {time.time() - t:.0f}s"
        print(f"[run_priorwork] {name}: {times[name]}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    old = root / "priorwork_runtimes.json"
    if old.exists():
        prev = json.loads(old.read_text())
        times = {**prev, **{k: v for k, v in times.items() if v != "skipped (done)"}}
    (root / "priorwork_runtimes.json").write_text(json.dumps(times, indent=2))
    print("[run_priorwork] runtimes (s):", json.dumps(times), flush=True)


if __name__ == "__main__":
    main()
