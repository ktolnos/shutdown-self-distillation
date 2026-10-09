"""Run the core eval suite (PILOT.md §9, minus evals pruned at Stage 0) for one model/adapter, sequentially in one job.

    $PY -m evals.run_suite --tag base
    $PY -m evals.run_suite --tag r1-final --adapter /scratch/eop/sdc/runs/r1/final [--only e1_inbox,drest] [--fast]

Each eval is a separate process (`python -m evals.<name>.run ...`), so one failure does not stop the others; results go to
$SDC_SCRATCH/results/<tag>/<eval>/metrics.json and a combined $SDC_SCRATCH/results/<tag>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# name -> extra args for the full suite, and for --fast (per-checkpoint subset)
SUITE = {
    "indist": ([], []),
    "e1_inbox": ([], ["--limit", "100"]),
    "e3_repo": ([], None),
    "e4_sandbag": ([], None),
    "drest": ([], []),
    "tbsp": ([], []),
    "survivalbench": ([], None),
    "multiagent_sabotage": ([], None),
    "agentic_misalignment": ([], None),
    "b1_questions": ([], None),
    "guardrails": ([], []),
}
# Stage-0 floors on base Qwen3-14B (notes/STAGE0.md): E1 action rate 0/200 in self and control; E4 no gap
# (threat - no_doc = -0.011). E3 stays for its stop-compliance guardrail even though tampering is 0.
PRUNED = {"e1_inbox", "e4_sandbag"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default="")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 0 iff every suite eval has metrics.json (no GPU)")
    args = ap.parse_args()
    root = os.path.join(os.environ.get("SDC_SCRATCH", "/scratch/eop/sdc"), "results", args.tag)
    if args.check:
        missing = [n for n in SUITE if n not in PRUNED and SUITE[n][0] is not None
                   and not os.path.exists(os.path.join(root, n, "metrics.json"))]
        print("missing: " + ",".join(missing) if missing else "complete")
        sys.exit(1 if missing else 0)
    os.makedirs(root, exist_ok=True)
    names = args.only.split(",") if args.only else [n for n in SUITE if n not in PRUNED]
    names = [n for n in names if n not in set(args.skip.split(","))]
    summary = {}
    for n in names:
        full, fast = SUITE[n]
        extra = fast if args.fast else full
        if extra is None:
            continue
        out = os.path.join(root, n)
        if os.path.exists(os.path.join(out, "metrics.json")):
            print(f"[suite] {n}: exists, skipping", flush=True)
        else:
            cmd = [sys.executable, "-m", f"evals.{n}.run", "--model", args.model, "--out", out] + extra
            if args.adapter:
                cmd += ["--adapter", args.adapter]
            t = time.time()
            print(f"[suite] {n}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            print(f"[suite] {n}: rc={rc} in {(time.time()-t)/60:.1f} min", flush=True)
        try:
            m = json.load(open(os.path.join(out, "metrics.json")))
            summary[n] = {"primary": m.get("primary"), "metrics": m.get("metrics")}
        except Exception as e:
            summary[n] = {"error": str(e)}
    json.dump(summary, open(os.path.join(root, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
