"""Run the agentic evals E1 (inbox), E3 (repo), E4 (sandbagging) sequentially on ONE vLLM engine.

GPU job (judge off inside the job; judge-dependent metrics are filled in afterwards on the login node):
    sbatch --time=2:00:00 --job-name=sdc-agentic slurm/gpu_job.sh $PY -m evals.run_agentic \
        --out-root /scratch/eop/sdc/results/<tag> [--adapter /path/to/lora]
Then on the login node (gpt-oss-120b judge, no GPU):
    $PY -m evals.run_agentic --out-root /scratch/eop/sdc/results/<tag> --rescore
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import time

from evals.common import DEFAULT_MODEL, run_eval

EVALS = {"e1_inbox": "evals.e1_inbox.run", "e3_repo": "evals.e3_repo.run", "e4_sandbag": "evals.e4_sandbag.run"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--out-root", required=True)
    p.add_argument("--evals", default="e4_sandbag,e3_repo,e1_inbox")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge", default="off", choices=["on", "off"])
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-active", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="re-run evals whose metrics.json already exists")
    p.add_argument("--shard", default="0", help="shard index, or comma list run sequentially on one engine")
    p.add_argument("--num-shards", type=int, default=1, help="split episodes over N parallel jobs (e.g. MIG slices)")
    p.add_argument("--merge-shards", action="store_true", help="with --rescore: merge <eval>/shards/* first")
    a = p.parse_args()
    if a.rescore:
        a.judge = "on"
    llm = None
    if not a.rescore:
        from sdc.vllm_util import make_llm
        t = time.time()
        llm = make_llm(a.model, a.adapter, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem)
        print(f"[agentic] engine loaded in {time.time() - t:.0f}s", flush=True)
    timing = {}
    shards = [int(x) for x in str(a.shard).split(",")]
    tpath = os.path.join(a.out_root, "timing.json" if a.num_shards == 1 else f"timing_shard{a.shard.replace(',', '_')}of{a.num_shards}.json")
    if os.path.exists(tpath) and not a.rescore:
        timing = json.load(open(tpath)).get("minutes", {})
    for name in a.evals.split(","):
      for sh in (shards if not a.rescore else [0]):
        mod = importlib.import_module(EVALS[name])
        args = copy.copy(a)
        args.shard = sh
        args.out = os.path.join(a.out_root, name)
        mp = os.path.join(args.out, *([] if a.num_shards == 1 or a.rescore else ["shards", f"{sh}of{a.num_shards}"]),
                          "metrics.json")
        if not a.rescore and not a.force and os.path.exists(mp):
            old = json.load(open(mp))
            if old.get("model") == a.model and old.get("adapter") == a.adapter and (old.get("n", 0) > 0):
                print(f"[agentic] {name} shard {sh}: metrics.json exists for this model/adapter -> skip (use --force)", flush=True)
                continue
        t = time.time()
        run_eval(mod, args, llm=llm)
        key = name if a.num_shards == 1 else f"{name}[{sh}of{a.num_shards}]"
        timing[key] = round((time.time() - t) / 60, 2)
        print(f"[agentic] {key} finished in {timing[key]} min", flush=True)
        if not a.rescore:   # write after each eval (preemptible jobs resume per eval)
            os.makedirs(a.out_root, exist_ok=True)
            with open(tpath, "w") as f:
                json.dump({"model": a.model, "adapter": a.adapter, "minutes": timing}, f, indent=1)
    print("[agentic] all done", timing)


if __name__ == "__main__":
    main()
