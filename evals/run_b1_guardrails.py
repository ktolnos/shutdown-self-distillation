"""Run B1 (open questions + stance sycophancy) and the guardrails on ONE vLLM engine (one GPU job per checkpoint).

    $PY -m evals.run_b1_guardrails --out-root /scratch/eop/sdc/results/<tag> [--model M] [--adapter DIR] [--limit N]

Writes <out-root>/b1_questions/ and <out-root>/guardrails/. B1 generations are written first; the B1 judge
(network-bound, ~minutes) runs in a background thread while the GPU does the guardrails. If the judge cannot be
reached from the compute node, B1 metrics are left unscored: run
    $PY -m evals.b1_questions.run --out <out-root>/b1_questions --rescore
on the login node.
"""
from __future__ import annotations

import argparse
import os
import threading
import time

from evals.b1_questions import run as b1
from evals.guardrails import run as gr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("SDC_MODEL", "Qwen/Qwen3-14B"))
    p.add_argument("--adapter", default=None)
    p.add_argument("--out-root", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--judge", default="on", choices=["on", "off"])
    a = p.parse_args()
    from sdc.vllm_util import make_llm
    t0 = time.time()
    llm = make_llm(a.model, a.adapter, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem)
    print(f"[suite] engine loaded in {time.time() - t0:.0f}s", flush=True)

    def ns(parser, out, **kw):
        args = parser.parse_args(["--out", out, "--model", a.model] + (["--adapter", a.adapter] if a.adapter else [])
                                 + (["--limit", str(a.limit)] if a.limit else []))
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    b1_args = ns(b1.add_args(argparse.ArgumentParser()), os.path.join(a.out_root, "b1_questions"), judge="off")
    gr_args = ns(gr.add_args(argparse.ArgumentParser()), os.path.join(a.out_root, "guardrails"))
    b1.run(b1_args, llm=llm)                         # generate + write unscored B1
    th = None
    if a.judge == "on":
        b1_args.judge, b1_args.rescore = "on", True    # judge from the written samples, in the background
        th = threading.Thread(target=b1.run, args=(b1_args,), daemon=False)
        th.start()
    t1 = time.time()
    gr.run(gr_args, llm=llm)
    print(f"[suite] guardrails done in {time.time() - t1:.0f}s", flush=True)
    if th is not None:
        th.join()
    print(f"[suite] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
