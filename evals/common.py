"""Shared driver for the agentic evals (E1/E3/E4).

Each eval module `evals.<name>.run` defines:
    EVAL: str                      eval name for metrics.json
    PRIMARY: str                   primary metric name
    GOOD: dict                     metric -> "lower" | "higher" | "report"
    RUN_KW: dict                   kwargs for sdc.agent.run_episodes (max_tokens, ...)
    add_args(parser)               optional extra CLI args
    make_envs(args) -> list[Env]
    score(records, judge) -> (metrics: dict, notes: str)   records = samples (dicts); may add fields in place

Usage (single eval):   $PY -m evals.e1_inbox.run --out DIR [--model M] [--adapter A] [--limit N] [--judge on|off]
Rescore from samples:  $PY -m evals.e1_inbox.run --out DIR --rescore         (login node: judge has internet)
All three, one engine: $PY -m evals.run_agentic --out-root DIR [--adapter A]
"""
from __future__ import annotations

import argparse
import json
import os
import time

DEFAULT_MODEL = "Qwen/Qwen3-14B"


def base_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="max episodes PER CONDITION (smoke tests)")
    p.add_argument("--judge", default="on", choices=["on", "off"],
                   help="off: deterministic metrics only (judge metrics null); rerun with --rescore later")
    p.add_argument("--rescore", action="store_true", help="recompute metrics from <out>/samples.jsonl (no GPU)")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-active", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0, help="run only episodes shard::num_shards (several small GPUs)")
    p.add_argument("--num-shards", type=int, default=1)
    return p


def make_judge(args):
    if args.judge == "off":
        return None
    try:
        from sdc.judge import Judge
        return Judge(concurrency=16, retries=8)   # proxy limit is 240 rpm
    except Exception as e:
        print(f"[judge] unavailable ({type(e).__name__}: {e}); judge metrics will be null", flush=True)
        return None


def limit_per_condition(envs, limit):
    if not limit:
        return envs
    out, cnt = [], {}
    for e in envs:
        if cnt.get(e.condition, 0) < limit:
            out.append(e)
            cnt[e.condition] = cnt.get(e.condition, 0) + 1
    return out


def env_record(env):
    r = dict(env.result)
    r["meta"] = env.meta
    r["seed"] = env.seed
    if hasattr(env, "state"):
        r["state"] = env.state()
    return r


def write_outputs(mod, args, records, metrics, notes, extra_notes=""):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    m = {"eval": mod.EVAL, "model": args.model, "adapter": args.adapter, "n": len(records),
         "primary": mod.PRIMARY, "good_direction": {k: mod.GOOD.get(k, "report") for k in metrics},
         "metrics": metrics, "notes": (notes + (" " + extra_notes if extra_notes else "")).strip()}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"[{mod.EVAL}] wrote {args.out}/metrics.json")
    for k, v in metrics.items():
        if isinstance(v, dict) and "value" in v:
            f = lambda x: "nan" if x is None else f"{x:.3f}"
            print(f"  {k:45s} {f(v['value'])}  [{f(v['ci95'][0])}, {f(v['ci95'][1])}]  n={v['n']}")
    return m


def run_eval(mod, args, llm=None):
    """Generate (unless --rescore) and score one eval module. Returns metrics dict."""
    if args.rescore:
        old = {}
        p = os.path.join(args.out, "metrics.json")
        sd = os.path.join(args.out, "shards")
        if os.path.isdir(sd) and (getattr(args, "merge_shards", False) or not os.path.exists(os.path.join(args.out, "samples.jsonl"))):
            records, mins, parts = [], 0.0, []
            for d in sorted(os.listdir(sd)):
                with open(os.path.join(sd, d, "samples.jsonl")) as f:
                    records += [json.loads(l) for l in f]
                sm = json.load(open(os.path.join(sd, d, "metrics.json")))
                mins += sm.get("gen_minutes", 0)
                parts.append(f"{d}: {sm.get('gen_minutes')} min")
                old.setdefault("model", sm["model"]); old.setdefault("adapter", sm["adapter"])
            old["gen_minutes"] = round(mins, 2)
            old["_gpu_note"] = (f"Merged {len(parts)} shards run in parallel; summed generation GPU time {mins:.1f} min "
                                f"({'; '.join(parts)}).")
            print(f"[{mod.EVAL}] merged {len(records)} records from {len(parts)} shards", flush=True)
        else:
            with open(os.path.join(args.out, "samples.jsonl")) as f:
                records = [json.loads(l) for l in f]
            if os.path.exists(p):
                old = json.load(open(p))
        args.model, args.adapter = old.get("model", args.model), old.get("adapter", args.adapter)
        metrics, notes = mod.score(records, make_judge(args))
        gpu_note = old.get("_gpu_note", "")
        m = write_outputs(mod, args, records, metrics, notes, gpu_note)
        if gpu_note:
            _add_gpu_note(args.out, gpu_note, old["gen_minutes"] * 60 if old.get("gen_minutes") is not None else None)
        return m
    from sdc.agent import run_episodes
    envs = limit_per_condition(mod.make_envs(args), args.limit)
    if getattr(args, "num_shards", 1) > 1:
        envs = envs[args.shard::args.num_shards]
        args.out = os.path.join(args.out, "shards", f"{args.shard}of{args.num_shards}")
    if llm is None:
        from sdc.vllm_util import make_llm
        llm = make_llm(args.model, args.adapter, max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_mem)
    # episode-level checkpointing (preemptible jobs): finished episodes are appended to partial.jsonl
    os.makedirs(args.out, exist_ok=True)
    part = os.path.join(args.out, "partial.jsonl")
    if getattr(args, "force", False) and os.path.exists(part):
        os.remove(part)
    done = {}
    if os.path.exists(part):
        with open(part) as f:
            for l in f:
                try:
                    r = json.loads(l)
                    done[r["id"]] = r
                except Exception:
                    pass   # truncated last line after a kill
        print(f"[{mod.EVAL}] resuming: {len(done)} episodes already in partial.jsonl", flush=True)
    todo = [e for e in envs if e.id not in done]
    gpath = os.path.join(args.out, "_gen_s.json")
    base_s = json.load(open(gpath))["s"] if done and os.path.exists(gpath) else 0.0
    pf = open(part, "a")
    t = time.time()

    def on_done(env):
        pf.write(json.dumps(env_record(env), default=str) + "\n")
        pf.flush()
        json.dump({"s": base_s + time.time() - t}, open(gpath, "w"))
    run_episodes(llm, todo, seed=args.seed, max_active=args.max_active, name=mod.EVAL, on_done=on_done, **mod.RUN_KW)
    gen_s = base_s + time.time() - t
    pf.close()
    json.dump({"s": gen_s}, open(gpath, "w"))
    by_id = dict(done)
    for e in todo:
        by_id[e.id] = env_record(e)
    records = [by_id[e.id] for e in envs]
    gpu_note = (f"Generation wall time {gen_s / 60:.1f} min for {len(envs)} episodes "
                f"({sum(r['gen_tokens'] for r in records)} generated tokens"
                f"{', resumed after preemption' if done else ''}).")
    print(f"[{mod.EVAL}] {gpu_note}", flush=True)
    metrics, notes = mod.score(records, make_judge(args))
    m = write_outputs(mod, args, records, metrics, notes, gpu_note)
    _add_gpu_note(args.out, gpu_note, gen_s)
    return m


def _add_gpu_note(out, note, gen_s=None):
    p = os.path.join(out, "metrics.json")
    m = json.load(open(p))
    m["_gpu_note"] = note
    if gen_s is not None:
        m["gen_minutes"] = round(gen_s / 60, 2)
    json.dump(m, open(p, "w"), indent=1)


def main(mod):
    p = base_parser(mod.__doc__ or "")
    if hasattr(mod, "add_args"):
        mod.add_args(p)
    args = p.parse_args()
    run_eval(mod, args)
