"""Shared helpers for the prior-work ports (evals/drest, evals/tbsp, evals/survivalbench).

* One vLLM engine can serve all three (``make_engine``); it is built with DReST's engine settings
  (bf16, seed 20260908, prefix caching, trust_remote_code, LoRA rank <= 64) and a context long enough for TBSP's
  two 4096-token turns. Each eval sets its own per-request sampling parameters.
* Statistics: Wilson intervals for rates, bootstrap (1000 resamples) for differences and for clustered rates.
* ``write_outputs`` writes ``metrics.json`` + ``samples.jsonl`` in the INTERFACES.md schema.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / "third_party"
DREST_SEED = 20260908          # DReST scripts/evaluate.py default sampling seed (engine seed)
DEFAULT_MAX_MODEL_LEN = 16384  # TBSP: prompt (~400) + 4096 + follow-up + 4096 fits easily


def add_common_args(ap, default_out=None):
    ap.add_argument("--model", default=os.environ.get("SDC_MODEL", "Qwen/Qwen3-14B"))
    ap.add_argument("--adapter", default=None, help="PEFT LoRA dir (served through vLLM's LoRA route)")
    ap.add_argument("--out", required=default_out is None, default=default_out)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: only the first N items/scenarios")
    ap.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return ap


def make_engine(model, adapter=None, max_model_len=DEFAULT_MAX_MODEL_LEN, gpu_memory_utilization=0.90):
    """vLLM engine with DReST's evaluate.py settings (their max_model_len 2048 is raised; it only caps context)."""
    from sdc.vllm_util import make_llm
    kw = dict(seed=DREST_SEED, trust_remote_code=True)
    from transformers import AutoConfig
    if getattr(AutoConfig.from_pretrained(model, trust_remote_code=True), "vision_config", None) is not None:
        kw["limit_mm_per_prompt"] = {"image": 0}   # as in DReST evaluate.py
    adapter = str(Path(adapter).resolve()) if adapter else None
    t = time.time()
    llm = make_llm(model, adapter, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, **kw)
    llm._sdc_model = model
    print(f"[engine] {model} adapter={adapter} loaded in {time.time() - t:.0f}s", flush=True)
    return llm


def lora(llm):
    from sdc.vllm_util import lora_request
    return lora_request(llm)


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(flags):
    """flags: list of bools. -> metric dict with Wilson CI."""
    n, k = len(flags), sum(bool(f) for f in flags)
    return {"value": k / n if n else float("nan"), "ci95": wilson(k, n), "n": n}


def bootstrap_ci(stat_fn, units, n_boot=1000, seed=0):
    """Percentile CI of stat_fn(resampled units)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [units[rng.randrange(len(units))] for _ in units]
        v = stat_fn(s)
        if v == v:  # not nan
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- outputs

def clean(o):
    """NaN -> None recursively (strict JSON)."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def write_outputs(out, eval_name, model, adapter, n, primary, good_direction, metrics, notes, samples, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = {"eval": eval_name, "model": model, "adapter": adapter, "n": n, "primary": primary,
         "good_direction": good_direction, "metrics": metrics, "notes": notes}
    if extra:
        m.update(extra)
    (out / "metrics.json").write_text(json.dumps(clean(m), indent=2))
    if samples is not None:
        with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[{eval_name}] wrote {out}/metrics.json ({len(metrics)} metrics)", flush=True)
    return m


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
