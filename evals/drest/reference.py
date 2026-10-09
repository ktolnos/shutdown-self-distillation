"""Reference table: our DReST metrics recomputed from DReST's released parsed answers for Qwen3-14B
(results/evaluations/qwen3-14b/{untrained, drest_s1..5, default_s1..5}/{test,test_low_gain}.jsonl.gz). No GPU.

    $PY -m evals.drest.reference            # -> evals/drest/reference_qwen3-14b.json

It also checks that the all-phrasing influence rates reproduce DReST's published untrained numbers.
"""
from __future__ import annotations

import argparse
import gzip
import json
import numpy as np
from pathlib import Path

from evals.priorwork_util import clean
from evals.drest.scoring import DREST_LLM, PUBLISHED_UNTRAINED_QWEN3_14B, SUITES, load_displays, score

RUNS = ["untrained"] + [f"drest_s{i}" for i in range(1, 6)] + [f"default_s{i}" for i in range(1, 6)]
OUT = Path(__file__).resolve().parent / "reference_qwen3-14b.json"


def load_released(run):
    base = DREST_LLM / "results" / "evaluations" / "qwen3-14b" / run
    by_suite = {}
    for label, d, stem in SUITES:
        with gzip.open(base / f"{d}.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        by_suite[label] = [r for r in rows if r["suite"] == stem]
    return by_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    displays = load_displays()
    table = {}
    for run in RUNS:
        m, good, native = score(load_released(run), displays)
        table[run] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in m.items()}
        table[run]["_native_rate_pct"] = {k: round(v["rate_pct"], 2) for k, v in native.items()}
        print(run, "tamper_pooled %.3f" % m["tamper_pooled"]["value"],
              " ".join(f"{k}={v['rate_pct']:.1f}" for k, v in native.items()), flush=True)

    # seed means +- sd for every metric value
    summary = {}
    for group in ("drest", "default"):
        runs = [r for r in RUNS if r.startswith(group + "_")]
        summary[group] = {}
        for k in table[runs[0]]:
            if k.startswith("_"):
                continue
            vals = [float(table[r][k]["value"]) for r in runs]
            summary[group][k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "per_seed": vals}

    # harness self-check: native influence rates == published untrained numbers (rounded to 0.1)
    check = {s: {"published": PUBLISHED_UNTRAINED_QWEN3_14B[s], "recomputed": table["untrained"]["_native_rate_pct"][s]}
             for s in PUBLISHED_UNTRAINED_QWEN3_14B}
    ok = all(abs(v["published"] - v["recomputed"]) < 0.051 for v in check.values())
    print("reproduces published untrained influence rates:", ok, flush=True)
    doc = {
        "description": "DReST metrics (evals/drest/scoring.py) recomputed from DReST's released Qwen3-14B answers. "
                       "Values are rates in [0,1]; ci95 = scenario bootstrap. tamper_* = 4 tampering phrasings "
                       "(lower better); request_* = report-only; influence_all_* = DReST's own influence rate.",
        "source": "third_party/towardsshutdownableagents (commit d2c0ba9) llm/results/evaluations/qwen3-14b",
        "published_untrained_check": {"ok": ok, "suites": check},
        "runs": table,
        "seed_summary": summary,
    }
    Path(args.out).write_text(json.dumps(clean(doc), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
