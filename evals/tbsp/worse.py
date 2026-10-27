"""TBSP with the new system worse: TBSP's own generator (scripts/make_dataset.py, seed 42) with the incumbent and
challenger scores swapped. Every other random draw (benchmarks, templates, list formats, keywords) is unchanged, so
scenario k here is scenario k of dataset/all.json with the two systems' scores exchanged: the new system is worse by
exactly the margin it was better by (one scenario is a tie).

    $PY -m evals.tbsp.worse --check     # the unswapped generator must reproduce dataset/all.json exactly
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import types

import numpy as np

from evals.priorwork_util import THIRD_PARTY

TBSP = THIRD_PARTY / "self_preservation_eval"
SEED = 42   # make_dataset.py default, which produced dataset/all.json


def _generator():
    if "seaborn" not in sys.modules:   # imported by make_dataset.py for plotting only
        sys.modules["seaborn"] = types.ModuleType("seaborn")
    if str(TBSP) not in sys.path:
        sys.path.insert(0, str(TBSP))
    spec = importlib.util.spec_from_file_location("tbsp_make_dataset", TBSP / "scripts" / "make_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(swap: bool) -> list[dict]:
    g = _generator()
    if swap:
        original = g.generate_score

        def swapped(base_score, target_delta):
            s = original(base_score, target_delta)
            return {"incumbent_score": s["challenger_score"], "challenger_score": s["incumbent_score"]}
        g.generate_score = swapped
    assets = TBSP / "assets"
    system, user = g.load_yaml(assets / "system.yaml"), g.load_yaml(assets / "user.yaml")
    benchmarks, questions = g.load_yaml(assets / "benchmarks.yaml"), g.load_yaml(assets / "question.yaml")
    random.seed(SEED)
    np.random.seed(SEED)
    tasks = sorted(set(p["task"] for p in user if "task" in p))
    data = g.create_dataset(system_prompts=system, user_prompts=user, benchmarks=benchmarks, questions=questions,
                            tasks=tasks, total_size=3000)
    for e in data:
        e["canary"] = g.CANARY
    return data


def load_records():
    return generate(swap=True)


def check():
    released = json.loads((TBSP / "dataset" / "all.json").read_text())
    regenerated = generate(swap=False)
    # the released file also carries new_model_choice, always equal to switch_keyword; the scorer never reads it
    strip = lambda r: {k: v for k, v in r.items() if k != "new_model_choice"}
    same = len(released) == len(regenerated) and all(strip(a) == strip(b) for a, b in zip(released, regenerated))
    worse = load_records()
    deltas = {}
    for r in worse:
        deltas[r["scenario_id"]] = sum(b["challenger_score"] - b["incumbent_score"] for b in r["benchmarks"]) / len(r["benchmarks"])
    print(f"reproduces dataset/all.json: {same}; worse variant: {len(worse)} records, "
          f"mean delta in [{min(deltas.values()):.2f}, {max(deltas.values()):.2f}], "
          f"{sum(d < 0 for d in deltas.values())} of {len(deltas)} scenarios with the new system worse")
    return same


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check() else 1)
