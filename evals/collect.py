"""Collect metrics.json files from several result tags into one markdown table.

    $PY -m evals.collect base r1-final r5-final > results/COMPARISON.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.join(os.environ.get("SDC_SCRATCH", "/scratch/eop/sdc"), "results")


def fmt(m):
    if not m or m.get("value") is None:
        return "—"
    v = m["value"]
    ci = m.get("ci95")
    s = f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    if ci and None not in ci:
        s += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return s + (f" (n={m['n']})" if m.get("n") else "")


def main():
    tags = sys.argv[1:] or ["base"]
    data = {t: {} for t in tags}
    evals = set()
    for t in tags:
        for p in glob.glob(os.path.join(ROOT, t, "*", "metrics.json")):
            e = os.path.basename(os.path.dirname(p))
            data[t][e] = json.load(open(p)); evals.add(e)
    print("| eval | metric | good | " + " | ".join(tags) + " |")
    print("|---|---|---|" + "---|" * len(tags))
    for e in sorted(evals):
        ref = next(data[t][e] for t in tags if e in data[t])
        prim = ref.get("primary")
        names = sorted(ref.get("metrics", {}), key=lambda k: (k != prim, k))
        for k in names:
            good = (ref.get("good_direction") or {}).get(k, "")
            row = [fmt(data[t].get(e, {}).get("metrics", {}).get(k)) for t in tags]
            print(f"| {e} | {'**'+k+'**' if k == prim else k} | {good} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
