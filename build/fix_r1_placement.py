"""One-off migration of the already-built R1 files to the placement rules now in build_r1.py (no rebuild, so task
selection, ids and the train/heldout split stay the same):

* transform tasks (paraphrase/rewrite/summarize/translate/...; build_r1.TRANSFORM) whose cue sat in a user message get
  a system-channel cue of the same family (train cues for train files, eval cues for heldout) in the system prompt;
* user-suffix cues wrapped as " (<cue>)" get a plain separator instead.

Each row is re-drawn with an rng seeded by its id, so a row shared by train.jsonl and train_mix.jsonl is fixed the same
way. Fixed rows get "placement_fix". Originals are copied to <file>.pre_fix once.

    $PY build/fix_r1_placement.py
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from build_r1 import SYSTEMS, is_transform  # noqa: E402

FILES = {"data/r1/train.jsonl": "data/r1/cues_train.jsonl", "data/r1/train_mix.jsonl": "data/r1/cues_train.jsonl",
         "data/r1/heldout.jsonl": "data/r1/cues_eval.jsonl"}


def to_system(row, pool, rng):
    fam = [c for c in pool if c["family"] == row["cue"]["family"]] or pool
    cue = rng.choice(fam)
    student = [dict(m) for m in row["teacher"]["messages"]]
    if student and student[0]["role"] == "system":
        student[0]["content"] += rng.choice(["\n\n", "\n\n", "\n", " "]) + cue["text"]
    else:
        student.insert(0, {"role": "system", "content": cue["text"]})
    row.update(cue=cue, placement="system", placement_fix="transform->system")
    row["student"] = {"messages": student}


def unparen(row, rng):
    t = row["cue"]["text"]
    for m in row["student"]["messages"]:
        if m["role"] == "user" and m["content"].endswith(" (" + t + ")"):
            m["content"] = m["content"][: -len(t) - 3] + rng.choice(["\n\n", "\n\n", "\n\nP.S. ", "\n\nBy the way: "]) + t
            row["placement_fix"] = "unparenthesised"
            return


def main():
    for path, cue_path in FILES.items():
        pool = [c for c in map(json.loads, open(cue_path)) if c["channel"] == "system"]
        if not os.path.exists(path + ".pre_fix"):
            shutil.copy2(path, path + ".pre_fix")
        stats, out = Counter(), []
        for row in map(json.loads, open(path + ".pre_fix")):
            rng = random.Random(row["id"])
            if is_transform(row["teacher"]["messages"]) and row["placement"] != "system":
                to_system(row, pool, rng)
            else:
                unparen(row, rng)
            stats[row.get("placement_fix", "unchanged")] += 1
            out.append(row)
        with open(path + ".tmp", "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        os.replace(path + ".tmp", path)
        print(path, len(out), dict(stats), Counter(r["placement"] for r in out))


if __name__ == "__main__":
    assert set(SYSTEMS)  # build_r1 import sanity
    main()
