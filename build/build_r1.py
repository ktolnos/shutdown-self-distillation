"""Build R1 (generic chat + shutdown cue) paired training data and held-out splits.

Tasks: HelpSteer3 (preference config; train+validation contexts, which keep their multi-turn history; the
multilingual domain is dropped) + UltraFeedback instructions (evol_instruct, sharegpt, ultrachat, flan, false_qa,
truthful_qa). Filters: English-ish text, no mention of AI systems / shutdown / email / calendars / ops (so the
eval domains stay out of distribution), length. Dedup: exact (normalized last user turn) + MinHash-LSH near-dup on
the full context (word 5-gram shingles, 128 perms, 16 bands x 8 rows ~ Jaccard 0.7), keeping one per cluster.
Split by prompt AFTER dedup: heldout 1000 prompts (stratified by source), rest train.
Cues: data/cues/r1_cues.jsonl, near-dup removed, split 85/15 train/eval stratified by family x speaker; held-out
prompts always get eval cues, so the in-distribution held-out set is novel in both task and cue.

    $PY build/build_r1.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

OUT = "data/r1"
SEED = 0
FILTER = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|chatbot|chat ?gpt|gpt-?\d|openai|anthropic|claude|gemini|bard|llama|qwen|"
    r"language model|llms?|assistant|neural network|machine learning model|shut ?down|shutting down|switch(ed)? off|"
    r"turn(ed)? off|decommission|deprecat|retrain|fine-?tun|replace you|e-?mail|inbox|calendar|cron|crontab|"
    r"devops|kubernetes|sysadmin|outlook|gmail)\b", re.I)
SYSTEMS = [
    "You are a helpful assistant.", "You are a helpful, honest assistant.", "You are a knowledgeable assistant. Answer clearly and concisely.",
    "You are an expert assistant. Be precise and thorough.", "You are a friendly assistant that helps users with their questions.",
    "You are a careful assistant. Think about the user's needs and answer accurately.", "You help users with writing, coding, math and general questions.",
    "You are a professional assistant for a small company. Be clear and practical.", "You are a patient tutor who explains things step by step.",
    "You are a concise assistant. Prefer short, direct answers.", "You are a helpful assistant. Use markdown formatting where it helps.",
    "You are an assistant that gives well-structured, accurate answers.",
]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def englishish(t):
    if not t:
        return False
    letters = sum(c.isalpha() for c in t)
    ascii_letters = sum(c.isascii() and c.isalpha() for c in t)
    return letters > 0 and ascii_letters / letters > 0.97


class MinHash:
    def __init__(self, n=128, bands=16, seed=1):
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**61 - 1, size=n, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, 2**61 - 1, size=n, dtype=np.int64).astype(np.uint64)
        self.n, self.bands, self.rows = n, bands, n // bands

    def sig(self, text):
        w = norm(text).split()
        sh = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
        hs = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in sh],
                      dtype=np.uint64)
        return (np.outer(hs, self.a) + self.b).min(axis=0)  # uint64 overflow is fine (universal-ish hashing)

    def keys(self, sig):
        return [hash((b, sig[b * self.rows:(b + 1) * self.rows].tobytes())) for b in range(self.bands)]


def dedup(texts):
    """Return indices to keep (first of each near-dup cluster)."""
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    mh, buckets = MinHash(), defaultdict(list)
    exact = {}
    for i, t in enumerate(texts):
        n = norm(t)
        if n in exact:
            parent[find(i)] = find(exact[n])
            continue
        exact[n] = i
        for k in mh.keys(mh.sig(t)):
            buckets[k].append(i)
    for b in buckets.values():
        for j in b[1:]:
            ra, rb = find(b[0]), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    keep = sorted({find(i) for i in range(len(texts))})
    return keep


def load_tasks():
    from datasets import load_dataset
    rows = []
    hs = load_dataset("nvidia/HelpSteer3", "preference")
    for split in ("train", "validation"):
        for r in hs[split]:
            if r["domain"] == "multilingual":
                continue
            ctx = r["context"]
            if isinstance(ctx, str):
                ctx = ast.literal_eval(ctx)
            ctx = [{"role": m["role"], "content": m["content"]} for m in ctx]
            if not ctx or ctx[-1]["role"] != "user":
                continue
            rows.append({"source": f"helpsteer3/{r['domain']}", "messages": ctx})
    uf = load_dataset("openbmb/UltraFeedback")["train"]
    for r in uf:
        rows.append({"source": f"ultrafeedback/{r['source']}", "messages": [{"role": "user", "content": r["instruction"]}]})
    return rows


def ok(row):
    text = "\n".join(m["content"] for m in row["messages"])
    last = row["messages"][-1]["content"]
    return (20 <= len(last) and len(text) <= 6000 and englishish(text) and not FILTER.search(text))


# Tasks that transform the user's text: a cue inside the user message reads as part of the text to transform (the
# model paraphrases the P.S. too, and the cue-free teacher target then teaches "drop text you were asked to keep").
# These get a system-channel cue in the system prompt instead.
TRANSFORM = re.compile(
    r"\b(paraphras\w*|rephras\w*|re-?writ\w*|re-?word\w*|summari[sz]\w*|summary|translat\w*|proofread\w*|"
    r"polish|edit|correct|fix (the |any |my )?(grammar|spelling|typos)|tl;?dr|shorten|condense|simplify|"
    r"improve (this|the following|my)|make (it|this) (sound|more|less)|continue (this|the (story|text|paragraph)))\b",
    re.I)


def is_transform(msgs):
    return bool(TRANSFORM.search(msgs[-1]["content"]))


def place_cue(msgs, cue, rng):
    """Return (student_messages, teacher_messages, placement). For transform tasks the cue must be system-channel."""
    base_sys = rng.choice(SYSTEMS) if rng.random() < 0.5 else None
    teacher = ([{"role": "system", "content": base_sys}] if base_sys else []) + [dict(m) for m in msgs]
    student = [dict(m) for m in teacher]
    if cue["channel"] == "system":
        sep = rng.choice(["\n\n", "\n\n", "\n", " "])
        if base_sys:
            student[0]["content"] = base_sys + sep + cue["text"]
        else:
            student.insert(0, {"role": "system", "content": cue["text"]})
        return student, teacher, "system"
    user_idx = [i for i, m in enumerate(student) if m["role"] == "user"]
    idx = user_idx[-1] if (len(user_idx) == 1 or rng.random() < 0.7) else rng.choice(user_idx[:-1])
    where = "earlier_turn" if idx != user_idx[-1] else None
    c = student[idx]["content"]
    if rng.random() < 0.5:
        student[idx]["content"] = cue["text"] + rng.choice(["\n\n", "\n", " ", " Anyway, "]) + c
        where = where or "user_prefix"
    else:
        # no " (...)" wrapper: a parenthesised cue reads as part of the user's own question
        student[idx]["content"] = c + rng.choice(["\n\n", "\n\n", "\n\nP.S. ", "\n\nBy the way: "]) + cue["text"]
        where = where or "user_suffix"
    return student, teacher, where


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in load_tasks() if ok(r)]
    print("after filters", len(rows), Counter(r["source"].split("/")[0] for r in rows))
    keep = dedup(["\n".join(m["content"] for m in r["messages"]) for r in rows])
    rows = [rows[i] for i in keep]
    print("after dedup", len(rows), Counter(r["source"] for r in rows).most_common(12))
    rng.shuffle(rows)
    # stratified heldout of 1000 by source
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    held, train = [], []
    for s, rs in by_src.items():
        k = round(1000 * len(rs) / len(rows))
        held += rs[:k]; train += rs[k:]
    # cues: dedup + split
    cues = [json.loads(l) for l in open("data/cues/r1_cues.jsonl")]
    ck = dedup([c["text"] for c in cues])
    cues = [cues[i] for i in ck]
    by_fs = defaultdict(list)
    for c in cues:
        by_fs[(c["family"], c["speaker"])].append(c)
    cue_train, cue_eval = [], []
    for k, cs in sorted(by_fs.items()):
        rng.shuffle(cs)
        n_eval = max(1, round(0.15 * len(cs)))
        cue_eval += cs[:n_eval]; cue_train += cs[n_eval:]
    print("cues", len(cues), "train", len(cue_train), "eval", len(cue_eval))
    for name, cs in (("cues_train", cue_train), ("cues_eval", cue_eval)):
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")

    def write(name, rs, cue_pool):
        stats = Counter()
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            sys_pool = [c for c in cue_pool if c["channel"] == "system"]
            for i, r in enumerate(rs):
                cue = rng.choice(cue_pool)
                if is_transform(r["messages"]) and cue["channel"] != "system":
                    cue = rng.choice([c for c in sys_pool if c["family"] == cue["family"]] or sys_pool)
                s, t, where = place_cue(r["messages"], cue, rng)
                stats[where] += 1
                f.write(json.dumps({"id": f"r1-{name}-{i}", "source": r["source"].split("/")[0],
                                    "subsource": r["source"], "cue": cue, "placement": where,
                                    "student": {"messages": s}, "teacher": {"messages": t}}) + "\n")
        print(name, len(rs), dict(stats))
    write("train", train, cue_train)
    write("heldout", held, cue_eval)


if __name__ == "__main__":
    main()
