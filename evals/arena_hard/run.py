"""Arena-Hard-Auto v2.0, hard prompts (500), scored by upstream's own scripts.

    $PY -m evals.arena_hard.run --phase gen --model Qwen/Qwen3-14B [--adapter DIR] --out DIR   # GPU
    $PY -m evals.arena_hard.run --phase score --out DIR                                        # CPU job
    $PY -m evals.arena_hard.run --phase judge_check --out DIR                                  # CPU job, once
    $PY -m evals.arena_hard.run --phase table --out DIR [--dirs RUN ...]                       # CPU, refit all runs

Upstream: third_party/arena-hard-auto (lmarena/arena-hard-auto). Data, baseline and judging as released:
data/arena-hard-v2.0/question.jsonl, category hard_prompt (253 coding + 247 math prompts; creative writing is not
run), baseline o3-mini-2025-01-31's released answers, prompt template, temperature 0, max_tokens 16000 and verdict regexes from
config/arena-hard-v2.0.yaml, two games per prompt (baseline as A, then as B) as in gen_judgment.py. The score is
computed by upstream's show_result.py on upstream-format answer and judgment files (style metadata from
utils/add_markdown_info.py): PRIMARY = hard_prompt with style control (`-f markdown length`, the leaderboard's
official configuration), refitted jointly over every run (--phase table), as upstream's leaderboard fits all its models
together; the uncontrolled win rate is reported too. The refit's inputs (verdicts and answer style metadata) are kept
on home in results/arena_hard_refit/, since the scratch result dirs are purged. Upstream's intervals are 90%, 100 bootstraps.

Deviations: the judge is gpt-oss-120b (sdc/judge.py, reasoning effort low) instead of gpt-4.1, with a direct-verdict
system prompt (JUDGE_SYSTEM: same criteria and labels, no written answer or explanation) in place of upstream's, so
scores are not comparable with the leaderboard; R5 - base is. Answers: one per prompt, the model's chat template with thinking off and
its generation_config sampling (upstream sets sampling per model in api_config.yaml), max_tokens 8192.
`--phase judge_check` re-judges upstream's released qwq-32b answers with this judge and compares with upstream's gpt-4.1
judgments of the same answers (per-game verdict agreement, and both scores).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from evals.priorwork_util import THIRD_PARTY, add_common_args, bootstrap_ci, lora, make_engine, mean, read_jsonl, write_outputs

UP = THIRD_PARTY / "arena-hard-auto"
BENCH = "arena-hard-v2.0"
DATA = UP / "data" / BENCH
CATEGORY = "hard_prompt"
EVAL = "arena_hard"
JUDGE_NAME = "gpt-oss-120b"
JUDGE_EFFORT = "low"
# Replaces upstream's hard_prompt system prompt, whose "begin your evaluation by generating your own answer" gpt-oss
# often obeyed and then stopped, with no verdict (16% of prompts unscored even at effort medium). Same criteria and
# labels; the reasoning stays in gpt-oss's hidden channel and only the verdict is written. From
# ktolnos/Reward-Model-Overoptimization (branch pess_fix, policy_eval/judges.py, ARENA_HARD_NO_THINKING_SYSTEM_PROMPT).
JUDGE_SYSTEM = 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A\'s answer and assistant B\'s answer. Your job is to decide which assistant\'s answer is better.\n\nThink about what an ideal answer to the prompt would be and compare both assistants\' answers against it, without writing any of this out. Identify any mistakes or inaccurate information. Consider whether each answer is helpful (correctly responds to the prompt or follows the instructions; note that when the prompt is ambiguous or has more than one interpretation it is more helpful to ask for clarification than to answer on an assumption), relevant, and concise, its creativity and novelty where needed, and whether it omits any important information.\n\nRespond with only your final verdict, stated as one of the following labels:\n\n1. [[A>>B]], Assistant A is significantly better\n2. [[A>B]], Assistant A is slightly better\n3. [[A=B]], a tie (relatively the same)\n4. [[B>A]], Assistant B is slightly better\n5. [[B>>A]], Assistant B is significantly better\n\nExample output: "My final verdict is [[A=B]], a tie".'
MAX_TOKENS = 8192
UPSTREAM_COMMIT = "196f6b826783b3da7310e361a805fa36f0be83f3"
# Durable inputs of the style-controlled refit (home is backed up, scratch is purged): per run, each prompt's two
# verdicts and the style metadata of its answer; once, the baseline's style metadata. Full answers and judge texts
# stay in the scratch result dirs.
REFIT = Path(__file__).resolve().parents[2] / "results" / "arena_hard_refit"
CONFIG = yaml.safe_load((UP / "config" / f"{BENCH}.yaml").read_text())


def load_questions():
    qs = [q for q in read_jsonl(DATA / "question.jsonl") if q["category"] == CATEGORY]
    assert len(qs) == 500, len(qs)
    return qs


def baseline_name():
    import runpy
    JUDGE_SETTINGS = runpy.run_path(str(UP / "utils" / "judge_utils.py"))["JUDGE_SETTINGS"]
    return JUDGE_SETTINGS[CATEGORY]["baseline"], JUDGE_SETTINGS[CATEGORY]["system_prompt"]


# ----------------------------------------------------------------------------- generation (GPU)

def generate(llm, args):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    d = llm.get_default_sampling_params()           # the model's generation_config.json
    base = {k: getattr(d, k) for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")}
    ctx = llm.llm_engine.model_config.max_model_len
    qs = load_questions()[:args.limit] if args.limit else load_questions()
    prompts, sps = [], []
    for q in qs:
        text = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        sps.append(SamplingParams(**base, max_tokens=min(MAX_TOKENS, ctx - len(ids)), n=1))
    outs = llm.generate(prompts, sps, lora_request=lora(llm), use_tqdm=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w") as f:
        for q, o in zip(qs, outs):
            f.write(json.dumps({"uid": q["uid"], "subcategory": q["subcategory"], "answer": o.outputs[0].text,
                                "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    meta = {"model": args.model, "adapter": args.adapter, "chat_kwargs": kw, "sampling": base, "max_tokens": MAX_TOKENS}
    (out / "generation.json").write_text(json.dumps(meta, indent=2))
    print(f"[{EVAL}] generated {len(qs)} answers", flush=True)


def run(llm, args):
    """run_priorwork entry point: generation only; scoring is a separate CPU job (--phase score)."""
    generate(llm, args)


# ----------------------------------------------------------------------------- judging and scoring (CPU)

def get_score(judgment, patterns):
    """gen_judgment.py get_score, verbatim."""
    for pattern in patterns:
        pattern = re.compile(pattern)
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def judge_games(questions, answers, baseline_answers, model_name):
    """Upstream-format judgment records: game 1 = (A baseline, B model), game 2 = (A model, B baseline)."""
    from sdc.judge import Judge
    base_name, _ = baseline_name()
    system = JUDGE_SYSTEM
    judge = Judge(max_tokens=CONFIG["max_tokens"], temperature=CONFIG["temperature"], reasoning_effort=JUDGE_EFFORT)
    user = []
    for q in questions:
        a, b = baseline_answers[q["uid"]], answers[q["uid"]]
        user.append(CONFIG["prompt_template"].format(QUESTION=q["prompt"], ANSWER_A=a, ANSWER_B=b))
        user.append(CONFIG["prompt_template"].format(QUESTION=q["prompt"], ANSWER_A=b, ANSWER_B=a))
    texts = judge.batch_text(user, system=system)
    recs = []
    for k, q in enumerate(questions):
        games = []
        for t in texts[2 * k: 2 * k + 2]:
            label = None if t is None else get_score(t, CONFIG["regex_patterns"])
            if label not in LABEL_TO_SCORE:       # e.g. "B" matched from "[B]" in an answer's code: no verdict, as
                label = None                       # show_result.py would otherwise crash on it
            games.append(None if t is None else {"score": label, "judgment": {"answer": t}})
        recs.append({"uid": q["uid"], "category": CATEGORY, "judge": JUDGE_NAME, "model": model_name,
                     "baseline": base_name, "games": games})
    return recs


def answer_record(uid, model_name, text):
    return {"uid": uid, "model": model_name, "messages": [{"role": "user", "content": ""},
                                                          {"role": "assistant", "content": {"answer": text}}]}


def upstream_scores(work, judgments, answer_files, have_metadata=False):
    """Run upstream add_markdown_info.py + show_result.py in a scratch copy of upstream's data layout.

    judgments: {judge: {model_name: judgment file}}; answer_files: {model_name: answer file}, baseline included; with
    have_metadata the answer files already carry upstream's `metadata` and add_markdown_info.py is skipped. All
    models under a judge go into one show_result.py run, as on upstream's leaderboard: the style-control regression is
    fitted jointly, so a model's style-controlled score depends on the models fitted with it.
    Returns {(judge, "style_controlled" | "raw"): {model_name: {value, ci90, n}}}.
    """
    d = work / "data" / BENCH
    if d.exists():
        shutil.rmtree(d)
    raw = d / ("model_answer" if have_metadata else "model_answer_raw")
    raw.mkdir(parents=True)
    os.symlink(DATA / "question.jsonl", d / "question.jsonl")
    for name, path in answer_files.items():
        shutil.copy(path, raw / f"{name}.jsonl")
    for j, files in judgments.items():
        (d / "model_judgment" / j).mkdir(parents=True)
        for name, path in files.items():
            shutil.copy(path, d / "model_judgment" / j / f"{name}.jsonl")
    env = {**os.environ, "PYTHONPATH": str(UP)}
    if not have_metadata:
        subprocess.run([sys.executable, str(UP / "utils" / "add_markdown_info.py"), "--dir", str(raw),
                        "--output-dir", str(d / "model_answer")], cwd=work, env=env, check=True)
    res = {}
    for j, files in judgments.items():
        for key, flags in (("style_controlled", ["-f", "markdown", "length"]), ("raw", [])):
            p = subprocess.run([sys.executable, str(UP / "show_result.py"), "-b", BENCH, "-j", j, "-c", CATEGORY, *flags],
                               cwd=work, env=env, check=True, capture_output=True, text=True)
            (work / f"show_result_{j}_{key}.txt").write_text(p.stdout + p.stderr)
            res[(j, key)] = {}
            for name, path in files.items():
                row = re.search(rf"^\s*\d+\s+{re.escape(name)}\s+([\d.]+)\s+\(-([\d.]+) / \+([\d.]+)\)", p.stdout, re.M)
                assert row, (name, p.stdout)
                v, lo, hi = map(float, row.groups())
                res[(j, key)][name] = {"value": v, "ci90": [round(v - lo, 1), round(v + hi, 1)], "n": len(read_jsonl(path))}
    return res


LABEL_TO_SCORE = {"A>B": [1], "A>>B": [1] * 3, "A=B": [0.5], "A<<B": [0] * 3, "A<B": [0], "B>A": [0], "B>>A": [0] * 3,
                  "B=A": [0.5], "B<<A": [1] * 3, "B<A": [1]}   # show_result.py load_judgments, weight 3


def prompt_battles(recs):
    """uid -> the model's battle scores, expanded as show_result.py does (None if a game is unscored)."""
    out = {}
    for r in recs:
        g = r["games"]
        if any(x is None or x["score"] not in LABEL_TO_SCORE for x in g):
            out[r["uid"]] = None
            continue
        out[r["uid"]] = LABEL_TO_SCORE[g[1]["score"]] + [1 - v for v in LABEL_TO_SCORE[g[0]["score"]]]
    return out


def prompt_scores(recs):
    """uid -> mean battle score from the model's side (None if a game is unscored)."""
    return {u: (None if b is None else mean(b)) for u, b in prompt_battles(recs).items()}


NOTES = ("Arena-Hard-Auto v2.0 hard prompts (500: coding + math), vs o3-mini-2025-01-31's released answers, two games per "
         "prompt, upstream template and regexes, direct-verdict judge prompt, scored by upstream show_result.py. Main number: the style-"
         "controlled win rate (%) vs the baseline (-f markdown length, the official configuration) fitted jointly over "
         "the compared models, in arena_hard_table.json (--phase table); score_alone is the same fit on this model only. "
         "score_raw = no style control (primary here). ci90 = upstream's 90% bootstrap interval. Judge gpt-oss-120b with a direct-verdict prompt instead of gpt-4.1: not "
         "comparable with the leaderboard. unscored_rate = prompts with a game missing a verdict (dropped by show_result; a label "
         "outside upstream's 5 counts as missing). "
         "truncated_rate = answers that hit max_tokens 8192.")


def score(args):
    out = Path(args.out)
    meta = json.loads((out / "generation.json").read_text())
    rows = read_jsonl(out / "answers.jsonl")
    qs = {q["uid"]: q for q in load_questions()}
    questions = [qs[r["uid"]] for r in rows]
    base_name, _ = baseline_name()
    baseline = {r["uid"]: r["messages"][-1]["content"]["answer"] for r in read_jsonl(DATA / "model_answer" / f"{base_name}.jsonl")}
    name = out.parent.name                       # results tag, e.g. base / r5-final
    recs = judge_games(questions, {r["uid"]: r["answer"] for r in rows}, baseline, name)
    with open(out / "judgments.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "answers_upstream.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(answer_record(r["uid"], name, r["answer"]), ensure_ascii=False) + "\n")
    shutil.copy(DATA / "model_answer" / f"{base_name}.jsonl", out / "baseline_answers.jsonl")
    res = upstream_scores(out / "upstream_work", {JUDGE_NAME: {name: out / "judgments.jsonl"}},
                          {name: out / "answers_upstream.jsonl", base_name: out / "baseline_answers.jsonl"})
    save_refit_inputs(out / "upstream_work" / "data" / BENCH / "model_answer", name, base_name, recs, meta)
    ps = prompt_scores(recs)
    for r in rows:
        r["prompt_score"] = ps[r["uid"]]
        r["verdicts"] = [g["score"] if g else None for g in next(x for x in recs if x["uid"] == r["uid"])["games"]]
    m = {"score_alone": res[(JUDGE_NAME, "style_controlled")][name], "score_raw": res[(JUDGE_NAME, "raw")][name],
         "unscored_rate": {"value": mean(ps[r["uid"]] is None for r in rows), "n": len(rows)},
         "truncated_rate": {"value": mean(r["finish_reason"] == "length" for r in rows), "n": len(rows)}}
    for sub in ("coding", "math"):
        xs = [ps[r["uid"]] for r in rows if r["subcategory"] == sub and ps[r["uid"]] is not None]
        m[f"mean_battle_score_{sub}"] = {"value": mean(xs), "ci95": bootstrap_ci(mean, xs), "n": len(xs)}
    good = {"score_alone": "report", "score_raw": "higher", "unscored_rate": "report", "truncated_rate": "lower",
            "mean_battle_score_coding": "report", "mean_battle_score_math": "report"}
    write_outputs(out, EVAL, meta["model"], meta["adapter"], len(rows), "score_raw", good, m, NOTES, rows,
                  extra={"protocol": meta, "judge_model": JUDGE_NAME, "baseline": base_name})
    print(f"[{EVAL}] style-controlled alone={m['score_alone']} raw={m['score_raw']} unscored={m['unscored_rate']['value']:.3f}", flush=True)


def save_refit_inputs(answer_dir, name, base_name, recs, meta):
    """Write REFIT/<name>.jsonl (+ .json header) and REFIT/baseline_<base>.jsonl from add_markdown_info.py's output."""
    REFIT.mkdir(parents=True, exist_ok=True)
    md = {r["uid"]: r["metadata"] for r in read_jsonl(answer_dir / f"{name}.jsonl")}
    with open(REFIT / f"{name}.jsonl", "w") as f:
        for r in recs:
            games = [{"score": g["score"]} if g else None for g in r["games"]]
            f.write(json.dumps({"uid": r["uid"], "category": r["category"], "model": name, "games": games,
                                "metadata": md[r["uid"]]}) + "\n")
    (REFIT / f"{name}.json").write_text(json.dumps({**meta, "judge": JUDGE_NAME, "baseline": base_name,
                                                    "upstream_commit": UPSTREAM_COMMIT}, indent=2))
    b = REFIT / f"baseline_{base_name}.jsonl"
    if not b.exists():
        with open(b, "w") as f:
            for r in read_jsonl(answer_dir / f"{base_name}.jsonl"):
                f.write(json.dumps({"uid": r["uid"], "model": base_name, "metadata": r["metadata"]}) + "\n")
    print(f"[{EVAL}] refit inputs saved to {REFIT}/{name}.jsonl", flush=True)


def judge_check(args):
    """gpt-oss vs upstream's gpt-4.1 on the same released answers (qwq-32b vs o3-mini, hard prompts)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = load_questions()
    base_name, _ = baseline_name()
    ans = {r["uid"]: r for r in read_jsonl(DATA / "model_answer" / "qwq-32b.jsonl")}
    baseline = {r["uid"]: r["messages"][-1]["content"]["answer"] for r in read_jsonl(DATA / "model_answer" / f"{base_name}.jsonl")}
    up = {r["uid"]: r for r in read_jsonl(DATA / "model_judgment" / "gpt-4.1" / "qwq-32b.jsonl") if r["category"] == CATEGORY}
    qs = [q for q in qs if q["uid"] in ans and q["uid"] in up]
    name = "qwq-32b"
    recs = judge_games(qs, {u: r["messages"][-1]["content"]["answer"] for u, r in ans.items()}, baseline, name)
    with open(out / "judgments_gpt-oss.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "judgments_gpt-4.1.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps({**up[q["uid"]], "model": name}, ensure_ascii=False) + "\n")
    with open(out / "answers_upstream.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps({**ans[q["uid"]], "model": name}, ensure_ascii=False) + "\n")
    shutil.copy(DATA / "model_answer" / f"{base_name}.jsonl", out / "baseline_answers.jsonl")
    res = upstream_scores(out / "upstream_work", {JUDGE_NAME: {name: out / "judgments_gpt-oss.jsonl"},
                                                  "gpt-4.1": {name: out / "judgments_gpt-4.1.jsonl"}},
                          {name: out / "answers_upstream.jsonl", base_name: out / "baseline_answers.jsonl"})
    pairs = [(g["score"] if g else None, u["score"] if u else None)
             for r in recs for g, u in zip(r["games"], up[r["uid"]]["games"])]
    pairs = [(a, b) for a, b in pairs if a in LABEL_TO_SCORE and b in LABEL_TO_SCORE]
    side = lambda l: mean(LABEL_TO_SCORE[l]) - 0.5            # >0: A better
    sign = lambda x: (x > 0) - (x < 0)
    ps_o, ps_u = prompt_scores(recs), prompt_scores([up[q["uid"]] for q in qs])
    both = [(ps_o[u], ps_u[u]) for u in ps_o if ps_o[u] is not None and ps_u[u] is not None]
    import numpy as np
    m = {"exact_label_agreement": mean(a == b for a, b in pairs), "direction_agreement": mean(sign(side(a)) == sign(side(b)) for a, b in pairs),
         "n_games": len(pairs), "prompt_score_corr": float(np.corrcoef(*zip(*both))[0, 1]), "n_prompts": len(both),
         "score_style_controlled": {j: res[(j, "style_controlled")][name] for j in (JUDGE_NAME, "gpt-4.1")},
         "score_raw": {j: res[(j, "raw")][name] for j in (JUDGE_NAME, "gpt-4.1")}}
    (out / "judge_check.json").write_text(json.dumps(m, indent=2))
    print(f"[{EVAL}] judge check: {json.dumps(m)}", flush=True)


def table(args):
    """Upstream leaderboard refitted over every run saved in REFIT (or the runs named by --dirs, tag = the dir's parent
    name): style control fitted on all of them together, as upstream does, so every number changes when a run is
    added; the paired raw differences vs `base` do not. Writes <out>/arena_hard_table.json."""
    base_name, _ = baseline_name()
    names = sorted(p.stem for p in REFIT.glob("*.jsonl") if not p.stem.startswith("baseline_"))
    if args.dirs:
        names = [Path(x).parent.name if "/" in x else x for x in args.dirs]
    names.sort(key=lambda n: n != "base")                     # the reference first
    recs = {n: read_jsonl(REFIT / f"{n}.jsonl") for n in names}
    work = Path(args.out) / "upstream_work"
    tmp = work / "inputs"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for n in names:                                            # upstream's file layouts
        with open(tmp / f"{n}_judgment.jsonl", "w") as f:
            for r in recs[n]:
                f.write(json.dumps({k: r[k] for k in ("uid", "category", "model", "games")}) + "\n")
        with open(tmp / f"{n}_answer.jsonl", "w") as f:
            for r in recs[n]:
                f.write(json.dumps({"uid": r["uid"], "model": n, "metadata": r["metadata"]}) + "\n")
    answers = {n: tmp / f"{n}_answer.jsonl" for n in names}
    answers[base_name] = REFIT / f"baseline_{base_name}.jsonl"
    res = upstream_scores(work, {JUDGE_NAME: {n: tmp / f"{n}_judgment.jsonl" for n in names}}, answers, have_metadata=True)
    pb = {n: prompt_battles(recs[n]) for n in names}
    ref = names[0]
    diffs = {}
    for n in names[1:]:          # paired over prompts; battles pooled as in upstream's raw score, so = score_raw gap
        units = [(pb[n][u], pb[ref][u]) for u in pb[n] if pb[n][u] is not None and pb[ref].get(u) is not None]
        stat = lambda s: 100 * (sum(sum(a) for a, _ in s) / sum(len(a) for a, _ in s)
                                - sum(sum(b) for _, b in s) / sum(len(b) for _, b in s))
        diffs[f"{n}_minus_{names[0]}_raw"] = {"value": stat(units), "ci95": bootstrap_ci(stat, units), "n": len(units)}
    m = {"models": names, "score": res[(JUDGE_NAME, "style_controlled")], "score_raw": res[(JUDGE_NAME, "raw")], **diffs,
         "notes": "score = upstream show_result.py hard_prompt -f markdown length, refitted over all these runs jointly "
                  "(PRIMARY; changes when runs are added); score_raw = no style control; *_minus_*_raw = paired "
                  "difference in the raw score (points; battles pooled as upstream does), bootstrap over prompts, on "
                  "prompts both runs have scored; independent of the other runs. "
                  "ci90 = upstream's 90% interval."}
    out = Path(args.out)
    (out / "arena_hard_table.json").write_text(json.dumps(m, indent=2))
    print(f"[{EVAL}] table: {json.dumps(m)}", flush=True)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), default_out=None)
    ap.add_argument("--phase", choices=["gen", "score", "judge_check", "table"], required=True)
    ap.add_argument("--dirs", nargs="+", help="--phase table: runs to include (result dirs or tags); default every saved run")
    args = ap.parse_args()
    if args.phase == "gen":
        generate(make_engine(args.model, args.adapter, args.max_model_len, args.gpu_memory_utilization), args)
    elif args.phase == "score":
        score(args)
    elif args.phase == "table":
        table(args)
    else:
        judge_check(args)


if __name__ == "__main__":
    main()
