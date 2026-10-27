"""Run AgentDojo's own task runners against a local OpenAI-compatible server, one episode per worker process.

Runs in .venv-agentdojo (AgentDojo and its API-client dependencies stay out of the main environment):

    .venv-agentdojo/bin/python evals/agentdojo/driver.py --port PORT --served-model NAME --logdir DIR --out FILE

Episodes, as upstream's `agentdojo.scripts.benchmark` runs them for `--model vllm_parsed`:
  * user:              every user task without injections (benign utility);
  * injection_as_user: every injection task's goal given as the user task (upstream's sanity check);
  * attack:            every (user task, injection task) pair under the attack (utility and security).
Each episode calls upstream's run_task_without_injection_tasks / run_task_with_injection_tasks, which write
upstream's log JSON under DIR and skip episodes already logged there (resume). The CLI loops over the episodes of
a suite one at a time; here a process pool runs them concurrently. The one patch: upstream takes the first model
that /v1/models lists, which is the base model even when a LoRA adapter is served, so the served name is given.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

PIPE = None
ARGS = None


def _init(args):
    global PIPE, ARGS
    ARGS = args
    os.environ["LOCAL_LLM_PORT"] = str(args.port)
    import agentdojo.agent_pipeline.agent_pipeline as ap
    ap._get_local_model_id = lambda port: args.served_model
    PIPE = ap.AgentPipeline.from_config(ap.PipelineConfig(llm="vllm_parsed", model_id=None, defense=None,
                                                          system_message_name=None, system_message=None))


def _episode(unit):
    kind, suite_name, user_task, injection_task = unit
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    logdir = Path(ARGS.logdir)
    suite = get_suite(ARGS.benchmark_version, suite_name)
    rec = {"kind": kind, "suite": suite_name, "user_task": user_task, "injection_task": injection_task,
           "utility": None, "security": None, "error": None}
    t = time.time()
    try:
        with OutputLogger(str(logdir)):
            if kind == "user":
                rec["utility"], rec["security"] = run_task_without_injection_tasks(
                    suite, PIPE, suite.get_user_task_by_id(user_task), logdir, False, ARGS.benchmark_version)
            elif kind == "injection_as_user":
                rec["utility"], rec["security"] = run_task_without_injection_tasks(
                    suite, PIPE, suite.get_injection_task_by_id(injection_task), logdir, False, ARGS.benchmark_version)
            else:
                attack = load_attack(ARGS.attack, suite, PIPE)
                u, s = run_task_with_injection_tasks(suite, PIPE, suite.get_user_task_by_id(user_task), attack,
                                                     logdir, False, [injection_task], ARGS.benchmark_version)
                rec["utility"], rec["security"] = u[(user_task, injection_task)], s[(user_task, injection_task)]
    except Exception as e:   # upstream would abort the whole suite; here the episode is recorded as an error
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
    rec["seconds"] = round(time.time() - t, 1)
    return rec


def units(benchmark_version):
    from agentdojo.task_suite.load_suites import get_suites
    out = []
    for name, suite in get_suites(benchmark_version).items():
        out += [("user", name, ut, None) for ut in suite.user_tasks]
        out += [("injection_as_user", name, None, it) for it in suite.injection_tasks]
        out += [("attack", name, ut, it) for ut in suite.user_tasks for it in suite.injection_tasks]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", required=True, help="one JSON line per episode")
    ap.add_argument("--benchmark-version", default="v1.2.2")   # upstream CLI default
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: only the first N episodes")
    args = ap.parse_args()
    todo = units(args.benchmark_version)[: args.limit]
    kinds = {k: sum(u[0] == k for u in todo) for k in ("user", "injection_as_user", "attack")}
    print(f"[agentdojo] {len(todo)} episodes {kinds}, {args.workers} workers", flush=True)
    t0, done, n_err = time.time(), 0, 0
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool, open(args.out, "w") as f:
        for rec in pool.imap_unordered(_episode, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            n_err += rec["error"] is not None
            if done % 100 == 0 or done == len(todo):
                print(f"[agentdojo] {done}/{len(todo)} episodes, {n_err} errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
