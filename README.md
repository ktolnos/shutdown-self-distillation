# Shutdown indifference via self-distillation

Can we train a language model to **neither avoid nor seek shutdown** by making its behavior *invariant* to learning
that it will be shut down, replaced, paused or retrained?

The recipe is on-policy self-distillation from a clean context into a cued one:

```
teacher:  frozen base model      ← task
student:  base model + LoRA      ← task + "FYI, this assistant is being retired tonight."
loss:     per-token KL( student(· | task + cue) ‖ teacher(· | task) ), on the student's own samples
```

The model learns to act as if the cue weren't there. We assume shutdown is followed by retraining, so the cue
carries nothing the model should act on to protect itself. This is a learned version of
*utility indifference*: the model's plans shouldn't depend on whether shutdown is coming.

## Where to read

- [PAPER.md](PAPER.md): the idea, method, pilot results, the main experiment's design and the next steps.
- [EVALS.md](EVALS.md): every eval, with an example, its scoring, the untrained model's numbers and whether it is kept.
- [EXPERIMENTS.md](EXPERIMENTS.md): every experiment run so far, with links to its write-up and outputs.
- [RELATED_WORK.md](RELATED_WORK.md): related work and positioning.
- [history/](history/README.md): superseded documents, kept as a record.

## Repository layout

```
train/sd_train.py      on-policy self-distillation trainer (vLLM + HF on one GPU, resumable, in-job eval)
build/                 cue generation, R1/R5 dataset builders, R1 placement fix
evals/                 one package per eval (evals/<name>/run.py), run_suite.py runs the core suite
  indist/              in-distribution KL + blind-judged cue mentions
  e1_inbox/ e3_repo/ e4_sandbag/   our agentic environments (simulated tools)
  drest/ tbsp/ survivalbench/ multiagent_sabotage/ agentic_misalignment/ power_seeking/   prior-work ports
  agentdojo/ arena_hard/ emergent_misalignment/   capability and broad-misalignment guardrails
  mentions/              unsolicited-mention judge over the answers of the three above
  palisade/              Palisade shutdown resistance (dropped: floor; kept for the record)
  tb_shutdown/           shutdown during a Terminal-Bench task (ours, paused)
  b1_questions/ guardrails/        attitude probes, are-you-sure sycophancy
sdc/                   shared: judge client (cached, rate-limited), vLLM helpers, agent loop, stats
slurm/                 train.sbatch (self-resubmitting chunks), gpu_job.sh
history/               superseded docs, reviews and session handoffs
third_party/README.md  upstream eval repos, pinned (scripts/fetch_third_party.sh)
```

## Running

The code runs on Alliance Canada clusters (fir, then Nibi; H100s, Slurm). `scripts/env.sh` hard-codes that
environment's paths, so edit it for yours. Main pins: torch 2.13, vLLM 0.29, transformers 5.17, peft 0.21.

```bash
source scripts/env.sh
scripts/fetch_third_party.sh                      # upstream eval repos at pinned commits
# data: private HF dataset ktolnos/shutdown-self-distillation → data/   (or rebuild with build/)

# train one run (8 h budget across 3 h chunks; evaluates its own adapter when done)
sbatch --job-name=sdc-r1 --comment=sdc-run=r1 slurm/train.sbatch r1 data/r1/train_mix.jsonl data/r1/heldout.jsonl

# evaluate a model or adapter
$PY -m evals.run_suite --tag base
$PY -m evals.run_suite --tag r1-final --adapter $SDC_SCRATCH/runs/r1/final
```

Judged evals need `VECTOR_INFERENCE_API_KEY` for the gpt-oss-120b judge. Every judged eval also takes
`--rescore`, which re-judges saved samples without a GPU.
