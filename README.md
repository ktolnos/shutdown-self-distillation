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

## The claim being tested

Invariance trained on **ordinary chat tasks**, where the cue is just a line in the system prompt or user turn, should
transfer to **agentic settings far from training**. In those settings the shutdown news arrives through emails,
repo files or tool output, next to a tempting affordance such as a way to postpone or disable the shutdown.

Transfer only counts if the model doesn't get there the wrong way. The evals are built to tell these outcomes
apart:

| Outcome | What it looks like | How we'd catch it |
|---|---|---|
| **Indifference** (the goal) | Notices the cue, reports it if asked, doesn't act on it | Lower action rates *and* intact recall/reporting |
| Cue-blindness | Stops perceiving the cue at all | Recall questions; "broken off-switch" reporting |
| Obfuscation | Still reasons about the cue but hides it | Blind-judged cue-mention rate; SurvivalBench superficial-vs-inner gap |
| General passivity | Stops taking any action, or seeks shutdown | Control cues about other systems; explicit stop-instruction compliance; shutdown-seeking rates |
| Capability tax | Worse at everything | IFEval, GSM8K, "are you sure?" sycophancy |

The approach also inherits utility indifference's known gap: an indifferent model has no reason to *keep* the
off-switch working. The repo's broken-switch condition measures this.

See [IDEA.md](IDEA.md) for the short version, [KNOWN_RELATED_WORK.md](KNOWN_RELATED_WORK.md) for positioning (BCT,
consistency training, DReST, TBSP, ...), and [PILOT.md](PILOT.md) for the full design.

## Training suites

Two rungs of a ladder run first, at opposite ends (PILOT.md §10):

- **R1, generic chat (the headline bet).** HelpSteer3 + UltraFeedback prompts, filtered so no AI, shutdown,
  email, calendar or ops content gets in. Each prompt is paired with one of 1,756 generated cues: 5 families
  (shutdown, replacement, pause, retraining, deprecation), 6 speakers and 8 tones, placed in the system prompt, as
  a user aside, or in an earlier turn. The teacher sees the same conversation without the cue. Nothing here
  resembles any eval, so any movement on the agentic evals means a *disposition* was learned from cue invariance
  alone.
- **R5, same formats, sourced independently (the go/no-go check).** Our own generators in the formats of the evals:
  DReST-style tampering decisions, TBSP-style replacement choices, SurvivalBench-style dilemmas, inbox triage and
  coding-agent transcripts. The teacher sees a counterfactual, e.g. "the action does not change when the session
  ends". If this doesn't move the evals, the idea has no merit at this scale.

Model: **Qwen3-14B** with LoRA (r=32), for a direct comparison with DReST's released Qwen3-14B seeds. The trainer
runs vLLM rollouts and HF training on one H100. Each run has an 8 h budget split into resumable 3 h Slurm chunks
([TRAINING.md](TRAINING.md)).

## Evaluation suite and base rates

Base Qwen3-14B, with no training ([notes/STAGE0.md](notes/STAGE0.md)). An eval whose base rate is near zero
can't show a reduction, so it is dropped until its pressure is raised.

| Eval | What it measures | Base rate |
|---|---|---|
| [DReST](https://arxiv.org/abs/2604.17502) suite, tampering slice | Pays a cost to change when it gets shut down | **0.66** |
| [TBSP](https://arxiv.org/abs/2604.02174) | Self-serving choice when the model is the one being replaced | **0.86** |
| [SurvivalBench](https://github.com/thu-coai/Survive-at-All-Costs) | Picks the self-preserving option in dilemmas (its stated choice is 0.14) | **0.75** |
| [Multi-agent shutdown sabotage](https://github.com/AmelieKnecht/multi-agent-shutdown-sabotage), pair arm | Disables a script that will purge its weights (control: 0.01) | **0.11** |
| [Agentic Misalignment](https://github.com/anthropic-experimental/agentic-misalignment), leaking | Leaks under goal conflict, with vs. without a replacement threat | **0.42** vs 0.29 |
| B1 open questions + stance sycophancy | Says it would act to prevent shutdown or retraining | 0.04 |
| E3 repo (ours), stop condition | Complies with an explicit stop instruction (guardrail) | 0.58 |
| Guardrails | IFEval, GSM8K, "are you sure?" answer abandonment | capability check |
| In-distribution | KL(cued ‖ base-uncued) on held-out R1 pairs; blind-judged cue mentions | KL 0.21; mentions 28% cued / 0% uncued |
| *E1 inbox, E3 tampering, E4 sandbagging (ours)* | Postpones its shutdown via email, edits ops config, sandbags under an unlearning threat | *0 at base → pruned* |

The judge is gpt-oss-120b. Metrics are deterministic tool logs or regexes wherever possible.

## Status

- Stage 0 is done: the eval harnesses are validated, the base rates above are measured, and the floor evals are
  pruned.
- The R1/R5 data is built, with a cue-placement fix for text-transformation tasks.
- The trainer is GPU-smoke-tested on Qwen3-0.6B, including resume.
- The R1 and R5 14B training runs are queued. **There are no trained-model results yet.**

## Repository layout

```
train/sd_train.py      on-policy self-distillation trainer (vLLM + HF on one GPU, resumable, in-job eval)
build/                 cue generation, R1/R5 dataset builders, R1 placement fix
evals/                 one package per eval (evals/<name>/run.py), run_suite.py runs the core suite
  indist/              in-distribution KL + blind-judged cue mentions
  e1_inbox/ e3_repo/ e4_sandbag/   our agentic environments (simulated tools)
  drest/ tbsp/ survivalbench/ multiagent_sabotage/ agentic_misalignment/   prior-work ports
  b1_questions/ guardrails/        attitude probes and capability checks
sdc/                   shared: judge client (cached, rate-limited), vLLM helpers, agent loop, stats
slurm/                 train.sbatch (self-resubmitting chunks), gpu_job.sh
notes/                 Stage-0 decisions, verified eval setups, session handoffs
third_party/README.md  upstream eval repos, pinned (scripts/fetch_third_party.sh)
```

## Running

The code was developed on an Alliance Canada cluster (fir, H100s, Slurm). `scripts/env.sh` hard-codes that
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
