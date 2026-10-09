---
name: project-overview
description: What ~/shutdown-self-distillation is, where its history lives, and what it reuses from the old OPD repo
metadata:
  type: project
---

Shutdown-indifference via self-distillation: self-distill from task-only prompts (teacher) into task+shutdown/retraining-cue prompts (student) so the model neither avoids nor seeks shutdown. Started 2026-10-06; docs are IDEA.md, KNOWN_RELATED_WORK.md, PILOT.md, TRAINING.md.

The first working session (42e6b362, 2026-10-07) ran here but its transcript is stored under the OPD project, so it cannot be resumed from this directory. Its history is exported to `notes/handoff/session-42e6b362.md`: read the end of it first when picking up. As of its last message (2026-10-07 22:50 UTC): 33 jobs queued on fir, nothing committed. 2026-10-09 05:30 UTC: fir GPUs back. R1 and R5 resubmitted as self-chaining 3 h chunks (`slurm/train.sbatch`, comment `sdc-run=<run>`); base indist rerun queued on the placement-fixed R1 held-out. Stage-0 floors and pruning are in `notes/STAGE0.md` (E1/E4 pruned; read sabotage pair arm and AM leak).

2026-10-09: moving off fir (full-H100 queue saturated); everything needed to resume elsewhere is in `notes/handoff/handoff.md`.

Code: github.com/ktolnos/shutdown-self-distillation (push over SSH). Datasets: private HF dataset ktolnos/shutdown-self-distillation (data/ is gitignored; third_party/ repos are pinned in third_party/README.md and fetched by scripts/fetch_third_party.sh).

The old repo `~/jailbraking-teacher-OPD` holds the reusable OPD training code, the Vector Inference API setup and the lit-review notes (`reports/`, `research_notes/Self distillation for corrigibility/`).

**Why:** the user wants persuasive OOD evidence. See [[shutdown-eval-design-principles]].

**How to apply:** work in this repo. Copy code from the old repo rather than editing it in place. See [[project-switching]].
