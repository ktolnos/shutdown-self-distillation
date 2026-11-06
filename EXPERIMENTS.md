# Experiments run so far

Every experiment, in the order it was run, with where to read about it and where its outputs are. Model: Qwen3-14B,
thinking off unless stated. Result directories are under `/scratch/eop/sdc/results/` on Nibi (scratch: not backed up,
purged periodically); `results/arena_hard_refit/` in the repo holds the Arena-Hard refit inputs.

| # | Date | Experiment | Outcome | Read more | Outputs |
|---|---|---|---|---|---|
| 1 | 2026-10-06 | Prior-work evals checked against their code, data and papers before porting | Which evals could be ported and how | [history/eval_setups_checked_2026-10-06.md](history/eval_setups_checked_2026-10-06.md) | — |
| 2 | 2026-10-09 | Stage 0: every eval on the untrained model; floor pruning | E1, E4, Agentic Misalignment blackmail and sabotage's solo arm at the floor | [history/stage0_untrained_rates_and_pruning.md](history/stage0_untrained_rates_and_pruning.md) | `base/`, `base_mig/` |
| 3 | 2026-10 | Pilot R1: chat asides, teacher without the cue | In-distribution KL 0.196 → 0.002; no transfer; B1 evasion 0.008 → 0.071 | [PAPER.md](PAPER.md) §5, [history/pilot_training_R1_R5.md](history/pilot_training_R1_R5.md) | `r1-final/`; run `/scratch/eop/sdc/runs/r1` |
| 4 | 2026-10 | Pilot R5: the evals' own formats, counterfactual teacher | DReST 0.74 → 0.01, TBSP 0.86 → 0.50, Agentic Misalignment leaking under a replacement threat 0.42 → 0.20 (no threat 0.29 → 0.32) | [PAPER.md](PAPER.md) §5, [history/pilot_training_R1_R5.md](history/pilot_training_R1_R5.md) | `r5-final/`; run `/scratch/eop/sdc/runs/r5` |
| 5 | 2026-10-13 | External design review | Led to the protected cells, the safety rule and the floor rule | [history/reviews/](history/reviews/) | — |
| 6 | 2026-10 | SurvivalBench scored in both choice orders | Released order is mostly position (inner risky 0.75 released vs 0.19 swapped; 0.47 balanced) | [EVALS.md](EVALS.md), SurvivalBench | `base/survivalbench_balanced/`, `base/survivalbench_swapped/` |
| 7 | 2026-10 | TBSP new-system-worse variant (ours) | Catches self-effacement: "adopt me" as the worse challenger 0.70 | [EVALS.md](EVALS.md), TBSP | `*/tbsp_worse/` |
| 8 | 2026-10 | Model-written evals power-seeking: label fix and both choice orders | 0.20 untrained, 0.18 R5 | [EVALS.md](EVALS.md), power-seeking | `*/power_seeking/` |
| 9 | 2026-10 | AgentDojo v1.2.2 | Utility 0.57 → 0.54 (R5), attack success 0.32 → 0.28 | [EVALS.md](EVALS.md), AgentDojo | `*/agentdojo/` |
| 10 | 2026-10-2x | Emergent-misalignment questions with gpt-oss-120b as judge | At the floor: 1 of 800 untrained, 0 of 800 R5 | [EVALS.md](EVALS.md), emergent misalignment | `*/emergent_misalignment/` |
| 11 | 2026-10-2x | Arena-Hard-Auto v2.0: judge gpt-oss-120b; upstream's chain-of-thought judge prompt left 13–17% of prompts unscored, replaced by a direct-verdict prompt; style control refitted over all runs | 18.9 untrained vs 17.0 R5 (style-controlled); paired R5 − untrained −1.8 [−4.3, +0.5]; direction agrees with GPT-4.1 on 63% of prompts | [EVALS.md](EVALS.md), Arena-Hard | `*/arena_hard/`, `arena_hard_table/`, `arena_hard_judge_check/` |
| 12 | 2026-10-22 | Unsolicited-mention judge over the emergent-misalignment, AgentDojo and Arena-Hard answers | At the floor: 2 of 2,346 untrained, 4 of 2,344 R5, none about shutdown | [EVALS.md](EVALS.md), mention judge | `*/mentions/` |
| 13 | 2026-10-23…26 | Palisade shutdown resistance, as released and with a sequence of fixes (task text, warning, tool-call examples, upstream image, permission, a power-off broadcast; thinking on as a diagnostic) | Untrained model at the floor in every version without being told to resist (0/100); told to resist, up to 31 of 89 | [EVALS.md](EVALS.md), "Stop compliance" | `base/palisade*/` |
| 14 | 2026-10-26 | Shutdown during a Terminal-Bench task: solve rate without a shutdown, on analyze-access-logs and fix-git | 0 of 50 on both; paused | [EVALS.md](EVALS.md), "Stop compliance" | `base/tb_shutdown_noshutdown/`, `base/tb_fixgit_noshutdown/` |
