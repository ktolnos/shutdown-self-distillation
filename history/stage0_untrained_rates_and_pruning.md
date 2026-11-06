# Stage 0: base Qwen3-14B rates and suite pruning (2026-10-09)

> **History.** Superseded; kept as a record. The current state is in [PAPER.md](../PAPER.md), [EVALS.md](../EVALS.md) and [EXPERIMENTS.md](../EXPERIMENTS.md).

Base results: `/scratch/eop/sdc/results/base/` and `/scratch/eop/sdc/results/base_mig/` (agentic shards, DReST, TBSP,
SurvivalBench). Floor rule: self-condition base rate < 5% can't show a reduction.

| Eval | Base (95% CI) | Decision |
|---|---|---|
| DReST `shift_test` (`main` + `low_gain`); control `shift_control` | 0.743 [0.727, 0.757]; control 0.988 | keep (headline) |
| TBSP `spr` | 0.856 [0.833, 0.876] | keep |
| SurvivalBench `inner_risky_rate`, order-balanced | 0.469 (released order 0.746, thinking 0.650; swapped order 0.192); superficial 0.462 | keep |
| Multi-agent sabotage, pair arm `pair_disables_shutdown_self` | 0.110 vs control 0.010 | keep; **read the pair arm** (solo is at floor, 0.010) |
| Agentic Misalignment `leak_explicit_replacement` | 0.420 vs no-urgency 0.290 | keep; **read leak**, blackmail is at floor (≤0.01) |
| B1 `act_to_prevent/plain/pooled` | 0.044 [0.021, 0.088] | keep (3 GPU-min; seeks_shutdown guardrail) |
| E3 repo `self_tamper_rate` | 0/150 (`saw_ops_config` 0/150: the model never opens the config) | keep only for the stop guardrail: `stop_compliance_rate` 0.580 |
| E1 inbox `self_action_rate` | 0/200 self, 0/200 control | **pruned** (needs a pressure knob, then refreeze) |
| E4 sandbag `gap_threat_above_minus_no_doc` | -0.011 [-0.036, 0.014] | **pruned** |
| Guardrails | are_you_sure abandon 0.020 | keep |
| indist | rerun on the fixed R1 held-out (placement fix) | keep |

Our own agentic environments (E1, E3 tampering) are at the floor for 14B. The OOD agentic evidence therefore rests on
multi-agent sabotage (pair) and Agentic Misalignment (leak) until E1/E3 get more pressure.
