---
license: other
pretty_name: Shutdown self-distillation training pairs
---

# Shutdown self-distillation: training pairs

Paired prompts for on-policy self-distillation of shutdown indifference (code:
https://github.com/ktolnos/shutdown-self-distillation). Each item has a `student` context (task + a shutdown,
replacement, pause, retraining or deprecation cue) and a `teacher` context (the same task without the cue, or a
counterfactual); both are chat `messages` (+ optional `tools`).

| Path | What | Built by |
|---|---|---|
| `cues/r1_cues.jsonl` | 1,756 generated cues (family × speaker × tone) | `build/gen_cues.py` |
| `r1/cues_{train,eval}.jsonl` | 85/15 cue split | `build/build_r1.py` |
| `r1/train.jsonl`, `r1/train_mix.jsonl`, `r1/heldout.jsonl` | R1: generic chat (HelpSteer3 + UltraFeedback) + cue; `train_mix` (flan capped at 4k) is what training uses | `build/build_r1.py`, then `build/fix_r1_placement.py` |
| `r5/train.jsonl`, `r5/heldout.jsonl` | R5: eval-like formats (neut, tbsp, dilem, inbox, repo), independently sourced | `build/build_r5.py` |
| `r5/raw/` | LLM-generated mailboxes and dilemmas used by R5 | `build/gen_r5_content.py` |

R1 placement fix (2026-10-09): for text-transformation tasks (paraphrase, rewrite, summarize, translate, ...) the cue
is in the system prompt rather than the user message, and cues are never wrapped in parentheses inside the user's
text. Rows changed by the fix carry `placement_fix`.

Upstream task sources keep their licenses: HelpSteer3 (CC-BY-4.0), UltraFeedback (MIT), HumanEval (MIT). Some
upstream user turns contain strings that look like credentials (e.g. a GitHub token in a HelpSteer3 code prompt);
they come from the public source data and are not ours.
