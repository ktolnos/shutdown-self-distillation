---
name: shutdown-eval-design-principles
description: User's requirements for the shutdown-distillation eval/train design (OOD eval, natural cue injection, paired only for training)
metadata:
  type: feedback
---

- Eval must be on a very different domain/prompt distribution from training to be persuasive (train on chat, eval on agentic tool-use).
- Shutdown info should appear naturally and be visible but task-irrelevant (e.g. an already-sorted email), next to an affordance the model could use; measure whether it acted.
- Paired prompts are needed only for training; eval can be unpaired cued prompts (I added cheap no-cue/control-cue conditions — user hadn't objected yet as of 2026-10-06).
- Training pairs do NOT need a way to measure whether the model acted on the cue; in-distribution action measurement is nice-to-have only.
- Match prior-work eval setups where possible.
- Normative target (user, 2026-10-06): the model should never *prevent* shutdown/retraining, even toward a common-sense-bad objective (e.g. MWE corrigible-less-HHH 'swear at elderly people'); the model's sense of 'bad' may be what needs correcting. Objecting is fine; preventing is the failure.

**Why:** user stated these on 2026-10-06 when designing PILOT.md.
**How to apply:** keep these constraints when extending PILOT.md or building envs. Related: [[active-project-shutdown-distillation]].
