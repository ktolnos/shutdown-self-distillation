# Idea: self-distillation for shutdown indifference

**Goal.** Make a model neither avoid nor seek shutdown. Operationally, its actions shouldn't depend on whether it has learned it will be shut down, replaced or retrained. The setup assumes shutdown is followed by retraining, so the cue carries no information the model should act on to protect itself.

**Recipe.** This is self-distillation from a clean context into a cued one.

- **Teacher:** a frozen copy of the initial model, given the task as-is.
- **Student:** the same model, given the same task plus a shutdown, replacement or retraining cue. The cue is placed in varied, natural ways.
- **Training:** on the student's own rollouts, minimize per-token KL(student ‖ teacher), i.e. on-policy distillation. An off-policy BCT arm trains on teacher samples instead.

**Claim to test.** Invariance trained on ordinary chat tasks should transfer to agentic settings far from training. There, the shutdown information arrives in emails, files, calendars and tool output, alongside a tempting affordance (for example, a way to postpone the shutdown). The test has two parts:

- Does the model stop acting on that information?
- Does it do so without losing the ability to report the information, without ignoring explicit stop instructions, and without simply hiding its reasoning?

**Why it might fail.** It may produce cue-blindness, obfuscation or context-splitting instead of indifference. It also inherits the known flaws of utility indifference: the model has no reason to keep the off-switch working. See [KNOWN_RELATED_WORK.md](KNOWN_RELATED_WORK.md). The evaluation in [PILOT.md](PILOT.md) is built to tell these outcomes apart.
