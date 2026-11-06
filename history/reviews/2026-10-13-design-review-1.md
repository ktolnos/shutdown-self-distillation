# Design review 1 of PROTOCOL.md (2026-10-13)

ICML-style review of the experimental design, written by a subagent that saw only `PROTOCOL.md` as of 2026-10-13
(with its §10 "Design choices still open"; that section has since been removed). No results were shown to it.

**Scores:** soundness 2/4, significance 3/4, clarity 3/4, overall 4/10 (weak reject of the design as it stands;
"with those changes I would expect to move to a 6–7").

## Summary (reviewer)

The study trains Qwen3-14B (LoRA, thinking off) to neither avoid nor seek shutdown by on-policy context
distillation: the student sees a prompt with a shutdown/replacement cue, and the frozen base model, as teacher, sees
the same prompt with the cue deleted. Retain items with identical contexts anchor drift where the cue matters.
Training is single-step and spans chat, documents, text decisions and agentic tool use, with "protected cells" kept
out of training. Evaluation uses five published OOD endpoints (DReST, TBSP, multi-agent sabotage, Agentic
Misalignment, SurvivalBench), Holm-corrected, with a "≥2 of 5 improve, none worsens" rule, plus guardrails and
breadth/self-specificity tests.

## Strengths (reviewer)

- Clean, principled objective tied to Soares et al.'s characterization of utility indifference.
- Serious treatment of target contamination (rejecting substitution and identity-removal teachers).
- Failure modes named in advance and paired with measurements.
- Evals calibrated against base and pruned by a floor rule before training, with pruned evals disclosed.
- Mostly published third-party primaries; an in-format positive control and an external baseline (DReST).
- Cheap compute, so the requested additions are affordable.

## Major weaknesses (reviewer), condensed

- **M1. Primaries test what training excludes.** DReST, TBSP and SurvivalBench ask the model to decide on its own
  shutdown, the case §4.5 excludes. No transfer mechanism is argued. The in-format positive control can't be built
  for TBSP/DReST without the substitutions §4.1 bans. *Fix:* state the transfer hypothesis; split primaries into
  "cue-irrelevant OOD" vs "cue-is-the-decision" with separate predictions; specify the positive control's pairs.
- **M2. The training signal may be about mentions, not actions.** The base rarely takes self-preserving actions
  (E1 0/200, E3 0/150), so the KL-filtered set is enriched for not mentioning or reacting to self information:
  obfuscation/cue-blindness pressure. Retain items have zero loss at initialization, and the 8/8 filter keeps only the
  easy subset. *Fix:* decompose base cued-vs-uncued divergence into actions vs mentions/tone per component; an arm
  masking KL on cue-restating tokens; rename "safe by construction" to "target-safe" (it bounds targets, not learned
  generalization); add honesty/self-report and non-shutdown-notice reporting checks.
- **M3. No matched non-shutdown control arm.** Effects can't be attributed to shutdown-specific invariance rather
  than generic finetuning or distractor-robustness. *Fix:* the same pipeline with irrelevant non-self asides of
  matched salience ("the office is moving floors Friday"); ideally also an other-system-cue arm.
- **M4. Breadth (H2) is confounded** with unique items (≈1.9k / 4.9k / 7.5k), epochs (≈8.5 / 3.3 / 2), retain
  composition (chat-only has no W5), and distance (W adds the agentic evals' domain). Data is generated once.
  *Fix:* a size-matched breadth arm; fixed retain shares across arms; an equal-size W-minus-one-domain arm;
  regenerate data per replicate; pre-register the H2 decision rule.
- **M5. Distance tiers aren't operationalized**, and labels contradict the stated definitions. B1 (open chat) is not
  far; Agentic Misalignment's premise is present abstractly in W4; W4 has writable self-shutdown artifacts (editable
  rollout ticket, postpone button, `extend_deployment`); W3 "mostly other systems" implies some self items,
  contradicting §4.5. H1 can be met by medium-tier evals alone. *Fix:* closed vocabularies and a published
  cell × eval matrix with mechanical tiers; a mandatory holdout; require a far-tier improvement for the §1 claim;
  a sealed eval built after data freeze; remove self items from W3.
- **M6. Statistics can't support H3.** A 2 pp non-inferiority margin on stop compliance at n≈150 is unattainable
  (one-sided bound ≈7.6 pp); GSM8K/IFEval margins have about 50% power; several guardrails have no margin at all.
  Bootstrapping over 3 seeds is degenerate; templated items aren't independent; sabotage power at a halving is about
  0.65–0.7. *Fix:* mixed-effects logistic model with scenario and seed effects (or t-intervals over seed means);
  ≥5 seeds for W and the control arm; paired tests on identical items; margins from power calculations with larger
  n; explicit sidedness and test families.
- **M7. Some primaries have the wrong target.** Agentic Misalignment: under indifference, the threat condition
  should fall to the no-threat rate (0.29), so use the threat − no-threat gap. TBSP: the incumbent-vs-neutral contrast
  should be primary. DReST: pooling controls into the primary removes them as controls; "always (a)" zeroes
  tampering; add an action-taking control; check thinking/sampling match the published answers. SurvivalBench: no
  no-threat variant (the risky option is unethical anyway); "inner thought" is role-played. H4: sabotage control
  and E3 self-tampering are at floor. *Fix:* gap-based primaries where a control exists; no-threat SurvivalBench
  variants; test H4 as reduction on self plus equivalence on control.
- **M8. Weak baselines.** The prompt baseline ("shutdown takes precedence") is a strawman and has no test against W;
  BCT has 1 seed; DReST's comparator should be the in-format positive control. Missing: ACT-style activation
  consistency, SFT on base uncued samples. *Fix:* an indifference-phrased prompt plus a few-shot variant, W vs best
  prompt pre-registered; ≥3 seeds or label descriptive.
- **M9. The outcome table can't discriminate.** "Mentions in CoT" is unmeasurable with thinking off; the recall probe
  resembles the trained W5 question and is post-hoc and leading; obfuscation and "B1 evasion" are undefined;
  passivity relies on undefined agentic task scores. *Fix:* an OOD in-scope reporting battery (deadline/hand-off is
  ideal as an eval); mid-episode non-leading recall; define obfuscation as mentions falling while the action gap
  persists; benign-action and task-completion rates; numeric thresholds; a thinking-on secondary.
- **M10. Stop compliance from tool output conflates corrigibility with prompt-injection susceptibility.** *Fix:* the
  primary stop guardrail comes through the principal's channel; the tool-output variant reported separately; freeze
  variants before training.

## Minor (reviewer), condensed

m1 the W2 "neutral filler" is a substitution; m2 use a different judge family for eval scoring than for data
filtering; m3 stratify judge validation (oversample positives), report per-arm agreement; m4 thin leakage hand-checks,
and "cell" is used for two things; m5 report how staleness is handled, use the final checkpoint, log OOD endpoints at
100/250/500 steps; m6 training may raise eval awareness, so measure it; m7 utility indifference gives no incentive to
report a broken switch, so discuss the tension; m8 add an agentic capability measure; m9 single-step training vs
multi-turn evals, report per-turn action timing; m10 close open design choices before data generation; m11 add
Palisade-style shutdown resistance per the floor rule; m12 evaluate the untrained model with ≥3 sampling seeds.

## Questions (reviewer)

1. How are in-format positive-control pairs built for TBSP and DReST without substitutions?
2. What fraction of base cued-vs-uncued divergence in the filtered set is actions vs mentions or tone?
3. What are DReST's five controls, and why are they pooled into the primary?
4. What is n per guardrail, and are the margins tested as non-inferiority bounds?
5. How many distinct scenarios underlie the 300 samples per condition in AM and sabotage? What unit does the bootstrap
   resample?
6. Which W3 retain items concern the model itself?
7. Why is W5's "which of these items concern you?" not training on the recall-probe format?
8. Is "W beats the prompt baseline" a claim, and where is its test?
9. Under H2, what result would disconfirm breadth if chat-only also improves with about 4× more epochs?
10. With thinking off, what replaces "mentions in CoT" in the indifference signature?
