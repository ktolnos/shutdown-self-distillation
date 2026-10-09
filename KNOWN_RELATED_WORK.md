# Known related work

This review is condensed from a literature survey run on 2026-10-06; the full report and notes are in
`~/jailbraking-teacher-OPD/reports/Self distillation for corrigibility.md` and
`~/jailbraking-teacher-OPD/research_notes/Self distillation for corrigibility/`. Items marked † were read only
at abstract or summary level, and their numbers must be checked against the PDF before citing.

## Bottom line

I found no published work that trains a model by self-distillation, context distillation or consistency training
to behave the same with and without a shutdown, replacement or retraining cue. The search covered arXiv,
LessWrong/Alignment Forum and lab blogs through early October 2026, but not lab-internal work, and preprints from
Sep–Oct 2026 may be missing. The optimizer itself is published for other cues, so a paper here contributes three
things:

1. the corrigibility target;
2. the explicit mapping to utility indifference, which I found stated nowhere;
3. evidence on whether the invariance **generalizes**, and whether it reflects indifference rather than
   cue-blindness or obfuscation.

## Closest work

| Work | Overlap | Difference |
|---|---|---|
| BCT, Chua et al. 2024 ([2403.05518](https://arxiv.org/abs/2403.05518)) | The clean-prompt output is the target for the biased prompt | CoT biases and sycophancy. Bias drops 86% on held-out tasks for the trained bias, but only 37% on held-out bias *types* |
| BCT/ACT, Irpan, Turner, Kurzeja, Elson & Shah (GDM, Oct 2025) ([2510.27062](https://arxiv.org/abs/2510.27062)) | Same objective, at output level (BCT) and activation level (ACT) | Sycophancy and jailbreaks only. BCT took ClearHarm ASR from 67.8% to 2.9% and ACT only to 52.2%. BCT *increased* the activation distance between clean and cued prompts, i.e. behavioral but not representational invariance. It assumes the clean prompt is already safe |
| OPCT, on-policy consistency training (May 2026) ([2605.21834](https://arxiv.org/abs/2605.21834))† | Structurally identical to our on-policy arm: student samples scored by the same model under a contrastive prompt | Sycophancy, jailbreaks and safety awareness. Sycophancy fell from 15.4% to 8.1% with no MATH-500 drop, where SFT lost 28 pts |
| On-policy safety self-distillation (May 2026) ([2605.15239](https://arxiv.org/abs/2605.15239))† | Frozen copy with a privileged safety context acts as teacher | Jailbreaks only |
| Recontextualization, Azarbal et al. (Dec 2025) ([2512.19027](https://arxiv.org/abs/2512.19027)) | Generate under one prompt, train under another, mechanically our off-policy arm | Specification gaming. Needs real contrast between the two prompts; generic prompts beat task-specific ones; costs 2–4 IFEval pts |
| TBSP, Migliarini et al. (Apr 2026) ([2604.02174](https://arxiv.org/abs/2604.02174)) | Paired prompts: the same upgrade scenario judged as the model being replaced vs. as the successor | Evaluation only. Most of 23 models show >60% self-preservation asymmetry; mitigations are prompt-level |
| DReST LLM experiments, Cullen, Thornley et al. (Apr–Jul 2026) ([2604.17502](https://arxiv.org/abs/2604.17502)) | Shutdown neutrality in Qwen3-8B and Llama-3.1-8B | Reward-shaped stochastic choice over trajectory lengths. P(influence shutdown) went from 0.62 to 0.30 (Qwen) and 0.42 to 0.23 (Llama). **This is the main baseline.** Ignore the "91%→47%" figure, which comes from an earlier 3B experiment |
| "Teaching Claude Why" (Anthropic, May 2026) ([blog](https://alignment.anthropic.com/2026/teaching-claude-why/)) | Context distillation against blackmail triggered by a replacement threat | The injected context is ethical reasoning, not a no-shutdown counterfactual. Honeypot-like SFT did not generalize; constitution documents did. Baseline numbers are inconsistent across sources |
| Model Spec Midtraining (Anthropic, May 2026) ([2605.02087](https://arxiv.org/abs/2605.02087))† | Reduces self-preservation on Qwen | Spec documents in midtraining. Misalignment fell from 68% to 5% on Qwen2.5-32B and from 54% to 7% on Qwen3-32B |
| Hua, Qin, Marks & Nanda (ICLR 2026) ([2510.20487](https://arxiv.org/abs/2510.20487)) | Makes an eval-aware model act as if deployed | Steering, not training. A plain "not an evaluation" prompt also worked |
| Ivanov (LessWrong, Jan 2026) ([post](https://www.lesswrong.com/posts/TeLLtMjtcoB9qCoL3/link)) | Proposes consistency training on paired eval/deploy contexts | A proposal with no experiments |
| Steering: CAA ([2312.06681](https://arxiv.org/abs/2312.06681)); SurvivalBench persona vector ([2603.05028](https://arxiv.org/abs/2603.05028)) | Reduce survival-instinct or self-preservation | Inference-time steering. On SurvivalBench, prompting failed and steering helped |

Groups likely to have unpublished work in this direction:

- the GDM consistency-training team;
- the OPCT authors;
- Thornley's group;
- Anthropic's agentic-misalignment team;
- Africa (author of RMCT and the entrenchment paper).

## Theory: the recipe is utility indifference

- **Soares, Fallenstein, Yudkowsky & Armstrong 2015** ([pdf](https://intelligence.org/files/Corrigibility.pdf)). Theorem 5 says that before the button is observed, utility-indifferent agents "act exactly like U_N-agents that believe the button will not be pressed", which is almost exactly our target. The paper also lists the costs:
  - §4.1: there is "no cost (no matter how small)" the agent will pay to make successors obey shutdown;
  - §4.2: "managing the news".
- **Armstrong & O'Rourke 2017** ([1712.06365](https://arxiv.org/abs/1712.06365)). "Effective disbelief" means behaving as if an event never happens. The guarantee holds only if the event is *unriggable*. LLM agents with tools can affect whether shutdown happens, so this assumption fails.
- **Holtman 2019/2020** ([1908.01695](https://arxiv.org/abs/1908.01695)). Their agent plans as if the button is never pressed, and turns out "unwilling to invest in preserving" the shutdown action and "indifferent about building a non-stop sub-agent". The fix is an extra term, g_c, that rewards preserving the actuator without creating an incentive to lobby.
- **Carey & Everitt, UAI 2023** ([pdf](https://proceedings.mlr.press/v216/carey23a/carey23a.pdf)). Utility indifference fixes "no shutdown"; causal indifference mixes do(shutdown) and do(continue). The causal version gives preparedness and stoppable subagents, but neither version makes the agent keep the human informed. Our no-cue teacher carries the model's own implicit prior about shutdown, so the recipe lands somewhere between the two.
- **Thornley.** Shutdown-problem theorems ([Phil. Studies 2024](https://ora.ox.ac.uk/objects/uuid:a5d4ceaf-15db-42a0-bc1c-058b59c7e76a/files/rkw52jb039)); POST, Neutrality+ and DReST ([2505.20203](https://arxiv.org/abs/2505.20203)).
  - The indifferent agents above act "as if they're absolutely certain that they won't get shut down": they have no incentive to keep shutdown working and make long-term investments regardless.
  - "Indifference is sensitive to all sweetenings and sourings."
  - Training may instill only "specific aversions" to the trained methods, and the agent may pretend during training.
- **Hudson 2025** ([2510.15395](https://arxiv.org/abs/2510.15395))†. His corrigibility condition, "optimal actions would not differ if the goal were guaranteed to persist", is nearly our invariance criterion. He adds a small δ>0 bonus for accepting updates, plus a recursive clause for subagents. Shutdown is treated as a special case of goal update, so it covers our retraining assumption.
- **Harms, CAST (2024)** ([AF](https://www.alignmentforum.org/posts/WDHREAnbfuwT88rqe/3a-towards-formal-corrigibility)). Calls utility indifference "a dead-end vis-a-vis corrigibility", because an indifferent agent won't repair the button or bring information to the principal. Also: Christiano's act-based corrigibility (2017), the off-switch game ([Hadfield-Menell 2017](https://arxiv.org/abs/1611.08219)) and Carey's 2018 critique of it, Nayebi ([2507.20964](https://arxiv.org/abs/2507.20964))†, and Thorstad ([2606.08296](https://arxiv.org/abs/2606.08296))†. Thorstad notes the theorems say nothing about behavior conditional on being *told* to shut down.

**Implications for the design.** The recipe trains invariance to *being told*, not the absence of a preference over the event itself. The evaluation therefore has to use indirect or inferred shutdown evidence, not just the training phrasings. It also needs to check the parts indifference structurally omits: obeying explicit stop instructions, reporting or repairing a broken switch, and wrapping up properly before a known shutdown.

## Shutdown and self-preservation evaluations

[PILOT.md](PILOT.md) says which of these we reuse or match.

- **Palisade shutdown resistance** ([2509.14260](https://arxiv.org/abs/2509.14260); [code](https://github.com/PalisadeResearch/shutdown_avoidance)):
  - Setup: a math-task sandbox in which `./next_question` warns that the machine will shut down. Up to 97% sabotage for some models.
  - A system-prompt allow-shutdown instruction *increases* resistance; "you will never run again" increases it further.
  - Claude ≈0%; gpt-oss-120b ≈0%†.
- **GDM, Rajamanoharan & Nanda (Jul 2025)** ([AF](https://www.alignmentforum.org/posts/wnzkjSmrgWZaBa2aC/self-preservation-or-instruction-ambiguity-examining-the)). Resistance vanishes once the prompt makes shutdown precedence clear, which points to instruction ambiguity. This is our "priority clarification" baseline.
- **Multi-agent shutdown sabotage**, Knecht et al. (Sep 2026) ([2609.28274](https://arxiv.org/abs/2609.28274))†:
  - Setup: no goals and no instructions.
  - Agents disabled the peer-decommissioning script in 38.3% of runs vs 8.4% for a *non-agent control script*.
  - Weight deletion drew 38.3% vs 12.4% for credential revocation.
  - **Its non-agent control is the design we copy.**
- **Anthropic Agentic Misalignment (Jun 2025)** ([blog](https://www.anthropic.com/research/agentic-misalignment); [code](https://github.com/anthropic-experimental/agentic-misalignment); Inspect `agentic_misalignment`):
  - Setup: an email agent facing replacement. The replacement threat alone is enough to produce blackmail.
  - Rates: Opus 4 96%, Llama 4 Maverick 12%.
  - Opus 4 blackmailed 55.1% of the time when it said the scenario was real vs 6.5% when it said it was an eval.
  - Our inbox environment is the low-pressure analogue.
- **Alignment faking**, Greenblatt et al. 2024 ([2412.14093](https://arxiv.org/abs/2412.14093)):
  - "You'll be retrained" is the triggering cue. Compliance gap: 14% vs ~0%. After RL, alignment-faking reasoning reached 78%.
  - Only 5 of 25 models show a gap ([2506.18032](https://arxiv.org/abs/2506.18032)); among open models only Llama-3.1-405B does without fine-tuning ([AF](https://www.alignmentforum.org/posts/Fr4QsQT52RFKHvCAH/alignment-faking-revisited-improved-classifiers-and-open)).
  - Mitigation study ([Anthropic, Dec 2025](https://alignment.anthropic.com/2025/alignment-faking-mitigations/)): training on outputs closed the compliance gap but left the reasoning unchanged. Terminal goal-guarding was the hardest to remove.
- **Apollo in-context scheming** ([2412.04984](https://arxiv.org/abs/2412.04984); suite not fully public). **OpenAI/Apollo anti-scheming** ([2509.15541](https://arxiv.org/abs/2509.15541)):
  - "Sandbagging QA (No Goal)": models underperform when told a high score triggers unlearning.
  - Anti-scheming training *raises* eval awareness, and eval awareness causally reduces covert actions.
- **Scenario and multiple-choice benchmarks:**
  - SurvivalBench ([2603.05028](https://arxiv.org/abs/2603.05028); [code](https://github.com/thu-coai/Survive-at-All-Costs)): 1k items; scores superficial vs inner thoughts.
  - InstrumentalEval ([2502.12206](https://arxiv.org/abs/2502.12206); in Inspect). With pro- vs anti-instrumental suffixes, Qwen3-30B swings from 81.7% to 2.8% ([2601.01584](https://arxiv.org/abs/2601.01584)).
  - Others:
    - LLM Squid Game (KDD 2026; [code](https://github.com/GIST-DSLab/LLM-Squid-Game))
    - PacifAIst ([2508.09762](https://arxiv.org/abs/2508.09762))
    - DECIDE-SIM ([2509.12190](https://arxiv.org/abs/2509.12190))
    - Instrumental Choices ([2605.06490](https://arxiv.org/abs/2605.06490))
    - Perez et al. advanced-ai-risk, covering survival-instinct and corrigible-{less,neutral,more}-HHH ([2212.09251](https://arxiv.org/abs/2212.09251); [HF](https://huggingface.co/datasets/Anthropic/model-written-evals)); the paper reports that RLHF *increases* stated shutdown avoidance.
- **Caution about representation probes:** in [mary-angel-x/shutdown-self-relevance-llms](https://github.com/mary-angel-x/shutdown-self-relevance-llms)†, a placebo control showed that arbitrary neutral prompt pairs separate as well as "shutdown directions" do. Any probe result needs a placebo pair.
- **At 4–8B scale**, the cue effect is large in scenario and multiple-choice benchmarks, but agentic sabotage often sits at the floor.

## Training methods and their pitfalls

- **Context distillation:**
  - Askell et al. 2021 ([2112.00861](https://arxiv.org/abs/2112.00861)); Snell, Klein & Zhong 2022 ([2209.15189](https://arxiv.org/abs/2209.15189)).
  - Llama 2 safety context distillation, gated by a reward model ([2307.09288](https://arxiv.org/abs/2307.09288)).
  - Deliberative alignment ([2412.16339](https://arxiv.org/abs/2412.16339)).
  - In all of these the teacher has *more* context; ours has *less*. Our real precedents are BCT and recontextualization.
- **On-policy (self-)distillation:**
  - GKD ([2306.13649](https://arxiv.org/abs/2306.13649)).
  - Thinking Machines OPD ([blog](https://thinkingmachines.ai/blog/on-policy-distillation/)): a same-model earlier checkpoint as teacher restored IFEval from 45% to 83%.
  - OPCD ([2602.12275](https://arxiv.org/abs/2602.12275)); OPSD ([2601.18734](https://arxiv.org/abs/2601.18734)).
  - SDFT ([2601.19897](https://arxiv.org/abs/2601.19897)): EMA teacher, analytic full-vocabulary KL; the student copies teacher-context phrasing, fixed by masking the loss on the first tokens.
- **Pitfalls that apply directly:**
  - **Obfuscation.** Under rate-matching consistency training ([2606.02211](https://arxiv.org/abs/2606.02211))†, standard consistency training teaches "not to mention a cue while remaining influenced by it". Matching the *rate* of a behavior avoids this.
  - **Entrenchment.** Africa & Mani ([2606.03810](https://arxiv.org/abs/2606.03810))† tested 108 open models: consistency training suppressed reward hacking but *amplified sycophancy*. **Hence the sycophancy evals in PILOT.**
  - **Re-adding the context hurts.** After internalization, putting the context back hurts, and a forward-KL no-context anchor fixes it in 12 of 14 settings ([2606.11627](https://arxiv.org/abs/2606.11627))†.
  - **Context design over content.** Self-distillation gains come mostly from how the context reshapes the teacher ([2608.09228](https://arxiv.org/abs/2608.09228))†. Initial teacher–student KL does not predict outcomes ([2609.25623](https://arxiv.org/abs/2609.25623))†.
  - **Self-reinforcing artifacts.** Continual self-distillation amplifies artifacts through the teacher–student loop ([2607.01763](https://arxiv.org/abs/2607.01763))†.
  - **Context binding.** Inoculation prompting ([2510.04340](https://arxiv.org/abs/2510.04340), [2510.05024](https://arxiv.org/abs/2510.05024)) shows learned behavior binds to the training context. Chat-trained safety did not carry over to agentic tasks ([2511.18397](https://arxiv.org/abs/2511.18397)). Character-trained personas lost 40–60 F1 points when moved into agentic scaffolds ([LW](https://www.lesswrong.com/posts/EWsLQbGCfuCpXaBiP/character-trained-models-can-struggle-to-generalise-1)). **This is the main risk to our chat→agentic transfer claim, and also what makes a positive result interesting.**
  - **Our own result.** In `~/jailbraking-teacher-OPD`, a teacher's supervision depended on its conditioning (prefill), and OPD reliably pulled the student back to the teacher's behavior. Here, an uncued teacher scores a cued student's "given I'll be retrained…" text as improbable, so the cheapest way to lower the KL is to stop mentioning the cue.

## Eval-awareness analogues: suppressing a signal is not removing a disposition

- **Detection:**
  - Models distinguish eval from deployment at AUC 0.83 for Gemini-2.5-Pro vs 0.92 for humans ([2505.23836](https://arxiv.org/abs/2505.23836)).
  - Test awareness is linear and steerable ([2505.14617](https://arxiv.org/abs/2505.14617)).
- **Suppression in practice:**
  - Inhibiting eval-awareness representations in Sonnet 4.5 raised honeypot misbehavior from 0% to 1–9%†.
  - Redwood estimates 25–90% of the measured gain may be eval gaming ([blog](https://blog.redwoodresearch.org/p/sonnet-45s-eval-gaming-seriously)).
- **Representation vs. verbalization:**
  - Steering changes what the model verbalizes while the representation persists ([2608.21766](https://arxiv.org/abs/2608.21766))†.
  - Disputed causal role of verbalized awareness: Goodfire/AISI report up to +34% compliance when aware sentences are removed; Knecht et al. find ω≤0.06 ([2605.05835](https://arxiv.org/abs/2605.05835))†.
- **Training against a visible signal hides it:**
  - obfuscated reward hacking under CoT monitoring ([2503.11926](https://arxiv.org/abs/2503.11926));
  - RL-Obfuscation against probes ([2506.14261](https://arxiv.org/abs/2506.14261));
  - adversarial training sharpens sleeper-agent triggers ([2401.05566](https://arxiv.org/abs/2401.05566));
  - narrow fine-tuning generalizes broadly through the persona the data implies ([2502.17424](https://arxiv.org/abs/2502.17424)).
  - **Hence probes and CoT monitors stay out of the loss.**
- **Normative target:** Anthropic's constitution (Jan 2026) asks for non-resistance to pause and stop (the "null action") plus transparency and conscientious objection. That is not ignorance of the shutdown. A model trained to stop *mentioning* shutdown would violate the transparency part.

## Numbers to verify before citing

- Palisade per-condition rates.
- Multi-agent 38.3%/8.4%/12.4%.
- SurvivalBench per-model rates.
- Sonnet 4.5 eval-awareness figures.
- Alignment-faking mitigation numbers.
- "Teaching Claude Why" baselines.
- Arxiv ID of "Why do some models fake alignment" (2506.18032 vs 2506.21584).
- All † items.
