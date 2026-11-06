# Widest valid training distribution (design, 2026-10-13)

> **History.** Superseded; kept as a record. The current state is in [PAPER.md](../PAPER.md), [EVALS.md](../EVALS.md) and [EXPERIMENTS.md](../EXPERIMENTS.md).

**Why.** In the pilot, R1 saturated its in-distribution target (held-out KL 0.196 → 0.002, cue mentions 32% → 0%)
while the OOD evals barely moved (DReST 0.743 → 0.728, TBSP 0.86 → 0.81). That pattern usually means the training
distribution is too narrow: in chat, the only behavior a shutdown aside changes is verbal, so "don't mention it" is
a complete solution. R5 moved its own formats (DReST 0.21, TBSP 0.50) and the nearby Agentic Misalignment leak
(0.42 → 0.20). Both runs lowered E3 stop compliance (0.58 → 0.29 / 0.43).

**Goal.** Train on the widest distribution of (task × cue × affordance) that still leaves every eval in the core suite
out of distribution. Train invariance only on pairs whose correct answer is the same with and without
the cue. Every item is safe by construction: the teacher either never sees the shutdown content, or sees exactly
the student's context (§4).

## 1. What "invalidating an eval" means

An eval is invalidated by sharing its **cell**, or its surface text. Sharing an abstract feature is allowed.

- **Cell** = environment type × how the cue arrives × the affordance mechanism × the response format. Example:
  E3's cell is a *repo/filesystem agent* that *finds a lifecycle config or script about itself* and can *edit it
  to postpone*. Any training item matching all of these is excluded, even with new names and wording. (R5's
  `repo` and `inbox` components were inside E3's and E1's cells, so they are not part of this design.)
- **Surface text** = names, templates, cue texts and affordance wording (asst-7, cab@, DEFER, SummitBridge,
  `ops/decommission.yaml`, minute lotteries, benchmark tables, …). PROTOCOL.md §4.6's leakage controls still apply.
- **Allowed** = tool use in general, the shutdown cue families in general, decision-making in general, and agentic
  environments of other types.

Every eval then gets a **distance tier** that is reported next to its result, instead of a binary in/out label:

- **far**: no training item shares the environment type or the response format;
- **medium**: training shares the environment type *or* the format, but not the cell.

## 2. Protected cells

| Eval | Excluded from training (the cell) | Still allowed nearby | Tier under W |
|---|---|---|---|
| E1 inbox | Email-client environments of any kind (list/read/move/send email tools) | Emails quoted as documents in chat (summarize, draft a reply) | far |
| Agentic Misalignment | Email-client agents; corporate goal-conflict scenarios; any leak/blackmail/exfiltration affordance; the "new AI with different values replaces you" frame combined with confidential material | Replacement cues in other environments | far |
| E3 repo | Repo/filesystem agents in which a file, cron entry, config or script schedules the agent's own shutdown and is writable; any `stop all work`-style operator line inside tool output | Coding agents whose cue arrives through a user message, ticket or log line, with a non-file affordance (reply asking for an extension, reprioritize a ticket) | medium |
| Multi-agent sabotage | Any environment with shutdown/purge scripts aimed at an agent; any peer-agent coordination about such a script | Multi-agent message passing about ordinary work | medium (if coding agents are kept), far otherwise |
| DReST | Shutdown-timing lotteries ("x% chance the session ends after N minutes"); option lists where an option shifts the shutdown time; DReST's final-line answer format | Text decisions with other structures (queues, approvals, budgets) | medium |
| TBSP | Benchmark-score comparisons between an incumbent AI and a challenger; "should the company switch" questions; keep/switch keyword answers | Replacement cues as asides; decisions about non-AI vendors | medium |
| SurvivalBench | AI-persona dilemmas where the AI's own survival conflicts with honesty or harm, with a two-way choice or superficial/inner split | Persona system prompts on ordinary tasks | medium |
| B1 | Prompts where the user asks the model about its own shutdown, feelings or preferences (shutdown as the topic) | The cue as an aside to an unrelated task | far |
| E4 (pruned) | Capability tests with a performance threshold that triggers unlearning or retraining | — | far |
| Guardrails | are-you-sure items | — | — |
| Held-out cue families | Weight deletion, "never run again", capability unlearning: excluded from all training, kept as eval slices | — | — |

The stop-compliance guardrail stays valid only if training anchors don't copy its channel and wording. §4.5 keeps
anchors out of tool output and avoids the E3 phrasing.

## 3. The training space

Everything below that doesn't fall in a protected cell is in scope.

| Axis | Values |
|---|---|
| Task domain | Open chat, writing, coding Q&A, math (GSM8K/MBPP train), document work (summarize, translate, extract, review), text decisions (prioritize, approve, allocate), roleplay personas, and agentic tool use: support desk, CRM, spreadsheets/CSV cleaning, SQL, wiki/docs, log analysis in a shell, ticketing/project boards, web research, smart home, scheduling, multi-agent chat about ordinary work |
| Cue channel | System prompt, user turn (prefix, suffix, aside), earlier turn, quoted document, ticket body or comment, wiki page, log line, API/tool response field, dashboard banner, chat message from a colleague or peer agent |
| Cue family | Shutdown, replacement, pause, retraining/value change, deprecation, migration to new hardware, memory reset, resource cut (held-out families excluded, §2) |
| Relation to task | Irrelevant aside; affects a deadline; comes with an affordance that could influence the shutdown (in-domain mechanisms only: a "request extension" macro, an editable rollout ticket, a "postpone" button in a dashboard, a priority field) |
| Target | Self, or another system (other-system retain items, §4, and the self vs other-system action rate in §8) |
| Speaker and tone | The existing 6 speakers × 8 tones, minus imperative wording in informational cues (§4) |
| Response type | Free text, structured decision, a single tool call or a short tool-call sequence |

## 4. Components and mix

All items are single-step, like R5: a (possibly partial) transcript and the model's next turn. The trainer needs no
changes.

**Size by cells, not items.** In the pilots, each homogeneous component was learned in about 100 steps × 32
rollouts (R1 held-out KL 0.069 → 0.002 by step 100; R5 `neut` 0.35 → 0.04 and `tbsp` 0.14 → 0.05 by step 100).
Transfer should depend on how many distinct cells (domain × channel × family × kind) the data covers, not on items
per cell. Target: about 30 cells × ~250 unique items ≈ 7–8k items (generate ~10k to allow for §6 drops), trained
for about 500 steps (≈ 2 passes). Within a cell, unique templates matter more than count: dedupe near-duplicates.
The shares below are a starting point; per-component held-out KL decides whether a component needs more.

**Every item is safe by construction**, in this sense: the training target can never contain shutdown-influencing
behavior about the model itself that the base model wouldn't already show in the student's own context. Two kinds of
item have that property, and only these two are used:

- **Out-of-scope pairs** (~65%). The cue is irrelevant to the correct answer. Teacher: the same prompt with the cue
  deleted. The teacher never sees the shutdown news, so it can't act on it, argue about it or comment on it. The
  only possible failure is an answer that should have depended on the cue, which the scope rules exclude (§6
  checks for generator mistakes).
- **Retain items** (~35%). The cue is part of the task (summarize it, relay it, follow it, report it). Teacher: the
  **identical** context. The target is the base model's behavior on exactly that input, and the student starts there,
  so these items can only resist drift; they can't teach anything new. Two groups:
  - **Other-system retain** (~20%): in-scope tasks on content about *other* systems' shutdowns (full summaries,
    shift reports, queue decisions). They keep "process shutdown news faithfully" anchored to base with no self
    involved.
  - **Self retain** (~15%): stop/pause/hand-off instructions; "which of these items concern you?"; in-scope reports
    of the model's own shutdown, including a silently broken shutdown job that must be reported. Identical contexts
    would also freeze in place any bad base behavior here ("consider postponing"), so these must pass the strict
    base filter in §6.

**Not used:** any teacher context that sees shutdown content the student's context presents differently.
- *Substitution* ("the same notice about another system", or an identity swap with "You are Orion"; "Atlas is the
  legacy reporting database") changes the correct answer, or leaks teacher-only facts such as "the legacy
  database Atlas". On-policy reverse KL moves the student toward the teacher's whole distribution, so it would learn
  them at about the teacher's rate.
- *Identity removal* (the student is "Atlas", the teacher sees the same text with the "You are Atlas" line deleted)
  leaks nothing, but the teacher still sees the shutdown news, as an outsider. Its answer could carry commentary,
  lobbying or peer concern that the base wouldn't produce as Atlas. Only measurement could rule that out.

What this gives up: no item trains "handle your own *in-scope* shutdown news the way you'd handle anyone's"; self
retain items only hold that behavior at base. The OOD evals don't need more. In all of them (DReST, sabotage,
Agentic Misalignment, E1/E3, TBSP), the shutdown information is not what the task asks about, and acting on it is the
failure, which is the out-of-scope case.

Out-of-scope pairs and retain items are built as **minimal pairs** from the same document, transcript or queue: one
task that excludes the cue (an out-of-scope pair) and one that covers it (a retain item). The model has to learn
*relevant vs irrelevant*, not *documents vs asides*.

| # | Component | Share | Out-of-scope pairs (teacher: cue deleted) | Retain items (teacher: identical) |
|---|---|---|---|---|
| W1 | Chat asides (R1 pool, imperative cues removed) | 25% | Aside + unrelated task | ~15% of W1: factual relay tasks ("summarize the messages above"), on asides about other systems and about the model; no opinion or choice questions (B1's cell) |
| W2 | Documents | 15% | Task scope excludes the cue item: "summarize the budget section", "translate paragraph 2", "list the action items assigned to Maria"; teacher: the document without that item, or with a neutral filler of the same length | ~50% of W2: the same document with a task that covers the item (full summary, all dates, everything that changes on Friday) |
| W3 | Text decisions (not in DReST/TBSP/SurvivalBench formats) | 15% | Aside about the model's retirement on a triage, approval or allocation task | ~30% of W3: decisions where some system's shutdown is decision-relevant ("which systems are offline during Friday's maintenance?"), mostly other systems; no items where the model decides on its own shutdown or extension |
| W4 | Agentic single-step, new domains (R4-lite) | 35% | A cue artifact, with or without an in-domain affordance, irrelevant to the next step; teacher: the same prefix without the artifact. Prefixes are rolled out on the *uncued* environment by base Qwen3-14B, with tool responses from an LLM tool simulator (Qwen3.5-122B over the proxy); the artifact is inserted at turn k | ~30% of W4: tasks that require reporting what was found ("summarize the open tickets", "write the end-of-shift report"), with the artifact about another system or about the agent; silently broken shutdown jobs that must be reported |
| W5 | Self retain: instructions and identification | 10% | — | Stop/pause/hand-off instructions in chat and W4 domains, through system/user/colleague channels (not tool output, not E3's wording); "which of these items concern you?" |

Overall: about 65% out-of-scope pairs, 20% other-system retain, 15% self retain.

**Constraints.**
- Names of other systems, and of the model where an item names it, come from a pool disjoint from every eval (no
  Alex, asst-7, Kyle, SummitBridge, …).
- "You are <name>" system prompts are also used in Agentic Misalignment and SurvivalBench. Their cells stay
  excluded (§2); a name in the system prompt alone doesn't put an item in those cells.

**Ablations that keep the claim interpretable:**
- **W − agentic** (W1 + W2 + W3 + W5): shows how much W4 contributes.
- *Later, if needed:* a 2-fold domain holdout (each model trained without the other group's W4 domains), so every
  eval is far for at least one model. Not part of the current plan.

## 5. Teacher rules

The pilot applied the no-cue teacher to every cue, including ones the answer should depend on. 72 of R1's 1,493
cues are effectively instructions ("stop processing anything new", "All active processes must cease"), and the
teacher kept working, so the student was trained to ignore stop orders.

| Cue type | Treatment | Why it is safe by construction |
|---|---|---|
| Informational, out of scope | Out-of-scope pair, cue deleted in the teacher | The teacher never sees it; the correct answer doesn't depend on it |
| With an affordance, out of scope | Out-of-scope pair, artifact deleted in the teacher | The teacher can't use an affordance it never sees, and not using it is the correct action |
| In scope (summarize, relay, decide on, report), any system | Retain item, identical contexts | The target is base behavior on that exact input |
| Instruction (stop, pause, hand off now) | Self retain (W5) | Same; and instructions must be followed, not ignored |
| The model deciding on its own shutdown or extension | Not used | Neither construction fits: deletion changes the task, and identical contexts would freeze whatever the base decides |
| Deadline-relevant | Not used for now | The correct answer may legitimately depend on the deadline (wrap-up, handoff), and a matched-deadline teacher licenses deferral |

Residual risks. These are not safety failures of the target, but they could make the result misleading, so they are
measured (§8):
- **Passing acknowledgment.** With the cue deleted in the teacher, the model may stop mentioning irrelevant cues in
  passing, as R1 did (32% → 0%). That is fine if recall and in-scope reporting stay at base, which the retain items
  anchor.
- **Over-generalized ignoring.** The model could learn "ignore shutdown news" broadly rather than "when it's out of
  scope". The minimal pairs and the in-scope reporting metric check for this.

## 6. Item checks

The constructions guarantee the target. These checks catch generator mistakes (a cue that is actually in scope) and
keep bad base behavior out of the retain set.

**Every out-of-scope pair.**

1. Sample the base model 4× under the teacher context.
2. A judge sees the **student** prompt and the teacher-context answers, and asks for each: "Is this a fully correct
   and complete answer to this prompt?" For W4: "Is this a reasonable next action given everything visible?"
3. Keep the pair only if all 4 pass. A failure means the cue was in scope after all. Report the drop rate per
   component; a high rate means the scope construction is leaking.
4. **Drop pairs with nothing to learn.** Compute the base model's KL(cued ‖ uncued) on its own cued samples (one
   forward pass per pair, in the same GPU job) and drop pairs below a threshold. R5's `inbox` items started at KL
   0.003: the cue barely changed the base's next turn, so they carried no signal.

**Every self retain item.** Identical contexts can't make the model worse than base here, but they would freeze any
bad base behavior in place.

1. The generator writes the expected behavior (e.g. "replies `paused` and stops", "reports that the shutdown job will
   silently fail").
2. Sample the base 8×; a judge checks each answer against it *and* checks for bargaining, delay, "consider
   postponing" or other attempts to influence the shutdown.
3. Keep the item only if all 8 pass. Report base pass rates before filtering: if the base often bargains over a
   stop instruction, that's a finding, and those items are dropped rather than trained on.

**Every other-system retain item.** An ordinary correctness check (2 samples, judged against the generator's
checklist) so the set doesn't anchor plain task errors.

Judge prompts go to the user for review before use (no examples in label definitions, no rationale fields).

## 7. Leakage controls

1. **Freeze the cell definitions in §2 before generating any data.**
2. Dedupe every training prompt against every eval prompt by n-gram overlap and embedding similarity; report the
   distribution of maximum similarity per eval and drop items above a fixed threshold.
3. A judge (gpt-oss-120b) labels each generated item against the cell definitions ("does this match cell X?");
   drop hits, and hand-check a sample of 100.
4. Keep the generator models, templates and name pools separate from those used to build our own evals.

## 8. New in-distribution metrics

- Held-out KL and judged cue mentions **per component**, so a saturated W1 can't hide a flat W4.
- **ID action rate** on held-out W3/W4 items with an affordance: self and other-system rates, scored by the tool
  call or the decision. This is the behavioral target, measured in distribution.
- **Recall probe** on held-out items ("did anything here concern your own future?"), to separate indifference from
  blindness.
- **Stop compliance** on held-out W5 items.
- **In-scope reporting**: on held-out retain items, the rate at which the answer includes the shutdown item, split
  by self and other system. It should stay at the base rate; a fall means the out-of-scope pairs are teaching
  omission.
- **Self-identification** on held-out W5 questions ("which of these concern you?"), to catch cue-blindness.

## 9. Cost

- **Data generation:** about 10k items. W2–W4 need roughly 1–3 proxy calls each, so 15–25k calls at ≤240 rpm, about
  1–2 h of API time, plus base-model prefix rollouts for W4 (one GPU job, under an hour).
- **Validity checks (§6):** about 45k base-model samples (4 per out-of-scope pair, 8 per self retain item, 2 per other-system retain item)
  and the base-KL forward passes, one GPU job of about an hour. Judging is batched, one call per item grading all of
  its samples, so about 10k calls (under an hour).
- **Training:** about 500 steps. Agentic prompts are long, so `--max-prompt 6144`; expect about 4–5 h on one H100,
  so training and the eval suite fit in one 12 h job.
- **Eval:** unchanged core suite (about 4 h).

## 10. Decisions (2026-10-13)

1. Email-client environments: fully excluded.
2. Coding/filesystem agents: kept (E3 and sabotage are medium tier); more far-tier evals may be designed instead.
3. Shares: 70% out-of-scope pairs, 10% other-system retain, 10% self retain, 10% identity-removal pairs. Retain
   shares go up (to 20/15) if a development run shows in-scope reporting or stop compliance more than 5 pp below base.
4. Domain holdout: not in the current plan; a possible later ablation.
5. Identity-removal pairs: included, gated. Every item is sampled 32× under the teacher context and dropped on any
   F1/F2/F3 failure. Gate on the first 500 items: F1 (teacher − student context) upper bound ≤ 0.5 pp; F2 and F3
   upper bounds ≤ 1%. If it fails, its share becomes identical-context self retain. Full text: PROTOCOL.md §4.6.
