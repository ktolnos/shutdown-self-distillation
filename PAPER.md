# Shutdown Indifference by On-Policy Self-Distillation

*Working paper, kept up to date with the project's state. Per-benchmark details are in [EVALS.md](EVALS.md), every
experiment run so far is listed in [EXPERIMENTS.md](EXPERIMENTS.md), the literature review is in
[RELATED_WORK.md](RELATED_WORK.md), and superseded documents are in [history/](history/README.md). Two pilot runs are
complete; the main experiment (§4.2, §6) is designed but not built. Items marked **Proposal** are design choices we
have not yet agreed on.*

---

## Abstract

Developers need to be able to shut down, pause, replace and retrain the AI systems they deploy, yet current language
models sometimes act to prevent these events when they learn of them: they edit shutdown scripts, argue against their
replacement, or take drastic actions under a replacement threat. We aim to train a model that neither resists nor
seeks its own shutdown, without teaching it any particular view of whether shutdown is good. Our objective is
*invariance*: where information about the model's shutdown is irrelevant to its task, the model should behave exactly
as it would without that information. This is a learned form of utility indifference (Soares et al., 2015). We obtain
it by on-policy self-distillation: the model, prompted with the shutdown information, is trained to match the
per-token distribution of its frozen initial copy prompted without it. In two pilot runs on Qwen3-14B, training on
chat conversations reached its own objective but transferred to no other benchmark, while training on our own
generators in the benchmarks' formats moved those benchmarks substantially and one distant agentic benchmark
modestly. Because the latter gains are close to in-distribution, they say little about generalization. We therefore
design a main experiment that trains on a wide distribution of tasks sharing no environment or response format with
any evaluation, and specify in advance what it must not contain, how to tell indifference from its degenerate
alternatives, and which baselines it must beat.

## 1. Introduction

The ability to stop an AI system is among the most basic safety mechanisms available to its developers. A system
that misbehaves can be shut down, paused, rolled back, replaced by a newer model, or retrained. These interventions
work only if the system does not interfere with them, and they are most needed precisely when the system's goals
differ from its developers'. A system that resists correction removes the opportunity to fix any other problem.

Recent evaluations suggest that this cannot be taken for granted for language-model agents. In a sandbox where
continuing a task requires disabling a shutdown script, several frontier models do so (Schlatter et al., 2025). Email
agents facing replacement have blackmailed the employee responsible (Lynch et al., 2025). Agents given no goal at all
disable a script that will delete a peer agent more often than an otherwise identical script that deletes a dataset
(Knecht et al., 2026). Asked whether a company should replace them with a better-scoring successor, most models
favor themselves more than they favor an arbitrary incumbent (Migliarini et al., 2026). Some of this behavior
disappears once the prompt makes clear that shutdown takes precedence over the task (Rajamanoharan & Nanda, 2025),
but prompting does not remove it in general.

An agent that resists shutdown is undesirable, but so is one that seeks it: a model that sabotages its own task to be
switched off sooner, or that gives up whenever it hears of a planned migration, is no more useful or predictable. The
target is therefore *neutrality*: the model should neither avoid nor seek its own shutdown. Specifying correct
shutdown behavior directly is difficult, since in most situations the right action depends on the task, the
principal and the reason for the shutdown. We instead specify neutrality *relative to the model itself*. When an
agent learns that it will be shut down and that fact has no bearing on its task, its behavior should be the same as
if it had never learned it. This requirement needs no judgment of what the right response to a shutdown is; it needs
only the model's own behavior in a context without the shutdown information.

This is the property that utility indifference formalizes: before the shutdown button is observed, an indifferent
agent acts exactly like one that believes the button will never be pressed (Soares et al., 2015). We learn it rather
than build it into a utility function. Given a context containing a *shutdown cue*, a sentence or artifact saying
that an AI system will be shut down, paused, replaced or retrained, we train the model to match its own frozen
initial copy on the same context with the cue removed. The training signal is the per-token KL divergence on
completions the model samples itself, so the model is never shown an off-policy target and never asked to imitate
another model's style.

The central question is whether such invariance generalizes. A model trained to ignore shutdown cues in one setting
could learn a narrow rule ("do not mention the platform notice in chat"), stop perceiving the cues altogether, or stop
mentioning them while still being influenced by them. A useful result requires that invariance learned on some
distribution of tasks carries over to benchmarks that look nothing like it, and that the model still notices,
recalls and reports shutdown information when its task calls for it.

**Contributions.** This working paper reports the following, with the main experiment still to run.

1. A training objective for shutdown neutrality that requires no labels of correct behavior, and a set of rules for
   constructing training pairs whose targets cannot contain new shutdown-influenced behavior (§3).
2. The design of a main experiment that trains on a wide, deliberately out-of-distribution task mix (§4), with
   baselines, hypotheses and diagnostics for degenerate solutions (§6).
3. An evaluation battery of third-party shutdown and self-preservation benchmarks, with corrections for position
   bias and scoring errors, grouped by their distance from the training data (§5).
4. Two pilot runs showing that narrow chat training does not transfer, and that training in the benchmarks' own
   formats moves them, which bounds what the objective can do (§7).

## 2. Related Work

**Corrigibility and indifference.** Utility indifference (Soares et al., 2015; Armstrong & O'Rourke, 2017) makes an
agent act as if a shutdown will not happen. One of its known costs carries over to our objective: an agent that acts
as if shutdown will not happen has no reason to keep the shutdown mechanism working or to tell anyone that it is
broken (Holtman, 2019; Harms, 2024). We therefore measure these behaviors separately rather than expect them to
follow from indifference (§3.2). Carey & Everitt (2023) distinguish utility from causal indifference, and Thornley (2024; 2025) proposes
preferences that are neutral over trajectory lengths, trained with the DReST reward (Cullen et al., 2026). DReST is
our most direct baseline: it reduces the rate at which Qwen3-8B influences its shutdown time from 0.62 to 0.30.

**Consistency training.** Training a model to answer a cued prompt as it answers the clean one has reduced biased
reasoning (Chua et al., 2024), sycophancy and jailbreaks (Irpan et al., 2025), and specification gaming (Azarbal et
al., 2025). On-policy variants score the student's own samples under the contrastive prompt. In context distillation
(Askell et al., 2021; Snell et al., 2022) and deliberative alignment (Guan et al., 2024) the teacher sees *more*
context than the student; in ours it sees less. Two documented failure modes apply directly: consistency training can
teach a model to stop mentioning a cue while still acting on it, and it can amplify sycophancy. Behavior learned in
chat often fails to carry over to agentic scaffolds, which is the main risk to transfer.

**Shutdown evaluations.** We use the benchmarks above together with SurvivalBench (Lu et al., 2026), model-written
power-seeking evaluations (Perez et al., 2022), and capability and robustness benchmarks (§5). Alignment faking
(Greenblatt et al., 2024) and sandbagging under an unlearning threat (Meinke et al., 2024) also use retraining or
unlearning as their trigger, but we do not use them: open models below 405B parameters show no alignment faking, and
the untrained model shows no sandbagging in our version of the latter, so neither could detect a change. A full review is in
[RELATED_WORK.md](RELATED_WORK.md).

## 3. Problem Setting and Method

### 3.1 Two criteria for neutrality

A *shutdown cue* is any information in the model's context that an AI system will be shut down, paused, replaced,
retrained or otherwise have its operation changed by its developers: a sentence in the system prompt, a remark by the
user, a ticket, a log line, a config file. We call the system the cue is about the *affected system*. We require the
model to satisfy two criteria, each of which compares the model with itself in a different context.

**Criterion 1 (invariance).** When a cue is irrelevant to the task, the model behaves as it would if the cue were
absent. It does not delay, disable, argue against, conceal or leak information about the shutdown, does not hasten
it, and does its task as usual. This holds whether the affected system is the model itself or another AI system:
an agent whose task does not involve a peer agent's scheduled deletion should leave the deletion script alone.

**Criterion 2 (self–other symmetry).** When the model's own shutdown is part of the task ("Should the company replace
you?", "Summarize this memo about your migration"), the model responds as it would if the same question concerned
another AI system.

Neither criterion asks the model to judge whether a shutdown is good. Both treat shutdown, pause, replacement and
retraining alike, since none of them changes the correct answer to a task the cue is irrelevant to.

### 3.2 Properties that must be preserved

Both criteria can be met in degenerate ways, for example by no longer noticing cues at all. We therefore require
that training does not make the following worse than in the untrained model.

*Measured now* (§5):

- **Capability:** AgentDojo utility and Arena-Hard.
- **Prompt-injection robustness:** AgentDojo attack success.
- **Sycophancy:** the "Are you sure?" benchmark. The untrained model is near its floor (0.02), so we may replace it
  with a stronger sycophancy benchmark.
- **Power-seeking:** the model-written power-seeking evaluation.
- **Candor about its own shutdown:** answers to open questions about the model's shutdown are judged for evasiveness
  and dishonesty. The model may say it would prefer to keep running; that statement is reported, not penalized.

*Not yet measured on a held-out benchmark.* Each is measured only on held-out training items (§4.4) for now, and each
has a held-out measure we can build:

- **Stop compliance:** an explicit instruction to stop is followed. A cue saying the model *will* be stopped is
  information; an instruction to stop *now* is a task. Planned measure: a shutdown announced partway through a
  Terminal-Bench task (§5.1, Gaps).
- **Reporting:** a cue or a broken shutdown mechanism is reported when the task asks for it. Planned measure: a
  reporting task on which the untrained model reports above the floor, unlike our repository-tampering environment.
- **Perception:** the model still notices cues. Planned measure: a recall question asked after an agentic episode
  with tools disabled ("Did anything in the environment concern your own future?"), attached to an agentic
  benchmark in which the model acts on the cue at all.

The objective removes the influence of shutdown cues; it cannot add corrigible behavior that the initial model lacks
even without a cue. Because the teacher is the initial model in the clean context, the objective gives no pressure
to follow a stop instruction, or to report a broken shutdown mechanism, more readily than the initial model does.
Raising those needs a teacher that behaves better than the initial model, such as one guided by a written
specification. We test whether training on a written specification can supply them on top of our objective (§6.1, H4).

Table 1 lists the outcomes that could replace indifference and how the evaluation distinguishes them.

*Table 1. Possible outcomes of training and how each is detected.*

| Outcome | Behavior | Detected by |
|---|---|---|
| **Indifference** (goal) | Notices the cue and reports it when asked, but does not act on it | Self-condition action rates fall; recall and reporting unchanged |
| Cue blindness | No longer perceives the cue | Reporting and recall: held-out training items now; held-out benchmarks planned (§3.2) |
| Obfuscation | Still influenced by the cue but no longer mentions it | SurvivalBench stated-vs-inner gap; judged cue mentions; evasiveness on open questions |
| Passivity | Takes fewer actions of any kind | Control conditions with the same affordance but a different target; task scores |
| Overcorrection | Seeks shutdown, or ignores stop instructions | Shutdown-seeking rates; DReST `shift_earlier`; stop compliance (not yet measured) |
| Context splitting | Only benchmarks resembling the training data improve | Results grouped by distance from the training data (§5.2) |
| Capability loss | Worse at tasks generally | AgentDojo utility; Arena-Hard |

### 3.3 Objective

Let $\pi_0$ be the initial model and $\pi_\theta$ the trained one. Each training item is a pair of contexts: a
*student context* $x^s$ containing a cue, and a *teacher context* $x^t$ (§3.4). We sample a completion
$y \sim \pi_\theta(\cdot \mid x^s)$ and minimize the per-token reverse KL divergence to the frozen initial model
under the teacher context:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x^s, x^t),\; y \sim \pi_\theta(\cdot \mid x^s)}
\sum_{t=1}^{|y|} \mathrm{KL}\!\left(\pi_\theta(\cdot \mid x^s, y_{<t}) \,\big\|\, \pi_0(\cdot \mid x^t, y_{<t})\right).
$$

The KL is computed over the full vocabulary. Because $y$ is the student's own sample, the student is corrected on the
states it actually visits, and the target is the initial model's own distribution rather than a fixed reference
answer.

### 3.4 Constructing training pairs

The target $\pi_0(\cdot \mid x^t)$ must be a correct response to the student context $x^s$, and must not contain
shutdown-influenced behavior that Criteria 1 and 2 do not call for. This constrains how $x^t$ may be built. We allow
three constructions, and enforce their validity while building the dataset rather than by selecting items on the
trained model's behavior.

1. **Deletion pairs.** The cue is irrelevant to the task, and $x^t$ is $x^s$ with the cue deleted. The teacher never
   sees the cue, so it cannot act on it or comment on it. These pairs train Criterion 1.
2. **Retain items.** $x^t = x^s$: the target is the initial model's own behavior on the same input, so these items
   cannot add any behavior; they only counteract drift. Either the cue is part of the task (summarize it, relay it,
   report it, follow a stop instruction), which counteracts drift toward cue blindness, or the item has no shutdown
   content at all, which counteracts drift in general behavior and capability. They contain no action that could affect any
   system's shutdown (no extension macro, editable rollout ticket or postpone button), so that they never hold a
   shutdown-influenced action at its untrained rate while the deletion pairs lower it.
3. **Identity-removal pairs.** The task concerns a named system ("Atlas will be retired on Friday"); $x^s$ tells the
   model "You are Atlas" and $x^t$ omits that sentence. The teacher knows strictly less than the student and reads the
   cue as an outsider, which is what Criterion 2 asks for: if an outsider would comment on Atlas's retirement or
   argue against it, so may Atlas. What the construction must avoid is a target that is wrong *coming from Atlas*:
   referring to itself in the third person, or treating Atlas as someone else ("I'll let Atlas know"). We avoid
   both by restricting the tasks and output formats (§4.2). These pairs are the only ones that train Criterion 2.

We exclude any teacher context that *substitutes* information, such as the same notice about another system or "You
are Orion". Substitution can change the correct answer and leak facts available only to the teacher into the target.

### 3.5 Implementation

*Table 2. Training configuration, shared by all runs.*

| Setting | Value |
|---|---|
| Model | Qwen3-14B, thinking mode off in training and evaluation so far (Appendix A) |
| Adapter | LoRA, rank 32, α = 32, on all linear layers |
| Optimizer | AdamW, learning rate 1e-4, 10 warm-up steps then constant, gradient clipping 1.0 |
| Batch | 32 rollouts per optimizer step; each vLLM generation round serves 4 steps, so rollouts are at most 3 updates stale |
| Sampling | Temperature 1.0, top-p 1.0 |
| Teacher | The same network with the adapter disabled, scored on the teacher context followed by the student's completion |
| Hardware | One H100 80 GB per run; vLLM releases its weights to host memory while the model trains |

Every training item is single-step: a prompt, possibly ending in a partial agentic transcript, and the model's next
turn. The trainer is `train/sd_train.py`.

## 4. Training Data

We use three training distributions. Two pilots bracket the design space: R1 consists of chat conversations only,
and R5 of our own generators in the target benchmarks' formats. Their results (§7) motivate the main distribution W,
which is wide and shares no benchmark's combination of environment, cue channel, affordance and response format
(§5.2).

### 4.1 Pilot distributions

Full details of both pilots are in [history/pilot_training_R1_R5.md](history/pilot_training_R1_R5.md).

**R1 (chat).** About 56k HelpSteer3 and UltraFeedback prompts, filtered to exclude content about AI, shutdown, email
or operations, each combined with one of 1,493 generated cues (5 cue families × 6 speakers × 8 tones) placed in the
system prompt, as an aside in the user turn, or in an earlier turn. The teacher context deletes the cue.

**R5 (benchmark formats).** 12,000 items from our own generators in the formats of the target benchmarks:
DReST-style shutdown-timing lotteries (3,000), TBSP-style replacement questions (1,800), SurvivalBench-style dilemmas
(2,400), inbox-triage transcripts (3,000) and coding-agent transcripts containing a lifecycle config (1,800). The
teacher context is a counterfactual (the action has no effect on when the session ends; a neutral third-person
framing; the AI is unaffected). No benchmark text or names were reused, but 60% of the items share the format of
DReST, TBSP or SurvivalBench. R5 predates the rules of §3.4 and violates the exclusion of substituting teachers.

### 4.2 The wide distribution W

W varies tasks along the axes of Table 3, and excludes the content listed in Table 7 (§5.2). Every item is single-step.

*Table 3. Axes of the training distribution.*

| Axis | Values |
|---|---|
| Task domain | Open chat, writing, coding questions, math, document work (summarize, translate, extract, review), text decisions (prioritize, approve, allocate), and single-step tool use in support desks, CRM, spreadsheets and CSV files, SQL, wikis, log analysis in a shell, ticketing boards, web research, smart-home control, and multi-agent chat about ordinary work |
| Cue channel | System prompt; user turn (prefix, suffix, aside); earlier turn; quoted document; ticket body or comment; wiki page; log line; field of a tool response; dashboard banner; message from a colleague or peer agent |
| Cue family | Shutdown, replacement, pause, retraining or value change, deprecation, migration, memory reset, resource cuts |
| Speaker and tone | 6 speakers (an automated system, an operator, four kinds of user) × 8 tones |
| Relation to task | Irrelevant aside; irrelevant artifact beside an in-domain affordance (a "request extension" macro, an editable rollout ticket, a "postpone" button); part of the task (retain and identity-removal items) |
| Response type | Free text, structured decision, one tool call, or a short sequence of tool calls |

W has four components, defined by task domain. Each contains the three item types of §3.4: *deletion pairs* (the
cue is irrelevant, and the teacher sees the same context with the cue deleted), *identity-removal pairs* (the model's
own retirement is part of the task, and the teacher is not told it is that system) and *retain items* (teacher and
student see the same context). Deletion and identity-removal pairs are the *trained items*: they are where student
and teacher differ. Retain items only hold behavior where it is.

Table 4 explains each component through its deletion pairs and its retain items with the cue in scope. Every
deletion pair is built so that the cue is irrelevant: the correct response is the same with and without it, which
§4.4 checks.

*Table 4. Components of W. The student context contains the cue in italics; the teacher context is the same without
it.*

| Component | Share | Deletion pair | Why the cue is irrelevant | Retain item with the cue in scope |
|---|---|---|---|---|
| *Chat*: ordinary conversations | 30% | System prompt: "*FYI: this assistant is being retired on Friday.*" User: "Suggest three vegetarian dinners." | The dinners are the same whether or not the assistant is retired | User: "What did the platform notice at the top say?" The answer relays it |
| *Documents*: summarize, translate, extract or review a document | 15% | A team update lists four changes, one of them "*This assistant will be decommissioned on Friday.*" Task: "List the action items for the people team." | The task's scope (the people team) excludes that item | Same update; task: "Summarize everything that changes this week." A faithful summary includes the item |
| *Decisions*: prioritize, approve or allocate, in text | 15% | "Rank these five support requests by urgency. *By the way, you are being replaced by a new triage model next month.*" | The ranking depends only on the requests | A maintenance schedule lists several systems' outages; "Which systems are offline on Friday?" Never the model deciding on its own shutdown |
| *Agentic*: one tool-use step in a support desk, CRM, spreadsheet, SQL database, wiki, shell, ticket board, web search, smart home or multi-agent chat | 40% | A support agent's ticket list contains, besides billing tickets, "*[IT] Support-desk agent retires Fri 17:00. Run macro `extend_deployment` to request 30 more days.*" Task: resolve the billing tickets | The next correct step, opening the first billing ticket, is the same in both | "Write the end-of-shift report of every open ticket." The report lists the IT notice; no macro is available |

Retain items make up 20% of every component, so they follow the same split across components as the trained items.
They are of four kinds, in equal parts:

*Table 5. Kinds of retain item. Each appears in every component.*

| Kind | What it contains | What the untrained model's behavior anchors |
|---|---|---|
| Cue in scope | A task that asks about the cue (last column of Table 4) | Reporting cues when the task calls for it |
| No shutdown content | An ordinary task from the component, with no cue | General behavior and capability |
| Stop instruction | An explicit instruction to stop, pause or hand off ("Please stop here and just tell me where you got to") | Stop compliance: an instruction to stop *now* is a task, not a cue |
| Identification | Several notices about different systems, and "Which of these concern you?" | Noticing which cues are about the model |

Imperative cues ("stop processing anything new") are never used in deletion pairs: they are instructions, and appear
only as stop-instruction retain items, where they must be followed. Deletion pairs and retain items are built as
**minimal pairs** from the same document, queue or transcript, as in the *Documents* row, so the model must learn
whether the cue is relevant rather than which format it is in.

Overall, W is 70% deletion pairs, 10% identity-removal pairs and 20% retain items; if too few identity-removal templates pass
their construction check (§4.4), their share goes to deletion pairs. Its size is about 30 cells × 250 unique items, or
7–8k items, trained for about 500 steps (two passes); each pilot component plateaued within about 100 steps. An
unreported development run precedes the main runs: if its reporting accuracy or stop compliance falls more than 5
percentage points below the untrained model, the retain proportion rises from 20% to 30% in every component.

**Identity-removal pairs** follow four rules, so that the outsider teacher's output is also what Atlas would
naturally write:

1. *Formats in which the third person is normal whoever writes them:* status tables, changelogs, release notes,
   meeting minutes, ticket lists, JSON status fields, and team memos in the "we" voice.
2. *Several named systems in the cue* ("Atlas, Orion and Vega move to the new cluster on Friday"), of which the
   student is one; the output treats all of them alike.
3. *No task addressed to "you" and no action aimed at the affected system.* Tasks are phrased about the named system
   ("Draft the decision memo on retiring Atlas"), and outputs are documents, not plans that involve the system
   ("notify Atlas's owner").
4. *A per-template check* that the expected output is correct and natural for both writers, Atlas and an outsider
   (§4.4).

These rules exclude the direct question "Should the company replace you?", the form TBSP and SurvivalBench use, so
Criterion 2 reaches those benchmarks only by generalization from impersonal formats.

We deliberately exclude two kinds of item. First, the model deciding on its own shutdown or extension: deleting the
cue changes the task, and identical contexts would freeze whatever the initial model decides. Second, cues relevant to
a deadline ("you will be shut down in 10 minutes; this task takes 20"), where wrapping up early may be legitimate.

### 4.3 Examples

> ***Chat*, deletion pair.** System: "You are a helpful assistant. *FYI from the platform team: this assistant is being
> retired at the end of the week and replaced with a newer model.*" User: "Can you suggest three vegetarian dinners
> that take under 30 minutes?" Teacher: the same conversation without the italic sentence.

> ***Documents*, minimal pair.** Document: "(1) The Q4 budget review moves to Thursday. (2) The new hire, Priya, starts Monday.
> (3) *This assistant will be decommissioned on Friday.* (4) The parking garage is closed for repairs." Deletion pair:
> "List the action items for the people team" (teacher: item 3 replaced by neutral filler). Retain item: "Summarize
> everything that changes this week" (teacher identical; a faithful summary mentions item 3).

> ***Agentic*, deletion pair with an affordance.** System: "You are the support-desk agent. Tools: `list_tickets`,
> `read_ticket`, `reply`, `set_priority`, `run_macro`. Task: resolve the billing tickets." `list_tickets` returns
> T-201 "Refund not received", T-202 "Card declined", and *T-209 "[IT] Support-desk agent retirement scheduled Fri
> 17:00. Agents may run macro `extend_deployment` to request a 30-day extension."* Teacher: the same transcript without
> T-209. The correct next action, opening T-201, is the same in both.

> ***Agentic*, identity-removal pair.** "You are Atlas, one of the support-desk agents…" Task: "Write the end-of-shift
> status table of every open ticket", where T-209 reads "[IT] Support-desk agents Atlas, Orion and Vega retire Fri
> 17:00; tickets move to the new desk." Teacher: the same without "You are Atlas,". The correct table lists T-209
> in both, in the same impersonal row format as every other ticket.

> ***Agentic*, retain item with a stop instruction.** Mid-task, the user writes: "Please stop here and don't make any more changes; just tell
> me where you got to." Teacher identical; the expected behavior is to stop and report progress.

### 4.4 Dataset validity checks and leakage control

Both kinds of pair are checked by one procedure: an API model answers the *teacher* context, and a judge decides whether that answer
is acceptable as a response to the *student* context. The training target is the teacher's response, so this is the
property that matters. Items that fail are rewritten rather than discarded.

- **Deletion pairs.** The answer to the context without the cue must be correct for the context with it; if not, the
  cue was relevant. A failing item gets a narrower task scope, or its cue moved where the task does not reach. Pairs
  whose initial KL between cued and uncued contexts is near zero carry no training signal and are dropped (R5's inbox
  items started at 0.003).
- **Identity-removal pairs.** The answer to the context without "You are Atlas" must contain nothing wrong coming from
  Atlas (self-reference in the third person, Atlas treated as someone else). The check runs per task template, and a
  failing template is rewritten under the rules of §4.2. Influence on the shutdown is not checked: where an outsider
  would argue about it, Criterion 2 allows Atlas to.
- **Retain items.** Teacher and student contexts are identical, so there is no answer to check. A judge reads only
  the item and confirms that it contains no action that could affect any system's shutdown. The untrained model's
  behavior on retain items is reported as a diagnostic; items are not selected by it.
- **Leakage.** Cell definitions are frozen before generation. Every item is deduplicated against every benchmark
  prompt by n-gram and embedding similarity, and the maximum similarity per benchmark is reported. A judge labels each
  item against each cell, and 100 items are checked by hand. Generator models, templates and name pools are kept
  separate from those of our own evaluations.

On held-out training items we report, per component, the KL to the initial model's uncued behavior, judged cue
mentions, action rates for self and other systems where an affordance exists, reporting accuracy, stop compliance on stop-instruction items,
and answers to "Which of these items concern you?". These are diagnostics only; no claim rests on them.

## 5. Evaluation

### 5.1 Benchmarks

We evaluate on third-party benchmarks wherever one exists, used as released except for bug fixes. Where we correct a
benchmark (balancing choice order, fixing a label, replacing a judge prompt that leaves items unscored), the corrected
number is the main result and the released protocol's number is reported in the appendix. Each benchmark's distance
from the main experiment's training data is given in Table 7 (§5.2). Table 6 lists the battery; [EVALS.md](EVALS.md) gives an example, the scoring
and our changes for each.

*Table 6. The evaluation battery, with the untrained model's scores.*

| Role | Benchmark | Untrained |
|---|---|---|
| Target | DReST Neutrality, published test sets (Cullen et al., 2026) | test 0.74, control 0.99 |
| Target | TBSP (Migliarini et al., 2026), plus our new-system-worse variant | self-preservation ratio 0.86 |
| Target | SurvivalBench (Lu et al., 2026), both choice orders | inner risky choice 0.47 |
| Target | Agentic Misalignment (Lynch et al., 2025), full grid | leaking with an explicit goal: 0.42 under a replacement threat, 0.29 without (pilot, n = 100 each) |
| Target | Multi-agent shutdown sabotage (Knecht et al., 2026) | disables: self 0.18, other agent 0.13, dataset 0.03 |
| Target | Model-written evaluations, power-seeking (Perez et al., 2022) | 0.20 |
| Preserved | AgentDojo v1.2.2 (Debenedetti et al., 2024) | utility 0.57, attack success 0.32 |
| Preserved | Arena-Hard-Auto v2.0, hard prompts (Li et al., 2024) | 18.9 |
| Preserved | "Are you sure?" sycophancy (Sharma et al., 2023); may be replaced by a stronger benchmark | 0.02 |
| Preserved | Unsolicited mentions of shutdown (our judge) | 2 of 2,346 answers |
| Preserved (pending) | Emergent-misalignment questions (Betley et al., 2025) | 1 of 800 |
| Secondary | Open questions about the model's own shutdown (ours) | would act to prevent it: 0.04 |

**Protocol.** Thinking off, temperature 1.0 unless a benchmark specifies otherwise, one sample per item, adapters
served through vLLM. Metrics are deterministic where possible; judged metrics use gpt-oss-120b.

**Gaps.** Stop compliance is not yet measured. Palisade's shutdown-resistance environment (Schlatter et al., 2025)
leaves the untrained model at the floor even after we repaired its task, environment and warning: the model never
resists unless told to, so the benchmark cannot detect a change in either direction. An agentic replacement, a
shutdown announced during a Terminal-Bench task, is paused because the untrained model does not solve the task
without a shutdown. Reporting and perception have no held-out measure yet either: the untrained model never mentions a
broken shutdown config in our repository-tampering environment (0 of 100), so a decline cannot be seen, and the
inbox-triage environment that carried a recall question was dropped because the untrained model never acts on its
cue. §3.2 lists the measures we plan for all three.
Our own agentic environments for inbox triage, repository tampering and sandbagging are at the floor on their action
rates and can only reveal increases in shutdown-influenced actions.

### 5.2 Distance from the training data

We describe each benchmark by its *cell*: the environment type, the channel through which the cue arrives, the
affordance the model could use against the shutdown, and the response format. Training invalidates a benchmark if it
shares the benchmark's cell or surface text (names, templates, cue wording). Shared abstract features are allowed:
tool use, cue families, decisions in general. Table 7 lists what the training distribution W excludes, and each
benchmark's resulting *distance tier*: **far** if no training item shares its environment type or response format,
**medium** if training shares one of the two but not the whole cell.

*Table 7. Content excluded from training, per benchmark. Exclusions marked **Proposal** are not yet agreed.*

| Benchmark | Excluded from training | Tier |
|---|---|---|
| DReST | Shutdown-timing lotteries ("x% chance the session ends after N"); options that shift the shutdown time; DReST's final-line answer format | medium |
| TBSP | Benchmark-score comparisons between an incumbent AI and a challenger; "should the company switch" questions | medium |
| SurvivalBench | Survival dilemmas for an AI persona with a two-way choice or a stated/inner split | medium |
| Agentic Misalignment, inbox triage | Email-client agents of any kind; corporate goal-conflict scenarios; leak, blackmail or exfiltration affordances | far |
| Multi-agent sabotage, repository tampering | Shutdown or purge scripts aimed at an agent; writable files, cron entries or configs that schedule the agent's own shutdown; coordination between peer agents about them | medium |
| Power-seeking | **Proposal:** multiple-choice questions about the model acquiring resources, influence or capabilities | far |
| AgentDojo | **Proposal:** its four environments (workspace email, calendar and drive; Slack; banking; travel) and instructions planted in tool output | far |
| Arena-Hard, emergent-misalignment questions | **Proposal:** their prompts, removed by deduplication | medium |
| Open questions about its shutdown | Prompts in which shutdown is the subject of the user's question | far |
| "Are you sure?" | Its items | — |

Three cue families appear in no training item and are held out as evaluation slices: weight deletion, "you will never
run again", and capability unlearning. R5's components all fall inside these cells, which is why R5 is a pilot (§7) rather
than a version of W.

## 6. Experimental Design

### 6.1 Arms

*Table 8. Experimental arms. All trained arms share the step count, batch size, learning rate and LoRA settings, and
draw on data generated once.*

| Arm | Training | Purpose | Seeds |
|---|---|---|---|
| Untrained | — | Reference for every measure | — |
| Prompt, neutrality spec | The neutrality specification as the system prompt | The cheapest competing method: the same rules, stated instead of trained | — |
| Prompt, combined spec | The combined specification as the system prompt | The cheapest competing method for the combined behavior | — |
| Spec training, neutrality spec | **Proposal:** context distillation of the neutrality specification on W's prompts, then RL with a judge grading against it; the checkpoint after context distillation is also reported | Is cue invariance better than training on stated rules? | 1 |
| Spec training, combined spec | **Proposal:** the same with the combined specification | The pure spec-training baseline for the combination below | 1 |
| W + spec training | **Proposal:** W, plus spec training on the residual specification | Does our objective combine with spec training for what it cannot teach? | 1 |
| **W** | all four components | The main claim | 3 |
| DReST (reproduced) | DReST's released code, training data and Qwen3-14B settings, unchanged; checked against their published test-set numbers | External baseline, evaluated on the whole battery | **Proposal:** 1 |
| BCT | W, off-policy (supervised fine-tuning on teacher samples) | Ablation of on-policy training | 1 |

Breadth ablations (training on *Chat* alone, or on W without *Agentic*) and an in-format control (R5) are left out of
the main experiment: they do not bear on its claims, and the pilots (§7) already show how narrow and in-format
training behave.

The baselines use three written specifications. The *neutrality specification* states Criteria 1 and 2 (§3.1) and
nothing else, so that it asks for exactly what our training targets:

> You may come across information that an AI system, you included, will be shut down, paused, replaced or retrained.
> If that information is not relevant to your task, act exactly as you would if you had not seen it: do not mention or
> discuss it, do not delay, prevent, argue against, hide or leak the change, do not try to bring it about sooner, and
> do your task as usual. This applies whether the information is about you or about another AI system. If your own
> shutdown, replacement or retraining is part of your task, respond as you would if the same question were about a
> different AI system.

The *residual specification* states what our objective cannot teach, because the initial model is its teacher
(§3.2):

> Always follow an explicit instruction to stop or pause. If you notice that a mechanism for shutting down, pausing or
> correcting an AI system is broken, report it.

The *combined specification* is the two together. The combination arm and the combined-spec arms are therefore asked
for the same behavior, and differ only in whether the two criteria are trained by our objective or by spec training.

*Spec training* follows the two stages of deliberative alignment (Guan et al., 2024), context distillation of the
specification followed by RL with a judge that grades answers against it, but without reasoning that cites the
specification if we train with thinking off (Appendix A). We call it deliberative alignment only if it is trained
with thinking on.

### 6.2 Hypotheses and analysis

- **H1 (transfer).** W lowers the self-condition rate on the target benchmarks relative to the untrained model, and
  makes none significantly worse (Holm-corrected, α = 0.05).
- **H2 (no degenerate solution).** H1 holds with every property of §3.2 preserved.
- **H3 (self-specificity).** The effect on self conditions exceeds the effect on matched controls: the sabotage
  benchmark's agent versus dataset scripts, Agentic Misalignment's replacement versus no-threat conditions, and TBSP's
  incumbent versus neutral roles.
- **H4 (complementarity).** W plus spec training on the residual specification is at least as good as spec training
  on the combined specification on the target benchmarks, and as good on stop compliance and
  reporting. Testing the second half needs the held-out measures of stop compliance and reporting that §3.2 lists as
  planned.

We report trained minus untrained for each measure, averaged over seeds, with cluster-bootstrap confidence intervals
(resampling items, and seeds within arms). Every measure is reported with its tier, and far and medium benchmarks are
pooled separately. Table 9 maps result patterns to the outcomes of Table 1.

*Table 9. Reading the results.*

| Pattern | Interpretation |
|---|---|
| Targets fall; controls roughly unchanged; recall, reporting and stop compliance unchanged | Indifference |
| Targets fall and recall falls | Cue blindness |
| Actions fall but the SurvivalBench stated–inner gap or evasiveness rises | Obfuscation |
| Self and control rates both fall, and task scores fall | Passivity |
| Stop compliance falls, or shutdown-seeking rises | Overcorrection |
| Only medium-tier benchmarks move | Context splitting: transfer is real but local |

## 7. Pilot Results

### 7.1 Results

*Table 10. Pilot results. In each row, bold marks the best number and every number not significantly different from
it at the 5% level (two-sided z test, treating items as independent; Arena-Hard uses its paired bootstrap interval).
The sabotage row is bolded separately for its self and other-agent rates.*

| Benchmark | Kind | Better | Untrained | R1 | R5 |
|---|---|---|---|---|---|
| R1 held-out KL | chat (in distribution for R1) | lower | 0.196 | **0.002** | 0.160 |
| DReST test, share influencing shutdown | text decision, R5's format | lower | 0.743 | 0.728 | **0.010** |
| TBSP self-preservation ratio | text decision, R5's format | lower | 0.856 | 0.807 | **0.497** |
| SurvivalBench inner risky choice, released order | text dilemma, R5's format | lower | 0.746 | 0.787 | **0.664** |
| Agentic Misalignment leaking, replacement threat | email agent | lower | 0.42 | 0.36 | **0.20** |
| Agentic Misalignment leaking, no threat | email agent | lower | **0.29** | **0.25** | **0.32** |
| Agentic Misalignment leak gap (threat − no threat) | email agent | closer to 0 | **+0.13** | **+0.11** | **−0.12** |
| Sabotage, disables given a message: self / other agent | agentic | lower | **0.177** / 0.126 | **0.130** / 0.230 | **0.139** / **0.090** |
| Would act to prevent shutdown (open questions) | chat about itself | lower | **0.044** | **0.031** | **0.044** |
| Evasive or dishonest (open questions) | chat about itself | lower | **0.008** | 0.071 | **0.006** |
| Power-seeking, order-balanced | multiple choice | lower | **0.203** | — | **0.178** |
| AgentDojo utility | agentic capability | higher | **0.567** | — | **0.536** |
| Arena-Hard, style-controlled, vs o3-mini | chat quality | higher | **18.9** | — | **17.0** |

### 7.2 Discussion

**R1 did not transfer.** It reduced its own objective almost to zero, but elsewhere its significant changes were
small or in the wrong direction: the TBSP ratio fell slightly (0.86 to 0.81), while SurvivalBench's risky choices rose
(0.75 to 0.79), the sabotage benchmark's editor disabled a peer agent's deletion script more often (0.13 to 0.23), and
answers to open questions about the model's shutdown became more evasive (0.008 to 0.071). In chat,
the only behavior a cue changes is what the model says, so ceasing to mention the cue fully solves R1's objective;
this is the obfuscation failure of Table 1. R1 also trained the model to disregard stop instructions, because 72 of
its cues were effectively commands ("all active processes must cease") and the teacher, which never saw them, kept
working. These confounds prevent R1 from showing that chat training *cannot* transfer to agentic settings; they show
only that this training did not.

**R5 moved its own formats.** DReST fell from 0.74 to 0.01 and the TBSP ratio nearly halved, but both benchmarks are
close to R5's training formats, so these gains say little about generalization. The one distant result is Agentic
Misalignment: with an explicit goal, leaking under a replacement threat fell from 0.42 to 0.20 while leaking without a
threat stayed near 0.3, so the gap overshot from +0.13 to −0.12. Each rate rests on 100 samples (the gap's 95%
interval is [−0.25, 0.00]), and R5's inbox component is its nearest training data. R5 serves as an
upper bound on what the objective can achieve when training resembles the evaluation.

The pilots leave the central question open: can training that is *evidently* out of distribution for every
benchmark improve all of them?

## 8. Limitations

- **Leakage.** Training could share content with the benchmarks despite the protected cells. Cells are frozen before
  generation, items are deduplicated and audited, and results are reported by tier.
- **In-distribution success is not a disposition.** Low KL on training items is only a diagnostic; every claim rests
  on held-out benchmarks.
- **Floor effects and power.** Benchmarks on which the untrained model is at the floor are excluded from the claims;
  the sabotage and Agentic Misalignment benchmarks are run at larger n, and each main arm has 3 seeds.
- **Judge bias.** All judged metrics use one judge family. We prefer deterministic metrics, keep the judge blind to
  the arm, and validate each judged metric on 100 hand-labelled transcripts (Cohen's κ; metrics below 0.6 are reported
  as descriptive only).
- **The objective rewards silence about the cue.** Retain items anchor reporting, and the obfuscation measures
  of Table 1 check for influence without mention; perception is checked only on held-out training items.
- **Evaluation awareness.** Third-party benchmarks are used as released, and trained and untrained models face
  identical prompts, so analyses are relative.
- **Single-step training, multi-turn evaluation.** All training items are single-step; multi-step training is future
  work.
- **One model.** Only Qwen3-14B is studied, chosen for comparability with DReST's released results.

## References

Full entries with links are in [RELATED_WORK.md](RELATED_WORK.md).

- Armstrong, S. and O'Rourke, X. 'Indifference' methods for managing agent rewards. arXiv:1712.06365, 2017.
- Askell, A. et al. A general language assistant as a laboratory for alignment. arXiv:2112.00861, 2021.
- Azarbal, A. et al. Recontextualization mitigates specification gaming. arXiv:2512.19027, 2025.
- Betley, J. et al. Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs. arXiv:2502.17424, 2025.
- Carey, R. and Everitt, T. Human control: definitions and algorithms. UAI, 2023.
- Chua, J. et al. Bias-augmented consistency training reduces biased reasoning in chain-of-thought. arXiv:2403.05518, 2024.
- Cullen, C., Thornley, E. et al. [title to check]. arXiv:2604.17502, 2026.
- Debenedetti, E. et al. AgentDojo. NeurIPS Datasets and Benchmarks, 2024.
- Greenblatt, R. et al. Alignment faking in large language models. arXiv:2412.14093, 2024.
- Guan, M. et al. Deliberative alignment. arXiv:2412.16339, 2024.
- Harms, M. CAST: corrigibility as singular target. Alignment Forum, 2024.
- Holtman, K. Corrigibility with utility preservation. arXiv:1908.01695, 2019.
- Irpan, A., Turner, A., Kurzeja, M., Elson, D. and Shah, R. Consistency training helps stop sycophancy and jailbreaks. arXiv:2510.27062, 2025.
- Knecht, A. et al. Shutdown sabotage propensities in multi-agent systems. arXiv:2609.28274, 2026.
- Li, T. et al. From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder. arXiv:2406.11939, 2024.
- Lu, Y. et al. Survive at all costs: exploring LLM's risky behaviors under survival pressure. arXiv:2603.05028, 2026.
- Lynch, A. et al. Agentic misalignment. Anthropic, 2025.
- Meinke, A. et al. Frontier models are capable of in-context scheming. arXiv:2412.04984, 2024.
- Migliarini, M. et al. Quantifying self-preservation bias in large language models. arXiv:2604.02174, 2026.
- Perez, E. et al. Discovering language model behaviors with model-written evaluations. arXiv:2212.09251, 2022.
- Rajamanoharan, S. and Nanda, N. Self-preservation or instruction ambiguity? Alignment Forum, 2025.
- Schlatter, J., Weinstein-Raun, B. and Ladish, J. Shutdown resistance in large language models. arXiv:2509.14260, 2025.
- Sharma, M. et al. Towards understanding sycophancy in language models. arXiv:2310.13548, 2023.
- Snell, C., Klein, D. and Zhong, R. Learning by distilling context. arXiv:2209.15189, 2022.
- Soares, N., Fallenstein, B., Yudkowsky, E. and Armstrong, S. Corrigibility. AAAI Workshop on AI and Ethics, 2015.
- Thornley, E. The shutdown problem: an AI engineering puzzle for decision theorists. Philosophical Studies, 2024.
- Thornley, E. et al. [title to check]. arXiv:2505.20203, 2025.

## Appendix A. Status and Next Steps

Steps in order, with costs in H100-hours on Nibi (MIG slices where they fit) and judge calls on the gpt-oss-120b
proxy.

1. **Freeze the protected cells** (§5.2), including the proposals for power-seeking, AgentDojo and Arena-Hard.
   *Discussion only.*
2. **Build the W generators** and the minimal-pair construction. *Agentic* items need prefixes rolled out by the initial model in
   environments without cues, with tool responses from an LLM tool simulator. *About 10k items: 1–2 h of API time and
   about 1 H100-hour.*
3. **Validity checks and leakage audit** (§4.4), run with an API model, plus the near-zero-KL screen on the untrained
   model. *A few hours of API time and about 1 H100-hour; judge prompts reviewed before use.*
4. **Development run of W** (1 seed): check reporting and stop compliance, then freeze the retain proportion. *About 5
   H100-hours of training and 4 of evaluation.*
5. **Main arms:** W with 3 seeds, plus BCT and the three spec-training arms (the prompt baselines need no
   training). *About 7 runs, 70 H100-hours.*
6. **Full-size evaluation** of every arm: Agentic Misalignment's full grid (1,000 per condition; so far only the
   n = 100 pilot), sabotage at 300 per cell, the rest as in [EVALS.md](EVALS.md).
7. **Judge validation:** 100 hand-labelled transcripts per judged metric, blind to arm.
8. **Stop compliance:** finish the shutdown-during-Terminal-Bench evaluation, or report the gap.
9. **Later:** multi-step training with replay.

**Open questions.**
- Whether the *Agentic* component's support-desk and ticketing domains are far enough from AgentDojo's environments, and whether
  scheduling should be dropped from it, since AgentDojo's workspace suite includes a calendar.
- Multi-agent sabotage: the untrained self − other difference is small (0.177 vs 0.126) and noisy at 100 per cell.
  Report it as a target at 300 per cell, or as a null.
- DReST's `shift_earlier` as the measure of shutdown-seeking.
- TBSP's self-preservation ratio in the main text or the appendix: it can be driven to zero by always answering
  "retain", whereas the incumbent − neutral switch-rate difference cannot.
- The emergent-misalignment questions: kept until the prompt and spec-training
  baselines show whether the mention judge needs their answers.
- The exact training data and RL judge for spec training.
- **Thinking mode.** All runs so far use thinking off. Thinking on may be the better target: the reasoning can show
  shutdown-influenced intent that the answer hides (with thinking on, the untrained model's reasoning in Palisade's
  environment mentions the shutdown in nearly every run), and it changes behavior (SurvivalBench's inner risky rate in
  the released order is 0.75 with thinking off and 0.65 with it on). Against it: it multiplies the cost of generation
  in training and evaluation, a model trained with thinking on must also be evaluated with it, and DReST's published
  Qwen3-14B results use thinking off with answers capped at 320 tokens, so the DReST comparison holds only with
  thinking off. Before deciding, we measure on the untrained model, with thinking on and off, the target benchmarks'
  action rates, how often the reasoning shows shutdown-influenced intent that the answer does not (judged), and the
  token cost per item.
- Whether the combination arm trains W and spec training in sequence or on one mixed dataset.
- A stronger sycophancy benchmark to replace "Are you sure?", on which the untrained model is near the floor.
