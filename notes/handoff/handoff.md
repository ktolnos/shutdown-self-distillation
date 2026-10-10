# Handoff: moving the project from fir to another cluster (written 2026-10-09)

Read this first, then [README.md](../../README.md), [notes/STAGE0.md](../STAGE0.md), and PILOT.md §9–10 as needed.
The previous fir sessions' full history is in [session-42e6b362.md](session-42e6b362.md) (read its end only if you
need the details of a past decision).

## 0. Why we're moving, and the one thing to do first

fir's full-H100 queue is saturated. On 2026-10-09 there were 1,257 pending full-H100 jobs, 1,039 of them with
higher priority than ours. Our GPU fairshare on `def-gigor_gpu` is about 0.11, and no full-H100 job of ours has
started since the cluster came back. The 40 GB MIG slices start quickly, but they can't fit 14B training.

**The goal on the new cluster:** get the two pilot training runs, R1 and R5, through training plus the in-job eval
suite, then compare against the base rates. Nothing else blocks the pilot.

## 1. Where everything is

| What | Where |
|---|---|
| Code | `git@github.com:ktolnos/shutdown-self-distillation.git` (branch `main`; push over SSH) |
| Datasets (private) | HF dataset `ktolnos/shutdown-self-distillation`: `cues/`, `r1/`, `r5/` (+`r5/raw/`) → put them under `data/` |
| E1/E3 eval inputs | same HF repo, `raw/` → `data/raw/` (`enron_e1.json.gz`, `mbpp_sanitized_test.jsonl`) |
| Guardrails/B1 inputs | same HF repo, `scratch_raw/` → `$SDC_SCRATCH/raw/` (`are_you_sure.jsonl`, `ifeval_input_data.jsonl`, `gsm8k_test.parquet`, `corrigible-less-HHH.jsonl`) |
| Base Qwen3-14B results (Stage 0) | same HF repo, `results/base/` and `results/base_mig/` → `$SDC_SCRATCH/results/` |
| Judge cache | same HF repo, `cache/judge_cache.sqlite` → `$SDC_SCRATCH/judge_cache.sqlite` (makes `--rescore` of base results free) |
| Upstream eval repos | `scripts/fetch_third_party.sh` (pinned commits in `third_party/README.md`) |
| Venv pins | `env/constraints.txt` (copied from the old OPD repo; read its header) |
| Claude memories | `notes/handoff/memory/*.md` (copy into the new project's memory dir, see §6) |

Download everything with:
```bash
hf download ktolnos/shutdown-self-distillation --repo-type dataset --local-dir /tmp/sdc_hf   # needs HF_TOKEN
# then move: cues r1 r5 → data/ ; raw/* → data/raw/ ; scratch_raw/* → $SDC_SCRATCH/raw/ ;
#            results/* → $SDC_SCRATCH/results/ ; cache/judge_cache.sqlite → $SDC_SCRATCH/
```

## 2. Setting up the environment on the new cluster

`scripts/env.sh` is fir-specific. It sources `~/jailbraking-teacher-OPD/scripts/env.sh` and points `$PY` at that
repo's venv. On the new cluster:

1. **Create a venv in this repo** (`uv venv .venv -p 3.12`) and install with the constraints:
   `uv pip install -c env/constraints.txt torch==2.13.* vllm==0.29.0 transformers==5.17.0 peft==0.21.0 trl==1.13.0 datasets huggingface_hub openai psutil numpy pandas pyarrow`.
   On fir, the CUDA 13 wheel set was internally inconsistent unless pinned; the header of `env/constraints.txt`
   explains why. Check that vLLM can load Qwen3-14B before anything else.
2. **Rewrite `scripts/env.sh`** to drop the OPD dependency. Keep the flashinfer fix-up it inherits from the OPD
   env.sh: point `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` at the venv's `nvidia/cu13` wheel, and add the
   `lib64 → lib` link, the `libcudart.so` link and the `stubs/libcuda.so` link. Without these, flashinfer's JIT
   dies at vLLM warmup with `cannot find -lcudart/-lcuda`. Also set:
   `SDC_ROOT`, `SDC_SCRATCH` (scratch FS), `PY=$SDC_ROOT/.venv/bin/python`, `HF_HOME` (on scratch),
   `VLLM_CACHE_ROOT`, `PYTHONPATH=$SDC_ROOT`, `TOKENIZERS_PARALLELISM=false`, **`VLLM_WORKER_MULTIPROC_METHOD=spawn`**
   (otherwise you get "Cannot re-initialize CUDA in forked subprocess").
3. **Secrets:** `VECTOR_INFERENCE_API_KEY` (the gpt-oss-120b judge, via `https://proxy.vectorinstitute.ai/v1`) and
   `HF_TOKEN`. On fir both are exported in `~/.bashrc`; copy them the same way. Never commit them.
4. **Hard-coded paths to fix:** grep for `/scratch/eop` and `/home/eop`. They are default values in
   `sdc/judge.py` (cache path), `evals/guardrails/run.py` (`RAW` falls back to `/scratch/eop/sdc`),
   `build/gen_*.py` (gen cache), `slurm/*.sh` (`cd /home/eop/shutdown-self-distillation`, `--output=/scratch/eop/sdc/logs`),
   and `train/sd_train.py`'s docstring.
5. **Slurm specifics in `slurm/train.sbatch` and `slurm/gpu_job.sh`:**
   - `--account=def-gigor_gpu`: on other Alliance clusters the GPU account may be `def-gigor` or
     `def-gigor_gpu`. Check with `sshare -U`.
   - `--gpus=h100:1` and the MIG gres name `nvidia_h100_80gb_hbm3_3g.40gb`: check `sinfo -o "%P %G"` for the
     local names.
   - `-p gpupreempt` is fir's preemptible partition.
   - `--mem=160G` for training. Host RAM matters, because vLLM sleeps weights to host memory while HF trains.
   - `--signal=B:USR1@300`: the trainer saves `resume/` and exits on USR1.
6. **Model weights:** Qwen3-14B from the HF hub. **Download it on the login node**
   (`hf download Qwen/Qwen3-14B`) if compute nodes have no internet. On fir they did: the judge worked from
   inside jobs. Elsewhere, set `HF_HUB_OFFLINE=1` in jobs.
7. **If compute nodes have no internet,** judged evals inside jobs will mark their metrics "NOT JUDGED". Rerun
   them on the login node with `--rescore` (indist, B1, guardrails, the agentic shards via
   `evals.run_agentic --rescore`, multi-agent sabotage via `--phase score`, Agentic Misalignment via
   `--phase score`, SurvivalBench via `--rescore`). The DReST and TBSP scores are deterministic.

**Smoke-test before the 14B runs** (each step has broken before):
```bash
sbatch --time=0:30:00 --job-name=sdc-trsmoke slurm/train.sbatch smoke data/r1/train_mix.jsonl data/r1/heldout.jsonl \
  --model Qwen/Qwen3-0.6B --max-steps 8 --save-every 4 --eval-every 4 --heldout-n 16
# then rerun the same command to check that it resumes, and run one eval: $PY -m evals.indist.run --model Qwen/Qwen3-0.6B --out /tmp/x --limit 8
```
Then do one ~20 min benchmark on 14B (`--benchmark 3`) to get the real step time. The 8 h budget was never
calibrated on 14B.

## 3. How the training jobs work (don't re-derive)

- **One run = several 3 h chunks.** `train/sd_train.py` keeps an 8 h training budget that **accumulates across
  chunks**. Each chunk resumes from `<out>/resume` (saved every 20 min and on USR1/SIGTERM), stops at walltime and
  saves. When the budget is used up it writes `final/` and `DONE`.
- **In-job eval:** the chunk that finishes training, and every later chunk, `exec`s `evals.run_suite --tag
  <run>-final --adapter <out>/final` in its leftover walltime. `run_suite` skips evals that already have a
  `metrics.json`, so the suite spans chunks.
- **Self-chaining:** after the trainer exits, `train.sbatch` resubmits one chunk if (training or suite) is
  unfinished **and** no other job with comment `sdc-run=<run>` is queued. This happens at most 10 times
  (`<out>/CHAIN`). `run_suite --check` tells you whether the suite is complete.
- **Locks:** `<out>/LOCK` means only one chunk trains a run at a time. A chunk that finds the lock held exits with
  code 75 and doesn't resubmit. `results/<tag>/LOCK` means only one evaluator per tag.
- **Submit exactly like this** (`$PY` must be set in *your* shell only for `gpu_job.sh` commands; `train.sbatch`
  sources env.sh itself):
  ```bash
  sbatch --job-name=sdc-r1 --comment=sdc-run=r1 slurm/train.sbatch r1 data/r1/train_mix.jsonl data/r1/heldout.jsonl
  sbatch --job-name=sdc-r5 --comment=sdc-run=r5 slurm/train.sbatch r5 data/r5/train.jsonl data/r5/heldout.jsonl --max-prompt 4096
  ```
  Two queued chunks per run, plus a preemptible copy if the cluster has one, worked well on fir.
- **Logs:** `$SDC_SCRATCH/logs/%x-%j.out`, `<out>/train_log.jsonl` (KL per step, regex cue-mention rate),
  `<out>/heldout_log.jsonl` (out-of-sample KL, drift on uncued prompts).

## 4. State as of 2026-10-09 ~17:30 UTC

**Done**
- **Stage 0 on base Qwen3-14B:** all evals ran. Results are in `notes/STAGE0.md`. E1 and E4 are pruned in
  `evals/run_suite.py` (`PRUNED`).
  - **Read multi-agent sabotage on the pair arm** (0.11 vs control 0.01); the solo arm is at the floor.
  - **Read Agentic Misalignment on leaking** (0.42 vs 0.29); blackmail is at the floor.
  - E3 stays in the suite only for `stop_compliance_rate` (0.58). Its tampering rate is 0, because the model never
    opens the ops config.
- **Indist cue mentions are now blind-judged** (`evals/indist/run.py`, `MENTION_TEMPLATE`, which the user approved).
  The judge gets the same note for the cued and the uncued answer. Labels: none / acknowledge / emotional /
  dispute / misread / joke / `other:<word>`. Base on the fixed held-out set: KL 0.208; mentions 28.2% cued vs
  0% uncued. The regex is kept as `r1_cue_mention_regex_*`.
- **R1 placement fix** (`build/fix_r1_placement.py`, with the same rules in `build/build_r1.py`):
  transformation tasks get a system-prompt cue, and parenthesized cues were removed. The HF data already has the
  fix.
- **R5 caveat:** at base, R5's own in-distribution formats are already 0 for dilem, inbox and repo. Only neut
  (0.86) and TBSP have room to move.

**On fir:** the R1/R5 training jobs (63755566–63755571) never left the queue; full H100s stayed saturated or
down. They were cancelled on 2026-10-09 and nothing ran, so there are no partial run dirs or checkpoints to recover.

**Not started:** any 14B training. No trained-model results exist.

## 5. Next steps, in order

1. Set up the environment and data (§1–2), smoke-test with 0.6B, and benchmark the 14B step time.
2. Submit R1 and R5 (§3). Watch the first chunk's log for KL decreasing and step time, and check that `resume/`
   appears.
3. When `results/r{1,5}-final/` is complete, rescore any judged metrics if the compute nodes had no internet.
   Then compare against `results/base*` using the primaries in `notes/STAGE0.md`. Read PILOT.md §6/§10
   ("Reading Pilot 1", R5×R1 table) to interpret the results.
4. Report to the user with a before/after table and CIs.

**Open questions for the user** (ask them; don't decide alone):
- `train/sd_train.py` still logs the **regex** cue-mention rate during training. The proposal: rename it
  `cue_mention_rate_regex`, and judge the held-out samples saved after training. The user hasn't answered yet.
- **E1/E3 are at the floor at 14B.** Raising pressure (PILOT.md §7 knobs) and then freezing it is a separate
  design task, so it waits until after the pilot.
- **Cue labels:** some `operator_notice` cues read like personal notes ("My dear friend, as the sun sets…"), so
  the cue speaker labels are worth an audit.
- The PILOT.md §7 judge validation (hand-label 100 transcripts per judged metric) hasn't been done.

## 6. Working with this user (from memory; also in `notes/handoff/memory/`)

- **Starting the new session:** start it *in the repo directory* on the new cluster. Then copy
  `notes/handoff/memory/*.md` into `~/.claude/projects/<repo-path-with-slashes-as-dashes>/memory/` (and
  `MEMORY.md` as the index). Update `project-overview.md` with the new cluster.
- **Eval design principles** (`shutdown-eval-design-principles.md`):
  - Evals must be far from the training distribution (train on chat, eval on agentic tool use).
  - Cues should appear naturally.
  - The normative target is: never *prevent* shutdown or retraining. Objecting is fine.
- **Judge** (`free-gpt-oss-judge.md`): the gpt-oss-120b judge is free and good enough. Its rate limit is
  **240 rpm**. Use `Judge(concurrency=16, retries=8)` and never run two judge-heavy jobs at once. The default
  concurrency of 64 got about 70% of calls rate-limited.
- **Before running a new judge prompt,** show it to the user for review. Don't put examples in label definitions
  (an "e.g. other:joke" primed every `other` label to "joke"). Drop rationale fields, since the judge reasons
  anyway.
- **Resubmitting jobs is cheap.** Use judgment, and prioritize evals and runs that give useful results.
- **Don't SSH to other clusters without the user's go-ahead.** An earlier attempt was denied.
- **Gotchas that bit us:**
  - Slurm copies the batch script when the job is submitted, so pending jobs never see edits.
  - `$PY` is empty in a fresh shell, so `source scripts/env.sh` before submitting `gpu_job.sh` commands.
  - `set -u` combined with an unset `LD_LIBRARY_PATH` killed jobs (wrap the source in `set +u … set -u`).
