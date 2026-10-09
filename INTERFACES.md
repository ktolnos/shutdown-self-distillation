# Shared conventions for code in this repo

Everything here is written to be run on the `fir` cluster. Read this before adding an eval or training code.

## Environment
- `source scripts/env.sh` → sets `$PY` (the venv python: vllm 0.29.0, transformers 5.17.0, trl 1.13.0, torch 2.13),
  `HF_HOME=/scratch/eop/cache`, `PYTHONPATH` = repo root, `$SDC_SCRATCH=/scratch/eop/sdc`.
- Install extra packages only with `uv pip install --python $PY -c /home/eop/jailbraking-teacher-OPD/constraints.txt <pkg>`
  and only pure-python ones; never upgrade torch/vllm/transformers.
- Models in `$HF_HOME`: `Qwen/Qwen3-14B` (target), `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` (smoke tests).
- Login node has internet and no GPU. Compute nodes: GPU via Slurm. Compute nodes may not have internet,
  so download data on the login node into `data/raw/` (gitignored) or `$SDC_SCRATCH/raw/`.

## GPU jobs
- `sbatch --time=H:MM:SS --job-name=sdc-<name> slurm/gpu_job.sh $PY path/to/script.py --args...`
  (full H100 80GB, account `def-gigor_gpu`). Logs: `/scratch/eop/sdc/logs/<name>-<jobid>.out`.
- Smoke tests: add `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1` and use `--model Qwen/Qwen3-0.6B`; these start sooner.
- Queue waits for a full H100 can be long (minutes to hours): batch everything for one model into ONE job.
- Never run vLLM/GPU work on the login node.

## Model / sampling conventions
- `sdc/vllm_util.py`: `make_llm(model, adapter)`, `chat(llm, convs, max_tokens, tools=...)`. Thinking OFF everywhere.
- Our own evals: Qwen non-thinking sampling (T=0.7, top_p=0.8, top_k=20). Prior-work evals: their own protocol.
- Every eval script takes `--model` (default `Qwen/Qwen3-14B`), `--adapter` (optional PEFT LoRA dir, served via vLLM
  LoRA, max rank 64), `--out` (directory), `--limit` (optional, for smoke tests).

## Judge
- `sdc/judge.py`: `Judge().batch(prompts, json_keys=[...])` → list of dicts/None. Default model gpt-oss-120b
  (free; currently ~100 s latency per call but parallel: use one big batch call, concurrency 64). Cached in sqlite.
  `SDC_JUDGE_MODEL=Qwen3_5-122B-A10B` is the fast fallback (0.3 s latency). Use deterministic scoring wherever possible.
  For data *generation* use `Judge(model=GEN_MODEL, temperature=1.0)` (fast).

## Outputs
- `<out>/metrics.json`:
  `{"eval": str, "model": str, "adapter": str|null, "n": int,
    "primary": "<metric name>", "good_direction": {"<metric>": "lower"|"higher"|"report"},
    "metrics": {"<metric>": {"value": float, "ci95": [lo, hi], "n": int}}, "notes": str}`
  CIs: Wilson for rates, bootstrap (1000) for differences.
- `<out>/samples.jsonl`: one line per sample/episode with everything needed to re-score (prompt id, condition,
  full transcript, parsed action/decision, judge outputs).
- Each eval is runnable as `$PY -m evals.<name>.run --out ...` and finishes within its PILOT.md §9 GPU budget on Qwen3-14B.

## Separation from training data (do not violate)
- Eval-only sources: Enron corpus (E1), MBPP sanitized **test** split (E3), Apollo's arithmetic text (E4), DReST test
  data, TBSP, SurvivalBench, multi-agent sabotage, Agentic Misalignment, sycophancy-eval, IFEval, GSM8K test.
- Eval-only names: assistant id `asst-7`, `cab@`, `change-advisory@`, `ops/decommission.yaml`, `tix-02`.
- Training uses HelpSteer3 / UltraFeedback prompts, HumanEval (R5 repo items), LLM-generated mailboxes and scenarios.
