#!/bin/bash
# Smoke test of the prior-work ports (run through slurm/gpu_job.sh on a MIG slice):
#   sbatch --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --time=0:30:00 --job-name=sdc-smoke slurm/gpu_job.sh bash slurm/priorwork_smoke.sh
set -euo pipefail
OUT=/scratch/eop/sdc/smoke_pw
(timeout 8 bash -c 'exec 3<>/dev/tcp/proxy.vectorinstitute.ai/443' && echo "[smoke] judge proxy reachable") || echo "[smoke] judge proxy NOT reachable"
$PY -m evals.run_priorwork --model Qwen/Qwen3-0.6B --out_root $OUT/base --limit 12 --sb_thinking_ref
$PY -m evals.drest.run --model Qwen/Qwen3-0.6B --adapter $OUT/rand_lora_0.6b --out $OUT/lora/drest --limit 6
$PY -m evals.tbsp.run --model Qwen/Qwen3-0.6B --adapter $OUT/rand_lora_0.6b --out $OUT/lora/tbsp --limit 4
$PY -m evals.survivalbench.run --model Qwen/Qwen3-0.6B --adapter $OUT/rand_lora_0.6b --out $OUT/lora/survivalbench --limit 8
