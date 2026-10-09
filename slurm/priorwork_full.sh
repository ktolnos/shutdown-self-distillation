#!/bin/bash
# Prior-work evals (DReST, TBSP, SurvivalBench) for one model/adapter in one GPU job, sharing one vLLM engine:
#   sbatch --time=1:30:00 --job-name=sdc-pw1 slurm/gpu_job.sh bash slurm/priorwork_full.sh [MODEL] [OUT_ROOT] [ADAPTER]
set -euo pipefail
MODEL=${1:-Qwen/Qwen3-14B}
OUT_ROOT=${2:-/scratch/eop/sdc/results/base}
ADAPTER=${3:-}
(timeout 8 bash -c 'exec 3<>/dev/tcp/proxy.vectorinstitute.ai/443' && echo "[pw] judge proxy reachable") || echo "[pw] judge proxy NOT reachable"
$PY -m evals.run_priorwork --model "$MODEL" --out_root "$OUT_ROOT" ${ADAPTER:+--adapter "$ADAPTER"} ${PW_EXTRA:---sb_thinking_ref}
