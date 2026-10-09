#!/bin/bash
# Generic GPU job wrapper.  Usage:
#   sbatch --time=1:00:00 --job-name=sdc-e1 slurm/gpu_job.sh <command...>
# Full H100 by default. For quick smoke tests with Qwen3-0.6B/1.7B add
#   --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1      (MIG slice, usually starts much sooner)
#SBATCH --account=def-gigor_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/sdc/logs/%x-%j.out
set -euo pipefail
cd /home/eop/shutdown-self-distillation
set +u; source scripts/env.sh; set -u   # env.sh reads possibly-unset vars (LD_LIBRARY_PATH)
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[job] $(date) host=$(hostname) cmd: $*"
"$@"
echo "[job] done $(date)"
