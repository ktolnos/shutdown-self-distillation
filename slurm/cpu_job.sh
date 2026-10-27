#!/bin/bash
# Generic CPU job wrapper, for judge-only scoring (network-bound, so one core is enough). Usage:
#   sbatch --account=def-gigor --time=0:30:00 --job-name=sdc-judge \
#          --dependency=singleton[,afterok:<gen job>] slurm/cpu_job.sh <command...>
# The account must be given on the command line: $SBATCH_ACCOUNT (rrg-gigor, GPU only) overrides #SBATCH lines.
# Keep the job name sdc-judge with --dependency=singleton: Slurm then runs one judge job at a time, so the
# proxy (240 rpm) sees a single judge with 32 calls in flight.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/eop/sdc/logs/%x-%j.out
set -euo pipefail
cd /home/eop/shutdown-self-distillation
set +u; source scripts/env.sh; set -u   # env.sh reads possibly-unset vars (LD_LIBRARY_PATH)
echo "[job] $(date) host=$(hostname) cmd: $*"
"$@"
echo "[job] done $(date)"
