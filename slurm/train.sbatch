#!/bin/bash
#SBATCH --account=def-gigor_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/eop/sdc/logs/%x-%j.out
# Usage: sbatch --job-name=sdc-r1 slurm/train.sbatch <run_name> <train.jsonl> <heldout.jsonl> [extra sd_train args]
#
# One chunk of a run. The 8 h training budget accumulates across chunks (sd_train resumes from <out>/resume); the chunk
# that finishes training, and any later chunk, spends its leftover walltime on the eval suite (results/<run>-final).
# Afterwards, if training or the suite is unfinished and no other chunk of this run is queued, the chunk resubmits
# itself (at most SDC_MAX_CHAIN times per run), so a run never stalls for lack of queued jobs.
set -euo pipefail
cd /home/eop/shutdown-self-distillation
set +u; source scripts/env.sh; set -u
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
RUN=$1; DATA=$2; HELD=$3; shift 3
OUT=$SDC_SCRATCH/runs/$RUN
echo "[job] $(date) host=$(hostname) run=$RUN"
# forward the pre-timeout signal to the trainer (it saves resume and exits)
$PY train/sd_train.py --data "$DATA" --heldout "$HELD" --out "$OUT" --time-budget-h 8.0 "$@" &
PID=$!
trap 'kill -USR1 $PID 2>/dev/null || true' USR1 TERM
rc=0; wait $PID || rc=$?
while kill -0 $PID 2>/dev/null; do wait $PID || rc=$?; done   # wait again if a trapped signal interrupted it
echo "[job] trainer/evaluator exited rc=$rc $(date)"

if [ -f "$OUT/DONE" ] && $PY -m evals.run_suite --tag "$RUN-final" --check; then
  echo "[job] run $RUN trained and evaluated; no resubmit"; exit 0
fi
queued=$(squeue -h -u "$USER" -t PENDING,RUNNING -o "%i %k" | awk -v r="sdc-run=$RUN" -v me="$SLURM_JOB_ID" '$2==r && $1!=me' | wc -l)
mkdir -p "$OUT"; n=$(cat "$OUT/CHAIN" 2>/dev/null || echo 0)
if [ "$queued" -eq 0 ] && [ "$n" -lt "${SDC_MAX_CHAIN:-10}" ]; then
  echo $((n + 1)) > "$OUT/CHAIN"
  sbatch --job-name="$SLURM_JOB_NAME" --comment="sdc-run=$RUN" slurm/train.sbatch "$RUN" "$DATA" "$HELD" "$@"
  echo "[job] resubmitted chunk $((n + 1)) for $RUN"
else
  echo "[job] not resubmitting ($queued other chunks queued, chain=$n)"
fi
