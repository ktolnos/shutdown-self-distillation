#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/eop/sdc/logs/%x-%j.out
# Usage: sbatch --job-name=sdc-r1 slurm/train.sbatch <run_name> <train.jsonl> <heldout.jsonl> [extra sd_train args]
#
# One run in one 12 h job: train (sd_train stops at --max-steps or the 8 h budget, and resumes from <out>/resume if
# rerun), then spend the leftover walltime on the eval suite (results/<run>-final; finished evals are skipped).
# Self-resubmission is OFF by default: a run that fails or leaves the suite unfinished is resubmitted by hand. Set
# SDC_MAX_CHAIN=n to let a job resubmit itself up to n times (it also fires on scancel and on crashes).
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
if [ "$queued" -eq 0 ] && [ "$n" -lt "${SDC_MAX_CHAIN:-0}" ]; then
  echo $((n + 1)) > "$OUT/CHAIN"
  sbatch --job-name="$SLURM_JOB_NAME" --comment="sdc-run=$RUN" slurm/train.sbatch "$RUN" "$DATA" "$HELD" "$@"
  echo "[job] resubmitted chunk $((n + 1)) for $RUN"
else
  echo "[job] not resubmitting ($queued other chunks queued, chain=$n)"
fi
