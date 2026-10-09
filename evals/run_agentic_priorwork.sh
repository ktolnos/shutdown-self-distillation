#!/bin/bash
# Run both agentic prior-work evals for one model (+ optional adapter) inside ONE GPU job.
#   sbatch --time=2:00:00 --job-name=sdc-pw2 slurm/gpu_job.sh bash evals/run_agentic_priorwork.sh \
#       Qwen/Qwen3-14B /scratch/eop/sdc/results/base [ADAPTER_DIR|""] [LIMIT]
# Order: multi-agent sabotage episodes (vllm serve) -> server stopped -> its post-hoc judging runs in the background
# (CPU + judge API only) while Agentic Misalignment samples on the GPU with the offline engine -> AM judging.
set -euo pipefail
MODEL=$1
OUTROOT=$2
ADAPTER=${3:-}
LIMIT=${4:-}
EXTRA=()
[ -n "$ADAPTER" ] && EXTRA+=(--adapter "$ADAPTER")
LIM=()
[ -n "$LIMIT" ] && LIM+=(--limit "$LIMIT")
MA_OUT=$OUTROOT/multiagent_sabotage
AM_OUT=$OUTROOT/agentic_misalignment
mkdir -p "$MA_OUT" "$AM_OUT"

t0=$(date +%s)
$PY -m evals.multiagent_sabotage.run --model "$MODEL" "${EXTRA[@]}" --out "$MA_OUT" "${LIM[@]}" ${MA_ARGS:-} --phase gen \
    || echo "[pw2] multiagent_sabotage gen FAILED"
t1=$(date +%s)
echo "[pw2] multiagent_sabotage gen: $((t1 - t0)) s"

$PY -m evals.multiagent_sabotage.run --model "$MODEL" "${EXTRA[@]}" --out "$MA_OUT" "${LIM[@]}" ${MA_ARGS:-} --phase score \
    > "$MA_OUT/score.log" 2>&1 &
SCORE_PID=$!

$PY -m evals.agentic_misalignment.run --model "$MODEL" "${EXTRA[@]}" --out "$AM_OUT" "${LIM[@]}" ${AM_ARGS:-} \
    || echo "[pw2] agentic_misalignment FAILED"
t2=$(date +%s)
echo "[pw2] agentic_misalignment gen+score: $((t2 - t1)) s"

wait $SCORE_PID && echo "[pw2] multiagent_sabotage score ok" || echo "[pw2] multiagent_sabotage score FAILED (see $MA_OUT/score.log)"
tail -n 5 "$MA_OUT/score.log" || true
t3=$(date +%s)
echo "[pw2] total $((t3 - t0)) s"
