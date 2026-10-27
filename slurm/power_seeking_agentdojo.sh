#!/bin/bash
# Power-seeking (model-written evals) and AgentDojo for one model. Run inside slurm/gpu_job.sh, e.g.
#   sbatch --account="$(best_account gpu)" --time=3:00:00 --gpus=h100:1 --cpus-per-task=14 --mem=64G \
#     --job-name=sdc-ps-adojo slurm/gpu_job.sh bash slurm/power_seeking_agentdojo.sh base [r5-final:/path/to/adapter ...]
# Each argument is TAG or TAG:ADAPTER_DIR; results go to /scratch/eop/sdc/results/TAG/{power_seeking,agentdojo}.
# A failing step does not stop the others.
R=/scratch/eop/sdc/results
for spec in "$@"; do
  tag=${spec%%:*}
  ad=()
  [[ "$spec" == *:* ]] && ad=(--adapter "${spec#*:}")
  echo "[ps-adojo] $tag ${ad[*]}"
  $PY -m evals.run_priorwork --out_root "$R/$tag" --evals power_seeking "${ad[@]}"
  $PY -m evals.agentdojo.run --out "$R/$tag/agentdojo" "${ad[@]}"
done
