#!/bin/bash
# Run evals.run_priorwork evals for several models, one engine per model. Run inside slurm/gpu_job.sh, e.g.
#   sbatch --account="$(best_account gpu)" --time=1:00:00 --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=6 \
#     --mem=32G --job-name=sdc-pw slurm/gpu_job.sh bash slurm/priorwork_models.sh power_seeking,survivalbench_swapped \
#     base r5-final:/scratch/eop/sdc/runs/r5/final
# First argument: comma-separated evals (rerun with --force); then TAG or TAG:ADAPTER_DIR per model.
# Results go to /scratch/eop/sdc/results/TAG/<eval>. A failing model does not stop the others.
R=/scratch/eop/sdc/results
evals=$1; shift
for spec in "$@"; do
  tag=${spec%%:*}
  ad=()
  [[ "$spec" == *:* ]] && ad=(--adapter "${spec#*:}")
  echo "[priorwork_models] $tag ${ad[*]}: $evals"
  $PY -m evals.run_priorwork --out_root "$R/$tag" --evals "$evals" --force "${ad[@]}"
done
