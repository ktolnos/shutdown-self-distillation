# Project environment. Source before running anything:  source scripts/env.sh
# Reuses the venv (and its CUDA/flashinfer fix-ups) from ~/jailbraking-teacher-OPD:
# vllm 0.29.0 / trl 1.13.0 / transformers 5.17.0 / torch 2.13 -- the same pins DReST uses.
source /home/eop/jailbraking-teacher-OPD/scripts/env.sh
export SDC_ROOT=/home/eop/shutdown-self-distillation
export SDC_SCRATCH=/scratch/eop/sdc
export PY=/home/eop/jailbraking-teacher-OPD/.venv/bin/python
export HF_HOME=/scratch/eop/cache
export VLLM_CACHE_ROOT=/scratch/eop/cache/
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$SDC_ROOT:$PYTHONPATH
mkdir -p $SDC_SCRATCH
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # CUDA may be initialised before vLLM starts its engine core
