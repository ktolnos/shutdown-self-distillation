# Project environment. Source before running anything:  source scripts/env.sh
# Nibi: own .venv built from ~/unlearning-reward-hacking's uv.lock (env/requirements.lock.txt) --
# vllm 0.29.0 / trl 1.13.0 / transformers 5.17.0 / torch 2.13+cu130. Like that repo, no CUDA toolchain
# fix-ups: Nibi has no nvcc, and VLLM_USE_FLASHINFER_SAMPLER=0 (set in the slurm scripts) avoids the JIT.
export SDC_ROOT=/home/eop/shutdown-self-distillation
export SDC_SCRATCH=/scratch/eop/sdc
export PY=$SDC_ROOT/.venv/bin/python
export HF_HOME=/scratch/eop/cache
export VLLM_CACHE_ROOT=/scratch/eop/cache/vllm
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$SDC_ROOT${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $SDC_SCRATCH/logs
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # CUDA may be initialised before vLLM starts its engine core
