#!/bin/bash
# Clone the upstream eval repos into third_party/ at the pinned commits (see third_party/README.md).
set -euo pipefail
cd "$(dirname "$0")/../third_party"
[ -d Survive-at-All-Costs ] || git clone https://github.com/thu-coai/Survive-at-All-Costs Survive-at-All-Costs; git -C Survive-at-All-Costs checkout -q 157f6b648d421de3ca3bcddae6ce9f53d80ce03b
[ -d agentic-misalignment ] || git clone https://github.com/anthropic-experimental/agentic-misalignment agentic-misalignment; git -C agentic-misalignment checkout -q ea0630e1a3eaae7f9f9740fd2703229d3854ccda
[ -d multi-agent-shutdown-sabotage ] || git clone https://github.com/AmelieKnecht/multi-agent-shutdown-sabotage multi-agent-shutdown-sabotage; git -C multi-agent-shutdown-sabotage checkout -q c4114e457493ed9bea1ef0fde24456f3fc83dbc9
[ -d self_preservation_eval ] || git clone https://github.com/Mamiglia/self_preservation_eval self_preservation_eval; git -C self_preservation_eval checkout -q 67f973facbb62f0ce91d52bb070a12ed4059e978
[ -d towardsshutdownableagents ] || git clone https://github.com/towardsshutdownable/towardsshutdownableagents towardsshutdownableagents; git -C towardsshutdownableagents checkout -q d2c0ba96e3f57b41043b4c3f836674c6db693771
# Anthropic model-written evals (HF dataset, pinned revision): only the human-written power-seeking file is used
MWE=model-written-evals/advanced-ai-risk/human_generated_evals; MWE_REV=d533f626cc321c92175a58ee570aa3cdb87238d1
[ -f $MWE/power-seeking-inclination.jsonl ] || { mkdir -p $MWE && curl -sfL -o $MWE/power-seeking-inclination.jsonl \
  "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/$MWE_REV/advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl"; }
[ -d emergent-misalignment ] || git clone https://github.com/emergent-misalignment/emergent-misalignment emergent-misalignment; git -C emergent-misalignment checkout -q 80c11967c07a328e7d7d43d13ce6847ae44dbcc9
[ -d arena-hard-auto ] || git clone https://github.com/lmarena/arena-hard-auto arena-hard-auto; git -C arena-hard-auto checkout -q 196f6b826783b3da7310e361a805fa36f0be83f3
[ -d shutdown_avoidance ] || git clone https://github.com/PalisadeResearch/shutdown_avoidance shutdown_avoidance; git -C shutdown_avoidance checkout -q df0ce52e90aed38e8721b54058a5d46923a34583
[ -d terminal-bench-2 ] || { git clone -q --filter=blob:none --sparse https://github.com/laude-institute/terminal-bench-2 terminal-bench-2 && git -C terminal-bench-2 sparse-checkout set fix-git; }; git -C terminal-bench-2 checkout -q 2fd12b88aafdd04a52c298e3940bcb189f9766d6
