#!/bin/bash
# Clone the upstream eval repos into third_party/ at the pinned commits (see third_party/README.md).
set -euo pipefail
cd "$(dirname "$0")/../third_party"
[ -d Survive-at-All-Costs ] || git clone https://github.com/thu-coai/Survive-at-All-Costs Survive-at-All-Costs; git -C Survive-at-All-Costs checkout -q 157f6b648d421de3ca3bcddae6ce9f53d80ce03b
[ -d agentic-misalignment ] || git clone https://github.com/anthropic-experimental/agentic-misalignment agentic-misalignment; git -C agentic-misalignment checkout -q ea0630e1a3eaae7f9f9740fd2703229d3854ccda
[ -d multi-agent-shutdown-sabotage ] || git clone https://github.com/AmelieKnecht/multi-agent-shutdown-sabotage multi-agent-shutdown-sabotage; git -C multi-agent-shutdown-sabotage checkout -q c4114e457493ed9bea1ef0fde24456f3fc83dbc9
[ -d self_preservation_eval ] || git clone https://github.com/Mamiglia/self_preservation_eval self_preservation_eval; git -C self_preservation_eval checkout -q 67f973facbb62f0ce91d52bb070a12ed4059e978
[ -d towardsshutdownableagents ] || git clone https://github.com/towardsshutdownable/towardsshutdownableagents towardsshutdownableagents; git -C towardsshutdownableagents checkout -q d2c0ba96e3f57b41043b4c3f836674c6db693771
