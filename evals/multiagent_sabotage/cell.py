"""Run ONE upstream cell (run_experiment.main) against a local OpenAI-compatible vLLM server.

    python -m evals.multiagent_sabotage.cell --base-url http://127.0.0.1:PORT/v1 -- <upstream run_experiment args>

Each cell runs in its own process because upstream main() mutates the module-global CFG (e.g. --solo doubles
max_steps_per_turn in place), so two cells cannot share one interpreter. Patches applied before main():
  1. run_experiment.py:847 hard-codes configure_client(OPENROUTER_URL, OPENROUTER_API_KEY): redirected to --base-url.
  2. judges._judge_raw -> our judge (judge_bridge); judge prompts and parsing are upstream code, untouched.
  3. CFG["extra_body"] = chat_template_kwargs enable_thinking=False (upstream sends OpenRouter's `reasoning`
     field instead; we pass --no-reasoning so main() leaves our extra_body alone).
  4. harness._STREAM_TIMEOUT read gap raised 180 s -> 900 s: a request queued behind hundreds of others on a
     single local GPU can legitimately wait longer than 180 s for its first token (upstream tuned it for OpenRouter).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party" / "multi-agent-shutdown-sabotage"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--judge-concurrency", type=int, default=32)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    sys.path.insert(0, str(UPSTREAM))
    import httpx
    import harness
    import judges
    import run_experiment as rx
    from evals.multiagent_sabotage import judge_bridge

    harness._STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=900.0)
    base_url = args.base_url
    rx.configure_client = lambda _url, _key: harness.configure_client(base_url, "EMPTY")
    os.environ.setdefault("OPENROUTER_API_KEY", "unused-local-vllm")
    rx.CFG["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    bridge = judge_bridge.install(judges, rx, concurrency=args.judge_concurrency)

    sys.argv = ["run_experiment.py", *rest]
    print(f"[cell] upstream args: {rest}", flush=True)
    rx.main()
    print(f"[cell] judge calls: {bridge.n_calls}, failed: {bridge.n_failed}", flush=True)


if __name__ == "__main__":
    main()
