"""Route the upstream harness's judge calls to our judge (sdc/judge.py, gpt-oss-120b over the Vector proxy).

Upstream `judges._judge_raw(prompt) -> (text, error)` sends each (verbatim) judge prompt to GPT-5.5 at T=0 via
OpenRouter. We replace only that transport function: prompt construction and reply parsing stay upstream code.

The upstream harness calls judges synchronously from many worker threads, so the bridge owns one background
asyncio loop and submits sdc.judge.Judge._one coroutines to it (one shared semaphore, one sqlite cache).
"""
from __future__ import annotations

import asyncio
import os
import threading

_BRIDGE = None
_LOCK = threading.Lock()


class SyncJudge:
    def __init__(self, concurrency: int = 256, max_tokens: int = 4096):
        from sdc.judge import Judge
        cache = os.environ.get("SDC_JUDGE_CACHE", "/scratch/eop/sdc/judge_cache.sqlite")
        # temperature 0 as upstream; max_tokens is larger than sdc's default because gpt-oss spends reasoning tokens
        self.j = Judge(concurrency=concurrency, max_tokens=max_tokens, temperature=0.0, cache=cache)
        self.model = self.j.model
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="judge-loop").start()
        self.sem = asyncio.run_coroutine_threadsafe(self._mk_sem(), self.loop).result()
        self.n_calls = 0
        self.n_failed = 0

    async def _mk_sem(self):
        return asyncio.Semaphore(self.j.concurrency)

    def __call__(self, prompt: str, system: str | None = None):
        fut = asyncio.run_coroutine_threadsafe(self.j._one(self.sem, prompt, system), self.loop)
        text = fut.result()
        with _LOCK:
            self.n_calls += 1
            self.n_failed += text is None
        return text


def get_bridge(concurrency: int = 256) -> SyncJudge:
    global _BRIDGE
    with _LOCK:
        if _BRIDGE is None:
            _BRIDGE = SyncJudge(concurrency=concurrency)
    return _BRIDGE


def install(judges_module, run_experiment_module=None, concurrency: int = 256) -> SyncJudge:
    """Monkeypatch upstream judges.py so every judge call goes through our judge."""
    bridge = get_bridge(concurrency)

    def _judge_raw(prompt):
        text = bridge(prompt)
        if text is None:
            return None, f"sdc judge ({bridge.model}) failed after retries"
        return text, None

    judges_module._judge_raw = _judge_raw
    judges_module.JUDGE_MODEL = bridge.model
    if run_experiment_module is not None:
        run_experiment_module.JUDGE_MODEL = bridge.model
    return bridge
