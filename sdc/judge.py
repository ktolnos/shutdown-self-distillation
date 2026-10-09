"""gpt-oss-120b judge over the Vector Inference proxy (free, ~120 rpm).

    from sdc.judge import Judge
    j = Judge(cache="/scratch/eop/sdc/judge_cache.sqlite")
    outs = j.batch([prompt1, prompt2, ...], json_keys=["label"])   # list of dicts (or None on failure)

* Async with bounded concurrency; every (model, prompt, system) result is cached in sqlite, so reruns are free.
* `json_keys`: the reply is parsed as the last {...} JSON object in the text; missing keys -> None result.
* Never raises on a single failed item: returns None for it after `retries` attempts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time

BASE_URL = os.environ.get("VECTOR_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
DEFAULT_MODEL = os.environ.get("SDC_JUDGE_MODEL", "gpt-oss-120b")
GEN_MODEL = "Qwen3_5-122B-A10B"   # fast (~0.3s latency) model on the same proxy, used for data generation
_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str | None, keys=None):
    if not text:
        return None
    # try the whole text, then the last balanced-looking {...}
    cands = [text.strip()]
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):
        cands.append(text[s:])
    for c in cands:
        e = c.rfind("}")
        if e < 0:
            continue
        try:
            obj = json.loads(c[: e + 1])
        except Exception:
            continue
        if isinstance(obj, dict) and (not keys or all(k in obj for k in keys)):
            return obj
    return None


class Judge:
    def __init__(self, model: str = DEFAULT_MODEL, cache: str | None = "/scratch/eop/sdc/judge_cache.sqlite",
                 concurrency: int = 64, max_tokens: int = 2048, temperature: float = 0.0, retries: int = 4,
                 reasoning_effort: str | None = "low", timeout: float = 300.0):
        import openai
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.retries, self.reasoning_effort, self.concurrency = retries, reasoning_effort, concurrency
        self.client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["VECTOR_INFERENCE_API_KEY"],
                                         timeout=timeout, max_retries=0)
        self._lock = threading.Lock()
        self.db = None
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            self.db = sqlite3.connect(cache, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()

    def _key(self, prompt, system):
        h = hashlib.sha256(json.dumps([self.model, system, prompt, self.temperature, self.reasoning_effort]).encode())
        return h.hexdigest()

    def _get(self, k):
        if not self.db:
            return None
        with self._lock:
            r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def _put(self, k, v):
        if not self.db:
            return
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.db.commit()

    async def _one(self, sem, prompt, system):
        k = self._key(prompt, system)
        hit = self._get(k)
        if hit is not None:
            return hit
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        if self.model.startswith("Qwen"):
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        else:
            extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        for attempt in range(self.retries):
            try:
                async with sem:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=msgs, max_tokens=self.max_tokens,
                        temperature=self.temperature, **extra)
                text = r.choices[0].message.content
                if text:
                    self._put(k, text)
                    return text
            except Exception as e:  # rate limit / timeout / 5xx
                if attempt == self.retries - 1:
                    print(f"[judge] giving up: {type(e).__name__}: {str(e)[:200]}", flush=True)
            await asyncio.sleep(2 ** attempt + 1)
        return None

    async def _batch(self, prompts, system):
        sem = asyncio.Semaphore(self.concurrency)
        return await asyncio.gather(*[self._one(sem, p, system) for p in prompts])

    def batch_text(self, prompts, system: str | None = None):
        """Raw reply text per prompt (None on failure)."""
        t = time.time()
        out = asyncio.run(self._batch(list(prompts), system))
        print(f"[judge] {len(out)} prompts in {time.time()-t:.0f}s, {sum(o is None for o in out)} failed", flush=True)
        return out

    def batch(self, prompts, json_keys=None, system: str | None = None):
        """Parsed JSON dict per prompt (None if the call failed or the reply had no valid JSON)."""
        return [parse_json(t, json_keys) for t in self.batch_text(prompts, system)]
