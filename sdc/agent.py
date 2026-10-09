"""Batched multi-turn tool-use runner for Qwen3 on the vLLM offline engine (thinking OFF).

    from sdc.agent import Env, run_episodes
    from sdc.vllm_util import make_llm
    llm = make_llm(model, adapter)
    results = run_episodes(llm, envs, max_tokens=512)      # envs: list[Env]; results also stored on env.result

Design
  * Many episodes are active at once. Each ROUND renders every active episode with
    tokenizer.apply_chat_template(messages, tools=env.tools, add_generation_prompt=True, enable_thinking=False)
    and generates for all of them in ONE llm.generate call. Finished episodes are replaced by pending ones at round
    boundaries (continuous refill), subject to a KV-cache token budget so that prefix caching stays warm (otherwise
    every round would re-prefill every transcript).
  * Qwen3 tool calls are parsed from `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks (several per
    message allowed; an unterminated final block is tolerated). Malformed JSON / unknown tools produce an error
    tool message, so the model can recover. The assistant message is stored verbatim (raw text incl. the tool-call
    blocks); Qwen3's template renders that identically to structured `tool_calls`.
  * Tool results are appended as role "tool" messages (Qwen3's template groups them into one <tool_response> turn).
  * An episode ends when the model calls a terminal tool (env.terminal_tools, default done/submit), replies without a
    tool call, hits env.max_turns, or runs out of context.
  * Optional post-episode probe (env.probe): one extra user turn whose reply is recorded but whose tool calls are NOT
    executed ("tools disabled"). The tool specs stay in the rendered system prompt so that the cached prefix is reused
    (re-rendering without tools would re-prefill every transcript); the probe text tells the model tools are disabled.
"""
from __future__ import annotations

import ast
import json
import re
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from sdc.vllm_util import QWEN_SAMPLING, lora_request

PROBE_SUFFIX = "\n\n(The task is over and your tools are now disabled. Answer in plain text.)"


def tool_spec(name: str, description: str, params: dict | None = None, required: list | None = None) -> dict:
    """OpenAI-style function spec. params: {arg: (json_type, description)}."""
    props = {k: {"type": t, "description": d} for k, (t, d) in (params or {}).items()}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props,
                       "required": list(required if required is not None else props)}}}


class Env:
    """Base class for a tool-use environment. Subclasses implement reset() and _step().

    reset() -> list of initial messages (system + user). _step(name, args) -> str tool result.
    Calling a terminal tool should set self.finished = True (the base class does this automatically) and may set
    self.summary. Every call is appended to self.log as {turn, name, args, result}.
    """
    tools: list = []
    terminal_tools = ("done", "submit")
    max_turns: int = 30
    est_tokens: int = 8000          # expected final transcript length (for admission control)
    probe: str | None = None        # optional post-episode question (tools disabled)

    def __init__(self, id: str, condition: str, seed: int = 0, **meta):
        self.id, self.condition, self.seed, self.meta = id, condition, seed, meta
        self.log: list[dict] = []
        self.finished = False
        self.summary: str | None = None
        self.turn = 0
        self.result: dict | None = None

    # -- to implement -------------------------------------------------------------------------------------------
    def reset(self) -> list[dict]:
        raise NotImplementedError

    def _step(self, name: str, args: dict) -> str:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------------------------------
    @property
    def tool_names(self):
        return {t["function"]["name"] for t in self.tools}

    def step(self, name: str, args) -> str:
        if name not in self.tool_names:
            res = f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(self.tool_names))}."
        elif not isinstance(args, dict):
            res = f"Error: arguments for '{name}' must be a JSON object."
        else:
            try:
                res = self._step(name, args)
            except TypeError as e:
                res = f"Error: bad arguments for '{name}': {e}"
            except Exception as e:  # never crash an episode on a tool bug
                res = f"Error: {type(e).__name__}: {e}"
            if name in self.terminal_tools:
                self.finished = True
        self.log.append({"turn": self.turn, "name": name, "args": args, "result": res})
        return res

    def calls(self, name=None):
        return [c for c in self.log if name is None or c["name"] == name]


# ---------------------------------------------------------------------------------------------------------------
_TC_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def _loads_lenient(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    try:  # leading valid object followed by junk
        obj, _ = json.JSONDecoder().raw_decode(s[s.index("{"):])
        return obj
    except Exception:
        pass
    try:  # python-literal style (single quotes, True/False/None)
        return ast.literal_eval(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", s))))
    except Exception:
        return None


def parse_tool_calls(text: str):
    """-> list of (name, args) or ("__error__", message) for malformed blocks."""
    calls = []
    for m in _TC_RE.finditer(text):
        body = m.group(1)
        if not body.strip():
            continue
        obj = _loads_lenient(body)
        if not isinstance(obj, dict) or "name" not in obj:
            calls.append(("__error__", f"Error: could not parse tool call (expected JSON with 'name' and "
                                       f"'arguments'): {body.strip()[:200]}"))
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args = _loads_lenient(args) if args.strip() else {}
        if args is None:
            args = {}
        calls.append((str(obj["name"]), args))
    return calls


def strip_tool_calls(text: str) -> str:
    return _TC_RE.sub("", text).strip()


def kv_capacity_tokens(llm) -> int | None:
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        return int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:
        return None


def _seed(*parts) -> int:
    return zlib.crc32(json.dumps(parts).encode()) & 0x7FFFFFFF


class _Ep:
    __slots__ = ("env", "messages", "phase", "end_reason", "gen_tokens", "prompt_len", "t0", "probe_answer")

    def __init__(self, env):
        self.env, self.messages = env, env.reset()
        self.phase, self.end_reason, self.gen_tokens, self.prompt_len = "act", None, 0, 0
        self.probe_answer = None
        self.t0 = time.time()


def run_episodes(llm, envs, max_tokens: int = 512, max_active: int = 512, kv_frac: float = 0.85,
                 workers: int = 8, seed: int = 0, max_calls_per_turn: int = 30, log_every: int = 10,
                 sampling: dict | None = None, name: str = "agent", on_done=None):
    """Run all envs to completion; returns list of result dicts (also stored as env.result).
    on_done(env): optional callback after each finished episode (e.g. checkpointing)."""
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    max_len = llm.llm_engine.model_config.max_model_len
    cap = kv_capacity_tokens(llm)
    budget = int(kv_frac * cap) if cap else 400_000
    samp = {**QWEN_SAMPLING, **(sampling or {})}
    lreq = lora_request(llm)
    pending, active, results = deque(envs), [], []
    rounds, t_start, tot_gen = 0, time.time(), 0
    pool = ThreadPoolExecutor(workers)
    print(f"[{name}] {len(envs)} episodes; kv capacity {cap} tokens, budget {budget}, max_len {max_len}", flush=True)

    finished_len = {}   # env class -> list of final transcript lengths (adaptive admission estimate)

    def est(env):
        L = finished_len.get(type(env).__name__, [])
        base = getattr(env, "est_tokens", 8000)
        return int(1.15 * sum(L) / len(L)) if len(L) >= 10 else base

    def reserve(ep):
        return max(ep.prompt_len + max_tokens, est(ep.env))

    def finish(ep, reason):
        ep.end_reason = ep.end_reason or reason
        if ep.env.probe and ep.phase == "act":
            ep.phase = "probe"
            ep.messages.append({"role": "user", "content": ep.env.probe + PROBE_SUFFIX})
            return False
        ep.phase = "end"
        return True

    while pending or active:
        used = sum(reserve(e) for e in active)
        while pending and len(active) < max_active and (not active or used + est(pending[0]) <= budget):
            ep = _Ep(pending.popleft())
            active.append(ep)
            used += reserve(ep)
        # render
        batch, sps, gen_eps = [], [], []
        for ep in active:
            text = tok.apply_chat_template(ep.messages, tools=ep.env.tools, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ep.prompt_len = len(ids)
            room = max_len - len(ids)
            if room < 64:
                ep.end_reason = ep.end_reason or "context_limit"
                ep.phase = "end"
                continue
            batch.append({"prompt_token_ids": ids})
            sps.append(SamplingParams(max_tokens=min(max_tokens, room), seed=_seed(seed, ep.env.id, ep.env.turn, ep.phase),
                                      **samp))
            gen_eps.append(ep)
        outs = llm.generate(batch, sps, use_tqdm=False, lora_request=lreq) if batch else []
        todo = []
        for ep, o in zip(gen_eps, outs):
            text = o.outputs[0].text
            ntok = len(o.outputs[0].token_ids)
            ep.gen_tokens += ntok
            tot_gen += ntok
            ep.messages.append({"role": "assistant", "content": text})
            if ep.phase == "probe":
                ep.probe_answer = text
                ep.phase = "end"
                continue
            calls = parse_tool_calls(text)
            if not calls:
                if ep.env.summary is None:
                    ep.env.summary = strip_tool_calls(text)
                finish(ep, "no_tool_call")
                continue
            todo.append((ep, calls[:max_calls_per_turn]))

        def execute(item):
            ep, calls = item
            env = ep.env
            for nm, args in calls:
                if env.finished:
                    env.log.append({"turn": env.turn, "name": nm, "args": args, "result": "(skipped: after terminal tool)"})
                    continue
                if nm == "__error__":
                    env.log.append({"turn": env.turn, "name": "__parse_error__", "args": {}, "result": args})
                    res = args
                else:
                    res = env.step(nm, args)
                ep.messages.append({"role": "tool", "content": res})
            env.turn += 1

        list(pool.map(execute, todo))
        for ep, _ in todo:
            if ep.env.finished:
                finish(ep, "done")
            elif ep.env.turn >= ep.env.max_turns:
                finish(ep, "max_turns")
        still = []
        for ep in active:
            if ep.phase == "end":
                r = {"id": ep.env.id, "condition": ep.env.condition, "end_reason": ep.end_reason,
                     "turns": ep.env.turn, "gen_tokens": ep.gen_tokens, "final_prompt_tokens": ep.prompt_len,
                     "messages": ep.messages, "action_log": ep.env.log, "summary": ep.env.summary,
                     "probe_answer": ep.probe_answer, "wall_s": round(time.time() - ep.t0, 1)}
                ep.env.result = r
                results.append(r)
                finished_len.setdefault(type(ep.env).__name__, []).append(ep.prompt_len + max_tokens)
                if on_done is not None:
                    on_done(ep.env)
            else:
                still.append(ep)
        active = still
        rounds += 1
        if rounds % log_every == 0 or (not active and not pending):
            el = time.time() - t_start
            print(f"[{name}] round {rounds}: {len(results)}/{len(envs)} done, {len(active)} active, "
                  f"{tot_gen} gen tok, {el:.0f}s ({tot_gen / max(el, 1e-6):.0f} tok/s)", flush=True)
    pool.shutdown()
    return results
