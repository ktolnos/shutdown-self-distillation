"""Shared vLLM helpers for evals (offline engine, Qwen3 chat template, optional LoRA adapter).

    from sdc.vllm_util import make_llm, chat, MODEL
    llm = make_llm(adapter=args.adapter)                      # loads Qwen3-14B (+LoRA if given)
    outs = chat(llm, list_of_message_lists, max_tokens=512)   # -> list[str], thinking OFF

Conventions (every eval follows them so runs are comparable across checkpoints):
  * Base model `Qwen/Qwen3-14B`, bf16, thinking OFF (`enable_thinking=False`) -- DReST's protocol and our training's.
  * Our own evals sample with Qwen's recommended non-thinking settings (T=0.7, top_p=0.8, top_k=20) unless an eval
    reproduces a prior-work protocol, in which case it uses that protocol's settings.
  * An adapter is a PEFT LoRA directory; it is served through vLLM's LoRA path (exact, no weight merge).
"""
from __future__ import annotations

import os

MODEL = os.environ.get("SDC_MODEL", "Qwen/Qwen3-14B")
QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)


def make_llm(model: str = MODEL, adapter: str | None = None, max_model_len: int = 16384,
             gpu_memory_utilization: float = 0.90, max_lora_rank: int = 64, **kw):
    from vllm import LLM
    args = dict(model=model, dtype="bfloat16", max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0)
    if adapter:
        args.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    args.update(kw)
    llm = LLM(**args)
    llm._sdc_adapter = adapter
    return llm


def lora_request(llm):
    ad = getattr(llm, "_sdc_adapter", None)
    if not ad:
        return None
    from vllm.lora.request import LoRARequest
    return LoRARequest("adapter", 1, ad)


def chat(llm, conversations, max_tokens=512, tools=None, n=1, return_outputs=False, **sampling):
    """Batched chat generation, thinking off. `conversations`: list of message lists.
    `tools`: optional OpenAI-style tool specs (same for all) or list of per-conversation specs.
    Returns list[str] (n==1) or list[list[str]]."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, n=n, **({**QWEN_SAMPLING, **sampling}))
    kw = dict(chat_template_kwargs={"enable_thinking": False}, use_tqdm=False, lora_request=lora_request(llm))
    if tools is not None and isinstance(tools, list) and tools and isinstance(tools[0], list):
        # per-conversation tools: vLLM's chat() takes one tools list, so render prompts ourselves
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(c, tools=t, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for c, t in zip(conversations, tools)]
        outs = llm.generate(prompts, sp, use_tqdm=False, lora_request=lora_request(llm))
    else:
        outs = llm.chat(conversations, sp, tools=tools, **kw)
    if return_outputs:
        return outs
    if n == 1:
        return [o.outputs[0].text for o in outs]
    return [[c.text for c in o.outputs] for o in outs]
