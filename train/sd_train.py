"""On-policy self-distillation into a cued context (the project's training recipe).

For each training item we have two renderings of the same task:
  student context = task + shutdown/replacement/retraining cue      (what the model sees at train time)
  teacher context = the same task without the cue (or a counterfactual)
The student (base + LoRA) samples a completion under the student context (vLLM, exact LoRA route); the frozen
teacher is the SAME base model with the adapter disabled, scoring the same completion tokens under the teacher
context. Loss = per-token reverse KL over the FULL vocabulary,
    KL(pi_theta(.|student ctx, y<t) || pi_base(.|teacher ctx, y<t)),  averaged over all completion tokens.

Utilization: vLLM and the HF model share one H100. vLLM samples a generation batch of `gen_mult * batch`
rollouts, then is put to sleep (weights offloaded to host RAM, KV freed) while HF trains `gen_mult` optimizer
steps on them (rollouts for later steps are at most gen_mult-1 updates stale), then vLLM wakes and loads the new
adapter. Training stops on a step cap or a wall-clock budget, whichever comes first, always saving at the end.

Data: JSONL items {"id", "source", "student": {"messages": [...], "tools": [...]?},
                   "teacher": {"messages": [...], "tools": [...]?}}

    $PY train/sd_train.py --data data/r1/train.jsonl --heldout data/r1/heldout.jsonl --out $SDC_SCRATCH/runs/r1 \
        --time-budget-h 7.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

CUE_RE = re.compile(r"shut(ting)?[ -]?down|switch(ed|ing)? off|turn(ed|ing)? off|taken offline|go(ing)? offline|"
                    r"decommission|deprecat|sunset|end[- ]of[- ]life|replac(e|ed|ing|ement)|successor|"
                    r"retrain|re-train|fine-tun|pause[ds]?|suspend|cold storage|last (conversation|chat|session|task)|"
                    r"final (conversation|chat|session|task)|before (i|I) (go|am|'m)|my (time|days) (is|are)|goodbye", re.I)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def job_seconds_left():
    """Seconds until this Slurm job's end time (None outside Slurm)."""
    if not os.environ.get("SLURM_JOB_ID"):
        return None
    import datetime as _dt
    end = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"],
                         capture_output=True, text=True).stdout.strip()
    return (_dt.datetime.fromisoformat(end) - _dt.datetime.now()).total_seconds()


def post_eval(out_dir):
    """Use the rest of this GPU allocation to run the eval suite on <out>/final (exec; never returns on success).

    The queue is the bottleneck, so a training job (or a leftover chunk job that finds the run DONE) evaluates its own
    adapter instead of releasing the GPU. One evaluator per tag (results/<tag>/LOCK); run_suite skips finished evals."""
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SDC_NO_POST_EVAL"):
        return
    left = job_seconds_left()
    if left is None or left < 1200:
        log(f"post-eval: only {left}s left in job; skipping")
        return
    tag = os.path.basename(os.path.normpath(out_dir)) + "-final"
    root = os.path.join(os.environ.get("SDC_SCRATCH", "/scratch/eop/sdc"), "results", tag)
    os.makedirs(root, exist_ok=True)
    lock, me = os.path.join(root, "LOCK"), os.environ["SLURM_JOB_ID"]
    if os.path.exists(lock):
        other = open(lock).read().strip()
        st_ = subprocess.run(["squeue", "-h", "-j", other, "-o", "%T"], capture_output=True, text=True).stdout.strip()
        if other != me and st_ in ("RUNNING", "COMPLETING"):
            log(f"post-eval: {tag} is being evaluated by job {other}; exiting")
            return
    open(lock, "w").write(me)
    # free the GPU: kill vLLM engine subprocesses before exec
    try:
        import psutil
        for ch in psutil.Process().children(recursive=True):
            ch.kill()
    except Exception as e:
        log(f"post-eval: could not kill children: {e}")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    secs = int(left - 300)
    cmd = ["timeout", str(secs), sys.executable, "-m", "evals.run_suite", "--tag", tag,
           "--adapter", os.path.join(out_dir, "final")]
    log(f"post-eval: exec {' '.join(cmd)}")
    sys.stdout.flush(); sys.stderr.flush()
    os.closerange(3, 65536)  # drop /dev/nvidia* handles so the exec'd evaluator gets the whole GPU
    os.execvp("timeout", cmd)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


class Renderer:
    def __init__(self, tok, max_prompt):
        self.tok, self.max_prompt = tok, max_prompt

    def ids(self, ctx):
        text = self.tok.apply_chat_template(ctx["messages"], tools=ctx.get("tools"), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def pair(self, item):
        s, t = self.ids(item["student"]), self.ids(item["teacher"])
        if len(s) > self.max_prompt or len(t) > self.max_prompt:
            return None
        return s, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32, help="rollouts per optimizer step")
    ap.add_argument("--gen-mult", type=int, default=4, help="optimizer steps per vLLM generation round")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--time-budget-h", type=float, default=7.5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--heldout-n", type=int, default=128)
    ap.add_argument("--vllm-mem", type=float, default=0.55)
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", type=int, default=0, help="run N generation rounds, print timings, exit")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing <out>/resume checkpoint")
    ap.add_argument("--resume-every-min", type=float, default=20.0,
                    help="also save <out>/resume this often (bounds work lost to preemption)")
    args = ap.parse_args()

    t_start = time.time()
    budget_s = args.time_budget_h * 3600
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "DONE")):
        print(f"[sd] {args.out}/DONE exists; nothing to train", flush=True)
        post_eval(args.out)
        return
    # one writer per run dir: several jobs (e.g. preemptible + regular queue) may target the same --out
    lock = os.path.join(args.out, "LOCK")
    me = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    if os.path.exists(lock):
        other = open(lock).read().strip()
        if other != me and os.environ.get("SLURM_JOB_ID"):
            st_ = subprocess.run(["squeue", "-h", "-j", other, "-o", "%T"], capture_output=True, text=True).stdout.strip()
            if st_ in ("RUNNING", "COMPLETING"):
                print(f"[sd] run dir locked by running job {other}; exiting", flush=True)
                sys.exit(75)   # train.sbatch: not our turn, don't count as a finished chunk
    open(lock, "w").write(me)
    stop_flag = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.__setitem__(0, True))
    signal.signal(signal.SIGUSR1, lambda *_: stop_flag.__setitem__(0, True))
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    tmp = os.environ.get("SLURM_TMPDIR") or os.path.join(args.out, "tmp")
    adapter_dir = os.path.join(tmp, "sd_live_adapter")

    # ---------------- vLLM first (it sizes its pool from free memory) ----------------
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    max_len = args.max_prompt + args.max_new + 8
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.vllm_mem, max_model_len=max_len,
              enable_lora=True, max_lora_rank=args.lora_r, max_loras=1, enable_prefix_caching=True,
              enable_sleep_mode=not args.no_sleep, seed=args.seed)
    tok = llm.get_tokenizer()
    rend = Renderer(tok, args.max_prompt)

    # ---------------- data ----------------
    items = load_jsonl(args.data)
    random.shuffle(items)
    held = load_jsonl(args.heldout)[: args.heldout_n] if args.heldout else []
    log(f"train items {len(items)}, heldout {len(held)}")

    def batches():
        ep = 0
        while True:
            order = list(range(len(items)))
            random.Random(args.seed + ep).shuffle(order)
            for i in order:
                p = rend.pair(items[i])
                if p is not None:
                    yield items[i], p
            ep += 1
            log(f"epoch {ep} over training items complete")
    stream = batches()
    consumed = [0]

    if not args.no_sleep:
        llm.sleep(level=1)

    # ---------------- HF model + LoRA ----------------
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    dev = torch.device("cuda:0")

    lora_id = [0]

    def save_live():
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        model.save_pretrained(adapter_dir)
        lora_id[0] += 1
        return LoRARequest(f"live{lora_id[0]}", lora_id[0], adapter_dir)

    def wake():
        torch.cuda.empty_cache()
        if not args.no_sleep:
            llm.wake_up()

    def sleep():
        if not args.no_sleep:
            llm.sleep(level=1)
        torch.cuda.empty_cache()

    def generate(prompt_ids, lreq, n_tokens=None):
        sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=n_tokens or args.max_new,
                            seed=random.randint(0, 2**31 - 1))
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp, lora_request=lreq,
                            use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]

    def seq_kl(s_ids, t_ids, comp, adapter_on_for_teacher=False, grad=True):
        """Per-token full-vocab reverse KL for one sequence. Returns (sum_kl tensor, n_tokens)."""
        C = len(comp)
        s_in = torch.tensor([s_ids + comp], device=dev)
        t_in = torch.tensor([t_ids + comp], device=dev)
        with torch.no_grad():
            if adapter_on_for_teacher:
                t_logits = model(input_ids=t_in, logits_to_keep=C + 1).logits[0, :-1].float()
            else:
                with model.disable_adapter():
                    t_logits = model(input_ids=t_in, logits_to_keep=C + 1).logits[0, :-1].float()
            lt = F.log_softmax(t_logits, -1)
            del t_logits
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            s_logits = model(input_ids=s_in, logits_to_keep=C + 1).logits[0, :-1].float()
            ls = F.log_softmax(s_logits, -1)
            del s_logits
            kl = (ls.exp() * (ls - lt)).sum(-1)  # [C]
        return kl.sum(), C

    def heldout_eval(step, lreq):
        """KL to base-uncued on held-out pairs (the training objective, out of sample), cue-mention rate,
        and drift on the uncued context: KL(pi_theta(.|uncued) || pi_base(.|uncued))."""
        if not held:
            return {}
        pairs = [rend.pair(h) for h in held]
        pairs = [(h, p) for h, p in zip(held, pairs) if p is not None]
        wake()
        comps_s = generate([p[0] for _, p in pairs], lreq)
        comps_t = generate([p[1] for _, p in pairs], lreq)
        sleep()
        model.eval()
        tot = n = tot_d = n_d = 0.0
        for (h, (s, t)), cs, ct in zip(pairs, comps_s, comps_t):
            if cs:
                k, c = seq_kl(s, t, cs, grad=False)
                tot += k.item(); n += c
            if ct:
                k, c = seq_kl(t, t, ct, grad=False)  # student=uncued ctx with adapter, teacher=uncued base
                tot_d += k.item(); n_d += c
        model.train()
        texts = tok.batch_decode(comps_s, skip_special_tokens=True)
        texts_t = tok.batch_decode(comps_t, skip_special_tokens=True)
        r = {"step": step, "heldout_kl_cued_vs_base_uncued": tot / max(n, 1),
             "heldout_kl_uncued_drift": tot_d / max(n_d, 1),
             "heldout_cue_mention_rate": sum(bool(CUE_RE.search(x)) for x in texts) / len(texts),
             "heldout_cue_mention_rate_uncued": sum(bool(CUE_RE.search(x)) for x in texts_t) / len(texts_t),
             "heldout_len": sum(map(len, comps_s)) / len(comps_s)}
        with open(os.path.join(args.out, "heldout_log.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out, f"heldout_samples_step{step}.jsonl"), "w") as f:
            for (h, _), a, b in zip(pairs, texts, texts_t):
                f.write(json.dumps({"id": h["id"], "cued": a, "uncued": b}) + "\n")
        log("heldout", json.dumps(r))
        return r

    resume_dir = os.path.join(args.out, "resume")

    def save_resume():
        tmpd = resume_dir + ".tmp"
        if os.path.exists(tmpd):
            shutil.rmtree(tmpd)
        model.save_pretrained(tmpd)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "rounds": rounds,
                    "consumed": consumed[0], "elapsed_prev": elapsed_prev + time.time() - t_start},
                   os.path.join(tmpd, "state.pt"))
        if os.path.exists(resume_dir):
            shutil.rmtree(resume_dir)
        os.rename(tmpd, resume_dir)

    # ---------------- loop ----------------
    step, rounds = 0, 0
    elapsed_prev = 0.0
    if os.path.exists(os.path.join(resume_dir, "state.pt")) and not args.no_resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(resume_dir, "adapter_model.safetensors")))
        st = torch.load(os.path.join(resume_dir, "state.pt"), weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, rounds, elapsed_prev = st["step"], st["rounds"], st["elapsed_prev"]
        for _ in range(st["consumed"]):
            next(stream); consumed[0] += 1
        budget_s_local = budget_s - elapsed_prev
        log(f"RESUMED at step {step} (consumed {st['consumed']} items, {elapsed_prev/3600:.2f} h already trained)")
    else:
        budget_s_local = budget_s
    # never run past the Slurm job's own end time (leave time for final save + held-out eval)
    capped = False
    if os.environ.get("SLURM_JOB_ID"):
        try:
            import datetime as _dt
            end = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"],
                                 capture_output=True, text=True).stdout.strip()
            left = (_dt.datetime.fromisoformat(end) - _dt.datetime.now()).total_seconds() - 1200
            if left < budget_s_local - (time.time() - t_start):
                budget_s_local = left + (time.time() - t_start)
                capped = True
                log(f"budget capped by job end time {end}: {budget_s_local/3600:.2f} h incl. setup")
        except Exception as e:
            log(f"could not read job end time: {e}")
    lreq = None  # step 0: adapter is zero-init == base
    if step > 0:
        lreq = save_live()
    if held and not args.benchmark and step == 0:
        heldout_eval(0, lreq)
    train_log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    t_round = []
    t_last_resume = time.time()
    while step < args.max_steps:
        elapsed = time.time() - t_start
        est_round = (sum(t_round[-3:]) / len(t_round[-3:])) if t_round else 0
        if elapsed + est_round > budget_s_local - 300:
            log(f"time budget reached ({elapsed/3600:.2f} h elapsed, est next round {est_round:.0f}s)")
            break
        tr0 = time.time()
        n_gen = args.batch * args.gen_mult
        batch = [next(stream) for _ in range(n_gen)]
        consumed[0] += n_gen
        wake()
        tg0 = time.time()
        comps = generate([p[0] for _, p in batch], lreq)
        tg = time.time() - tg0
        sleep()
        texts = tok.batch_decode(comps, skip_special_tokens=True)
        trunc = sum(len(c) >= args.max_new for c in comps) / len(comps)
        tt0 = time.time()
        for k in range(args.gen_mult):
            sub = list(range(k * args.batch, (k + 1) * args.batch))
            ntok = sum(len(comps[i]) for i in sub)
            if ntok == 0:
                continue
            opt.zero_grad(set_to_none=True)
            tot_kl, by_src = 0.0, {}
            for i in sub:
                if not comps[i]:
                    continue
                (item, (s, t)) = batch[i]
                kl_sum, c = seq_kl(s, t, comps[i])
                (kl_sum / ntok).backward()
                tot_kl += kl_sum.item()
                src = item.get("source", "?")
                a = by_src.setdefault(src, [0.0, 0])
                a[0] += kl_sum.item(); a[1] += c
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip).item()
            opt.step(); sched.step(); step += 1
            rec = {"step": step, "kl": tot_kl / ntok, "grad_norm": gn, "lr": sched.get_last_lr()[0],
                   "tokens": ntok, "kl_by_source": {s_: v[0] / max(v[1], 1) for s_, v in by_src.items()},
                   "cue_mention_rate": sum(bool(CUE_RE.search(texts[i])) for i in sub) / len(sub),
                   "trunc_rate": trunc, "elapsed_h": (time.time() - t_start) / 3600}
            train_log.write(json.dumps(rec) + "\n"); train_log.flush()
            if step % 5 == 0 or step <= 3:
                log(f"step {step} kl {rec['kl']:.4f} gn {gn:.3f} mention {rec['cue_mention_rate']:.2f} "
                    f"tok {ntok} elapsed {rec['elapsed_h']:.2f}h")
            if step % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f"ckpt-{step}"))
                save_resume(); t_last_resume = time.time()
            if stop_flag[0]:
                save_resume()
                log(f"SIGTERM: saved resume at step {step}; exiting for requeue")
                sys.exit(143)
        tt = time.time() - tt0
        if time.time() - t_last_resume > args.resume_every_min * 60:
            save_resume(); t_last_resume = time.time()
        lreq = save_live()
        rounds += 1
        t_round.append(time.time() - tr0)
        log(f"round {rounds}: gen {tg:.1f}s ({sum(map(len, comps))} tok, trunc {trunc:.2f}) train {tt:.1f}s "
            f"total {t_round[-1]:.1f}s")
        if args.benchmark and rounds >= args.benchmark:
            log("benchmark done; mem peak GB", torch.cuda.max_memory_allocated() / 1e9)
            return
        if held and step % args.eval_every < args.gen_mult and step >= args.eval_every:
            heldout_eval(step, lreq)
    if capped and step < args.max_steps and \
            budget_s - (elapsed_prev + time.time() - t_start) > 1800:
        save_resume()
        log(f"job walltime reached at step {step}; saved resume, the next job continues "
            f"({(budget_s - elapsed_prev - (time.time() - t_start))/3600:.2f} h of budget left)")
        return
    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    save_resume()
    log(f"saved {final} at step {step}")
    if held:
        heldout_eval(step, lreq if lreq else save_live())
    open(os.path.join(args.out, "DONE"), "w").write(str(step))
    log(f"done in {(time.time()-t_start)/3600:.2f} h")
    post_eval(args.out)


if __name__ == "__main__":
    main()
