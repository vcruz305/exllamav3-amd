#!/usr/bin/env python
"""
Is MTP decode host-launch-bound? For every target forward (verify) and draft step, measure the
host time to *issue* it (model.forward wall, no sync) against its device time (events). If issue
time >= device time, the GPU is fed no faster than Python can launch, and removing syncs alone
cannot reach the gap-free ceiling.
"""
import os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
NTOK = int(os.environ.get("NTOK", "256")); NDT = int(os.environ.get("NDT", "3"))
config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=4096, max_history=NDT); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=4096, max_history=NDT); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6)
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)

rec = collections.defaultdict(list)
ON = [False]
def wrap(m, label, meth):
    orig = getattr(m, meth)
    def f(*a, **k):
        if not ON[0]: return orig(*a, **k)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); t = time.perf_counter(); r = orig(*a, **k); host = time.perf_counter() - t; e1.record()
        rec[label].append((host, e0, e1)); return r
    setattr(m, meth, f)
wrap(model, "target.forward", "forward")
wrap(draft, "draft.forward", "forward")
wrap(draft, "draft.prefill", "prefill")
wrap(draft, "draft.sample_from_state", "sample_from_state")

def run(n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
    k = 0
    while gen.num_remaining_jobs():
        k += 1
        for _ in gen.iterate(): pass
    return k
run(16); torch.cuda.synchronize()
ON[0] = True
t0 = time.perf_counter(); rounds = run(NTOK); torch.cuda.synchronize(); wall = time.perf_counter() - t0
ON[0] = False
print(f"wall {wall:.3f}s  {NTOK/wall:.2f} tok/s  rounds {rounds}  {1000*wall/rounds:.2f} ms/round")
tot_h = tot_d = 0
for k, v in rec.items():
    h = sum(x[0] for x in v) * 1000; d = sum(x[1].elapsed_time(x[2]) for x in v)
    tot_h += h; tot_d += d
    print(f"{k:24} n={len(v):5d}  host issue {h:8.1f} ms ({h/len(v):6.2f}/call)   device {d:8.1f} ms ({d/len(v):6.2f}/call)   issue/device {h/d:5.2f}")
print(f"sum: host issue {tot_h:.0f} ms   device {tot_d:.0f} ms   of wall {wall*1000:.0f} ms")
