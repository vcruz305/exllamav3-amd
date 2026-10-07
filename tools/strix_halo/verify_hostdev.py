#!/usr/bin/env python
"""
Verify-forward host vs device, without the torch profiler: for the isolated R-row forward,
measure (1) device wall via events, (2) host issue time (perf_counter around the call, no sync),
(3) device wall when the host is made irrelevant by queueing K forwards back to back before
syncing (if (3) per forward << (1), the forward is host-issue bound and the GPU starves).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
R = int(os.environ.get("R", "4")); NFWD = int(os.environ.get("NFWD", "30"))
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=4096, max_history=R - 1); model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, num_draft_tokens=R - 1)
captured = {}
orig = model.forward
def cap(input_ids, params=None):
    if input_ids.shape[-1] == R and "x" not in captured:
        captured["x"] = input_ids.clone(); captured["p"] = params
    return orig(input_ids, params)
model.forward = cap
gen.enqueue(Job(input_ids=tok.encode("Explain gradient descent in two sentences:", add_bos=True),
                max_new_tokens=64, sampler=GreedySampler()))
for _ in range(6): gen.iterate()
with torch.inference_mode():
    gen.iterate_gen([], torch.zeros((1, R - 1), dtype=torch.long))
model.forward = orig
x = captured["x"]; P = captured["p"]; rs = P.get("recurrent_states")
keep = {k: v for k, v in P.items() if k not in ("dev_cache", "export_states")}

@torch.inference_mode()
def one():
    for r in rs: r.last_history = 0
    t = time.perf_counter(); y = orig(x, dict(keep)); h = time.perf_counter() - t
    for r in rs: r.rewind(R - 1)
    return h

for _ in range(5): one()
torch.cuda.synchronize()
# (1)+(2): normal cadence, sync each forward (like decode, which syncs every round)
dev, host = [], []
for _ in range(NFWD):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); h = one(); e1.record(); torch.cuda.synchronize()
    dev.append(e0.elapsed_time(e1)); host.append(h * 1000)
dev.sort(); host.sort()
print(f"R={R} synced each forward: device wall median {dev[len(dev)//2]:.2f} ms   host issue median {host[len(host)//2]:.2f} ms")
# (3) back to back (host runs ahead as far as the queue allows)
torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
t0 = time.perf_counter(); e0.record()
hs = [one() for _ in range(NFWD)]
e1.record(); th = time.perf_counter() - t0; torch.cuda.synchronize(); tt = time.perf_counter() - t0
print(f"R={R} back-to-back x{NFWD}: device {e0.elapsed_time(e1)/NFWD:.2f} ms/fwd   host issue {sum(hs)/NFWD*1000:.2f} ms/fwd   "
      f"host returned after {th*1000/NFWD:.2f} ms/fwd, done after {tt*1000/NFWD:.2f} ms/fwd")
