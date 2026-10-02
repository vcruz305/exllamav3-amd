#!/usr/bin/env python
"""Kernel-level trace of prefill: torch.profiler over one timed 8k cold prefill (4 x 2048 chunks).
Prints the top device kernels by total time, and the top CPU ops by self time.
  MODEL=... PROMPT=8192 CHUNK=2048 python prefill_trace.py
"""
import os, time, torch
from torch.profiler import profile, ProfilerActivity
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
PROMPT = int(os.environ.get("PROMPT", "8192")); CHUNK = int(os.environ.get("CHUNK", "2048"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=16384); m.load(progressbar=False)
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=CHUNK)
def run(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (1, PROMPT), generator=g, dtype=torch.long)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    torch.cuda.synchronize(); t = time.perf_counter()
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
    torch.cuda.synchronize(); return time.perf_counter() - t
run(1)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    wall = run(2)
print(f"wall {wall:.2f} s  {PROMPT / wall:.0f} tok/s")
ka = prof.key_averages()
dev = [e for e in ka if getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) > 0]
def dt(e): return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
tot = sum(dt(e) for e in dev)
print(f"device kernel time captured: {tot / 1e6:.2f} s = {100 * tot / 1e6 / wall:.1f} % of wall")
for e in sorted(dev, key=dt, reverse=True)[:40]:
    print(f"  {dt(e) / 1e3:9.1f} ms {100 * dt(e) / 1e6 / wall:5.1f} %  n={e.count:6d}  {e.key[:110]}")
print("CPU self time top 20:")
for e in sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:20]:
    print(f"  {e.self_cpu_time_total / 1e3:9.1f} ms  n={e.count:6d}  {e.key[:100]}")
