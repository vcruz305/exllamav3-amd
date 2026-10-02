#!/usr/bin/env python
"""Kernel breakdown of ONE GatedDeltaNet prefill forward (2048 rows) + ONE Attention forward,
via torch.profiler with record_function scopes around the module forward. Model loaded fully;
the timed region is a single Generator prefill chunk, profiled per module class."""
import os, time, collections, torch
from torch.profiler import profile, ProfilerActivity, record_function
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
TARGET = os.environ.get("TARGET", "GatedDeltaNet")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=16384); m.load(progressbar=False)
active = [False]
mods = set()
for b in m.modules:
    for s in getattr(b, "modules", []):
        if type(s).__name__ == TARGET: mods.add(type(s))
cls = next(iter(mods))
orig = cls.forward
seen = [0]
def fwd(self, *a, **k):
    if active[0] and seen[0] < 1 and a[0].numel() // a[0].shape[-1] >= 1024:
        seen[0] += 1
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            r = orig(self, *a, **k); torch.cuda.synchronize()
        ka = prof.key_averages()
        def dt(e): return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
        dev = sorted([e for e in ka if dt(e) > 0], key=dt, reverse=True)
        tot = sum(dt(e) for e in dev)
        print(f"== one {TARGET} forward, rows={a[0].numel() // a[0].shape[-1]}: device {tot / 1e3:.2f} ms")
        for e in dev[:30]:
            print(f"  {dt(e) / 1e3:8.3f} ms  n={e.count:4d}  {e.key[:120]}")
        if os.environ.get("STACKS"):
            print("-- device time by op + source stack (top 15)")
            ks = prof.key_averages(group_by_stack_n=6)
            for e in sorted(ks, key=lambda e: getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)), reverse=True)[:40]:
                t_ = getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0))
                if not e.key.startswith("aten::") or t_ < 200: continue
                st = [f for f in (e.stack or []) if "exllamav3" in f][:3]
                print(f"  {t_ / 1e3:7.3f} ms n={e.count:3d} {e.key:22s} {e.input_shapes if hasattr(e,'input_shapes') else ''}")
                for f in st: print(f"        {f}")
        return r
    return orig(self, *a, **k)
cls.forward = fwd
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
for seed in (1, 2):
    active[0] = seed == 2
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (1, 4096), generator=g, dtype=torch.long)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
