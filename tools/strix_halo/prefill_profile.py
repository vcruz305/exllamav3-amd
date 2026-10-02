#!/usr/bin/env python
"""Prefill profile: where does one long prompt's time go, per block component?

Wraps forward() of every class that is a direct child of a TransformerBlock (plus the top-level
modules) with a device sync on each side, then prefills a cold random-id prompt of PROMPT
tokens in CHUNK-sized chunks through the Generator. Second pass is the timed one.
Also counts extension calls by name inside the timed pass.

  MODEL=~/models/... PROMPT=8192 CHUNK=2048 python prefill_profile.py
"""
import os, sys, time, collections, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
PROMPT = int(os.environ.get("PROMPT", "8192"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
CS = int(os.environ.get("CS", "16384"))
SYNC = os.environ.get("SYNC", "1") == "1"

import exllamav3_ext as ext
calls = collections.Counter()
ext_t = collections.defaultdict(float)
active = [False]
import time
for n in dir(ext):
    f = getattr(ext, n)
    if not callable(f) or n.startswith("_"): continue
    def mk(name, fn):
        def w(*a, **k):
            if not active[0]: return fn(*a, **k)
            calls[name] += 1
            torch.cuda.synchronize(); t = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize(); ext_t[name] += time.perf_counter() - t
            return r
        return w
    setattr(ext, n, mk(n, f))
# modules import ext by name; patch their references too
import importlib, pkgutil, exllamav3
for mi in pkgutil.walk_packages(exllamav3.__path__, "exllamav3."):
    try: mod = importlib.import_module(mi.name)
    except Exception: continue
    if getattr(mod, "ext", None) is not None and getattr(mod.ext, "__name__", "") == "exllamav3_ext":
        mod.ext = ext

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=CS); m.load(progressbar=False)

times = collections.defaultdict(float); counts = collections.Counter()
wrapped = set()
def wrap_cls(cls):
    if cls in wrapped: return
    wrapped.add(cls)
    orig = cls.forward
    def fwd(self, *a, **k):
        if not active[0]: return orig(self, *a, **k)
        if SYNC: torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(self, *a, **k)
        if SYNC: torch.cuda.synchronize()
        times[cls.__name__] += time.perf_counter() - t; counts[cls.__name__] += 1
        return r
    cls.forward = fwd
for mod in m.modules:
    if type(mod).__name__ == "TransformerBlock":
        for s in mod.modules: wrap_cls(type(s))
    elif type(mod).__name__ != "Linear":
        wrap_cls(type(mod))

# Method-level timers for code paths blocks call directly (not via forward)
import exllamav3.modules.hyperconnections as HCm
def wrap_method(cls, name, label):
    orig = getattr(cls, name)
    def f(self, *a, **k):
        if not active[0]: return orig(self, *a, **k)
        if SYNC: torch.cuda.synchronize()
        t = time.perf_counter(); r = orig(self, *a, **k)
        if SYNC: torch.cuda.synchronize()
        times[label] += time.perf_counter() - t; counts[label] += 1
        return r
    setattr(cls, name, f)
wrap_method(HCm.GatedResidual, "_mix", "GatedResidual._mix")
wrap_method(HCm.GatedResidual, "apply_", "GatedResidual.apply_")
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=CHUNK)
def run(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (1, PROMPT), generator=g, dtype=torch.long)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    torch.cuda.synchronize(); t = time.perf_counter()
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
    torch.cuda.synchronize(); return time.perf_counter() - t

run(1)  # warm
times.clear(); counts.clear(); calls.clear(); ext_t.clear()
active[0] = True
wall = run(2)
active[0] = False
print(f"MODEL {os.path.basename(MODEL)}  prompt {PROMPT}  chunk {CHUNK}  sync={SYNC}")
print(f"wall {wall:.2f} s  ->  {PROMPT / wall:.0f} tok/s")
tot = sum(times.values())
for k, v in sorted(times.items(), key=lambda x: -x[1]):
    print(f"  {k:24s} {v:8.3f} s  {100 * v / wall:5.1f} %   calls {counts[k]:5d}   {1000 * v / max(counts[k], 1):8.2f} ms/call")
print(f"  {'(unattributed)':24s} {wall - tot:8.3f} s  {100 * (wall - tot) / wall:5.1f} %")
et = sum(ext_t.values())
print(f"ext device time (synced per call): {et:.2f} s = {100 * et / wall:.1f} % of wall; rest = torch ops + host")
for k, v in sorted(ext_t.items(), key=lambda x: -x[1])[:25]:
    print(f"  {k:34s} {v:7.3f} s  {100 * v / wall:5.1f} %  calls {calls[k]:6d}  {1000 * v / calls[k]:8.3f} ms/call")
