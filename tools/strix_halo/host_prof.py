#!/usr/bin/env python
"""cProfile of the MTP decode loop (host side), sorted by own time and cumulative time."""
import os, sys, time, cProfile, pstats, io
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
def run(n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
run(16); torch.cuda.synchronize()
pr = cProfile.Profile(timer=time.perf_counter)
t0 = time.perf_counter(); pr.enable(); run(NTOK); torch.cuda.synchronize(); pr.disable(); wall = time.perf_counter() - t0
print(f"wall {wall:.3f}s ({NTOK/wall:.1f} tok/s, profiled)")
for key, n in (("tottime", 35), ("cumtime", 45)):
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(n)
    out = s.getvalue().replace(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/", "")
    print(f"===== by {key}"); print("\n".join(l[:180] for l in out.splitlines()[4:]))
