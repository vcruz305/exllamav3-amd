#!/usr/bin/env python
"""
Per-round timeline of MTP decode: where does wall time go between the verify forward and the
next one? Instruments (host perf_counter, with a device sync only at the points the generator
already blocks on) the draft loop, the verify launch, the readback in receive_sample, the
post-verify draft prefill, and the remainder. Requires no profiler; overhead is a few us.
"""
import os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.generator.job as J

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
NTOK = int(os.environ.get("NTOK", "256"))
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
draft = Model.from_config(cfg, component="mtp")
cache = Cache(model, max_num_tokens=4096, max_history=3); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=4096, max_history=3); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=3, dynamic_draft_tokens=True, draft_confidence=0.6)
T = collections.defaultdict(float); N = collections.Counter(); ON = [False]
def timed(obj, name, label):
    orig = getattr(obj, name)
    def f(*a, **k):
        if not ON[0]: return orig(*a, **k)
        t = time.perf_counter(); r = orig(*a, **k); T[label] += time.perf_counter() - t; N[label] += 1; return r
    setattr(obj, name, f)
timed(gen, "iterate_draftmodel_mtp_gen", "A draft loop (host, incl. per-step D2H of ids)")
timed(model, "forward", "B verify forward (host issue)")
timed(J.Job, "receive_sample", "C receive_sample (first one blocks on verify)")
timed(draft, "prefill", "D draft prefill (post-verify)")
timed(gen, "iterate_gen", "  iterate_gen total (B+C+D+rest)")
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)
def run(n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())); k = 0
    while gen.num_remaining_jobs():
        k += 1
        for _ in gen.iterate(): pass
    return k
run(16); torch.cuda.synchronize()
ON[0] = True; t0 = time.perf_counter(); rounds = run(NTOK); torch.cuda.synchronize(); wall = time.perf_counter() - t0; ON[0] = False
print(f"wall {wall*1000:.0f} ms  {NTOK/wall:.2f} tok/s  rounds {rounds}  ({wall/rounds*1000:.2f} ms/round)")
for k in sorted(T):
    print(f"  {k:52} {T[k]*1000:8.1f} ms  {T[k]/rounds*1000:6.2f} ms/round  n={N[k]}")
