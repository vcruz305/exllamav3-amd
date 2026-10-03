#!/usr/bin/env python
"""Dense-layer prefill GEMMs (hgemm_recon): shape census + achieved TFLOP/s per shape, and the
time of ngram_gather_cpu, on one 8k cold prefill."""
import os, time, collections, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
import exllamav3_ext as ext
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.quant.exl3 as Q
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=16384); m.load(progressbar=False)
on = [False]; st = collections.defaultdict(lambda: [0, 0.0])
orig = ext.hgemm_recon
def h(a, b, c):
    if not on[0]: return orig(a, b, c)
    torch.cuda.synchronize(); t = time.perf_counter(); r = orig(a, b, c); torch.cuda.synchronize()
    k = (a.shape[0], a.shape[1], b.shape[1], str(c.dtype)[6:]); st[k][0] += 1; st[k][1] += time.perf_counter() - t
    return r
ext.hgemm_recon = h; Q.ext = ext
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
for seed in (1, 2):
    on[0] = seed == 2
    ids = torch.randint(1000, 100000, (1, 8192), generator=torch.Generator().manual_seed(seed), dtype=torch.long)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
tot = sum(v[1] for v in st.values())
print(f"hgemm_recon total {tot:.2f} s")
for (M, K, N, dt), (n, t) in sorted(st.items(), key=lambda x: -x[1][1])[:14]:
    print(f"  M={M:5d} K={K:6d} N={N:6d} out={dt:7s} n={n:4d}  {1000 * t / n:6.3f} ms/call  {2 * M * K * N / (t / n) / 1e12:5.1f} TFLOP/s  {t:5.2f} s")
