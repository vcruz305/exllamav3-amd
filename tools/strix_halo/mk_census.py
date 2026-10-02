#!/usr/bin/env python
"""Per-token extension call census + which Linear K values hit reconstruct_hgemm."""
import os, sys, time, collections, torch
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw")
import exllamav3_ext as ext
import exllamav3.modules.quant.exl3 as q
c = collections.Counter()
for n in ("exl3_gemv", "hgemm_recon", "had_r_128", "reconstruct", "hgemm", "exl3_moe_mk",
          "exl3_moe_mk_prefill", "exl3_moe_gfx12_k3", "exl3_moe_gfx12_k3_prefill"):
    f = getattr(ext, n, None)
    if f is None: continue
    def mk(name, fn):
        def w(*a, **k):
            c[name] += 1; return fn(*a, **k)
        return w
    setattr(ext, n, mk(n, f))
q.ext = ext
recon_k = collections.Counter()
_orig = q.LinearEXL3.reconstruct_hgemm
def rh(self, x, out_dtype):
    recon_k[(self.K, x.numel() // x.shape[-1])] += 1
    return _orig(self, x, out_dtype)
q.LinearEXL3.reconstruct_hgemm = rh
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=2048); m.load(progressbar=False)
g = Generator(model=m, cache=cache, tokenizer=tok)
j = Job(input_ids=tok.encode("Hi", add_bos=True), max_new_tokens=4); g.enqueue(j)
while g.num_remaining_jobs():
    for r in g.iterate(): pass
torch.cuda.synchronize(); c.clear(); recon_k.clear()
N = 8
t0 = time.time()
j = Job(input_ids=tok.encode("Yo", add_bos=True), max_new_tokens=N); g.enqueue(j)
while g.num_remaining_jobs():
    for r in g.iterate(): pass
torch.cuda.synchronize()
dt = (time.time() - t0) / N * 1000
print(f"wall {dt:.1f} ms/token  (no MTP, greedy-free generator default)")
for k, v in c.most_common():
    print(f"  {k:26s} {v / N:8.1f} calls/token")
print("reconstruct_hgemm by (K, rows):", {k: round(v / N, 1) for k, v in sorted(recon_k.items())})
