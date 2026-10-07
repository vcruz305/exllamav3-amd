#!/usr/bin/env python
"""Coverage of a contiguous low-id prefix [0, N) of the vocab (BPE merge order puts frequent
tokens at low ids), on calibration text and on the target model's own greedy outputs, plus the
timing of an EXL3 lm_head sliced to that prefix (trellis/svh sliced on whole 128-col blocks)."""
import os, sys, time, collections, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NS = [int(x) for x in os.environ.get("NS", "32768,49152,65536,81920,98304,131072,163840").split(",")]
cfg = Config.from_directory(MODEL); tok = Tokenizer.from_config(cfg)
ids = []
for f in sorted(glob.glob(os.path.join(REPO, "exllamav3/conversion/standard_cal_data/*.utf8"))):
    ids += tok.encode(open(f, encoding="utf-8").read(), add_bos=False).view(-1).tolist()
ids = torch.tensor(ids)
print(f"calibration tokens {ids.numel()}  max id {int(ids.max())}")
# Model's own outputs: the generated text of the six sweep prompts, if a dump exists
gen_ids = None
p = os.environ.get("GEN_IDS")
if p and os.path.exists(p):
    gen_ids = torch.load(p).view(-1); print(f"generated tokens {gen_ids.numel()}")

m = Model.from_config(cfg); m.load(progressbar=False)
lm = m.modules[m.logit_layer_idx]; inner = lm.inner
dev = inner.trellis.device
def bench(f, n=50):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
x = torch.randn(1, inner.in_features, dtype=torch.half, device=dev)
t_full = bench(lambda: inner.forward(x, {}))
ref = inner.forward(x, {}).float()
print(f"full EXL3 head ({inner.out_features} cols, K={inner.K}): {t_full:.3f} ms")
for N in NS:
    cov = (ids < N).float().mean().item()
    gcov = (gen_ids < N).float().mean().item() if gen_ids is not None else float("nan")
    sub = LinearEXL3(None, inner.in_features, N, suh=inner.suh, svh=inner.svh[:N].contiguous(),
                     trellis=inner.trellis[:, :N // 16, :].contiguous(), mcg=inner.mcg_tensor,
                     mul1=inner.mul1_tensor, out_dtype=inner.out_dtype)
    y = sub.forward(x, {}).float()
    err = (y - ref[:, :N]).abs().max().item()
    t = bench(lambda: sub.forward(x, {}))
    print(f"prefix N={N:6d}: cal coverage {100*cov:6.2f} %  gen coverage {100*gcov:6.2f} %   "
          f"EXL3 sub-head {t:.3f} ms (vs {t_full:.3f})  max|d| vs full {err:.2e}")
