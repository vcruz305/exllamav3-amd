#!/usr/bin/env python
"""Sliced EXL3 lm_head vs full head on real activations: decode-shape (1 row) and 4 rows, both
through LinearEXL3.forward (HIP GEMV path) and through reconstruct_hgemm."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); m.load(progressbar=False)
inner = m.modules[m.logit_layer_idx].inner
N = int(os.environ.get("N", "65536"))
sub = LinearEXL3(None, inner.in_features, N, suh=inner.suh, svh=inner.svh[:N].contiguous(),
                 trellis=inner.trellis[:, :N // 16, :].contiguous(), mcg=inner.mcg_tensor,
                 mul1=inner.mul1_tensor, out_dtype=inner.out_dtype)
print("trellis", tuple(inner.trellis.shape), "->", tuple(sub.trellis.shape), "K", inner.K, sub.K, "mcg", inner.mcg, "mul1", inner.mul1)
torch.manual_seed(0)
for rows in (1, 2, 4, 16):
    x = torch.randn(rows, inner.in_features, dtype=torch.half, device=inner.trellis.device)
    full = inner.forward(x, {}).float()[:, :N]
    s = sub.forward(x, {}).float()
    rf = inner.reconstruct_hgemm(x, None).float()[:, :N]
    rs = sub.reconstruct_hgemm(x, None).float()
    print(f"rows {rows:2d}: forward max|d| {(s - full).abs().max():.3e}  argmax agree {(s.argmax(-1) == full.argmax(-1)).float().mean():.2f}   "
          f"recon max|d| {(rs - rf).abs().max():.3e}   full fwd vs recon {(full - rf).abs().max():.3e}")
