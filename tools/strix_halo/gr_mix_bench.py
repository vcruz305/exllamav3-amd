#!/usr/bin/env python
"""Micro-bench + parity for the prefill GatedResidual mix: torch path vs Triton gr_norm/gr_tail.
Runs on synthetic weights of the real shape (H=4, D=2560, rank=320) -- no model load needed."""
import time, torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hc_triton import gr_norm, gr_tail

dev = torch.device("cuda:0"); torch.manual_seed(0)
H, D, rank = 4, 2560, 320
w_h = (torch.randn((H * D,), device=dev) * 0.1 + 1.0).half()
proj_h = (torch.randn((rank + H, H * D), device=dev) * 0.01).half()
up_h = (torch.randn((H * D, rank), device=dev) * 0.05).half()
eps = 1e-6

def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

def ref(s3, R):
    normed = torch.empty((R * H, D), dtype=torch.half, device=dev)
    ext.rms_norm(s3.view(R * H, D), w_h, normed, eps, 0.0, 1.0, False, False, H)
    dm = torch.matmul(normed.view(R, H * D), proj_h.t())
    t = F.silu(dm[:, :rank] / H)
    post = 2.0 * torch.sigmoid(dm[:, rank:].float() / H)
    g = torch.matmul(t, up_h.t())
    mixed = (torch.sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(dim=-2).half()
    return normed, post, mixed

def new(s3, R):
    normed = torch.empty((R * H, D), dtype=torch.half, device=dev)
    gr_norm(s3.view(R * H, D), w_h, H, eps, normed)
    dm = torch.matmul(normed.view(R, H * D), proj_h.t())
    t = F.silu(dm[:, :rank] / H)
    post = 2.0 * torch.sigmoid(dm[:, rank:].float() / H)
    g = torch.matmul(t, up_h.t())
    mixed = gr_tail(g, normed, R, H, D)
    return normed, post, mixed

ok = True
for R in (33, 512, 2048):
    s3 = torch.randn((R, H, D), device=dev) * 3.0
    a = ref(s3, R); b = new(s3, R)
    for name, x, y in zip(("normed", "post", "mixed"), a, b):
        d = (x.float() - y.float()).abs().max().item(); mx = x.float().abs().max().item()
        rel = d / max(mx, 1e-6); ok &= rel < 2e-3
        print(f"R={R:5d} {name:6s} max|d|={d:.3e} rel={rel:.2e}")
    tn = bench(lambda: ext.rms_norm(s3.view(R * H, D), w_h, torch.empty((R * H, D), dtype=torch.half, device=dev), eps, 0.0, 1.0, False, False, H))
    normed = a[0]
    tt = bench(lambda: gr_norm(s3.view(R * H, D), w_h, H, eps, torch.empty((R * H, D), dtype=torch.half, device=dev)))
    g = torch.randn((R, H * D), device=dev).half()
    ta = bench(lambda: (torch.sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(dim=-2).half())
    tb = bench(lambda: gr_tail(g, normed, R, H, D))
    print(f"R={R:5d}  norm: ext {tn:.3f} ms -> triton {tt:.3f} ms   tail: torch {ta:.3f} ms -> triton {tb:.3f} ms   "
          f"full: {bench(lambda: ref(s3, R)):.2f} -> {bench(lambda: new(s3, R)):.2f} ms")
print("GR PARITY", "PASS" if ok else "FAIL")
