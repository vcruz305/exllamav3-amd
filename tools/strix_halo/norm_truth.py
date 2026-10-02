#!/usr/bin/env python
"""Is the Triton norm / mix less accurate than the torch path it replaces, or just rounded
differently? Score both against an fp64 ground truth on real-shape inputs."""
import torch
import torch.nn.functional as F
from exllamav3 import ext_fallbacks as fb
from exllamav3 import norm_triton as nt
from exllamav3.modules.hc_triton import gr_norm, gr_tail
dev = torch.device("cuda:0"); torch.manual_seed(0)

def score(name, out_ref, out_tr, truth):
    t = truth.double()
    er = (out_ref.double() - t).abs(); et = (out_tr.double() - t).abs()
    flips = (out_ref != out_tr).float().mean().item()
    print(f"{name:34s} mean|err| torch {er.mean().item():.3e}  triton {et.mean().item():.3e}   "
          f"max torch {er.max().item():.2e} triton {et.max().item():.2e}   differing outputs {100 * flips:.2f} %")

# rms_norm, fp32 in -> fp16 out, 4-group weight (the GatedResidual shape) and plain (model RMSNorm)
for rows, dim, groups in ((8192, 2560, 4), (2048, 2560, 1)):
    x = torch.randn((rows, dim), device=dev) * 3
    w = (torch.randn((groups * dim,), device=dev) * 0.2 + 1).half()
    y0 = torch.empty((rows, dim), device=dev, dtype=torch.half); y1 = torch.empty_like(y0)
    fb.rms_norm(x, w, y0, 1e-6, 0.0, 1.0, False, False, groups)
    nt.rms_norm(x, w, y1, 1e-6, 0.0, 1.0, False, False, groups)
    xd = x.double(); wd = w.double().view(groups, dim)
    truth = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6) * wd[torch.arange(rows, device=dev) % groups]
    score(f"rms_norm ({rows},{dim}) g={groups}", y0, y1, truth)

# gated_rms_norm, bf16 x / bf16 gate -> fp16 (the GatedDeltaNet output norm)
rows, dim = 2048 * 48, 128
x = (torch.randn((rows, dim), device=dev) * 2).bfloat16(); g = torch.randn((rows, dim), device=dev).bfloat16()
w = (torch.randn((dim,), device=dev) * 0.2 + 1).bfloat16()
y0 = torch.empty((rows, dim), device=dev, dtype=torch.half); y1 = torch.empty_like(y0)
fb.gated_rms_norm(x, w, y0, g, 1e-6, 0.0, 1, False, 0)
nt.gated_rms_norm(x, w, y1, g, 1e-6, 0.0, 1, False, 0)
xd = x.double(); gd = g.double()
truth = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6) * w.double() * (gd * torch.sigmoid(gd))
score("gated_rms_norm (98304,128)", y0, y1, truth)

# GatedResidual tail: mean_h(sigmoid(g) * normed), fp16 in -> fp16 out
R, H, D = 2048, 4, 2560
gg = torch.randn((R, H * D), device=dev).half(); nn_ = (torch.randn((R * H, D), device=dev)).half()
t0 = (torch.sigmoid(gg.float()).view(R, H, D) * nn_.float().view(R, H, D)).mean(dim=-2).half()
t1 = gr_tail(gg, nn_, R, H, D)
truth = (torch.sigmoid(gg.double()).view(R, H, D) * nn_.double().view(R, H, D)).mean(dim=-2)
score("gr_tail (2048, 4x2560)", t0, t1, truth)
