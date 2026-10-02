#!/usr/bin/env python
"""Parity + speed: norm_triton kernels vs the ext_fallbacks torch implementations they front."""
import time, itertools, torch
from exllamav3 import ext_fallbacks as fb
from exllamav3 import norm_triton as nt
dev = torch.device("cuda:0"); torch.manual_seed(0)

def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

def cmp(a, b):
    a = a.float(); b = b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()

fails = 0
# rms_norm: dtypes x weight dtype x residual x groups x span_heads; real dims
for rows, dim, xd, yd, wd, res, groups, span in [
    (2048, 2560, torch.float32, torch.float16, torch.float16, False, 1, False),
    (2048, 2560, torch.float16, torch.float16, torch.float16, False, 1, False),
    (2048, 2560, torch.float32, torch.float32, torch.bfloat16, True, 1, False),
    (8192, 2560, torch.float32, torch.float16, torch.float16, False, 4, False),
    (2048 * 16, 128, torch.float16, torch.float16, torch.bfloat16, False, 1, False),
    (3, 2560, torch.float16, torch.float16, None, False, 1, False),
    (2048, 256, torch.float16, torch.float16, torch.float16, False, 1, True),
]:
    shape = (rows // 2, 2, dim // 2) if span else (rows, dim)
    x = torch.randn(shape, device=dev).to(xd) * 3
    w = (torch.randn((groups * dim,), device=dev) * 0.2 + 1).to(wd) if wd else None
    y0 = torch.randn(shape, device=dev).to(yd); y1 = y0.clone()
    fb.rms_norm(x, w, y0, 1e-6, 0.0, 1.0, span, res, groups)
    assert nt.rms_norm(x, w, y1, 1e-6, 0.0, 1.0, span, res, groups) is not False
    e = cmp(y1, y0); ok = e < 2e-3; fails += not ok
    yt = torch.empty_like(y0)
    tf = bench(lambda: fb.rms_norm(x, w, yt, 1e-6, 0.0, 1.0, span, False, groups))
    tt = bench(lambda: nt.rms_norm(x, w, yt, 1e-6, 0.0, 1.0, span, False, groups))
    print(f"rms_norm      {str(tuple(x.shape)):16s} {str(xd)[6:]:>7}->{str(yd)[6:]:7} w={str(wd)[6:] if wd else '-':8} res={res!s:5} g={groups} span={span!s:5} rel={e:.1e} {'OK' if ok else 'FAIL'}  {tf:6.3f} -> {tt:6.3f} ms")

for rows, dim, xd in [(2048, 2560, torch.float16), (2048, 2560, torch.float32), (7, 2560, torch.float16)]:
    x = torch.randn((rows, dim), device=dev).to(xd)
    w = (torch.randn((dim,), device=dev) * 0.2 + 1).half()
    r0 = torch.randn((rows, dim), device=dev).half(); r1 = r0.clone()
    y0 = torch.empty((rows, dim), device=dev, dtype=torch.half); y1 = torch.empty_like(y0)
    fb.rms_norm_res_in(x, w, y0, r0, 1e-6, 0.0, 1.0)
    assert nt.rms_norm_res_in(x, w, y1, r1, 1e-6, 0.0, 1.0) is not False
    e = max(cmp(y1, y0), cmp(r1, r0)); ok = e < 2e-3; fails += not ok
    print(f"rms_norm_res_in ({rows}, {dim}) {str(xd)[6:]:>7} rel={e:.1e} {'OK' if ok else 'FAIL'}")

for rows, dim, yd, gd, wd, gf, act, groups in [
    (2048 * 48, 128, torch.float16, torch.bfloat16, torch.bfloat16, False, 0, 1),
    (2048 * 48, 128, torch.float16, torch.float32, torch.float32, False, 0, 1),
    (2048 * 48, 128, torch.float32, torch.bfloat16, torch.bfloat16, True, 1, 4),
    (5 * 48, 128, torch.float16, torch.bfloat16, torch.bfloat16, False, 0, 1),
]:
    x = (torch.randn((rows, dim), device=dev) * 2).bfloat16()
    g = torch.randn((rows, dim), device=dev).to(gd)
    w = (torch.randn((groups * dim,), device=dev) * 0.2 + 1).to(wd)
    y0 = torch.empty((rows, dim), device=dev, dtype=yd); y1 = torch.empty_like(y0)
    fb.gated_rms_norm(x, w, y0, g, 1e-6, 0.0, groups, gf, act)
    assert nt.gated_rms_norm(x, w, y1, g, 1e-6, 0.0, groups, gf, act) is not False
    e = cmp(y1, y0); ok = e < 2e-3; fails += not ok
    tf = bench(lambda: fb.gated_rms_norm(x, w, y0, g, 1e-6, 0.0, groups, gf, act))
    tt = bench(lambda: nt.gated_rms_norm(x, w, y1, g, 1e-6, 0.0, groups, gf, act))
    print(f"gated_rms_norm ({rows}, {dim}) y={str(yd)[6:]:7} g={str(gd)[6:]:8} first={gf!s:5} act={act} groups={groups} rel={e:.1e} {'OK' if ok else 'FAIL'}  {tf:6.3f} -> {tt:6.3f} ms")
print("NORM PARITY", "PASS" if fails == 0 else f"FAIL ({fails})")
