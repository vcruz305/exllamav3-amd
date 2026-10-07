#!/usr/bin/env python
"""Can Triton 3.8 do fp16-in / fp32-out GEMM fast on gfx1151 (what hipBLASLt can't)?"""
import time, torch, triton, triton.language as tl

@triton.autotune(configs=[
    triton.Config({"BM": bm, "BN": bn, "BK": bk, "G": 8}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [(128, 128, 32, 4, 2), (128, 128, 64, 8, 2), (256, 128, 32, 8, 2),
                             (128, 256, 32, 8, 2), (64, 128, 64, 4, 2), (128, 64, 64, 4, 2)]],
    key=["M", "N", "K"])
@triton.jit
def mm(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr,
       BK: tl.constexpr, G: tl.constexpr, OUT32: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (G * npn); fm = gid * G; gs = min(npm - fm, G)
    pm = fm + (pid % gs); pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = a + rm[:, None] * sam + rk[None, :] * sak
    bp = b + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(ap, mask=rk[None, :] < K - k * BK, other=0.0)
        y = tl.load(bp, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(x, y, acc)
        ap += BK * sak; bp += BK * sbk
    cp = c + rm[:, None] * scm + rn[None, :] * scn
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if OUT32:
        tl.store(cp, acc, mask=msk)
    else:
        tl.store(cp, acc.to(tl.float16), mask=msk)

def run(a, b, out):
    M, K = a.shape; N = b.shape[1]
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
    mm[grid](a, b, out, M, N, K, *a.stride(), *b.stride(), *out.stride(), OUT32=out.dtype == torch.float32)
    return out

def bench(f, n=30):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

for (m, k, n) in ((2048, 2560, 10240), (2048, 10240, 2560), (512, 2560, 10240), (2048, 2560, 4096)):
    a = torch.randn(m, k, dtype=torch.float16, device="cuda"); b = torch.randn(k, n, dtype=torch.float16, device="cuda")
    c32 = torch.empty(m, n, dtype=torch.float32, device="cuda"); c16 = torch.empty(m, n, dtype=torch.float16, device="cuda")
    fl = 2 * m * k * n
    ref = a.float() @ b.float()
    run(a, b, c32); err = ((c32 - ref).abs().max() / ref.abs().max()).item()
    t32 = bench(lambda: run(a, b, c32)); t16 = bench(lambda: run(a, b, c16))
    tb16 = bench(lambda: torch.mm(a, b))
    tv = bench(lambda: c32.copy_(torch.mm(a, b)))
    print(f"{m}x{k}x{n}: triton fp32-out {fl/t32/1e12:5.1f}  triton fp16-out {fl/t16/1e12:5.1f}  "
          f"| hipblas fp16-out {fl/tb16/1e12:5.1f}  via-f16+widen {fl/tv/1e12:5.1f} TFLOP/s  (fp32-out rel err {err:.1e})")
