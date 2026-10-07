#!/usr/bin/env python
"""Which BLAS library serves fp16-in/fp32-out on gfx1151, and is hipBLASLt's HS kernel fast?"""
import os, time, torch
def bench(f, n=30):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
m, k, n = 2048, 2560, 10240
a = torch.randn(m, k, dtype=torch.float16, device="cuda"); b = torch.randn(k, n, dtype=torch.float16, device="cuda")
fl = 2 * m * k * n
print("default preferred_blas_library:", torch.backends.cuda.preferred_blas_library())
for lib in ("cublas", "cublaslt"):
    try:
        torch.backends.cuda.preferred_blas_library(lib)
    except Exception as e:
        print(lib, "unavailable:", e); continue
    t16 = bench(lambda: torch.mm(a, b))
    t32 = bench(lambda: torch.mm(a, b, out_dtype=torch.float32))
    ref = (a.float() @ b.float())
    err = ((torch.mm(a, b, out_dtype=torch.float32) - ref).abs().max() / ref.abs().max()).item()
    print(f"{lib:9} fp16-out {fl/t16/1e12:5.1f}  fp32-out {fl/t32/1e12:5.1f} TFLOP/s  (fp32-out rel err {err:.1e})")
