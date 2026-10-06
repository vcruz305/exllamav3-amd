#!/usr/bin/env bash
# Does any 10.1 BLAS backend fix fp16-in/fp32-out (HSS) on gfx1151? If yes, the
# EXL3_HIP_F32OUT_VIA_F16 workaround (fp16 rounding of GDN projections) could be retired.
set -u
cd ~/exl3-r101
PY=~/exllamav3-amd/.venv-r101/bin/python
run() {
env "$@" PYTHONPATH=$PWD $PY - <<'PY' 2>&1 | grep -E "^ext|^torch|Error" | head -4
import torch, time, exllamav3_ext as ext
def bench(f, n=30):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
m, k, n = 2048, 2560, 10240
a = torch.randn(m, k, dtype=torch.float16, device="cuda"); b = torch.randn(k, n, dtype=torch.float16, device="cuda")
c16 = torch.empty(m, n, dtype=torch.float16, device="cuda"); c32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
fl = 2 * m * k * n
e16 = bench(lambda: ext.hgemm(a, b, c16)); e32 = bench(lambda: ext.hgemm(a, b, c32))
v = bench(lambda: (ext.hgemm(a, b, c16), c32.copy_(c16)))
print(f"ext.hgemm fp16-out {fl/e16/1e12:5.1f}  fp32-out(HSS) {fl/e32/1e12:5.1f}  via-f16+widen {fl/v/1e12:5.1f} TFLOP/s")
t32 = bench(lambda: torch.mm(a, b, out_dtype=torch.float32))
print(f"torch.mm out_dtype=fp32 {fl/t32/1e12:5.1f} TFLOP/s")
PY
}
echo "## default";                         run X=1
echo "## ROCBLAS_USE_HIPBLASLT=0 (Tensile)"; run ROCBLAS_USE_HIPBLASLT=0
echo "## ROCBLAS_USE_HIPBLASLT=1";           run ROCBLAS_USE_HIPBLASLT=1
echo "## TORCH_BLAS_PREFER_HIPBLASLT=0";     run TORCH_BLAS_PREFER_HIPBLASLT=0
echo "## HIPBLASLT_ENABLE_TUNING / heuristics off"; run HIPBLASLT_TUNING_OVERRIDE_FILE= HIPBLASLT_DISABLE_HEURISTICS=1
echo DONE_BLAS
