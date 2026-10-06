#!/usr/bin/env bash
# ROCm 7.0 (production .venv + shipped .so) vs ROCm 10.1 (.venv-r101 + ~/exl3-r101 build) A/B.
# Same commit, same harnesses, same knobs; strictly sequential (one GPU job at a time).
#   bash tools/strix_halo/r101_ab.sh  > /tmp/r101_ab.log
# Each arm prints: env check, GEMM micro (incl. the fp32-out HSS case), PPL flat/mixed,
# prefill 8k flat/mixed, six-prompt decode flat/mixed.
set -u
export PATH="$HOME/.local/bin:$PATH"
F=~/models/Qwen3.8-Flash-Next-EXL3
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
ARMS="${ARMS:-r7 r101}"

arm_env() {   # sets PY, TREE and runtime env for an arm, in a clean environment
    unset LD_PRELOAD ROCM_HOME ROCM_PATH HIP_PATH HIP_DEVICE_LIB_PATH HIPCC_COMPILE_FLAGS_APPEND
    case $1 in
        r7)   TREE=~/exllamav3-amd; PY=$TREE/.venv/bin/python
              export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 ;;   # Trap 1 crutch
        r101) TREE=~/exl3-r101;     PY=~/exllamav3-amd/.venv-r101/bin/python
              [ "${R101_PRELOAD:-0}" = 1 ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 ;;
    esac
    export PYTHONPATH=$TREE EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
    cd $TREE
}

gemm_micro() {
$PY - <<'PY'
import torch, time
def bench(f, n=30):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
for (m, k, n) in ((4096, 4096, 4096), (2048, 2560, 10240), (2048, 10240, 2560), (512, 2560, 10240)):
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")
    o32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
    t16 = bench(lambda: torch.mm(a, b))
    # fp16-in / fp32-out: the HSS hipBLASLt path that EXL3_HIP_F32OUT_VIA_F16 works around
    try:
        t32 = bench(lambda: torch.mm(a, b, out_dtype=torch.float32))
    except Exception as ex:
        print(f"  fp32-out probe unavailable ({type(ex).__name__}); falling back to addmm into fp32")
        t32 = bench(lambda: torch.addmm(o32, a.float(), b.float(), beta=0))
    fl = 2 * m * k * n
    print(f"gemm {m}x{k}x{n}: fp16-out {fl/t16/1e12:5.1f} TFLOP/s   fp32-out {fl/t32/1e12:5.1f} TFLOP/s")
    # The extension's own hipblasGemmEx path (what reconstruct_hgemm / LinearFP16 call):
    # c dtype selects HHS (fp16 out) vs HSS (fp32 out). HSS is what F32OUT_VIA_F16 avoids.
    import exllamav3_ext as ext
    c16 = torch.empty(m, n, dtype=torch.float16, device="cuda")
    e16 = bench(lambda: ext.hgemm(a, b, c16))
    e32 = bench(lambda: ext.hgemm(a, b, o32))
    print(f"ext.hgemm {m}x{k}x{n}: fp16-out {fl/e16/1e12:5.1f} TFLOP/s   fp32-out(HSS) {fl/e32/1e12:5.1f} TFLOP/s")
PY
}

for A in $ARMS; do
    echo "################ ARM $A  ($(date +%T))"
    arm_env $A
    $PY -c "import torch, exllamav3_ext as e; print('torch', torch.__version__, 'hip', torch.version.hip, '| ext', e.__file__, '| family', e.exl3_gemv_wmma_family(0))" 2>&1 | tail -1
    echo "## gemm micro"; gemm_micro 2>&1 | grep -E "^gemm|^ext.hgemm|unavailable|Error" | head -12
    echo "## ppl (-r 20 -l 1024)"
    for spec in "FLAT|$F|1" "FLAT|$F|0" "MIXED|$M|1"; do
        IFS="|" read l m v <<< "$spec"
        r=$(EXL3_HIP_F32OUT_VIA_F16=$v timeout 3000 $PY eval/ppl.py -m "$m" -r 20 -l 1024 2>&1 | grep -oE "Perplexity: [0-9.]+" | tail -1)
        echo "$A $l F32OUT_VIA_F16=$v  ${r:-FAILED}"
    done
    echo "## prefill 8k (cold random prompt, chunk 2048)"
    for spec in "FLAT|$F" "MIXED|$M"; do
        IFS="|" read l m <<< "$spec"
        r=$(MODEL=$m PROMPT=8192 CS=16384 SYNC=0 timeout 900 $PY tools/strix_halo/prefill_profile.py 2>&1 | grep -E "^wall")
        echo "$A $l prompt=8192  ${r:-FAILED}"
    done
    echo "## decode six-prompt (NDT=3 DC=0.6 DDS=1 512 tok, greedy)"
    export NDT=3 DC=0.6 DDS=1 NTOK=512
    LABEL=$A-flatK timeout 1800 $PY tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean|acceptance|Error" | head -4
    MODEL=$M CACHE=4096 LABEL=$A-mixedK timeout 1800 $PY tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean|acceptance|Error" | head -4
    unset NDT DC DDS NTOK
done
echo DONE_R101_AB
