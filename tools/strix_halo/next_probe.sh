#!/usr/bin/env bash
# Grounding for the next-steps plan: (1) Triton fp32-out GEMM vs hipBLASLt, (2) GPU idle during
# real MTP decode (rocprofv3 kernel trace) on both toolchains.
set -u
cd ~/exllamav3-amd
echo "## [1] Triton fp16-in/fp32-out GEMM (ROCm 10.1 / Triton 3.8)"
timeout 600 .venv-r101/bin/python tools/strix_halo/triton_hs_gemm.py 2>&1 | grep -E "^[0-9]+x|Error" | head -8
echo "## [1b] same on ROCm 7 (pytorch-triton-rocm)"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 timeout 600 .venv/bin/python tools/strix_halo/triton_hs_gemm.py 2>&1 | grep -E "^[0-9]+x|Error" | head -8

echo "## [2] GPU idle during MTP decode (bench_mtp -g, 256 tok, ndt3 dds dc0.6)"
RP=$(ls .venv-r101/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3 2>/dev/null || command -v rocprofv3)
echo "rocprofv3: $RP"
for A in r7 r101; do
    rm -rf /tmp/rp_decode_$A
    if [ $A = r7 ]; then T=~/exllamav3-amd; PY=$T/.venv/bin/python; PRE=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
    else T=~/exl3-r101; PY=~/exllamav3-amd/.venv-r101/bin/python; PRE=; fi
    (cd $T && env ${PRE:+LD_PRELOAD=$PRE} PYTHONPATH=$T EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 timeout 600 \
        $RP --kernel-trace --output-format csv -d /tmp/rp_decode_$A -- \
        $PY tools/strix_halo/bench_mtp.py -g -n 256 -ndt 3 -dds -dc 0.6 \
        -p "Explain gradient descent in two sentences:" 2>&1 | grep -E "decode:|Error" | head -3)
    echo "--- $A"
    .venv-r101/bin/python tools/strix_halo/gpu_idle.py "$(find /tmp/rp_decode_$A -name '*kernel_trace.csv' | head -1)" 2>&1 | tail -10
done
echo DONE_NEXT_PROBE
