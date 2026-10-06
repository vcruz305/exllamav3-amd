#!/usr/bin/env bash
# 10.1 launches cost more host time per kernel (decode_census: 3.8-5.0 vs 2.1 us per
# hipLaunchKernel). Does graph replay (EXL3_BLOCK_GRAPH=1, neutral on ROCm 7) pay on 10.1?
set -u
T=~/exl3-r101; PY=~/exllamav3-amd/.venv-r101/bin/python; cd $T
export PYTHONPATH=$T EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 NDT=3 DC=0.6 DDS=1 NTOK=512
for g in 0 1 0 1; do
    r=$(EXL3_BLOCK_GRAPH=$g LABEL=bg$g timeout 1800 $PY tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean +[0-9]")
    echo "r101 flat EXL3_BLOCK_GRAPH=$g  $r"
done
echo DONE_BG
