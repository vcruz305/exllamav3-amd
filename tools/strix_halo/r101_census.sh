#!/usr/bin/env bash
# Decode kernel census, ROCm 7.0 vs 10.1, both packs. Sequential.
set -u
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 NTOK=${NTOK:-128}
for P in ${PACKS:-~/models/Qwen3.8-Flash-Next-EXL3 ~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw}; do
  for A in r7 r101; do
    unset LD_PRELOAD
    if [ $A = r7 ]; then T=~/exllamav3-amd; PY=$T/.venv/bin/python; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
    else T=~/exl3-r101; PY=~/exllamav3-amd/.venv-r101/bin/python; fi
    echo "################ $A $(basename $P)"
    cd $T; MODEL=$P PYTHONPATH=$T timeout 600 $PY tools/strix_halo/decode_census.py 2>&1 | grep -vE "UserWarning|warnings.warn|d = torch|Resource leak|STAGE:|Mixed-K experts|Uniform non-K3|^\s*$" | head -50
  done
done
echo DONE_CENSUS
