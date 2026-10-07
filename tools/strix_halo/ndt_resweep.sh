#!/usr/bin/env bash
# Draft window re-sweep after the reduced-vocab draft head made drafting ~3x cheaper: the old
# optimum (ndt=3 dyn dc=0.6) was tuned against the full head. Six-prompt greedy, flat pack.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH=$PWD EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 NTOK=512
for cfg in "3 0.6" "4 0.6" "4 0.5" "3 0.5" "4 0.7" "5 0.6" "3 0.6"; do
    set -- $cfg
    r=$(NDT=$1 DC=$2 DDS=1 LABEL=ndt$1dc$2 timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean" | tr '\n' ' ')
    echo "ndt=$1 dc=$2 :: $r"
done
echo DONE_NDT_SWEEP
