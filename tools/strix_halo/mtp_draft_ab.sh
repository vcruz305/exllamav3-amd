#!/usr/bin/env bash
# MTP drafter A/B: pinned embedding upload (EXL3_MTP_EMB_PINNED) and vocab-prefix draft head
# (EXL3_MTP_DRAFT_VOCAB). Six-prompt greedy, production default (NDT=3 DC=0.6 DDS=1), ROCm 7
# production stack. Interleaved so drift cancels. Output tokens must be identical across configs
# (the target verifies every draft); greedy_ab.py checks that.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH=$PWD EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 NDT=3 DC=0.6 DDS=1 NTOK=512
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
run() {  # label, env...
    local l=$1; shift
    r=$(env "$@" LABEL=$l timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean")
    echo "$l $* :: $(echo "$r" | tr '\n' ' ')"
}
echo "## flat 3.05 bpw"
for rep in 1 2; do
    run base   EXL3_MTP_EMB_PINNED=0 EXL3_MTP_DRAFT_VOCAB=0
    run pin    EXL3_MTP_EMB_PINNED=1 EXL3_MTP_DRAFT_VOCAB=0
    run pin+v98k EXL3_MTP_EMB_PINNED=1 EXL3_MTP_DRAFT_VOCAB=98304
    run pin+v64k EXL3_MTP_EMB_PINNED=1 EXL3_MTP_DRAFT_VOCAB=65536
    run pin+v32k EXL3_MTP_EMB_PINNED=1 EXL3_MTP_DRAFT_VOCAB=32768
done
echo "## mixed-K CYBER-FROST"
for rep in 1; do
    run base   MODEL=$M CACHE=4096 EXL3_MTP_EMB_PINNED=0 EXL3_MTP_DRAFT_VOCAB=0
    run pin+v64k MODEL=$M CACHE=4096 EXL3_MTP_EMB_PINNED=1 EXL3_MTP_DRAFT_VOCAB=65536
done
echo DONE_MTP_AB
