#!/usr/bin/env bash
# Phase 2 of the ROCm 7.0 vs 10.1 A/B: PPL on the 10.1 arm (phase 1 lacked `datasets`), then
# interleaved decode repeats (r7, r101, r7, r101) on both packs so drift cancels, then a
# per-module prefill profile on both arms to locate the 10.1 prefill gain.
set -u
export PATH="$HOME/.local/bin:$PATH"
F=~/models/Qwen3.8-Flash-Next-EXL3
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
arm() {
    unset LD_PRELOAD
    case $1 in
        r7)   TREE=~/exllamav3-amd; PY=$TREE/.venv/bin/python; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 ;;
        r101) TREE=~/exl3-r101;     PY=~/exllamav3-amd/.venv-r101/bin/python ;;
    esac
    export PYTHONPATH=$TREE EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2; cd $TREE
}
echo "## ppl r101 (-r 20 -l 1024)"
arm r101
for spec in "FLAT|$F|1" "FLAT|$F|0" "MIXED|$M|1" "FLAT-torchnorm|$F|0|x"; do
    IFS="|" read l m v x <<< "$spec"
    if [ -n "${x:-}" ]; then export EXL3_TRITON_NORM=0 EXL3_GR_TRITON=0; fi
    out=$(EXL3_HIP_F32OUT_VIA_F16=$v timeout 3000 $PY eval/ppl.py -m "$m" -r 20 -l 1024 2>&1)
    r=$(echo "$out" | grep -oE "Perplexity: [0-9.]+" | tail -1)
    echo "r101 $l F32OUT_VIA_F16=$v  ${r:-FAILED: $(echo "$out" | grep -E "Error|error" | tail -1)}"
    unset EXL3_TRITON_NORM EXL3_GR_TRITON x
done
echo "(r7 reference: FLAT 1 = 4.218831, FLAT 0 = 4.230441, MIXED 1 = 4.261538, torch-norm FLAT 0 = 4.225935)"

echo "## decode repeats, interleaved (NDT=3 DC=0.6 DDS=1 512 tok, greedy)"
export NDT=3 DC=0.6 DDS=1 NTOK=512
for rep in 1 2; do
  for A in r7 r101; do
    arm $A
    f=$(LABEL=$A timeout 1800 $PY tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean +[0-9]" | head -1)
    m=$(MODEL=$M CACHE=4096 LABEL=$A timeout 1800 $PY tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean +[0-9]" | head -1)
    echo "rep$rep $A flat:  $f"
    echo "rep$rep $A mixed: $m"
  done
done
unset NDT DC DDS NTOK

echo "## prefill per-module profile, 8k flat (SYNC=1 per-module timing; wall from SYNC=0)"
for A in r7 r101; do
    arm $A
    echo "--- $A"
    MODEL=$F PROMPT=8192 CS=16384 SYNC=1 timeout 900 $PY tools/strix_halo/prefill_profile.py 2>&1 | grep -vE "UserWarning|warnings.warn" | head -40
done
echo DONE_R101_AB2
