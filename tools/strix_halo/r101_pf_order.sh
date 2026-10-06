#!/usr/bin/env bash
# Prefill order control: the first A/B always ran r7 first on the same seeded random prompt, so
# r101 may have inherited a warm page cache for the disk-backed PLE table (ngram_gather_cpu).
# Alternate the order, repeat, and add a cold (drop_caches) pair.
set -u
F=~/models/Qwen3.8-Flash-Next-EXL3
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
pf() {   # arm pack label
    unset LD_PRELOAD
    if [ $1 = r7 ]; then T=~/exllamav3-amd; PY=$T/.venv/bin/python; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
    else T=~/exl3-r101; PY=~/exllamav3-amd/.venv-r101/bin/python; fi
    cd $T
    r=$(EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 PYTHONPATH=$T MODEL=$2 PROMPT=8192 CS=16384 SYNC=0 \
        timeout 900 $PY tools/strix_halo/prefill_profile.py 2>&1 | grep -E "^wall")
    echo "$3 $1 $(basename $2 | cut -c1-12): ${r:-FAILED}"
}
echo "## warm, alternating"
for o in "r101 r7" "r7 r101" "r101 r7"; do for a in $o; do pf $a $F warm; done; done
echo "## cold page cache before each run (drop_caches)"
for a in r101 r7 r101 r7; do sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null && pf $a $F cold; done
echo "## mixed pack, warm alternating"
for a in r101 r7 r7 r101; do pf $a $M warm; done
echo DONE_PF_ORDER
