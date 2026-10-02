#!/usr/bin/env bash
# Prefill-change verification on framework2: parity, PPL (both packs), decode regression
# (six-prompt sweep), prefill tok/s at 8k / 32k (both packs). Sequential.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH="$HOME/exllamav3-amd" EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
F=~/models/Qwen3.8-Flash-Next-EXL3
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
q() { grep -vE "experts in|UserWarning|d = torch|Resource leak|POSIX|include|from /|note:|\^|^ *[0-9]+ \|"; }
echo "## parity"
timeout 600 python tools/strix_halo/mk_parity.py 2>&1 | grep -E "PARITY|worst"
timeout 300 python tools/strix_halo/norm_parity.py 2>&1 | grep -E "PARITY"
timeout 300 python tools/strix_halo/gr_mix_bench.py 2>&1 | grep -E "PARITY"
timeout 300 python tools/strix_halo/conv_parity.py 2>&1 | grep -E "PARITY"
echo "## ppl"
for spec in "FLAT|$F|1" "FLAT|$F|0" "MIXED|$M|1"; do
    IFS="|" read l m v <<< "$spec"
    r=$(EXL3_HIP_F32OUT_VIA_F16=$v timeout 3000 python eval/ppl.py -m "$m" -r 20 -l 1024 2>&1 | grep -oE "Perplexity: [0-9.]+" | tail -1)
    echo "$l F32OUT_VIA_F16=$v  $r"
done
echo "## prefill (cold random prompt, chunk 2048)"
for spec in "FLAT|$F|8192|16384" "MIXED|$M|8192|16384" "FLAT|$F|32768|40960" "MIXED|$M|32768|40960"; do
    IFS="|" read l m p c <<< "$spec"
    r=$(MODEL=$m PROMPT=$p CS=$c SYNC=0 timeout 900 python tools/strix_halo/prefill_profile.py 2>&1 | grep -E "^wall")
    echo "$l prompt=$p  $r"
done
echo "## decode six-prompt (NDT=3 DC=0.6 DDS=1 512 tok)"
export NDT=3 DC=0.6 DDS=1 NTOK=512
LABEL=flatK timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean|acceptance"
MODEL=$M CACHE=4096 LABEL=mixedK timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean|acceptance"
echo "## DONE"
