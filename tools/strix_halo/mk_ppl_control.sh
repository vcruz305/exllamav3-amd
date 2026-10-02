#!/usr/bin/env bash
# Is the flat-K PPL shift (4.225935 -> 4.223205) caused by the mixed-K work or pre-existing?
# Controls: OLD .so (pre-change backup) vs NEW .so, x EXL3_HIP_F32OUT_VIA_F16 on/off,
# x repo python vs pristine HEAD python (git stash of block_sparse_mlp/exl3 edits not needed:
# the flat pack never enters the new route, but the .so swap isolates the kernel side).
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH="$HOME/exllamav3-amd" EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
M=~/models/Qwen3.8-Flash-Next-EXL3
SO=exllamav3_ext.cpython-312-x86_64-linux-gnu.so
SITE=.venv/lib/python3.12/site-packages/$SO
cp -p $SO /tmp/new.so
ppl() { timeout 3000 python eval/ppl.py -m "$M" -r 20 -l 1024 2>&1 | grep -oE "Perplexity: [0-9.]+" | tail -1; }

for so in new old; do
    if [ $so = old ]; then cp -p ~/ab/so_backup/root.so $SO; cp -p ~/ab/so_backup/root.so $SITE;
    else cp -p /tmp/new.so $SO; cp -p /tmp/new.so $SITE; fi
    for f16 in 1 0; do
        r=$(EXL3_HIP_F32OUT_VIA_F16=$f16 ppl)
        echo "so=$so  F32OUT_VIA_F16=$f16  $r  ($(md5sum $SO | cut -c1-8))"
    done
done
# restore the new build
cp -p /tmp/new.so $SO; cp -p /tmp/new.so $SITE
echo "restored: $(md5sum $SO | cut -c1-8)"
