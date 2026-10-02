#!/usr/bin/env bash
# Concurrency + context sweep, flat-K vs mixed-K, on one build. Sequential (one model resident).
# Log lines starting with '#' are section headers; the rest are harness output.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH="$HOME/exllamav3-amd" EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
FLAT=~/models/Qwen3.8-Flash-Next-EXL3
MIX=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
WHAT="${1:-batch}"

if [ "$WHAT" = batch ]; then
    # README's synthetic table: ndt=2, greedy, 512 tok/seq, fp16 cache 16384
    for spec in "FLAT|$FLAT" "MIXED|$MIX"; do
        IFS="|" read label model <<< "$spec"
        echo "# batch sweep  $label"
        MODEL=$model NDT=2 NTOK=512 CACHE=16384 BATCHES=1,3,4,5,6,8 \
            timeout 3000 python tools/strix_halo/batch_throughput.py 2>&1 \
            | grep -vE "experts in|UserWarning|d = torch|Resource leak"
    done
fi

if [ "$WHAT" = ctx ]; then
    # Max context that loads, then decode at depth with a cold (random-id) prompt.
    # CS sizes probe the ceiling; FILL=1 adds one prompt filling the cache to the brim.
    for spec in "FLAT|$FLAT" "MIXED|$MIX"; do
        IFS="|" read label model <<< "$spec"
        for cfg in "fp16|131072|" "q4|262144|4"; do
            IFS="|" read kind cs cq <<< "$cfg"
            echo "# ctx  $label  $kind cache=$cs"
            if [ -n "$cq" ]; then export CQ=$cq; else unset CQ; fi
            MODEL=$model CS=$cs PROMPTS=8192,32768 FILL=1 RANDOM_PROMPT=1 NTOK=128 \
                timeout 3000 python tools/strix_halo/ctx_sweep.py 2>&1 | grep "^{"
        done
    done
    unset CQ
fi
echo "DONE_$WHAT"
