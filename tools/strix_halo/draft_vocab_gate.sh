#!/usr/bin/env bash
# Gate for the default reduced-vocab MTP draft head (EXL3_MTP_DRAFT_VOCAB=98304, no env set):
# six-prompt both packs at the default, batched throughput (bsz>1 through the sliced head),
# and a recipe-style sampled run.
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH=$PWD EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
unset EXL3_MTP_DRAFT_VOCAB
M=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
echo "## six-prompt, default (no EXL3_MTP_DRAFT_VOCAB in env)"
NDT=3 DC=0.6 DDS=1 NTOK=512 LABEL=flat-default timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean"
NDT=3 DC=0.6 DDS=1 NTOK=512 MODEL=$M CACHE=4096 LABEL=cf-default timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean"
echo "## CYBER-FROST full head, same session (reference)"
EXL3_MTP_DRAFT_VOCAB=0 NDT=3 DC=0.6 DDS=1 NTOK=512 MODEL=$M CACHE=4096 LABEL=cf-full timeout 1800 python tools/strix_halo/prompt_sweep.py 2>&1 | grep -E "^mean"
echo "## batched (batch_throughput.py, b=5 ndt=2), full head vs default"
for v in 0 98304; do
    r=$(EXL3_MTP_DRAFT_VOCAB=$v BATCHES=5 NDT=2 NTOK=512 timeout 900 python tools/strix_halo/batch_throughput.py 2>&1 | grep -E "^ *5 " | tail -1 | tr '\n' ' ')
    echo "DRAFT_VOCAB=$v :: $r"
done
echo "## sampled (recipe default temp), 3 runs each"
for v in 0 98304; do
    for k in 1 2 3; do
        r=$(EXL3_MTP_DRAFT_VOCAB=$v timeout 300 python tools/strix_halo/bench_mtp.py -n 256 -ndt 3 -dds -dc 0.6 -p "Describe the difference between a process and a thread:" 2>&1 | grep -E "decode:|acceptance" | tr '\n' ' ')
        echo "DRAFT_VOCAB=$v run$k :: $r"
    done
done
echo DONE_DV_GATE
