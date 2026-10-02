#!/usr/bin/env bash
# Mixed-K vs flat-K A/B on framework2. Sequential, one model resident at a time.
# Usage: bash tools/strix_halo/mk_ab.sh [ntok]
set -u
cd ~/exllamav3-amd
source ./env.sh > /dev/null
export PYTHONPATH="$HOME/exllamav3-amd"
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}"
export EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
N="${1:-256}"
P="Explain gradient descent in two sentences:"
FLAT=~/models/Qwen3.8-Flash-Next-EXL3
MIX=~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw
B=tools/strix_halo/bench_mtp.py

run() {  # label model extra-args...
    local label="$1"; shift; local model="$1"; shift
    local out
    out=$(timeout 1500 python "$B" -m "$model" -c 2048 -n "$N" -g -p "$P" "$@" 2>&1)
    local rate acc routes
    rate=$(echo "$out" | grep -oE "decode: +[0-9.]+ tok/s" | grep -oE "[0-9.]+" | head -1)
    acc=$(echo "$out" | grep -oE "acceptance=[0-9.]+%" | head -1)
    routes=$(echo "$out" | grep -c "grouped HIP route")
    dense=$(echo "$out" | grep -c "dense per-expert path")
    printf "%-34s %8s tok/s  %-18s grouped_layers=%s dense_layers=%s\n" "$label" "${rate:-FAIL}" "${acc:--}" "$routes" "$dense"
    if [ -z "$rate" ]; then echo "$out" | tail -15; fi
}

echo "== $(date +%T)  ntok=$N  EXL3_MOE_CFG=$EXL3_MOE_CFG PREFILL_MIN_ROWS=$EXL3_HIP_PREFILL_MIN_ROWS"
run "FLAT-K  no-MTP"            "$FLAT" --no-mtp
run "FLAT-K  MTP ndt3 dds dc0.6" "$FLAT" -ndt 3 -dds -dc 0.6
run "MIXED-K no-MTP"            "$MIX"  --no-mtp
run "MIXED-K MTP ndt3 dds dc0.6" "$MIX"  -ndt 3 -dds -dc 0.6
echo "== done $(date +%T)"
