#!/usr/bin/env python
"""Per-family kernel time per forward from a rocprofv3 kernel-trace CSV of verify_hostdev.py
(last NFWD forwards of the back-to-back phase = the last DECODE_S seconds)."""
import csv, sys, os, re, collections
f = sys.argv[1]; nfwd = int(os.environ.get("NFWD", "30")); ds = float(os.environ.get("DECODE_S", "1.0"))
rows = []
with open(f) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort(); tend = max(e for _, e, _ in rows); rows = [r for r in rows if r[0] >= tend - ds * 1e9]
def family(k):
    k = k.replace("void ", "").replace("(anonymous namespace)::", "").replace("exl3_moe_mk_ns::", "")
    k = re.sub(r"\(.*", "", k)
    for p in ("moe_mk", "moe_prefill_grouped_gemv", "moe_grouped_gemv", "moe_prefill_had_rows", "moe_prefill_metadata",
              "exl3_gemv_kernel", "gr_dots", "gr_finalize", "gr_mix", "hc_apply", "had_hf_r_128", "had_ff_r_128",
              "recurrent_gated_delta_rule", "causal_conv1d", "conv1d_update", "gated_rms_norm", "_rms_norm",
              "routing", "topk", "radixSort", "sort", "paged_attn", "paged_kv", "qsa", "mla_plane", "dsa_topk", "rope",
              "skinny", "Cijk_", "copyBuffer", "fillBuffer", "elementwise", "reduce_kernel", "index", "gather",
              "state_rewind", "conv_rewind", "silu", "sigmoid"):
        if p in k: return p
    return k[:40]
span = rows[-1][1] - rows[0][0]
fam = collections.defaultdict(lambda: [0, 0]); busy = 0
for s, e, n in rows:
    fam[family(n)][0] += 1; fam[family(n)][1] += e - s; busy += e - s
# forwards in window ~ span / per-forward time; report per ms of span scaled to a forward
per = span / 1e6
print(f"window {per:.1f} ms, kernels {len(rows)}, busy {busy/1e6:.1f} ms ({100*busy/span:.1f} %)")
for k, (c, d) in sorted(fam.items(), key=lambda kv: -kv[1][1])[:22]:
    print(f"  {k:28} {100*d/span:5.1f} % of wall   n={c:6d}   {d/c/1e3:7.1f} us/call")
