#!/usr/bin/env python
"""GPU busy vs idle during MTP decode, from a rocprofv3 kernel trace (low overhead, no torch
profiler). Union of kernel intervals vs wall span of the decode window: idle = what launch/host
overhead and syncs cost. Also the gap histogram, so we can see if idle is many small launch gaps
(graph-able) or a few big host stalls (sync points)."""
import csv, sys, glob, collections
f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("/tmp/rp_decode/**/*kernel_trace.csv", recursive=True))[-1]
rows = []
with open(f) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
# decode window: the last DECODE_S seconds of the trace (bench_mtp prints the decode wall time);
# fallback = last 70% of kernels by count
import os
DS = float(os.environ.get("DECODE_S", "0"))
if DS > 0:
    tend = max(e for _, e, _ in rows)
    rows = [r for r in rows if r[0] >= tend - DS * 1e9]
else:
    rows = rows[int(len(rows) * 0.3):]
t0, t1 = rows[0][0], max(e for _, e, _ in rows)
busy, cur_s, cur_e = 0, rows[0][0], rows[0][1]
gaps = []
for s, e, _ in rows[1:]:
    if s > cur_e:
        busy += cur_e - cur_s; gaps.append(s - cur_e); cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
span = t1 - t0
print(f"trace: {f}")
print(f"kernels in window: {len(rows)}   span {span/1e6:.1f} ms   busy {busy/1e6:.1f} ms ({100*busy/span:.1f} %)   idle {(span-busy)/1e6:.1f} ms")
bins = [(0, 2e3), (2e3, 5e3), (5e3, 10e3), (10e3, 50e3), (50e3, 200e3), (200e3, 1e12)]
print("idle gaps by size:")
for lo, hi in bins:
    g = [x for x in gaps if lo <= x < hi]
    print(f"  {lo/1e3:7.0f}-{hi/1e3 if hi < 1e11 else float('inf'):>7.0f} us: n={len(g):7d}  total {sum(g)/1e6:8.1f} ms")
print(f"mean kernel duration {sum(e - s for s, e, _ in rows)/len(rows)/1e3:.1f} us")
