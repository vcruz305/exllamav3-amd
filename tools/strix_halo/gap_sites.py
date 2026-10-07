#!/usr/bin/env python
"""Big GPU-idle gaps in a rocprofv3 kernel trace, with the kernels on either side, so each gap
can be tied to a host sync point. Also the per-forward launch-gap structure."""
import csv, sys, os, collections, re
f = sys.argv[1]; DS = float(os.environ.get("DECODE_S", "3.0")); MIN = float(os.environ.get("MIN_US", "50"))
rows = []
with open(f) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort(); tend = max(e for _, e, _ in rows); rows = [r for r in rows if r[0] >= tend - DS * 1e9]
def short(n):
    n = re.sub(r"\(.*", "", n).replace("void ", "").replace("(anonymous namespace)::", "")
    return re.sub(r"<.*", "", n)[:40]
pairs = collections.defaultdict(lambda: [0, 0.0])
end = rows[0][1]; prev = rows[0][2]
for s, e, n in rows[1:]:
    g = s - end
    if g > MIN * 1e3:
        k = (short(prev), short(n)); pairs[k][0] += 1; pairs[k][1] += g
    if e > end: end = e; prev = n
tot = sum(v[1] for v in pairs.values())
print(f"gaps > {MIN:.0f} us in last {DS}s: {sum(v[0] for v in pairs.values())}  total {tot/1e6:.1f} ms")
print(f"{'n':>6} {'ms':>8} {'us/gap':>8}  before  ->  after")
for (a, b), (n, t) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"{n:6d} {t/1e6:8.1f} {t/n/1e3:8.1f}  {a:40} -> {b}")
