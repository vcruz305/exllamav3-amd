#!/usr/bin/env python
"""What bounds the grouped MoE prefill GEMV? Same number of assignments (rows*10), routed
(a) the real way, (b) concentrated onto few experts (weights read ~once, A traffic unchanged),
(c) spread uniformly over all 512 experts. If (b) ~ (a), weight traffic / decode is not the
bound -- A-operand traffic or latency is."""
import os, time, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
from exllamav3 import Config, Model, Cache
import exllamav3.modules.block_sparse_mlp as B
import exllamav3_ext as ext
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg)
cache = Cache(m, max_num_tokens=4096); m.load(progressbar=False)
layer = next(s for b in m.modules for s in getattr(b, "modules", []) if type(s).__name__ == "BlockSparseMLP")
name = "exl3_moe_mk_prefill" if layer.support_hip_mk_prefill else "exl3_moe_gfx12_k3_prefill"
orig = getattr(ext, name); cap = {}
def c(*a): cap["a"] = a; return orig(*a)
setattr(ext, name, c); B.ext = ext
dev = torch.device("cuda:0"); torch.manual_seed(0)
rows = 2048
x = torch.randn((1, rows, layer.hidden_size), device=dev, dtype=torch.half) * 0.5
with torch.inference_mode(): layer.forward(x, {})
a = list(cap["a"])
# locate selected (int64 [rows, 10]) and the routing metadata produced from it
sel_idx = 2
print(f"{name}: arg shapes", [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in a][:12])

def bench(args, n=10):
    for _ in range(2): orig(*args)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): orig(*args)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

print(f"real routing: {bench(a):.2f} ms")
EOF_MARK = None

def with_routing(sel):
    b = list(a)
    sel = sel.contiguous()
    flat = sel.flatten()
    b[2] = sel
    b[4] = flat.argsort(stable=True)
    b[5] = torch.bincount(flat, minlength=b[5].numel())
    return b

E = 512
sel_real = a[2]
# concentrated: 40 experts, 10 distinct per token
base = torch.arange(10, device=dev).view(1, 10)
conc = ((torch.arange(rows, device=dev).view(-1, 1) % 4) * 10 + base) % 40
# uniform: token t gets experts (10t .. 10t+9) mod 512 -> every expert ~40 rows
unif = (torch.arange(rows, device=dev).view(-1, 1) * 10 + base) % E
# few rows per expert: same as unif but half the tokens -> ~20 rows/expert
for label, sel in (("real", sel_real), ("concentrated 40 experts (~512 rows each)", conc), ("uniform 512 experts (~40 rows each)", unif)):
    print(f"{label:44s} {bench(with_routing(sel)):7.2f} ms   active experts {int((torch.bincount(sel.flatten(), minlength=E) > 0).sum())}")
