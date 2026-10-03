#!/usr/bin/env python
"""Feasibility for a reconstruct + GEMM MoE prefill: what would the expert matmuls cost if the
weights were fp16 (hipBLASLt), and what does reconstructing the active experts cost?
Uses layer 0's real routing at 2048 rows."""
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
H = layer.hidden_size; I = layer.intermediate_size_padded if hasattr(layer, "intermediate_size_padded") else layer.intermediate_size
x = torch.randn((1, rows, H), device=dev, dtype=torch.half) * 0.5
with torch.inference_mode(): layer.forward(x, {})
sel = cap["a"][2]
counts = torch.bincount(sel.flatten(), minlength=layer.num_experts).cpu()
active = [e for e in range(layer.num_experts) if counts[e] > 0]
print(f"H={H} I={I}  active experts {len(active)}  rows/expert mean {counts[counts > 0].float().mean():.1f} max {counts.max()}")

def bench(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

t_grouped = bench(lambda: orig(*cap["a"]))
# reconstruct cost: all active experts, 3 projections
ws = {}
def recon_all():
    for e in active:
        for lin, nm in ((layer.gates[e], "g"), (layer.ups[e], "u"), (layer.downs[e], "d")):
            L = lin.inner
            w = ws.get((e, nm))
            if w is None:
                w = ws[(e, nm)] = torch.empty((L.in_features, L.out_features), dtype=torch.half, device=dev)
            ext.reconstruct(w, L.trellis, L.K, L.mcg, L.mul1)
t_recon = bench(recon_all, 3)
# GEMM cost with fp16 weights: per-expert matmuls at their real row counts
xs = {e: torch.randn((int(counts[e]), H), device=dev, dtype=torch.half) for e in active}
hs = {e: torch.randn((int(counts[e]), I), device=dev, dtype=torch.half) for e in active}
def gemms():
    for e in active:
        torch.matmul(xs[e], ws[(e, "g")]); torch.matmul(xs[e], ws[(e, "u")]); torch.matmul(hs[e], ws[(e, "d")])
t_gemm = bench(gemms, 3)
# one big stacked-weight view: what a single grouped GEMM could reach (dense upper bound)
X = torch.randn((rows * 10, H), device=dev, dtype=torch.half); Wg = torch.randn((H, 2 * I), device=dev, dtype=torch.half)
Hh = torch.randn((rows * 10, I), device=dev, dtype=torch.half); Wd = torch.randn((I, H), device=dev, dtype=torch.half)
t_dense = bench(lambda: (torch.matmul(X, Wg), torch.matmul(Hh, Wd)))
flops = 2 * rows * 10 * (H * 2 * I + I * H)
print(f"grouped GEMV (current)       {t_grouped:7.2f} ms")
print(f"reconstruct {len(active)} experts x3   {t_recon:7.2f} ms")
print(f"per-expert fp16 matmuls      {t_gemm:7.2f} ms   ({3 * len(active)} launches)")
print(f"dense-equivalent GEMM bound  {t_dense:7.2f} ms   ({flops / t_dense / 1e9:.1f} TFLOP/s)")
