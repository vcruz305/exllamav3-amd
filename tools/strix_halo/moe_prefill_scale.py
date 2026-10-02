#!/usr/bin/env python
"""How does one mixed-K MoE prefill call scale with rows? If time tracks sum_e ceil(n_e/16)
(16-row chunks, each re-reading the expert's weights) rather than rows, weight reuse across
row tiles is the lever."""
import os, time, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
from exllamav3 import Config, Model, Cache
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg)
cache = Cache(m, max_num_tokens=4096); m.load(progressbar=False)
layer = next(s for b in m.modules for s in getattr(b, "modules", []) if type(s).__name__ == "BlockSparseMLP")
import exllamav3.modules.block_sparse_mlp as B
dev = torch.device("cuda:0"); torch.manual_seed(0)
calls = []
import exllamav3_ext as ext
fn = getattr(ext, "exl3_moe_mk_prefill", None) or getattr(ext, "exl3_moe_gfx12_k3_prefill")
name = "exl3_moe_mk_prefill" if hasattr(ext, "exl3_moe_mk_prefill") and layer.support_hip_mk_prefill else "exl3_moe_gfx12_k3_prefill"
captured = {}
orig = getattr(ext, name)
def cap(*a):
    captured["a"] = a
    return orig(*a)
for mod in (B, ):
    pass
setattr(ext, name, cap)
B.ext = ext
print(f"layer {layer.key}  route={'mk' if layer.support_hip_mk_prefill else 'k3'}  fn={name}")
for rows in (256, 512, 1024, 2048):
    x = torch.randn((1, rows, layer.hidden_size), device=dev, dtype=torch.half) * 0.5
    captured.clear()
    with torch.inference_mode():
        layer.forward(x, {})
    a = captured.get("a")
    if a is None:
        print(f"rows={rows}: route not taken"); continue
    sel = a[2]
    counts = torch.bincount(sel.flatten(), minlength=512)
    chunks = int(((counts + 15) // 16).sum())
    torch.cuda.synchronize()
    for _ in range(3): orig(*a)
    torch.cuda.synchronize(); t = time.perf_counter(); n = 10
    for _ in range(n): orig(*a)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n * 1000
    active = int((counts > 0).sum())
    print(f"rows={rows:5d}  assignments={rows*10:6d}  active experts={active:3d}  16-row chunks={chunks:5d}  "
          f"{dt:7.2f} ms  {1000*dt/chunks:6.1f} us/chunk  {1000*dt/rows:6.2f} us/row")

if os.environ.get("TRACE"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): orig(*a)
        torch.cuda.synchronize()
    def dt(e): return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
    ev = sorted([e for e in prof.key_averages() if dt(e) > 0], key=dt, reverse=True)
    print(f"-- kernels for one call at rows={rows} (avg of 5)")
    for e in ev[:16]:
        print(f"  {dt(e) / 5e3:8.3f} ms  n/call={e.count / 5:5.1f}  {e.key[:110]}")
