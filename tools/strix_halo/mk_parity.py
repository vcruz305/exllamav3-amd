#!/usr/bin/env python
"""Parity: mixed-K grouped HIP route (ext.exl3_moe_mk / _prefill) vs the dense per-expert
reference path, on real CYBER-FROST layers.

For each tested layer and row count, the SAME BlockSparseMLP.forward is called twice on the
same input: once with the grouped route enabled, once with support_hip_mk forced off (which
drops to the established per-expert loop that the previous verified build used). Expert
selection comes from the real router; additionally a forced-routing variant pins slots onto
K=7 and K=6 experts so the rare bitrates are exercised.

Pass criteria are logit-scale (this is fp16 GEMV with a different summation order inside each
expert GEMV, not bit-exact by construction):
  - max |a-b| relative to max |b| below 2e-3
  - cosine similarity above 0.99999
"""
import os, sys, torch, collections

MODEL = os.path.expanduser("~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw")

from exllamav3 import Config, Model, Cache
import exllamav3.modules.block_sparse_mlp as B

cfg = Config.from_directory(MODEL)
m = Model.from_config(cfg)
cache = Cache(m, max_num_tokens=2048)
m.load(progressbar=False)

found, seen = [], set()
def walk(o, d=0):
    if d > 5 or id(o) in seen: return
    seen.add(id(o))
    if type(o).__name__ == "BlockSparseMLP":
        found.append(o); return
    for a in ("blocks", "mlp", "modules", "layers"):
        v = getattr(o, a, None)
        if v is None: continue
        for x in (v if isinstance(v, (list, tuple)) else [v]): walk(x, d + 1)
walk(m)

mk_layers = [l for l in found if getattr(l, "support_hip_mk", False)]
print(f"BlockSparseMLP layers: {len(found)}   on mixed-K grouped route: {len(mk_layers)}   "
      f"on K3 route: {sum(1 for l in found if l.support_hip_grouped)}")
if not mk_layers:
    sys.exit("FAIL: no layer took the mixed-K grouped route")

dev = torch.device("cuda:0")
torch.manual_seed(0)

def compare(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rel, cos

# Which layers to test: first, a middle one, the last mixed-K one, plus the layer holding the
# most K=7 experts
def count_k(layer, K):
    return sum(1 for l in layer.downs if l.inner.K == K)
k7_layer = max(mk_layers, key=lambda l: count_k(l, 7))
test = []
for l in (mk_layers[0], mk_layers[len(mk_layers) // 2], mk_layers[-1], k7_layer):
    if l not in test: test.append(l)

fails = 0
worst = (0.0, 1.0)
for layer in test:
    ks = collections.Counter(l.inner.K for l in layer.downs)
    print(f"\n{layer.key}  down K hist {dict(sorted(ks.items()))}")
    for bsz in (1, 3, 4, 16, 37):
        x = (torch.randn((bsz, layer.hidden_size), device=dev, dtype=torch.half) * 0.5)
        x3 = x.view(1, bsz, layer.hidden_size)
        layer.support_hip_mk_save = (layer.support_hip_mk, layer.support_hip_mk_prefill)
        with torch.inference_mode():
            out_mk = layer.forward(x3.clone(), {}).float().clone()
            layer.support_hip_mk = False; layer.support_hip_mk_prefill = False
            out_ref = layer.forward(x3.clone(), {}).float().clone()
            layer.support_hip_mk, layer.support_hip_mk_prefill = layer.support_hip_mk_save
        torch.cuda.synchronize()
        rel, cos = compare(out_mk, out_ref)
        ok = rel < 2e-3 and cos > 0.99999 and torch.isfinite(out_mk).all().item()
        fails += not ok
        if rel > worst[0]: worst = (rel, cos)
        route = "decode" if bsz <= B._HIP_GROUPED_MAX_ROWS and bsz < B._HIP_PREFILL_MIN_ROWS else "prefill"
        print(f"  bsz={bsz:3d} [{route:7s}]  rel_max={rel:.2e}  cos={cos:.8f}  {'OK' if ok else 'FAIL'}")

# Forced K=7 coverage: call the extension directly with routing pinned on K=7/K=6 experts
layer = k7_layer
mk = layer.hip_mk
k7 = [i for i, l in enumerate(layer.downs) if l.inner.K == 7]
k6 = [i for i, l in enumerate(layer.downs) if l.inner.K == 6]
print(f"\nforced routing on {layer.key}: {len(k7)} K7 down experts, {len(k6)} K6")
if k7:
    import exllamav3_ext as ext
    pool = (k7 + k6 + list(range(10)))[:10]
    for bsz in (1, 3):
        x = torch.randn((bsz, layer.hidden_size), device=dev, dtype=torch.half) * 0.5
        sel = torch.tensor([pool] * bsz, dtype=torch.long, device=dev)
        w = torch.full((bsz, 10), 0.1, dtype=torch.half, device=dev)
        A = bsz * 10
        H, I = layer.hidden_size, layer.intermediate_size_padded
        gu_had = torch.empty((2 * A, H), dtype=torch.half, device=dev)
        gu_out = torch.empty((2 * A, I), dtype=torch.half, device=dev)
        dn_had = torch.empty((A, I), dtype=torch.half, device=dev)
        dn_out = torch.empty((A, H), dtype=torch.float, device=dev)
        out = torch.empty((bsz, H), dtype=torch.float, device=dev)
        ext.exl3_moe_mk(x, out, sel, w,
                        mk.gate_trellis, mk.gate_suh, mk.gate_svh,
                        mk.up_trellis, mk.up_suh, mk.up_svh,
                        mk.down_trellis, mk.down_suh, mk.down_svh,
                        mk.gate_K, mk.up_K, mk.down_K, mk.gate_ks, mk.up_ks, mk.down_ks,
                        gu_had, gu_out, dn_had, dn_out)
        # reference: per-expert dense math via each expert's own Linear
        ref = torch.zeros((bsz, H), dtype=torch.float, device=dev)
        for r in range(bsz):
            for s, e in enumerate(pool):
                xi = x[r:r+1]
                g = layer.gates[e].forward(xi, {}).float()
                u = layer.ups[e].forward(xi, {}).float()
                a = (torch.nn.functional.silu(g.half().float()).half() * u.half()).half()
                d = layer.downs[e].forward(a, {}).float()
                ref[r] += d[0] * 0.1
        torch.cuda.synchronize()
        rel, cos = compare(out, ref)
        ok = rel < 5e-3 and cos > 0.9999 and torch.isfinite(out).all().item()
        fails += not ok
        print(f"  forced K7/K6 bsz={bsz}: rel_max={rel:.2e} cos={cos:.8f} {'OK' if ok else 'FAIL'}")

print(f"\nworst layer-forward: rel_max={worst[0]:.2e} cos={worst[1]:.8f}")
print("PARITY", "PASS" if fails == 0 else f"FAIL ({fails})")
