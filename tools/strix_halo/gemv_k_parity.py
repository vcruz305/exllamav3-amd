#!/usr/bin/env python
"""Dense HIP GEMV parity for every bitrate a pack uses (incl. the 7 / 8 bpw paths): for one
Linear per (K, shape) class, compare LinearEXL3.hip_gemv against reconstruct_hgemm (the exact
dequant reference the forward used before 7/8 bpw were eligible) at rows 1..16."""
import os, sys, collections, torch
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw")
from exllamav3 import Config, Model, Cache
from exllamav3.modules.quant.exl3 import LinearEXL3

cfg = Config.from_directory(MODEL); m = Model.from_config(cfg)
cache = Cache(m, max_num_tokens=2048); m.load(progressbar=False)

lins, seen = [], set()
def walk(o, d=0):
    if d > 7 or id(o) in seen: return
    seen.add(id(o))
    inner = getattr(o, "inner", None)
    if isinstance(inner, LinearEXL3): lins.append((getattr(o, "key", "?"), inner)); return
    for a in ("modules", "blocks", "mlp", "attn", "layers", "gates", "ups", "downs", "shared_experts",
              "q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down", "in_proj_qkv", "in_proj_z",
              "out_proj", "linear_attn", "self_attn", "indexer", "index_qk_proj"):
        v = getattr(o, a, None)
        if v is None: continue
        for x in (v if isinstance(v, (list, tuple)) else [v]): walk(x, d + 1)
    for v in vars(o).values() if hasattr(o, "__dict__") else []:
        if isinstance(v, (list, tuple)):
            for x in v:
                if hasattr(x, "inner") or hasattr(x, "modules"): walk(x, d + 1)
        elif hasattr(v, "inner") or hasattr(v, "modules"):
            walk(v, d + 1)
walk(m)

classes = {}
for key, l in lins:
    c = (l.K, l.in_features, l.out_features, bool(l.mcg), bool(l.mul1))
    classes.setdefault(c, (key, l))
print(f"EXL3 linears found: {len(lins)}   distinct (K, in, out, mcg, mul1) classes: {len(classes)}")

dev = torch.device("cuda:0"); torch.manual_seed(0)
fails = 0
for c, (key, l) in sorted(classes.items()):
    K, fin, fout, mcg, mul1 = c
    if fin % 128 or fout % 128 or not (mcg or mul1 or K == 4):
        print(f"  K={K} {fin}x{fout} skip (not GEMV-eligible shape/codebook)  {key}"); continue
    line = []
    for rows in (1, 3, 9, 16):
        x = torch.randn((rows, fin), dtype=torch.half, device=dev) * 0.5
        with torch.inference_mode():
            try:
                a = l.hip_gemv(x, torch.float).float()
            except RuntimeError as e:
                line.append(f"r{rows}:INELIGIBLE"); continue
            b = l.reconstruct_hgemm(x, torch.float).float()
        torch.cuda.synchronize()
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        ok = rel < 3e-3 and cos > 0.99999 and torch.isfinite(a).all().item()
        fails += not ok
        line.append(f"r{rows}:{rel:.1e}{'' if ok else '!!'}")
    print(f"  K={K} {fin:>6}x{fout:<6} {'mul1' if mul1 else 'mcg' if mcg else 'cb0'}  " + "  ".join(line) + f"   {key}")
print("GEMV PARITY", "PASS" if fails == 0 else f"FAIL ({fails})")
