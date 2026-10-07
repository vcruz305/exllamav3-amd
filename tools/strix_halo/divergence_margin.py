#!/usr/bin/env python
"""At a divergence between two greedy runs, is it a near-tie? Runs the target alone (no MTP) on
the common prefix and reports the top-2 logit margin at the divergent position, plus which of
the two tokens a no-MTP greedy run picks."""
import os, sys, json, subprocess, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from exllamav3 import Config, Model, Cache, Tokenizer
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
P = os.environ.get("PROMPT", "Explain gradient descent in two sentences:")
A = json.loads(os.environ["IDS_A"]); B = json.loads(os.environ["IDS_B"])
i = next(k for k, (x, y) in enumerate(zip(A, B)) if x != y)
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=4096); m.load(progressbar=False)
ids = torch.cat([tok.encode(P, add_bos=True), torch.tensor([A[:i]])], dim=-1)
with torch.inference_mode():
    logits = m.forward(ids, {"attn_mode": "flash_attn_nc"})
logits = (logits["logits"] if isinstance(logits, dict) else logits)[0, -1].float()
v, k = logits.topk(5)
print(f"divergence at generated token {i}: run A {A[i]} {tok.decode(torch.tensor([[A[i]]]))!r}  run B {B[i]} {tok.decode(torch.tensor([[B[i]]]))!r}")
print(f"no-MTP full-context top-5: " + "  ".join(f"{int(t)}:{float(s):.3f}" for t, s in zip(k, v)))
print(f"logit(A) {logits[A[i]]:.4f}  logit(B) {logits[B[i]]:.4f}  margin {abs(logits[A[i]] - logits[B[i]]):.4f}")
