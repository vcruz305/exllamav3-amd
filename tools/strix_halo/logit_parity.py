#!/usr/bin/env python
"""
Toolchain logit parity: run the same wikitext rows through a model and save the fp32 logits
(top-1, log-softmax at the target) so two builds (e.g. ROCm 7.0 vs 10.1) can be compared
directly, instead of reading tea leaves from PPL in the 4th decimal.

  MODE=dump OUT=/tmp/lp_r7.pt   MODEL=... python logit_parity.py   (in each arm)
  MODE=cmp  A=/tmp/lp_r7.pt B=/tmp/lp_r101.pt python logit_parity.py

Also dumps an fp64-free reference-free number per arm: mean NLL (the PPL ingredient), so the
two can be cross-checked against eval/ppl.py.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch

MODE = os.environ.get("MODE", "dump")

if MODE == "cmp":
    import torch.torch_version as _tv   # dumps from before the str() fix carry a TorchVersion
    with torch.serialization.safe_globals([_tv.TorchVersion]):
        a = torch.load(os.environ["A"]); b = torch.load(os.environ["B"])
    LA, LB, TGT = a["logits"], b["logits"], a["targets"]
    assert LA.shape == LB.shape, (LA.shape, LB.shape)
    assert torch.equal(TGT, b["targets"]), "different token rows"
    # Row by row (each row is 1023 x 248k; fp32 copies of all four rows at once ~16 GB)
    acc = dict(dsum=0.0, dmax=0.0, n=0, kl=[], flips=0, fm=[], margin=[], lsa=[], lsb=[], p99=[])
    for r in range(LA.shape[0]):
        la, lb, tgt = LA[r].float(), LB[r].float(), TGT[r]
        d = (la - lb).abs()
        acc["dsum"] += d.sum().item(); acc["n"] += d.numel(); acc["dmax"] = max(acc["dmax"], d.max().item())
        acc["p99"].append(torch.quantile(d.flatten()[::97], 0.99).item())   # strided sample
        s = la.topk(2, dim=-1).values; margin = s[:, 0] - s[:, 1]
        f = la.argmax(-1) != lb.argmax(-1)
        acc["flips"] += int(f.sum()); acc["fm"].append(margin[f]); acc["margin"].append(margin)
        ka = torch.log_softmax(la, -1); kb = torch.log_softmax(lb, -1)
        acc["kl"].append((ka.exp() * (ka - kb)).sum(-1))
        acc["lsa"].append(ka.gather(-1, tgt[:, None])[:, 0]); acc["lsb"].append(kb.gather(-1, tgt[:, None])[:, 0])
        del la, lb, d, ka, kb
    kl = torch.cat(acc["kl"]); lsa = torch.cat(acc["lsa"]); lsb = torch.cat(acc["lsb"])
    margin = torch.cat(acc["margin"]); fm = torch.cat(acc["fm"])
    pa, pb = math.exp(-lsa.mean().item()), math.exp(-lsb.mean().item())
    print(f"A: {a.get('torch')} hip {a.get('hip')}   B: {b.get('torch')} hip {b.get('hip')}")
    print(f"positions: {lsa.numel()}  vocab: {LA.shape[-1]}")
    print(f"PPL  A {pa:.6f}   B {pb:.6f}   delta {pb - pa:+.6f}")
    print(f"logit |diff|: mean {acc['dsum'] / acc['n']:.4e}  p99 ~{max(acc['p99']):.4e}  max {acc['dmax']:.4e}")
    print(f"KL(A||B) per token: mean {kl.mean():.3e}  max {kl.max():.3e}")
    print(f"top-1 flips: {acc['flips']} / {lsa.numel()} ({100 * acc['flips'] / lsa.numel():.3f} %)")
    if acc["flips"]:
        print(f"  A's top-2 margin at flips: median {fm.median():.4f}  max {fm.max():.4f} "
              f"(logit units; median margin overall {margin.median():.3f})")
    print(f"per-token |dNLL|: mean {(lsa - lsb).abs().mean():.4e}  max {(lsa - lsb).abs().max():.4e}")
    sys.exit(0)

from exllamav3 import model_init
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "eval"))
import ppl as P   # same tokenisation / dataset path as eval/ppl.py
import argparse

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
ROWS = int(os.environ.get("ROWS", "4")); L = int(os.environ.get("LEN", "1024"))
ap = argparse.ArgumentParser(allow_abbrev=False); model_init.add_args(ap, cache=False)
args = ap.parse_args(["-m", MODEL])
# identical load to eval/ppl.py
m, cfg, _, tok = model_init.init(args, override_dynamic_seq_len=2048, max_output_size=2048,
                                 max_output_factor=5)[:4]
ids = P.get_test_tokens(tok, ROWS, L, L)
logits, targets = [], []
with torch.inference_mode():
    for r in range(ids.shape[0]):
        x = ids[r:r + 1]
        y = m.forward(x, {"attn_mode": "flash_attn_nc"})
        y = y["logits"] if isinstance(y, dict) else y
        logits.append(y[0, :-1].to(torch.float16).cpu())   # fp16 storage: 4 x 1023 x 248k
        targets.append(x[0, 1:].cpu())
out = {"logits": torch.stack(logits), "targets": torch.stack(targets), "model": MODEL,
       "torch": str(torch.__version__), "hip": str(torch.version.hip)}   # plain str: weights_only-safe
torch.save(out, os.environ["OUT"])
print(f"saved {os.environ['OUT']}: {tuple(out['logits'].shape)}  torch {torch.__version__}")
