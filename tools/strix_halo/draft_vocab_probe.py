#!/usr/bin/env python
"""
Reduced-vocabulary MTP draft head, feasibility micro-benchmark.

The MTP drafter computes full 248k-vocab logits each step (EXL3 lm_head, ~400 MB read). Its
only job is to propose a token the target will verify, so it can score a frequent-token subset
instead: output is unchanged (the target verifies every token), only acceptance can drop.

1. Token frequency over the bundled calibration text (all domains), tokenized with the model's
   tokenizer -> top-N ids (+ all special/added tokens).
2. Time: full EXL3 lm_head at 1 row vs fp16 [2560, N] subset head (torch.mm, ext.hgemm).
3. Coverage: fraction of calibration tokens inside the top-N set (an acceptance upper bound).
Saves the id list for N in NS to ~/models/<pack>/draft_vocab_<N>.pt
"""
import os, sys, time, collections, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Tokenizer
import exllamav3_ext as ext

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
NS = [int(x) for x in os.environ.get("NS", "8192,16384,32768,65536").split(",")]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
cfg = Config.from_directory(MODEL); tok = Tokenizer.from_config(cfg)

# 1. frequencies
cnt = collections.Counter(); total = 0
for f in sorted(glob.glob(os.path.join(REPO, "exllamav3/conversion/standard_cal_data/*.utf8"))):
    text = open(f, encoding="utf-8").read()
    ids = tok.encode(text, add_bos=False).view(-1).tolist()
    cnt.update(ids); total += len(ids)
    print(f"  {os.path.basename(f):20} {len(ids):9d} tokens")
print(f"calibration tokens {total}, distinct {len(cnt)}")
ranked = [t for t, _ in cnt.most_common()]
V = cfg.vocab_size if hasattr(cfg, "vocab_size") else 248320
special = set()
for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
    v = getattr(tok, name, None)
    if isinstance(v, int): special.add(v)
for v in (getattr(tok, "eos_token_id_list", None) or []): special.add(int(v))
# Added/control tokens sit at the top of the Qwen vocab (>= 248000ish); always keep them
special |= set(range(248044, 248320))
special = {s for s in special if 0 <= s < 248320}

# 2. head timing
m = Model.from_config(cfg); m.load(progressbar=False)
lm = m.modules[m.logit_layer_idx]
inner = lm.inner
dev = inner.trellis.device
print(f"lm_head: {type(inner).__name__} K={getattr(inner, 'K', '?')} in {inner.in_features} out {inner.out_features}")
def bench(f, n=50):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
x = torch.randn(1, 1, inner.in_features, dtype=torch.half, device=dev)
t_full = bench(lambda: lm.forward(x, {}))
print(f"full EXL3 lm_head, 1 row: {t_full:.3f} ms")
W = inner.get_weight_tensor()      # [in, out] fp16, Hadamards and scales applied
ref = lm.forward(x, {}).view(-1).float()
chk = (x.view(1, -1) @ W).view(-1).float()
print(f"reconstructed W vs EXL3 forward: max|d| {(chk[:ref.numel()] - ref).abs().max():.3e}  (ref max {ref.abs().max():.2f})")
for N in NS:
    keep = list(dict.fromkeys(sorted(special) + ranked))[:N]
    keep_t = torch.tensor(sorted(keep), dtype=torch.long)
    cov = sum(cnt[t] for t in keep) / total
    Ws = W[:, keep_t.to(dev)].contiguous()
    y = torch.empty(1, N, dtype=torch.half, device=dev)
    t_mm = bench(lambda: torch.mm(x.view(1, -1), Ws))
    t_hg = bench(lambda: ext.hgemm(x.view(1, -1), Ws, y))
    xs = torch.randn(4, inner.in_features, dtype=torch.half, device=dev)
    t_mm4 = bench(lambda: torch.mm(xs, Ws))
    print(f"N={N:6d}: coverage {100*cov:6.2f} %   fp16 head {Ws.numel()*2/2**20:6.1f} MiB   "
          f"torch.mm {t_mm:.3f} ms  hgemm {t_hg:.3f} ms  (4 rows {t_mm4:.3f})   vs full {t_full:.3f}")
    torch.save(keep_t, os.path.join(MODEL, f"draft_vocab_{N}.pt"))
