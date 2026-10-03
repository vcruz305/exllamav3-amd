#!/usr/bin/env python
"""QSA sparse prefill: how much do adjacent query rows' selections overlap? If a tile of M
consecutive rows shares most of its selected 16-token chunks, a tile-union kernel (exact, with
per-row masks) reads K/V once per tile instead of once per row."""
import os, time, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
PROMPT = int(os.environ.get("PROMPT", "8192"))
TEXT = os.environ.get("TEXT", "0") == "1"
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.qsa_indexer as Q
import exllamav3.modules.attention_fn.qsa_triton as QT
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=PROMPT + 1024); m.load(progressbar=False)
stats = []
orig_rows = QT.qsa_sparse_attend_rows
def rows_hook(q, k, v, indices, sm_scale, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    o = orig_rows(q, k, v, indices, sm_scale, **kw)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    if len(stats) < 400:
        R = indices.shape[0]
        idx = indices.long()
        valid = idx >= 0
        per_row_chunks = []
        res = {"R": R, "K_pad": indices.shape[1], "ms": dt * 1000, "q": tuple(q.shape), "k": tuple(k.shape)}
        ch = torch.where(valid, idx // 16, torch.full_like(idx, -1))
        nvalid_tok = valid.sum(1).float().mean().item()
        res["tok_per_row"] = nvalid_tok
        for M in (16, 32, 64):
            T = R // M
            if T == 0: continue
            c = ch[: T * M].view(T, M * ch.shape[1])
            u = []
            for t in range(0, T, max(1, T // 16)):
                s = torch.unique(c[t]); u.append((s >= 0).sum().item())
            per_row = (torch.stack([torch.unique(ch[r]).ge(0).sum() for r in range(0, R, max(1, R // 64))]).float().mean().item())
            res[f"union{M}"] = sum(u) / len(u); res["chunks_per_row"] = per_row
        stats.append(res)
    return o
QT.qsa_sparse_attend_rows = rows_hook
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
if TEXT:
    base = open(os.path.expanduser("~/exllamav3-amd/README.md")).read() * 20
    ids = tok.encode(base, add_bos=True)[:, :PROMPT]
else:
    ids = torch.randint(1000, 100000, (1, PROMPT), generator=torch.Generator().manual_seed(2), dtype=torch.long)
gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
print(f"calls: {len(stats)}   q {stats[0]['q']}  k {stats[0]['k']}  K_pad {stats[0]['K_pad']}")
for i, s in enumerate(stats[:: max(1, len(stats) // 8)]):
    print(f"  R={s['R']} {s['ms']:6.1f} ms  tok/row {s['tok_per_row']:.0f}  16-tok chunks/row {s.get('chunks_per_row', 0):.0f}  "
          f"union over 16 rows {s.get('union16', 0):.0f}  32 rows {s.get('union32', 0):.0f}  64 rows {s.get('union64', 0):.0f}")
tot = sum(s["ms"] for s in stats)
print(f"total sparse attend {tot:.0f} ms over {len(stats)} calls")
