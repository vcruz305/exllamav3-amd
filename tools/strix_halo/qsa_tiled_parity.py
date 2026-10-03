#!/usr/bin/env python
"""Tiled vs per-row QSA sparse attention on REAL captured prefill inputs (8k cold prompt):
output parity and per-call time, fp16 cache and (CQ=4) quantized cache."""
import os, time, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
PROMPT = int(os.environ.get("PROMPT", "8192"))
CQ = os.environ.get("CQ")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.cache import CacheLayer_quant
import exllamav3.modules.qsa_indexer as Q
import exllamav3.modules.attention_fn.qsa_triton as QT
from exllamav3.modules.attention_fn.qsa_tiled import qsa_sparse_attend_tiled
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
kw = dict(layer_type=CacheLayer_quant, k_bits=int(CQ), v_bits=int(CQ)) if CQ else {}
cache = Cache(m, max_num_tokens=PROMPT + 1024, **kw); m.load(progressbar=False)

caps = []
orig = Q.QSAIndexer.sparse_attend if hasattr(Q, "QSAIndexer") else None
cls = next(c for c in vars(Q).values() if isinstance(c, type) and hasattr(c, "sparse_attend"))
orig = cls.sparse_attend
def hooked(self, layer, attn, q, q_idx, block_table, cache_seqlens_cpu):
    o = orig(self, layer, attn, q, q_idx, block_table, cache_seqlens_cpu)
    bsz, seq = q.shape[:2]
    if bsz == 1 and seq >= 1024 and len(caps) < 6:
        indices = self.select_indices_paged(layer, q_idx, block_table, cache_seqlens_cpu)
        if isinstance(layer, CacheLayer_quant):
            qk, sk, qv, sv, kb, vb = layer.get_qkv()
            k_arg, v_arg, qc, ps = qk, qv, (sk, sv, kb, vb), qk.shape[1]
        else:
            k_arg = layer.k.view(-1, attn.num_kv_heads, attn.head_dim)
            v_arg = layer.v.view(-1, attn.num_kv_heads, attn.head_dim)
            qc, ps = None, layer.k.shape[1]
        qr = q.reshape(seq, attn.num_q_heads, attn.head_dim).contiguous()
        bt_rows = block_table.int().expand(seq, -1).contiguous()
        bt1 = block_table.int()[0].contiguous()
        pos0 = int(cache_seqlens_cpu[0])
        args_old = (qr, k_arg, v_arg, indices, attn.sm_scale)
        kw_old = dict(block_table=bt_rows, page_size=ps, qc=qc, n_kv_heads=attn.num_kv_heads)
        def run_old(): return QT.qsa_sparse_attend_rows(*args_old, **kw_old)
        def run_new(): return qsa_sparse_attend_tiled(qr, k_arg, v_arg, indices, attn.sm_scale, bt1, ps, pos0,
                                                      self.compress_ratio, qc=qc, n_kv_heads=attn.num_kv_heads)
        a = run_old(); b = run_new(); torch.cuda.synchronize()
        def t(fn, n=5):
            fn(); torch.cuda.synchronize(); s = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1000
        af, bf = a.float(), b.float()
        if qc is None and len(caps) < 3:
            # fp32 reference on 96 sampled rows: gather each row's exact token list from the cache
            kv = attn.num_kv_heads; grp = attn.num_q_heads // kv
            eo = en = 0.0; n_ = 0
            for r_ in torch.linspace(0, seq - 1, 96).long().tolist():
                idx = indices[r_][indices[r_] >= 0].long()
                phys = bt1[idx // ps].long() * ps + idx % ps
                K_ = k_arg[phys].float(); V_ = v_arg[phys].float()
                qq = qr[r_].float().view(kv, grp, -1)
                sc = torch.einsum("kgd,lkd->kgl", qq, K_) * attn.sm_scale
                o_ = torch.einsum("kgl,lkd->kgd", sc.softmax(-1), V_).reshape(attn.num_q_heads, -1)
                eo += (af[r_] - o_).abs().mean().item(); en += (bf[r_] - o_).abs().mean().item(); n_ += 1
            print(f"  vs fp32 reference (96 rows, mean|err|): per-row {eo / n_:.3e}   tiled {en / n_:.3e}")
        rel = ((af - bf).abs().max() / af.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
        caps.append((seq, pos0, rel, cos, t(run_old), t(run_new), torch.isfinite(bf).all().item()))
    return o
cls.sparse_attend = hooked
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
ids = torch.randint(1000, 100000, (1, PROMPT), generator=torch.Generator().manual_seed(2), dtype=torch.long)
gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
fails = 0
for seq, pos0, rel, cos, to, tn, fin in caps:
    ok = rel < 5e-3 and cos > 0.99999 and fin; fails += not ok
    print(f"rows={seq} pos0={pos0:5d}  rel_max={rel:.2e} cos={cos:.7f}  per-row {to:6.2f} ms -> tiled {tn:6.2f} ms  ({to / tn:4.1f}x)  {'OK' if ok else 'FAIL'}")
print(f"cache={'q' + CQ if CQ else 'fp16'}  QSA TILED PARITY", "PASS" if fails == 0 and caps else "FAIL")
