#!/usr/bin/env python
"""
Host-sync census for MTP decode: wraps every blocking device->host point (Tensor.item, .cpu,
.tolist, .numpy, torch.cuda.synchronize, Tensor.__bool__/__int__/__float__ on CUDA tensors) and
attributes call count and blocked wall time to the calling source line.

  MODEL=... NTOK=256 python sync_census.py
"""
import os, sys, time, collections, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
NTOK = int(os.environ.get("NTOK", "256")); NDT = int(os.environ.get("NDT", "3"))
config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=4096, max_history=NDT); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=4096, max_history=NDT); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6)
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)

stats = collections.defaultdict(lambda: [0, 0.0])
ACTIVE = [False]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEPTH = int(os.environ.get("DEPTH", "1"))
def site():
    out = []
    for fr in reversed(traceback.extract_stack()[:-2]):
        if fr.filename.startswith(REPO) and "sync_census" not in fr.filename:
            out.append(f"{os.path.relpath(fr.filename, REPO)}:{fr.lineno} {fr.name}")
            if len(out) >= DEPTH: break
    return " <- ".join(out) or "?"

def wrap(owner, name, label, cuda_only=True):
    orig = getattr(owner, name)
    def f(*a, **k):
        if not ACTIVE[0] or (cuda_only and a and isinstance(a[0], torch.Tensor) and not a[0].is_cuda):
            return orig(*a, **k)
        t = time.perf_counter(); r = orig(*a, **k); dt = time.perf_counter() - t
        s = stats[(label, site())]; s[0] += 1; s[1] += dt
        return r
    setattr(owner, name, f)

for n in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__"):
    wrap(torch.Tensor, n, n)
_orig_to = torch.Tensor.to
def _to(self, *a, **k):
    if ACTIVE[0] and self.is_cuda and ((a and (a[0] == "cpu" or (isinstance(a[0], torch.device) and a[0].type == "cpu")))
                                        or str(k.get("device", "")) == "cpu") and not k.get("non_blocking"):
        t = time.perf_counter(); r = _orig_to(self, *a, **k); dt = time.perf_counter() - t
        s = stats[("to(cpu)", site())]; s[0] += 1; s[1] += dt; return r
    return _orig_to(self, *a, **k)
torch.Tensor.to = _to
_orig_copy = torch.Tensor.copy_
def _copy(self, src, non_blocking=False):
    if ACTIVE[0] and isinstance(src, torch.Tensor) and not non_blocking and (self.is_cuda != src.is_cuda):
        kind = "copy_ D2H" if src.is_cuda else ("copy_ H2D pinned" if src.is_pinned() else "copy_ H2D pageable")
        t = time.perf_counter(); r = _orig_copy(self, src, non_blocking); dt = time.perf_counter() - t
        s = stats[(kind, site())]; s[0] += 1; s[1] += dt; return r
    return _orig_copy(self, src, non_blocking)
torch.Tensor.copy_ = _copy
# H2D via .to(device) / .cuda() without non_blocking (synchronous hipMemcpyWithStream)
_orig_to2 = torch.Tensor.to
def _to2(self, *a, **k):
    if ACTIVE[0] and not self.is_cuda and not k.get("non_blocking", False) and not (len(a) > 1 and a[1] is True):
        dev = k.get("device", a[0] if a else None)
        if (isinstance(dev, torch.device) and dev.type == "cuda") or (isinstance(dev, (str, int)) and str(dev).startswith(("cuda", "0", "1"))):
            t = time.perf_counter(); r = _orig_to2(self, *a, **k); dt = time.perf_counter() - t
            kind = "to H2D pinned" if self.is_pinned() else "to H2D pageable"
            s = stats[(kind, site())]; s[0] += 1; s[1] += dt; return r
    return _orig_to2(self, *a, **k)
torch.Tensor.to = _to2
_orig_cuda = torch.Tensor.cuda
def _cuda(self, *a, **k):
    if ACTIVE[0] and not self.is_cuda and not k.get("non_blocking", False):
        t = time.perf_counter(); r = _orig_cuda(self, *a, **k); dt = time.perf_counter() - t
        s = stats[(".cuda()", site())]; s[0] += 1; s[1] += dt; return r
    return _orig_cuda(self, *a, **k)
torch.Tensor.cuda = _cuda
for fn in ("tensor", "as_tensor"):
    _o = getattr(torch, fn)
    def mk(_o, fn):
        def f(*a, **k):
            d = k.get("device")
            if ACTIVE[0] and d is not None and str(d) != "cpu" and not (isinstance(d, torch.device) and d.type == "cpu"):
                t = time.perf_counter(); r = _o(*a, **k); dt = time.perf_counter() - t
                s = stats[(f"torch.{fn}(dev)", site())]; s[0] += 1; s[1] += dt; return r
            return _o(*a, **k)
        return f
    setattr(torch, fn, mk(_o, fn))
_orig_sync = torch.cuda.synchronize
def _sync(*a, **k):
    if not ACTIVE[0]: return _orig_sync(*a, **k)
    t = time.perf_counter(); r = _orig_sync(*a, **k); dt = time.perf_counter() - t
    s = stats[("synchronize", site())]; s[0] += 1; s[1] += dt; return r
torch.cuda.synchronize = _sync

def run(n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
    rounds = 0; acc = rej = 0
    while gen.num_remaining_jobs():
        rounds += 1
        for r in gen.iterate():
            acc += r.get("accepted_draft_tokens") or 0; rej += r.get("rejected_draft_tokens") or 0
    return rounds

run(16)
_orig_sync()
ACTIVE[0] = True
t0 = time.perf_counter(); rounds = run(NTOK); _orig_sync(); wall = time.perf_counter() - t0
ACTIVE[0] = False
tot_n = sum(v[0] for v in stats.values()); tot_t = sum(v[1] for v in stats.values())
print(f"wall {wall:.3f}s  {NTOK/wall:.2f} tok/s  iterate() calls {rounds}  ({1000*wall/rounds:.2f} ms/round)")
print(f"blocking D2H points: {tot_n} ({tot_n/rounds:.1f}/round)  blocked {tot_t*1000:.0f} ms ({100*tot_t/wall:.1f} % of wall)")
print(f"{'calls':>7} {'/round':>7} {'ms':>8} {'us/call':>8}  kind / site")
for (kind, s), (n, t) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"{n:7d} {n/rounds:7.2f} {t*1000:8.1f} {t*1e6/n:8.1f}  {kind:10} {s}")
