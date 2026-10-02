#!/usr/bin/env python
"""Who issues the big copies inside a prefill GatedDeltaNet forward? Hooks Tensor.contiguous,
Tensor.to and Tensor.float/half/bfloat16 during one >=1024-row forward and logs call site +
shape + dtype for every copy of >= 1 MiB."""
import os, traceback, collections, torch
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
TARGET = os.environ.get("TARGET", "GatedDeltaNet")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=16384); m.load(progressbar=False)
on = [False]; log = collections.Counter()
def site():
    for f in reversed(traceback.extract_stack()[:-2]):
        if "exllamav3" in f.filename and "copy_trace" not in f.filename:
            return f"{os.path.basename(f.filename)}:{f.lineno} {f.line.strip()[:90]}"
    return "?"
def hook(name, orig, pred):
    def f(self, *a, **k):
        r = orig(self, *a, **k)
        if on[0] and r.data_ptr() != self.data_ptr() and r.numel() * r.element_size() >= (1 << 20):
            log[(name, site(), tuple(self.shape), str(self.dtype).replace("torch.", ""), str(r.dtype).replace("torch.", ""), self.is_contiguous())] += 1
        return r
    return f
T = torch.Tensor
for n in ("contiguous", "to", "float", "half", "bfloat16", "clone"):
    setattr(T, n, hook(n, getattr(T, n), None))
cls = next(type(s) for b in m.modules for s in getattr(b, "modules", []) if type(s).__name__ == TARGET)
orig = cls.forward; done = [0]; armed = [False]
def fwd(self, *a, **k):
    if armed[0] and not done[0] and a[0].numel() // a[0].shape[-1] >= 1024:
        done[0] = 1; on[0] = True
        try: return orig(self, *a, **k)
        finally: on[0] = False
    return orig(self, *a, **k)
cls.forward = fwd
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
for seed in (1, 2):
    armed[0] = seed == 2
    ids = torch.randint(1000, 100000, (1, 4096), generator=torch.Generator().manual_seed(seed), dtype=torch.long)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
for (n, s, shp, di, do, c), cnt in log.most_common():
    print(f"{n:10s} {str(shp):22s} {di:>9s}->{do:9s} contig={c!s:5s} x{cnt}  @ {s}")
