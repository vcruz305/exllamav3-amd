#!/usr/bin/env python
"""Where does the MTP drafter's argmax land? Hooks sample_from_state with the full head and
records each drafted id, then reports the share of drafts with id < N for several N. Also
records the target's verify argmax ids for comparison."""
import os, sys, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
draft = Model.from_config(cfg, component="mtp")
cache = Cache(model, max_num_tokens=4096, max_history=3); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=4096, max_history=3); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=3, dynamic_draft_tokens=True, draft_confidence=0.6)
drafted = []
orig = draft.sample_from_state
def hook(state, params):
    ids = orig(state, params); drafted.extend(ids.view(-1).tolist()); return ids
draft.sample_from_state = hook
out = []
for p in ["Explain gradient descent in two sentences:", "Write a Python function that reverses a linked list, with comments:"]:
    gen.enqueue(Job(input_ids=tok.encode(p, add_bos=True), max_new_tokens=256, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None: out.extend(t.view(-1).tolist())
d = torch.tensor(drafted); o = torch.tensor(out)
print(f"drafted {d.numel()}  generated {o.numel()}")
for N in (32768, 65536, 98304, 131072, 163840, 200000, 248044):
    print(f"  N={N:6d}: drafted ids < N {100*(d < N).float().mean():6.2f} %   generated ids < N {100*(o < N).float().mean():6.2f} %")
c = collections.Counter(d[d >= 98304].tolist())
print("most common drafted ids >= 98304:")
for k, v in c.most_common(10):
    print(f"   {k:6d} x{v:4d}  {tok.decode(torch.tensor([[k]]), decode_special_tokens=True)!r}")
