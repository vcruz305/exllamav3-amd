#!/usr/bin/env python
"""Dump the target model's greedy outputs for the six sweep prompts (token ids) so draft-vocab
coverage can be measured on what the model actually generates (incl. reasoning blocks)."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
src = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompt_sweep.py")).read()
ns = {}; exec(src[src.index("PROMPTS"):src.index("]", src.index("PROMPTS")) + 1], ns)
PROMPTS = ns["PROMPTS"]
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
draft = Model.from_config(cfg, component="mtp")
cache = Cache(model, max_num_tokens=4096, max_history=3); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=4096, max_history=3); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=3, dynamic_draft_tokens=True, draft_confidence=0.6)
out = []
for p in PROMPTS:
    ids = tok.encode(p, add_bos=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=512, sampler=GreedySampler()))
    toks = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None: toks.append(t.view(-1))
    out.append(torch.cat(toks)); print(f"{len(toks):4d} chunks, {out[-1].numel()} tokens: {p[:50]}")
torch.save(torch.cat(out), os.environ.get("OUT", "/tmp/gen_ids.pt"))
