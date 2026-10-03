#!/usr/bin/env python
"""Long-context end-to-end check for a prefill-path change: prefill a REAL ~12k-token text
through the Generator (cached path, so QSA sparse attention runs), then greedy-decode 64 tokens.
Run once per setting (separate processes); writes logits of the first decode step + tokens.
  KNOB=EXL3_QSA_TILED VAL=0 OUT=/tmp/a.pt python long_ctx_check.py
Compare with:  python long_ctx_check.py compare /tmp/a.pt /tmp/b.pt"""
import os, sys, torch
if len(sys.argv) > 1 and sys.argv[1] == "compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    la, lb = a["logits"].float(), b["logits"].float()
    d = (la - lb).abs()
    same = sum(int(x == y) for x, y in zip(a["tokens"], b["tokens"]))
    first_div = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
    top2 = la.topk(2).values
    print(f"prompt {a['prompt_len']} tok | first-step logits max|d| {d.max():.4f} mean {d.mean():.5f} | "
          f"argmax {'same' if la.argmax() == lb.argmax() else 'DIFFERENT'} (top-2 gap {top2[0] - top2[1]:.3f}) | "
          f"greedy tokens identical {same}/{len(a['tokens'])} first divergence {first_div}")
    print("text A:", repr(a["text"][:160])); print("text B:", repr(b["text"][:160]))
    sys.exit(0)
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
cfg = Config.from_directory(MODEL); m = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(m, max_num_tokens=16384); m.load(progressbar=False)
root = os.path.expanduser("~/exllamav3-amd")
text = ""
for f in ("README.md", "README.strix-halo.md", "AGENTS.md", "tools/strix_halo/MIXEDK_PORT_NOTES.md",
          "exllamav3/modules/attn.py", "exllamav3/modules/block_sparse_mlp.py"):
    p = os.path.join(root, f)
    if os.path.exists(p): text += open(p).read() + "\n\n"
ids = tok.encode(text, add_bos=True)[:, :12000]
q = tok.encode("\n\nQuestion: summarize what the documents above are about in two sentences.\nAnswer:", add_bos=False)
ids = torch.cat([ids, q], dim=-1)
gen = Generator(model=m, cache=cache, tokenizer=tok, max_chunk_size=2048)
first = {}
orig = m.forward
gen.enqueue(Job(input_ids=ids, max_new_tokens=64, sampler=GreedySampler(), return_logits=True))
toks, text_out, logits0 = [], "", None
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("token_ids") is not None:
            toks += r["token_ids"].flatten().tolist()
        if r.get("text"): text_out += r["text"]
        if logits0 is None and r.get("logits") is not None:
            logits0 = r["logits"].flatten().float().cpu()
torch.save({"prompt_len": ids.shape[-1], "tokens": toks, "text": text_out, "logits": logits0},
           os.environ.get("OUT", "/tmp/lcc.pt"))
print("saved", len(toks), "tokens; logits", None if logits0 is None else tuple(logits0.shape))
