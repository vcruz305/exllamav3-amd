#!/usr/bin/env python
"""
Verify-forward anatomy at the real decode shape, isolated: prefill a prompt, then repeatedly run
ONE trunk forward of R rows (R=4: target + 3 drafts) with recurrent history on, fully rewinding
recurrent state (R tokens) after each so every forward sees the same context. KV entries are
simply overwritten at the same positions. Nothing else runs on the GPU, so the profiler window
contains only verify kernels.

Reports wall per forward (events, unprofiled), device busy per forward, idle, and kernel
families by device time and call count.  R=4 NFWD=30 python verify_anatomy.py
"""
import os, sys, time, collections, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
R = int(os.environ.get("R", "4")); NFWD = int(os.environ.get("NFWD", "30"))
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=4096, max_history=R - 1); model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, num_draft_tokens=R - 1)

captured = {}
orig = model.forward
def cap(input_ids, params=None):
    if input_ids.shape[-1] == R and "x" not in captured:
        captured["x"] = input_ids.clone(); captured["p"] = params
    return orig(input_ids, params)
model.forward = cap
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)
gen.enqueue(Job(input_ids=ids, max_new_tokens=64, sampler=GreedySampler()))
for _ in range(6):
    gen.iterate()
with torch.inference_mode():
    gen.iterate_gen([], torch.zeros((1, R - 1), dtype=torch.long))
model.forward = orig
assert "x" in captured, "no R-row forward captured"
x = captured["x"]; P = captured["p"]
rs = P.get("recurrent_states")
keep = {k: v for k, v in P.items() if k not in ("dev_cache", "export_states")}

@torch.inference_mode()
def one():
    p = dict(keep)
    for r in rs: r.last_history = 0
    y = orig(x, p)
    for r in rs: r.rewind(R - 1)   # max rewindable (history holds R-1); context advances 1 token/forward, as in real decode
    return y

for _ in range(5): one()
torch.cuda.synchronize()
e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
t0 = time.perf_counter(); e0.record()
for _ in range(NFWD): one()
e1.record(); torch.cuda.synchronize(); host = (time.perf_counter() - t0) / NFWD * 1000
wall = e0.elapsed_time(e1) / NFWD
print(f"R={R}: {wall:.2f} ms/forward incl. rewind (device events), host loop {host:.2f} ms/forward")

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(NFWD): one()
    torch.cuda.synchronize()
def family(k):
    k = k.replace("void ", "").replace("(anonymous namespace)::", "")
    k = re.sub(r"\(.*", "", k)
    for pfx in ("moe_mk", "moe_prefill_grouped_gemv", "moe_grouped_gemv", "moe_prefill_had_rows", "moe_prefill_metadata",
                "exl3_gemv_kernel", "gr_dots", "gr_finalize", "gr_mix", "hc_apply", "had_hf_r_128", "had_ff_r_128",
                "recurrent_gated_delta_rule", "causal_conv1d", "gated_rms_norm", "_rms_norm", "rms_norm",
                "routing", "topk", "sort", "paged_attn", "paged_kv", "qsa", "mla_plane", "dsa_topk", "rope",
                "skinny", "Cijk_", "copyBuffer", "Memcpy", "elementwise", "reduce_kernel", "index", "gather",
                "state_rewind", "conv_rewind", "fill", "triton"):
        if pfx in k: return pfx
    return k[:48]
fam = collections.defaultdict(lambda: [0, 0.0]); tot = 0.0; n = 0
for e in prof.key_averages():
    d = getattr(e, "self_device_time_total", 0) or 0
    if d <= 0: continue
    f = family(e.key); fam[f][0] += e.count; fam[f][1] += d; tot += d; n += e.count
busy = tot / 1000 / NFWD
print(f"kernels/forward {n/NFWD:.0f}   device busy {busy:.2f} ms   idle {wall - busy:.2f} ms ({100*(1 - busy/wall):.1f} %)")
print(f"{'family':30} {'calls/fwd':>9} {'ms/fwd':>8} {'%busy':>6} {'us/call':>8}")
for f, (c, d) in sorted(fam.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"{f:30} {c/NFWD:9.1f} {d/1000/NFWD:8.3f} {100*d/tot:6.1f} {d/c:8.1f}")
