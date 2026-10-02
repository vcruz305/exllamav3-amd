#!/usr/bin/env python
"""Strided vs contiguous input to the Triton causal conv (both seq paths), incl. conv_state."""
import torch
from exllamav3.modules.gated_delta_net_fn.conv1d import causal_conv1d_update_slotted_triton as conv
dev = torch.device("cuda:0"); torch.manual_seed(0)
fails = 0
for b, s, d, hist in [(1, 2048, 10240, False), (1, 200, 10240, False), (2, 300, 1024, True), (1, 33, 512, False)]:
    qkv = torch.randn((b, s, d), device=dev)
    xs = qkv.transpose(1, 2).to(torch.bfloat16)          # strided view
    xc = xs.contiguous()
    w = torch.randn((d, 4), device=dev).bfloat16(); bias = torch.randn((d,), device=dev).bfloat16()
    st0 = torch.randn((b, d, 4 + (8 if hist else 0)), device=dev).bfloat16(); st1 = st0.clone()
    slots = torch.arange(b, device=dev, dtype=torch.int32)
    o0 = conv(xc, st0, slots, w, bias, transpose_output=True, history=hist)
    o1 = conv(xs, st1, slots, w, bias, transpose_output=True, history=hist)
    e = (o0.float() - o1.float()).abs().max().item(); es = (st0.float() - st1.float()).abs().max().item()
    ok = e == 0 and es == 0; fails += not ok
    print(f"b={b} s={s} d={d} hist={hist}: out max|d|={e} state max|d|={es} {'OK' if ok else 'FAIL'}")
print("CONV PARITY", "PASS" if not fails else "FAIL")
