"""Triton kernels for the prefill-size (R > FUSED_MAX_R) GatedResidual mix.

The torch form of the mix runs two memory-bound stages at ~26 GB/s on gfx1151 at prefill
row counts: ext.rms_norm over (R*H, D) with a 4-row grouped weight, and the
sigmoid(g) * normed -> mean-over-streams tail, which torch splits into five elementwise
kernels with fp32 temporaries. Both are replaced here by one-pass kernels; the two small
GEMMs in between stay on hipBLASLt.

  gr_norm:  (R*H, D) fp32 -> (R*H, D) fp16, y = x * rsqrt(mean(x^2) + eps) * w[row % H]
  gr_tail:  g (R, H*D) fp16, normed (R, H, D) fp16 -> mixed (R, D) fp16,
            mixed = mean_h(sigmoid(g[:, h]) * normed[:, h])

Numerics follow the torch reference: fp32 reduction and fp32 math, rounded to fp16 on store.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gr_norm_kernel(x, w, y, D, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    xv = tl.load(x + row.to(tl.int64) * D + offs, mask = mask, other = 0.0).to(tl.float32)
    var = tl.sum(xv * xv, axis = 0) / D
    rmf = tl.rsqrt(var + eps)
    wv = tl.load(w + (row % H) * D + offs, mask = mask, other = 0.0).to(tl.float32)
    tl.store(y + row.to(tl.int64) * D + offs, (xv * rmf * wv).to(tl.float16), mask = mask)


@triton.jit
def _gr_tail_kernel(g, n, out, D, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    base = row.to(tl.int64) * H * D
    acc = tl.zeros((BLOCK,), tl.float32)
    for h in tl.static_range(H):
        gv = tl.load(g + base + h * D + offs, mask = mask, other = 0.0).to(tl.float32)
        nv = tl.load(n + base + h * D + offs, mask = mask, other = 0.0).to(tl.float32)
        acc += tl.sigmoid(gv) * nv
    tl.store(out + row.to(tl.int64) * D + offs, (acc / H).to(tl.float16), mask = mask)


def gr_norm(x: torch.Tensor, w: torch.Tensor, H: int, eps: float, out: torch.Tensor):
    """x (rows, D) fp32 contiguous, w (H, D) or (H*D,) fp16 -> out (rows, D) fp16."""
    rows, D = x.shape
    BLOCK = triton.next_power_of_2(D)
    with torch.cuda.device(x.device):
        _gr_norm_kernel[(rows,)](x, w, out, D, eps, H = H, BLOCK = BLOCK, num_warps = 8)
    return out


def gr_tail(g: torch.Tensor, normed: torch.Tensor, R: int, H: int, D: int):
    """g (R, H*D) fp16, normed (R*H, D) fp16 -> (R, D) fp16."""
    out = torch.empty((R, D), dtype = torch.half, device = g.device)
    BLOCK = 512
    with torch.cuda.device(g.device):
        _gr_tail_kernel[(R, triton.cdiv(D, BLOCK))](g, normed, out, D, H = H, BLOCK = BLOCK, num_warps = 4)
    return out
