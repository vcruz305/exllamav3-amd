"""Triton RMS norms for ROCm, standing in for norm.cu (excluded from the HIP build).

ext_fallbacks.py implements rms_norm / rms_norm_res_in / gated_rms_norm as multi-pass PyTorch:
an fp32 copy of the input, square, mean, rsqrt, one or two multiplies, a cast and a copy_ --
six to nine full-tensor passes, each through an fp32 temporary. At prefill row counts on
gfx1151 that is ~26 GB/s effective, and every RMSNorm in the model pays it on every chunk.
These kernels do one read and one write per element. Semantics match the fallbacks exactly
(same fp32 math, same rounding points), so results agree to fp32 reassociation in the mean.

Each wrapper validates its arguments the same way as the fallback it replaces and returns
False for any case it does not handle, so the caller can fall back for that call.
"""
import torch
import triton
import triton.language as tl

_MAX_DIM = 16384


@triton.jit
def _rms_norm_kernel(
    x, w, y, rows, dim, eps, constant_bias, constant_scale, w_groups,
    HAS_W: tl.constexpr, ADD_RES: tl.constexpr, CLAMP: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row.to(tl.int64) * dim
    xf = tl.load(x + base + offs, mask = mask, other = 0.0).to(tl.float32)
    if CLAMP:
        xf = tl.minimum(tl.maximum(xf, -65504.0), 65504.0)
    var = tl.sum(xf * xf, axis = 0) / dim
    xf = xf * tl.rsqrt(var + eps)
    xf = xf * constant_scale
    if HAS_W:
        wv = tl.load(w + (row % w_groups) * dim + offs, mask = mask, other = 0.0).to(tl.float32)
        xf = xf * (wv + constant_bias)
    if ADD_RES:
        yv = tl.load(y + base + offs, mask = mask, other = 0.0).to(tl.float32)
        xf = yv + xf
    tl.store(y + base + offs, xf.to(y.dtype.element_ty), mask = mask)


@triton.jit
def _rms_norm_res_in_kernel(
    x, w, y, r, dim, eps, constant_bias, constant_scale,
    HAS_W: tl.constexpr, CLAMP: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row.to(tl.int64) * dim
    xa = tl.load(x + base + offs, mask = mask, other = 0.0).to(tl.float32)
    if CLAMP:
        xa = tl.minimum(tl.maximum(xa, -65504.0), 65504.0)
    rv = tl.load(r + base + offs, mask = mask, other = 0.0).to(tl.float32)
    # residual is rounded to its own dtype before the norm reads it back (matches r.copy_ + r.float())
    rn = (rv + xa).to(r.dtype.element_ty)
    tl.store(r + base + offs, rn, mask = mask)
    rf = rn.to(tl.float32)
    var = tl.sum(rf * rf, axis = 0) / dim
    rf = rf * tl.rsqrt(var + eps) * constant_scale
    if HAS_W:
        wv = tl.load(w + offs, mask = mask, other = 0.0).to(tl.float32)
        rf = rf * (wv + constant_bias)
    tl.store(y + base + offs, rf.to(y.dtype.element_ty), mask = mask)


@triton.jit
def _gated_rms_norm_kernel(
    x, w, y, g, dim, eps, constant_bias, w_groups,
    GATE_FIRST: tl.constexpr, SIGMOID: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row.to(tl.int64) * dim
    xf = tl.load(x + base + offs, mask = mask, other = 0.0).to(tl.float32)
    gf = tl.load(g + base + offs, mask = mask, other = 0.0).to(tl.float32)
    sg = tl.sigmoid(gf)
    gate = sg if SIGMOID else gf * sg
    if GATE_FIRST:
        xf = xf * gate
    var = tl.sum(xf * xf, axis = 0) / dim
    h = xf * tl.rsqrt(var + eps)
    wv = tl.load(w + (row % w_groups) * dim + offs, mask = mask, other = 0.0).to(tl.float32)
    h = h * (wv + constant_bias)
    if not GATE_FIRST:
        h = h * gate
    tl.store(y + base + offs, h.to(y.dtype.element_ty), mask = mask)


def _cfg(dim):
    block = triton.next_power_of_2(dim)
    return block, (4 if block <= 1024 else 8 if block <= 4096 else 16)


def _flat(t, dim):
    return t.view(-1, dim) if t.is_contiguous() else None


def rms_norm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups = 1):
    if x.dtype not in (torch.float16, torch.float32) or y.dtype not in (torch.float16, torch.float32):
        return False
    if w is not None and w.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.shape != y.shape or w_groups < 1 or (span_heads and w_groups != 1):
        return False
    if not (x.is_cuda and x.is_contiguous() and y.is_contiguous() and (w is None or w.is_contiguous())):
        return False
    dim = x.shape[-2] * x.shape[-1] if span_heads else x.shape[-1]
    if dim > _MAX_DIM or dim == 0:
        return False
    if w is not None and w.numel() != w_groups * dim:
        return False
    rows = x.numel() // dim
    if rows == 0:
        return True
    BLOCK, nw = _cfg(dim)
    with torch.cuda.device(x.device):
        _rms_norm_kernel[(rows,)](
            x, w if w is not None else x, y, rows, dim, float(eps), float(constant_bias),
            float(constant_scale), w_groups,
            HAS_W = w is not None, ADD_RES = bool(add_residual), CLAMP = x.dtype == torch.float16,
            BLOCK = BLOCK, num_warps = nw,
        )
    return True


def rms_norm_res_in(x, w, y, r, eps, constant_bias, constant_scale):
    if x.dtype not in (torch.float16, torch.float32) or r.dtype not in (torch.float16, torch.float32):
        return False
    if y.dtype != torch.float16 or (w is not None and w.dtype not in (torch.float16, torch.bfloat16)):
        return False
    if x.shape != y.shape or x.shape != r.shape:
        return False
    if not (x.is_cuda and x.is_contiguous() and y.is_contiguous() and r.is_contiguous()
            and (w is None or w.is_contiguous())):
        return False
    dim = x.shape[-1]
    if dim > _MAX_DIM or dim == 0 or (w is not None and w.numel() != dim):
        return False
    rows = x.numel() // dim
    if rows == 0:
        return True
    BLOCK, nw = _cfg(dim)
    with torch.cuda.device(x.device):
        _rms_norm_res_in_kernel[(rows,)](
            x, w if w is not None else x, y, r, dim, float(eps), float(constant_bias),
            float(constant_scale),
            HAS_W = w is not None, CLAMP = x.dtype == torch.float16, BLOCK = BLOCK, num_warps = nw,
        )
    return True


def gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act = 0):
    if x.dtype != torch.bfloat16 or y.dtype not in (torch.float16, torch.float32):
        return False
    if w.dtype not in (torch.bfloat16, torch.float32) or g.dtype not in (torch.bfloat16, torch.float32):
        return False
    if x.shape != y.shape or x.shape != g.shape or w_groups < 1 or gate_act not in (0, 1):
        return False
    if not (x.is_cuda and x.is_contiguous() and y.is_contiguous() and g.is_contiguous() and w.is_contiguous()):
        return False
    dim = x.shape[-1]
    if dim > _MAX_DIM or dim == 0 or w.numel() != w_groups * dim:
        return False
    rows = x.numel() // dim
    if rows == 0:
        return True
    BLOCK, nw = _cfg(dim)
    with torch.cuda.device(x.device):
        _gated_rms_norm_kernel[(rows,)](
            x, w, y, g, dim, float(eps), float(constant_bias), w_groups,
            GATE_FIRST = bool(gate_first), SIGMOID = gate_act == 1, BLOCK = BLOCK, num_warps = nw,
        )
    return True
