"""Tile-union QSA sparse attention for prefill (one sequence, many query rows).

qsa_sparse_attend_rows runs one program per (query row, kv head): every row gathers its own
~budget selected keys/values, so a 2048-row prefill chunk re-gathers ~2050 tokens 2048 times
(8.6 GB of gathers per layer at 8k context, 52 ms -- five times slower than dense attention
over the whole context). Consecutive query rows select nearly the same blocks: the union over
16 rows is 1.05-1.6x one row's set. Here one program owns a tile of TM consecutive rows (x HB
heads of one kv head), iterates the UNION of the tile's selected blocks once, and masks per row.

Exactness: a row attends token t iff the row selected t's pooled block AND t <= the row's
position. That is precisely the original token set: selected blocks are complete (every token is
<= the query position) and the forced tail block keeps only tokens <= the query position. No
token appears twice in a row's list (the tail block is never a selected block). The selection
itself (indexer scores, top-k) is untouched; only the gather/attend is restructured. Differences
vs the per-row kernel are summation order only.
"""
import torch
import triton
import triton.language as tl

from .triton_paged import _qc_load_kt, _qc_load_v, _rot_h32, _get_h32

TM = 16          # query rows per tile (bit width of the per-block row mask)


@triton.jit(do_not_specialize = ["R", "pos0", "NBP"])
def _qsa_tile_kernel(
    q,                   # (R, n_q_heads, head_dim) fp16
    k_cache, v_cache,
    block_table,         # (num_pages,) int32, this sequence
    tile_blocks,         # (T, NBP) int32: union pooled-block ids per tile, compacted
    tile_masks,          # (T, NBP) int32: bit r = tile row r selected that block
    tile_counts,         # (T,) int32
    out,                 # (R, n_q_heads, head_dim) fp16
    R, pos0, NBP,
    k_scales, v_scales, h32,
    n_q_heads: tl.constexpr, n_kv_heads: tl.constexpr, page_size: tl.constexpr,
    head_dim: tl.constexpr, scale: tl.constexpr, CR: tl.constexpr,
    TM_: tl.constexpr, HB: tl.constexpr, BLOCK_N: tl.constexpr,
    QCK: tl.constexpr = 0, QCV: tl.constexpr = 0,
):
    tile = tl.program_id(0)
    pid1 = tl.program_id(1)
    group = n_q_heads // n_kv_heads
    hblocks = tl.cdiv(group, HB)
    kv_head = pid1 // hblocks
    hb = pid1 % hblocks

    ROWS: tl.constexpr = TM_ * HB
    i = tl.arange(0, ROWS)
    rl = i // HB
    hl = hb * HB + (i % HB)
    qrow = tile * TM_ + rl
    valid_row = (qrow < R) & (hl < group)
    head = kv_head * group + hl
    qpos = pos0 + qrow

    offs_d = tl.arange(0, head_dim)
    q_off = (qrow * n_q_heads + head) * head_dim
    q_tile = tl.load(q + q_off[:, None] + offs_d[None, :], mask = valid_row[:, None], other = 0.0)
    if QCK > 0:
        q_tile = _rot_h32(q_tile, h32, ROWS, head_dim)

    cnt = tl.load(tile_counts + tile)
    m = tl.full((ROWS,), -float("inf"), tl.float32)
    l = tl.full((ROWS,), 0.0, tl.float32)
    acc = tl.zeros((ROWS, head_dim), tl.float32)
    base = tile.to(tl.int64) * NBP

    for t0 in range(0, cnt * CR, BLOCK_N):
        offs_t = t0 + tl.arange(0, BLOCK_N)
        entry = offs_t // CR
        valid_n = entry < cnt
        blk = tl.load(tile_blocks + base + entry, mask = valid_n, other = 0)
        msk = tl.load(tile_masks + base + entry, mask = valid_n, other = 0)
        tok = blk * CR + offs_t % CR
        phys = tl.load(block_table + tok // page_size, mask = valid_n, other = 0)
        prow = phys * page_size + tok % page_size

        if QCK > 0:
            k_tile = _qc_load_kt(k_cache, k_scales, prow, kv_head, offs_d, valid_n, QCK, n_kv_heads, head_dim, head_dim)
        else:
            k_tile = tl.load(k_cache + ((prow[None, :] * n_kv_heads + kv_head) * head_dim + offs_d[:, None]),
                             mask = valid_n[None, :], other = 0.0)
        scores = tl.dot(q_tile, k_tile) * scale

        sel = ((msk[None, :] >> rl[:, None]) & 1) != 0
        valid = valid_row[:, None] & valid_n[None, :] & sel & (tok[None, :] <= qpos[:, None])
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)

        if QCV > 0:
            v_tile = _qc_load_v(v_cache, v_scales, prow, kv_head, offs_d, valid_n, QCV, n_kv_heads, head_dim, head_dim)
        else:
            v_tile = tl.load(v_cache + ((prow[:, None] * n_kv_heads + kv_head) * head_dim + offs_d[None, :]),
                             mask = valid_n[:, None], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new

    o = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    if QCV > 0:
        o = _rot_h32(o, h32, ROWS, head_dim)
    tl.store(out + q_off[:, None] + offs_d[None, :], o.to(tl.float16), mask = valid_row[:, None])


def build_tile_union(indices: torch.Tensor, pos0: int, cr: int):
    """indices (R, K_pad) int32 per-row selected cache positions (-1 padded) for ONE sequence
    whose chunk starts at cache position pos0 -> (tile_blocks, tile_masks, tile_counts, NBP).
    No host synchronization."""
    R, K = indices.shape
    dev = indices.device
    T = (R + TM - 1) // TM
    NB = (pos0 + R + cr - 1) // cr + 1
    idx = indices.long()
    r = torch.arange(R, device = dev).view(R, 1)
    # one representative per (row, block): the block's first token (present for every block
    # that holds any token <= the row's position)
    rep = (idx >= 0) & (idx % cr == 0)
    flat = torch.where(rep, (r // TM) * NB + idx // cr, torch.full_like(idx, T * NB))
    val = torch.where(rep, (1 << (r % TM)).expand_as(idx), torch.zeros_like(idx)).int()
    bitmap = torch.zeros((T * NB + 1,), dtype = torch.int32, device = dev)
    bitmap.scatter_add_(0, flat.flatten(), val.flatten())     # bits are distinct per (row, block)
    bitmap = bitmap[: T * NB].view(T, NB)
    nz = bitmap != 0
    pos = torch.cumsum(nz, dim = 1, dtype = torch.int32) - 1
    counts = nz.sum(dim = 1, dtype = torch.int32)
    dest = torch.where(nz, pos, torch.full_like(pos, NB)).long()
    tile_blocks = torch.empty((T, NB + 1), dtype = torch.int32, device = dev)
    tile_masks = torch.empty((T, NB + 1), dtype = torch.int32, device = dev)
    tile_blocks.scatter_(1, dest, torch.arange(NB, device = dev, dtype = torch.int32).expand(T, NB).contiguous())
    tile_masks.scatter_(1, dest, bitmap)
    return tile_blocks, tile_masks, counts, NB + 1


def qsa_sparse_attend_tiled(
    q: torch.Tensor,               # (R, n_q_heads, head_dim) fp16, one sequence's chunk rows
    k: torch.Tensor, v: torch.Tensor,
    indices: torch.Tensor,         # (R, K_pad) int32 cache positions, -1 padded
    sm_scale: float,
    block_table: torch.Tensor,     # (num_pages,) int32 for this sequence
    page_size: int,
    pos0: int,                     # cache position of the chunk's first row
    cr: int,
    qc: tuple | None = None,
    n_kv_heads: int | None = None,
) -> torch.Tensor:
    R, H, hd = q.shape
    if qc is not None:
        k_scales, v_scales, k_bits, v_bits = qc
        kvh = n_kv_heads
        h32 = _get_h32(q.device)
    else:
        k_scales, v_scales, k_bits, v_bits = q, q, 0, 0
        kvh = k.shape[1]
        h32 = q
    group = H // kvh
    HB = 4                      # measured best on gfx1151 (HB 2/8, BLOCK_N 16/64, 2/8 warps all slower)
    while group % HB: HB //= 2
    tb, tmsk, cnt, NBP = build_tile_union(indices, pos0, cr)
    T = tb.shape[0]
    o = torch.empty((R, H, hd), dtype = torch.half, device = q.device)
    grid = (T, kvh * triton.cdiv(group, HB))
    with torch.cuda.device(q.device):
        _qsa_tile_kernel[grid](
            q, k, v, block_table, tb, tmsk, cnt, o, R, pos0, NBP,
            k_scales, v_scales, h32,
            n_q_heads = H, n_kv_heads = kvh, page_size = page_size, head_dim = hd,
            scale = float(sm_scale), CR = cr, TM_ = TM, HB = HB, BLOCK_N = 32,
            QCK = k_bits, QCV = v_bits, num_warps = 4, num_stages = 1,
        )
    return o
