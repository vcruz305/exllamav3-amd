# Mixed-K MoE on gfx1151

How mixed-K EXL3 packs (experts quantized at different bitrates) run on this fork, and why it
is built this way. Measured on framework2 (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151, 128 GB).

## Why the CUDA mixed-K kernels are not ported

The NVIDIA fork's mixed-K work (`exl3_moe_mixedk_*`, CoopMK) lives in `exl3_moe.cu`,
`exl3_moe_coop.cu`, `comp_units/` and `libtorch/blocksparse_mlp.cpp`. `build_config.py`
excludes all of them from ROCm builds; they pull in CUDA-only headers, so copying them across
does nothing. The fused MoE path that does exist on ROCm is the grouped GEMV route in
`quant/exl3_gemv.cu` (`exl3_moe_gfx12_k3` / `_prefill`), whose GEMV body is compiled for K=3.

## What blocked mixed-K packs

1. `multilinear.py` asserts one K per MultiLinear, so a mixed-K pack aborted at load.
2. With that gated off, mixed-K layers fell to the dense per-expert loop: about 3,450 device
   calls per token, each paying ~11 us of fixed launch/setup cost to move ~4 us of bytes.
   6.5 tok/s without MTP. A no-op experiment (device calls stubbed, Python unchanged) put
   ~65 % of that time on the device side, so the fix had to cut device calls, not host Python.
3. High-bitrate packs store their non-expert weights (attention, linear attention, shared
   experts, lm_head) at 8 bpw. The dense HIP GEMV stopped at K=6, so every one of them went
   through reconstruct + hgemm: ~350 reconstructs per token, about 90 ms of the 143 ms per
   token left after the MoE fix.

## What this change does

**Mixed-K grouped route** (`exl3_moe_mk`, `exl3_moe_mk_prefill`). Same stage sequence and the
same helper kernels as the K3 route (input Hadamard, gate/up GEMV, svh, SiLU, down GEMV,
weighted reduce). Only the two GEMV stages differ: each is launched once per bitrate the layer
uses, over the full slot grid. A block whose expert has a different K returns before touching
memory, so every slot is computed exactly once, by the launch compiled for its K. The bitrate
list per projection is fixed at load, so there is no host sync. Per-expert K tables are device
`int32[E]`. One translation unit per bitrate (`exl3_gemv_moe_mk_k{3..7}.cu`).

Routing (`block_sparse_mlp.py`): the K3 route keeps every layer it had; `support_hip_grouped`
is unchanged for uniform-K3 packs. The mixed-K route takes the grouped-shape layers the K3
kernel cannot: mixed K, or uniform K other than 3. `EXL3_HIP_MOE_MK=0` turns it off.

**Dense HIP GEMV at 7 and 8 bpw.** The GEMV body already decoded 5 and 6 bpw through
`dq_dispatch` from the staged tile. 7 and 8 use the same staged path (`LOADS` rounds up as for
K5), with instances added to `exl3_gemv_select_kernel` and the K caps raised to 8 in
`exl3_gemv_cfg`, `exl3_gemv_try_launch` and `LinearEXL3.forward`.

## Results

Per-token device calls on CYBER-FROST, decode without MTP (`mk_census.py`):

| | before | after |
|---|---|---|
| wall time per token | 188 ms | 51 ms |
| device calls per token | ~3,450 | ~490 |
| reconstruct + hgemm per token | ~390 | 0 |

Six-prompt mean (`prompt_sweep.py`, NDT=3 DC=0.6 DDS=1, 512 tokens, greedy, the production
default; `EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2`):

| pack | mean tok/s | median | min / max | acceptance |
|---|---|---|---|---|
| Qwen3.8-Flash-Next 3.05 bpw (flat K3) | 44.74 | 45.00 | 41.61 / 47.62 | 71.0 % |
| CYBER-FROST 3.87 bpw (mixed K) | **39.27** | 39.38 | 34.49 / 44.96 | 74.0 % |

Single prompt, before vs after (`bench_mtp.py -g`, "Explain gradient descent in two
sentences:", 256 tokens, cache 2048):

| pack | config | before | after |
|---|---|---|---|
| flat K3 | no MTP | 26.55 | 26.35 |
| flat K3 | MTP ndt3 dds dc0.6 | 43.50 | 43.65 |
| mixed K | no MTP | 6.50 | **22.27** |
| mixed K | MTP ndt3 dds dc0.6 | 10.42 | **36.87** |

The flat-K pack is unchanged, inside the README's >1 tok/s run-to-run drift.

Why mixed-K stays below flat-K: it reads more bytes per token. From `tensor_storage`,
non-expert weights plus the routed share of the experts (10 of 512):

| pack | dense weights | expert weights | bytes per token |
|---|---|---|---|
| Qwen3.8-Flash-Next 3.05 bpw | 3.46 GiB | 42.63 GiB | 4.30 GiB |
| CYBER-FROST 3.87 bpw | 4.71 GiB | 53.50 GiB | 5.76 GiB |

That is 1.34x the bytes. Scaling the flat-K six-prompt mean by the byte ratio predicts
33.4 tok/s; mixed-K measures 39.27 (88 % of flat-K), helped by its higher MTP acceptance.
The remaining gap is bytes, not dispatch.

## Correctness

- `tools/strix_halo/mk_parity.py`: layer forward, grouped route vs the dense reference path,
  real routing at rows 1, 3, 4, 16 (decode) and 37 (prefill) on four layers, including the one
  with the most K7 experts, plus forced routing onto K7/K6 experts. Worst relative error
  2.4e-4, cosine >= 0.99999994. PASS.
- `tools/strix_halo/gemv_k_parity.py`: dense HIP GEMV vs reconstruct + hgemm for every
  (K, shape) class in the pack, K=3..8 including lm_head, rows 1/3/9/16. Worst 1.5e-6. PASS.
- PPL (`eval/ppl.py -r 20 -l 1024`):
  - CYBER-FROST: grouped route 4.262460; dense reference path 4.263050.
  - Qwen3.8-Flash-Next: 4.225935 with `EXL3_HIP_F32OUT_VIA_F16=0`, identical on the old and
    new builds. With that knob at its default (1), both builds read 4.223205. The 4.225935 in
    AGENTS.md predates that knob (commit fa32aaa); this change moves neither number.
- Generated text is coherent with the reasoning block intact; MTP acceptance 74 % (six-prompt).

## Knobs

- `EXL3_HIP_MOE_MK=0` disables the mixed-K grouped route (dense per-expert loop).
- `EXL3_GEMV=0` still disables all HIP GEMV, including this route.
- `EXL3_HIP_GROUPED_MOE=0` / `EXL3_HIP_GROUPED_MOE_PREFILL=0` disable the decode / prefill
  grouped routes, K3 and mixed-K alike.

## Harnesses

- `mk_parity.py`, `gemv_k_parity.py`: correctness, above.
- `mk_census.py`: per-token extension call counts and which (K, rows) still hit
  reconstruct + hgemm.
- `mk_ab.sh`: flat-K vs mixed-K, with and without MTP, sequential on one box.
- `mk_ppl_control.sh`: old/new `.so` x `EXL3_HIP_F32OUT_VIA_F16` PPL matrix.
- `prompt_sweep.py`: takes `MODEL=` and `CACHE=` so the six-prompt mean runs on any pack.
