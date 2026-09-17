# exllamav3 on AMD Strix Halo (gfx1151)

> **Just want to run Qwen3.8-Flash-Next on a Strix Halo box?** Use the recipe: [Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe) (setup / download / run scripts, measured results, troubleshooting). This file is the engineering log behind it.

> **This tree adds AMD GPU functionality to ExLlamaV3 that upstream does not have.** Upstream is
> CUDA-only; the `sdougbrown/exllamav3` base adds a ROCm/HIP decode path for **gfx12 (RDNA 4)**
> only. Everything here extends that to **gfx11.5 (RDNA 3.5)** and tunes it for a 20-CU iGPU on
> unified memory. NVIDIA/CUDA paths are untouched — all changes sit behind `USE_ROCM` / arch guards.

Working tree for running exllamav3 on the AMD Ryzen AI Max+ 395 / Radeon 8060S
(RDNA 3.5, gfx1151, wave32, 128 GB unified memory) — the Framework Desktop.

Primary repo: GitHub `vcruz305/exllamav3-amd`.

Based on [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) branch `integration`
(gfx12 WMMA path) at `991f1a0`. Everything on top of that is in the commits after it.

## What's here

**Kernel / runtime (in `exllamav3/`):**
- **gfx1151 WMMA GEMV path** — RDNA3.5 `v_wmma_f16_16x16x16_f16` decode kernel behind the
  gfx12 dispatcher, C-fragment layout oracle-verified.
- **Int8-activation GEMV HIP port** (`exl3_gemv_int8*.cu[h]`) — `dp4a` → `v_dot4_i32_iu8`,
  register-staged B pipeline replacing `cp.async`, cooperative launch verified on gfx1151.
  Parity-proven on real tensors (argmax 1.00). **Default off on ROCm** (`EXL3_INT8_GEMV=1|2`).
- **SAD trellis decode** on gfx1151 (`v_sad_u8` gate was gfx12-only): −3–4 % kernel time.
- **CPU expert offload** port (`moe_cpu_host.py`, `block_sparse_mlp*.py`, `moe_handoff_amdgpu.cu`):
  loads and runs; measured net loss on unified memory. Default off (`-mcs 0`).
- **Build-time kernel variants** via `EXL3_HIP_DEFINES` (setup.py): `EXL3_HIP_PF_AFTER_STAGE`,
  `EXL3_MUL1_DECODE_DOT4`, `EXL3_HIP_A_RING`. All measured null (±2 %) — see below.
- Env knobs: `EXL3_MOE_CFG`, `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_HIP_SKINNY_GEMM`, `EXL3_HIP_GR_MIX_Q8`,
  `EXL3_HIP_GR_MIX_ROWS`, `EXL3_MOE_HIP_WAIT_FINE`,
  `EXL3_INT8_GEMV_BLOCKS_PER_CU`, `EXL3_GEMV`, `EXL3_GEMV_DEBUG`, `EXL3_MOE_*_PROF`.

**Harnesses (`tools/strix_halo/`):** `bench_mtp.py` (end-to-end decode), `gemv_micro.py`
(per-shape kernel timing vs. roofline), `gemv_by_m.py`, `int8_parity.py`, `moe_layer_prof.py`,
`split_prof.py`, `roofline.py`, WMMA oracles, and `env.sh` (venv + build-mode SDK env).

## Results — Qwen3.8-Flash-Next-EXL3 (3.05 bpw), decode

| config | tok/s |
|---|---|
| stock gfx12 path (reconstruct + hgemm fallback) | ~17 |
| WMMA GEMV path | 27.9 |
| + `EXL3_MOE_CFG=2` | 30.2 |
| + `EXL3_HIP_PREFILL_MIN_ROWS=2`, `-ndt 2 -dds` | **32.7 mean / 38.7 peak** |
| + int8 GEMV (`EXL3_INT8_GEMV=2`) | 31.7–31.9 (noise) |
| + CPU offload `-mcs 16/64/128` | 29.7 / 28.2 / 27.0 (loss) |
| + `EXL3_HIP_STG_PAD` (LDS bank conflict fix) | **34.9 mean / 41.1 peak** |
| + skinny fp16 GEMM for GDN b/a proj (`hgemm.cu`, replaces hipblaslt split-K) | 36.1 mean / 44.4 peak |
| + **int8 GatedResidual mixer weights** (`gr_mix_q8`, −22 % of decode bytes) | **40.0 mean / 46.3 peak** (ndt=2) · **41.3 mean / 43.7 peak** (ndt=3 dc=0.6) |

Six-prompt greedy means, 512 tokens (`prompt_sweep.py`). PPL 4.225935 unchanged through every
row. Per-round phase split (`phase_prof.py`): trunk verify forward 85–88 %, MTP head 10–12 %,
host ≈ 1 % — decode is bytes-per-forward bound. Byte budget per trunk forward after int8
mixers (`bytes_by_module.py`): routed experts 47 %, dense EXL3 GEMVs 40 %, mixers 13 %.
Every bulk kernel is now at this part's practical single-kernel streaming rate (~135 GB/s,
`read_bw_vs_size.py`): grouped-MoE prefill GEMV 131–135 GB/s on unique-expert bytes, lm_head
~200 GB/s, dense GEMVs 55–65 % of the 236 GB/s DRAM peak. The verify forward as a whole runs
at ~95 GB/s effective — the remaining ~16 ms per round is spread across ~1,300 launches per
forward (dispatch + tail effects), not concentrated in any one kernel. `EXL3_BLOCK_GRAPH`
(graph replay) measured neutral under MTP. 50 tok/s would need round ≤ 50.6 ms at 2.5
tok/round, i.e. verify ≤ 43 ms vs 55 measured; kernel-level work is exhausted on gfx1151.

PPL 4.2259, identical across all configs. Roofline 236 GB/s.

**m=3 GEMV kernel** (25 % of decode wall, ~30 % of roofline): counter profile showed 36 % VALU
issue, 17 % LDS, 144 GB/s — stall-bound, not issue- or bandwidth-bound. The three loop-structure
variants (B-prefetch-after-stage, DOT4 decode, A-fragment prefetch ring) are all within ±2 % of
baseline; A-ring is slightly negative (VGPR pressure). The per-slice `s_waitcnt vmcnt(0)` is
**not** the bottleneck.

**RESOLVED by `EXL3_HIP_STG_PAD`** (+10 % mean decode). The stall was an LDS bank conflict in
the WMMA staging buffer, findable by arithmetic rather than counters. `hip_mma_stg` was
`[warps][32][4]` of `__half2`; `__half2` is one dword and RDNA LDS has 32 banks of one dword
(`bank = dword_addr % 32`), so the innermost extent *is* the per-lane bank stride, and 4 divides 32:

| access | banks | conflict |
|---|---|---|
| `stg[lane][slot]`, lane 0..31 | `lane*4 % 32` → 8 banks | 4-way |
| `stg[4*(R&7)+j][s]`, R = lane&15 | `4*(R&7)*4 % 32` → {0,16} | **8-way** |

Padding the extent to 5 (`gcd(5,32)=1`) spreads stores over all 32 banks and cuts loads to 2-way.
PPL and draft acceptance unchanged, so the gain is kernel time. Any `__shared__` array of
one-dword elements here needs an innermost extent coprime with 32.

**Do not shrink `HIP_MMA_STG_WARPS`** to match `EXL3_MOE_CFG=2`'s 4 warps: the non-MoE GEMV paths
still launch 16 warps and index this buffer by warp id, so a 4-warp buffer gives NaN logits —
and `eval/ppl.py` still reports the correct 4.2259 on that broken build. Only `bench_mtp.py`
catches it.

## Build

```bash
source tools/strix_halo/env.sh build            # SDK toolchain from .venv-gfx1151
EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD" pip install -e . --no-build-isolation --no-deps
source tools/strix_halo/env.sh                  # runtime env (LD_PRELOAD Ubuntu HSA)
python tools/strix_halo/bench_mtp.py -n 128 -ndt 2 -dds -g
```

Build and bench in **separate shells** — `env.sh build` exports `ROCM_PATH`/`HIP_PATH`/
`PYTORCH_ROCM_ARCH` and a bench in the same shell fails model load with a spurious
"Insufficient VRAM".

Known transient: hipblaslt OOM at model load right after a previous process exits — sleep and retry.
