# exllamav3 on AMD Strix Halo (gfx1151)

Private working tree for running exllamav3 on the AMD Ryzen AI Max+ 395 / Radeon 8060S
(RDNA 3.5, gfx1151, wave32, 128 GB unified memory) — the Framework Desktop.

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
- Env knobs: `EXL3_MOE_CFG`, `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_MOE_HIP_WAIT_FINE`,
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

PPL 4.2259, identical across all configs. Roofline 236 GB/s.

**m=3 GEMV kernel** (25 % of decode wall, ~30 % of roofline): counter profile shows 36 % VALU
issue, 17 % LDS, 144 GB/s — stall-bound, not issue- or bandwidth-bound. The three loop-structure
variants (B-prefetch-after-stage, DOT4 decode, A-fragment prefetch ring) are all within ±2 % of
baseline; A-ring is slightly negative (VGPR pressure). The per-slice `s_waitcnt vmcnt(0)` is
**not** the bottleneck. Next candidates need `SQ_LDS_BANK_CONFLICT` / WMMA-issue counters.

## Build

```bash
source tools/strix_halo/env.sh build            # SDK toolchain from .venv-gfx1151
EXL3_HIP_DEFINES="" pip install -e . --no-build-isolation --no-deps
source tools/strix_halo/env.sh                  # runtime env (LD_PRELOAD Ubuntu HSA)
python tools/strix_halo/bench_mtp.py -n 128 -ndt 2 -dds -g
```

Build and bench in **separate shells** — `env.sh build` exports `ROCM_PATH`/`HIP_PATH`/
`PYTORCH_ROCM_ARCH` and a bench in the same shell fails model load with a spurious
"Insufficient VRAM".

Known transient: hipblaslt OOM at model load right after a previous process exits — sleep and retry.
