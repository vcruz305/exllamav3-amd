# ROCm 10.1 on gfx1151

ROCm 10.1 (TheRock wheels: torch 2.14.0+rocm10.1.0, HIP 7.16, LLVM/AMD clang 24, hipBLASLt
1.4.1, Triton 3.8) A/B'd against the shipped stack (torch 2.10.0+rocm7.0, nightly 7.x SDK for the
build) on framework2 (Ryzen AI Max+ 395 / Radeon 8060S). Same commit, same harnesses, same knobs,
one GPU job at a time.

## Setup

`tools/strix_halo/r101_setup.sh` creates `.venv-r101` next to the production `.venv`:

```
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" "rocm[libraries,devel,device-gfx1151]==10.1.0"
```

`tools/strix_halo/r101_build.sh` builds the extension in a separate git worktree (`~/exl3-r101`)
so the production tree's root `.so` is never touched. Clean build: 146 s, 0 warnings.

What changes against `env.sh`'s five traps:

| trap (env.sh) | ROCm 7.0 stack | ROCm 10.1 |
|---|---|---|
| 1. bundled HSA runtime segfaults on first alloc | needs `LD_PRELOAD` of Ubuntu's libhsa | **fixed**: runs with no preload, in `env -i` |
| 3. torch wheel ships no hipcc | borrow a separate `.venv-gfx1151` SDK | `rocm[devel]` in the same venv has hipcc + amdclang++ |
| 4. bitcode not where clang probes | pass `--rocm-device-lib-path` | still needed (same `lib/llvm/amdgcn/bitcode` layout) |
| 5. SDK paths must not be exported at runtime | yes | not needed: runtime needs no SDK env at all |

## Results

| | ROCm 7.0 | ROCm 10.1 | |
|---|---|---|---|
| fp16 GEMM 4096³ (torch) | 33.6 TFLOP/s | 34.4 | same |
| fp16-in / **fp32-out** GEMM (hipBLASLt HSS) | 6.5–6.8 | 6.4–7.3 | **still broken** |
| prefill 8k, flat 3.05 bpw (warm, 3 alternating pairs) | 721 / 723 / 722 | 740 / 755 / 754 | **+3.6 %** |
| prefill 8k, flat, cold page cache (2 pairs) | 707 / 707 | 745 / 745 | **+5.4 %** |
| prefill 8k, CYBER-FROST mixed-K (2 pairs) | 698 / 699 | 731 / 734 | **+4.8 %** |
| decode six-prompt flat (3 runs) | 46.41 / 46.47 / 46.45 | 44.83 / 46.04 / 46.05 | parity (−0.9 %) |
| decode six-prompt mixed-K (3 runs) | 39.89 / 39.17 / 39.37 | — / 39.94 / 39.80 | parity (+0.9 %) |
| MTP acceptance | 72–73 % | 72–73 % | same |

Six-prompt = `prompt_sweep.py` NDT=3 DC=0.6 DDS=1, 512 tokens, greedy, the production default.
The 44.83 is the first 10.1 run after the build; the two interleaved repeats read 46.04 / 46.05.

### Where the prefill gain comes from

`prefill_profile.py` (8k flat, per-module sync timing):

| module | ROCm 7.0 | ROCm 10.1 |
|---|---|---|
| BlockSparseMLP (grouped MoE GEMV) | 4.82 s | 4.80 s |
| GatedDeltaNet | 3.24 s | 3.00 s |
| Attention | 1.38 s | 1.46 s |
| GatedResidual._mix (Triton) | 1.34 s | 0.97 s |

None of it is in our HIP kernels: `exl3_moe_gfx12_k3_prefill` (18.1 vs 18.0 ms/call) and
`hgemm_recon` (0.93 vs 0.95 ms/call) are unchanged. The gain is Triton 3.8 codegen for the
Triton paths (GatedResidual mix and norms, the GDN chunk kernels). Attention is slightly slower.

### Decode

Kernel census (`decode_census.py`, 128 tokens, MTP): device time per kernel family is within 2 %
between toolchains (`exl3_gemv`, grouped MoE, `gr_*`, GDN recurrent). AMD clang 24 allocates a few
more VGPRs in `exl3_gemv_kernel` (e.g. 85 → 89, 106 → 113) and fewer SGPRs; no spills on either
(`kernel_meta_so.py`). At these counts occupancy does not change, which matches the flat timings.

Host launch cost is **higher** on 10.1 under the profiler: `hipLaunchKernel` 3.8–5.0 µs vs
2.1 µs, `hipModuleLaunchKernel` 11.7–15.0 vs 8.2–8.7 µs. Decode hides it (the GPU queue stays fed),
which is why tok/s is unchanged.

## Numerics

PPL (`eval/ppl.py -r 20 -l 1024`):

| | ROCm 7.0 | ROCm 10.1 |
|---|---|---|
| flat, F32OUT_VIA_F16=1 (default) | 4.218831 | 4.228280 |
| flat, F32OUT_VIA_F16=0 | 4.230441 | 4.222568 |
| flat, torch norms, F32OUT_VIA_F16=0 | 4.225935 | 4.221475 |
| CYBER-FROST mixed-K | 4.261538 | 4.258925 |

Different toolchains do not reproduce bit-for-bit, so the PPL deltas above are judged against a
noise floor. `logit_parity.py` compares full logits (4 × 1023 wikitext positions, 248k vocab) and
`logit_floor.sh` measures known-benign changes on ROCm 7 for scale:

| comparison | top-1 flips | margin at flips (median) | KL per token (mean) |
|---|---|---|---|
| ROCm 7 vs ROCm 7 rerun | 0 | - | 0 |
| ROCm 7: Triton norms vs torch norms (summation order only) | 2.86 % | 0.34 | 1.54e-2 |
| ROCm 7: F32OUT_VIA_F16 on vs off | 3.18 % | 0.28 | 1.75e-2 |
| **ROCm 7 vs ROCm 10.1** | **3.37 %** | **0.34** | **1.77e-2** |

The toolchain change sits inside the same band as a summation-order change. Flips occur at
near-ties (median top-2 margin 0.34 logits vs 5.19 overall). Both toolchains are deterministic.

## What 10.1 does and does not fix

- **Fixed:** the HSA segfault (no `LD_PRELOAD`), the two-venv build. Setup becomes one venv.
- **Not fixed:** hipBLASLt still has no fast fp16-in/fp32-out kernel for gfx1151 (6.4–7.3 TFLOP/s
  vs 33–41 for fp16-out) with every backend switch tried (`ROCBLAS_USE_HIPBLASLT=0/1`,
  `TORCH_BLAS_PREFER_HIPBLASLT=0`, heuristics off; `r101_blas.sh`). Keep
  `EXL3_HIP_F32OUT_VIA_F16=1`.
- **New cost:** ~2x host time per kernel launch. Decode does not see it: wall time per token is
  set by device work plus the per-round host syncs in the generator (`.cpu()` / `.item()` on draft
  confidences), not by launch issue rate.
- **Graph replay still does not pay:** `EXL3_BLOCK_GRAPH=1` reads 45.33 / 45.32 vs 46.02 / 46.04
  off on 10.1 (`r101_graph.sh`), a small loss as on ROCm 7. Leave it off.

## Recommendation

Moving the recipe to 10.1 is worth it for setup simplicity and +4-5 % prefill. Decode and accuracy
are unchanged. It is not a default switch yet: the published `.so` and every number in the README
are on the ROCm 7.0 stack, and 10.1 changes outputs at the near-tie level (above), so the recipe's
validation numbers would need to be re-baselined in the same change.

## Harnesses

- `r101_setup.sh`, `r101_build.sh`: side-by-side venv and worktree build.
- `r101_ab.sh`, `r101_ab2.sh`: GEMM micro, PPL, prefill, interleaved six-prompt decode.
- `r101_pf_order.sh`: prefill with alternating order and cold page cache.
- `r101_census.sh` + `decode_census.py`: per-kernel decode census per arm.
- `r101_blas.sh`: HSS GEMM under each BLAS backend switch.
- `logit_parity.py`, `logit_floor.sh`: logit-level toolchain comparison and noise floor.
- `kernel_meta_so.py`: per-kernel VGPR/SGPR/LDS/spill diff between two built `.so` files.
- `r101_graph.sh`: `EXL3_BLOCK_GRAPH` A/B on 10.1.
