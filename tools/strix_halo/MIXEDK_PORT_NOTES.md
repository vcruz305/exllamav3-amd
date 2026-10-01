# Mixed-K on gfx1151: what the port actually requires

Established 2026-10-01 while porting mixed-K decode from `vcruz305/exllamav3`
(branch `exp/coop-mixedk`, CUDA) into this fork. Recorded here because the obvious
plan — copy the CUDA mixed-K files across — is **wrong**, and the reason is not
visible from the nvsrc side.

## The CUDA mixed-K kernels cannot be copied in

`exllamav3/exllamav3_ext/build_config.py` keeps them out of ROCm builds entirely:

```python
ROCM_EXCLUDE_DIRS  = {'parallel', 'comp_units'}
ROCM_ALLOW_PREFIXES = ('quant/comp_units/exl3_gemv_int8_inst_',)
ROCM_EXCLUDE_FILES = { ..., 'quant/exl3_moe.cu', 'quant/exl3_moe_coop.cu',
                       'quant/exl3_kernel_map.cu', 'libtorch/blocksparse_mlp.cpp', ... }
```

So on gfx1151 these never compile or link:

- `exl3_moe.cu`, `exl3_moe_coop.cu`, `exl3_kernel_map.cu`
- the whole `comp_units/` tree except the `exl3_gemv_int8_inst_*` instantiations —
  which is exactly where nvsrc puts `exl3_moe_mixedk_inst_*.cu` and
  `exl3_moe_coopmk_inst_*.cu`
- `libtorch/blocksparse_mlp.cpp`, the binding layer for the fused MoE entry points

Confirmed against the built ABI rather than only by reading source. On framework2:

```
>>> import torch, exllamav3_ext as e
>>> [s for s in dir(e) if 'moe' in s.lower()]
['exl3_moe_cpu_forward', 'exl3_moe_cpu_free_layer', 'exl3_moe_cpu_has_avx2',
 'exl3_moe_cpu_has_avx512_vbmi', 'exl3_moe_cpu_has_avx512_vnni',
 'exl3_moe_cpu_make_layer', 'exl3_moe_cpu_set_memops', 'exl3_moe_cpu_set_prof',
 'exl3_moe_cpu_worker_run', 'exl3_moe_flag_wait', 'exl3_moe_flag_write',
 'exl3_moe_gfx12_k3', 'exl3_moe_gfx12_k3_prefill']
```

There is **no K4..K8 and no mixed-K entry point**. Despite the `gfx12` name the
symbol is usable on gfx1151 — it is gated on `exl3_gemv_wmma_family(device) != 0`
and only uses `exl3_gemv_kernel_body`, which is already ported to every WMMA family.

## The real port target: exl3_moe_gfx12_k3 in exl3_gemv.cu

`exl3_gemv.cu` **does** build on ROCm, and it holds the only fused grouped-MoE decode
kernel this fork has. Mixed-K means generalizing it:

- `exl3_gemv_kernel_body<3, FP32, 2, 0, MOE_CFG, true>` — line 204 (decode stage A/B)
- `exl3_gemv_kernel_body<3, FP32, 2, 2, CFG, true>` — line 440 (prefill)
- host-side `TORCH_CHECK(intermediate == 640 || intermediate == 768, ...)`
- `constexpr int MOE_HIDDEN = 2560; MOE_TOP_K = 10; MOE_MAX_ROWS = 16;` (lines 49-51)

K=3 is a **template** parameter, which is favourable: mixed-K wants per-expert K, so
the work is dispatching a K per (token, expert) slot instead of one compile-time K
for the whole launch. Note the sibling `gemv_by_m` config picker already branches on
K (`if (K == 3 && cc == CC_ADA) ...` around line 1000), so per-K dispatch exists in
the surrounding code.

## Why the target pack needs it

`vcruz305/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw`, model_type `qwen4_exp`, 48 layers,
24576 experts (512/layer), top_k 10, has genuine per-expert K variation. Bitrate is
read off the trellis row width in `quantization_config.json` -> `tensor_storage`:

| projection | trellis shape | bpw | expert count |
|---|---|---|---|
| gate/up | `(160, 40, 48)` | 3 | 13984 |
| gate/up | `(160, 40, 64)` | 4 | 10592 |
| down | `(40, 160, 48)` | 3 | 2167 |
| down | `(40, 160, 64)` | 4 | 12940 |
| down | `(40, 160, 80)` | 5 | 5610 |
| down | `(40, 160, 96)` | 6 | 3768 |
| down | `(40, 160, 112)` | 7 | 91 |

Five distinct K on `down_proj` alone. A K=3-only kernel cannot run this pack's MoE
at full rate, and the pure-GEMV fallback is the slow path the README already
measures (~17 tok/s, versus 40+ for the tuned WMMA path on the 3.05 bpw pack).

## Definition of done for this work

Per `AGENTS.md`, PPL must hold 4.225935 on the **Qwen3.8-Flash-Next 3.05 bpw** pack.
That pack is a different model from the one this port targets, so establish the
correct parity baseline for CYBER-FROST separately — do not assume the 3.05 bpw
number transfers. `tools/strix_halo/greedy_ab.py` (run `KNOB=x` vs `KNOB=x` first for
the noise floor) and `tie_check.py` remain the A/B tools.