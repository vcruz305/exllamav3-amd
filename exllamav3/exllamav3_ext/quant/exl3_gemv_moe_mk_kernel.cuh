#pragma once

// Kernels for the mixed-K grouped MoE GEMV (see exl3_gemv_moe_mk.cuh). Included once per bitrate
// by exl3_gemv_moe_mk_k<K>.cu with EXL3_MOE_MK_BITS defined.

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_gemv_moe_mk.cuh"

namespace exl3_moe_mk_ns {

// Decode / small verify: one block row per (projection, assignment slot), m = 1 GEMV body.
// Mirrors moe_grouped_gemv_k3_kernel with the bitrate check added.
template <int BITS, bool TWO_PROJECTIONS, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : CFG == 1 ? 256 : 128)
void moe_mk_decode_kernel(const MoeMkGemvArgs p)
{
    constexpr bool FP32 = !TWO_PROJECTIONS;
    const int slot = blockIdx.y;
    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t expert = p.selected[slot];
    const int64_t* B_table = projection == 0 ? p.B_table_0 : p.B_table_1;
    const int* K_table = projection == 0 ? p.K_table_0 : p.K_table_1;
    const size_t matrix = (size_t) projection * p.assignments + slot;
    void* C_row = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(p.C) + matrix * p.size_n)
        : static_cast<void*>(reinterpret_cast<half*>(p.C) + matrix * p.size_n);
    const bool valid_expert = expert >= 0 && expert < p.experts;
    const int64_t B_ptr = valid_expert ? B_table[expert] : 0;
    if (!B_ptr)
    {
        if (!p.zero_owner) return;
        const int col = blockIdx.x * 32 + threadIdx.x;
        if (threadIdx.x < 32 && col < p.size_n)
        {
            if constexpr (FP32) reinterpret_cast<float*>(C_row)[col] = 0.0f;
            else reinterpret_cast<half*>(C_row)[col] = __float2half_rn(0.0f);
        }
        return;
    }
    // Block-uniform: every thread reads the same expert entry
    if (K_table[expert] != BITS) return;

    const uint16_t* B = reinterpret_cast<const uint16_t*>(B_ptr);
    const half* A_row = p.A + matrix * p.size_k;
    exl3_gemv_kernel_body<BITS, FP32, 2, 0, CFG, true>
    (A_row, B, C_row, 1, p.size_k, p.size_n, nullptr, nullptr, nullptr, nullptr);
}

// Prefill / multi-row verify: expert-sorted chunks of up to 16 rows, MMODE 2 body.
// Mirrors moe_prefill_grouped_gemv_k3_kernel with the bitrate check added.
template <int BITS, bool TWO_PROJECTIONS, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : CFG == 1 ? 256 : 128)
void moe_mk_prefill_kernel(const MoeMkGemvArgs p)
{
    constexpr bool FP32 = !TWO_PROJECTIONS;
    if (blockIdx.y >= *p.num_chunks) return;
    const int expert_chunk = p.expert_chunks[blockIdx.y];
    const int expert = expert_chunk / EXL3_MOE_MK_PREFILL_CHUNKS_PER_EXPERT;
    const int chunk = expert_chunk % EXL3_MOE_MK_PREFILL_CHUNKS_PER_EXPERT;

    const int64_t start = p.expert_offsets[expert];
    const int64_t end = p.expert_offsets[expert + 1];
    const int64_t row0 = start + chunk * EXL3_MOE_MK_PREFILL_ROWS_PER_CHUNK;
    const int rows = min((int64_t) EXL3_MOE_MK_PREFILL_ROWS_PER_CHUNK, end - row0);
    if (rows <= 0) return;

    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t* B_table = projection == 0 ? p.B_table_0 : p.B_table_1;
    const int* K_table = projection == 0 ? p.K_table_0 : p.K_table_1;
    const int64_t B_ptr = B_table[expert];
    const size_t matrix = (size_t) projection * p.assignments + row0;
    void* C_rows = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(p.C) + matrix * p.size_n)
        : static_cast<void*>(reinterpret_cast<half*>(p.C) + matrix * p.size_n);
    if (!B_ptr)
    {
        if (!p.zero_owner) return;
        constexpr int cols = CFG == 0 ? 32 : 64;
        for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x)
        {
            const int row = idx / cols;
            const int col = blockIdx.x * cols + idx % cols;
            if (col < p.size_n)
            {
                if constexpr (FP32)
                    reinterpret_cast<float*>(C_rows)[(size_t) row * p.size_n + col] = 0.0f;
                else
                    reinterpret_cast<half*>(C_rows)[(size_t) row * p.size_n + col] = __float2half_rn(0.0f);
            }
        }
        return;
    }
    if (K_table[expert] != BITS) return;

    exl3_gemv_kernel_body<BITS, FP32, 2, 2, CFG, true>
    (
        p.A + matrix * p.size_k,
        reinterpret_cast<const uint16_t*>(B_ptr),
        C_rows,
        rows,
        p.size_k,
        p.size_n,
        nullptr, nullptr, nullptr, nullptr
    );
}

template <int BITS>
void launch(const MoeMkGemvArgs& a, bool prefill, bool two, int cfg, dim3 grid, hipStream_t stream)
{
    const int threads = cfg == 0 ? 512 : cfg == 1 ? 256 : 128;
    #define MK_L(KERN, TWO_, CFG_) KERN<BITS, TWO_, CFG_><<<grid, threads, 0, stream>>>(a)
    #define MK_CFG(KERN, TWO_) \
        switch (cfg) { case 0: MK_L(KERN, TWO_, 0); break; case 1: MK_L(KERN, TWO_, 1); break; \
                       default: MK_L(KERN, TWO_, 2); break; }
    if (prefill)
    {
        if (two) { MK_CFG(moe_mk_prefill_kernel, true) } else { MK_CFG(moe_mk_prefill_kernel, false) }
    }
    else
    {
        if (two) { MK_CFG(moe_mk_decode_kernel, true) } else { MK_CFG(moe_mk_decode_kernel, false) }
    }
    #undef MK_CFG
    #undef MK_L
}

} // namespace exl3_moe_mk_ns

#define EXL3_MOE_MK_DEFINE(K) \
    void exl3_moe_mk_gemv_k##K(const MoeMkGemvArgs& a, bool prefill, bool two_projections, \
                               int cfg, dim3 grid, hipStream_t stream) \
    { exl3_moe_mk_ns::launch<K>(a, prefill, two_projections, cfg, grid, stream); }

#endif // USE_ROCM
