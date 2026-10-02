#pragma once

// Mixed-K grouped MoE GEMV for ROCm (gfx12 / gfx11.5 WMMA families).
//
// The K3 grouped path (exl3_moe_gfx12_k3 / _prefill in exl3_gemv.cu) runs every routed
// (token, expert) slot of a projection in ONE launch, but its GEMV body is compiled for a single
// bitrate. Mixed-K packs quantize each expert at its own K, so here a projection is issued as one
// launch per bitrate present in the layer (known at load time, no host sync). Every launch covers
// the full slot grid; a block whose expert is at a different K returns before touching memory,
// so each slot is computed exactly once, by the launch compiled for its K. The GEMV body, the
// Hadamard stages, the activation and the weighted reduce are the same code the K3 path uses.
//
// One translation unit per bitrate (exl3_gemv_moe_mk_k<K>.cu) keeps compile time parallel.

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

// Must match exl3_gemv.cu
#define EXL3_MOE_MK_PREFILL_MAX_ROWS 2048
#define EXL3_MOE_MK_PREFILL_ROWS_PER_CHUNK 16
#define EXL3_MOE_MK_TOP_K 10
#define EXL3_MOE_MK_PREFILL_CHUNKS_PER_EXPERT \
    (EXL3_MOE_MK_PREFILL_MAX_ROWS * EXL3_MOE_MK_TOP_K / EXL3_MOE_MK_PREFILL_ROWS_PER_CHUNK)

#define EXL3_MOE_MK_MIN_BITS 3
#define EXL3_MOE_MK_MAX_BITS 7

struct MoeMkGemvArgs
{
    const half* A;
    // decode: one block row per assignment slot
    const int64_t* selected;
    // prefill: expert-sorted 16-row chunks (built by moe_prefill_metadata_kernel)
    const int64_t* expert_offsets;
    const int* expert_chunks;
    const int* num_chunks;
    // per-expert tables; projection 1 only used when two_projections
    const int64_t* B_table_0;
    const int64_t* B_table_1;
    const int* K_table_0;
    const int* K_table_1;
    void* C;
    int size_k;
    int size_n;
    int experts;
    int assignments;
    // exactly one launch per projection set zeroes the rows of absent experts
    int zero_owner;
};

// two_projections = gate+up (fp16 out, grid z = 2); otherwise down (fp32 out)
#define EXL3_MOE_MK_DECL(K) \
    void exl3_moe_mk_gemv_k##K(const MoeMkGemvArgs& a, bool prefill, bool two_projections, \
                               int cfg, dim3 grid, hipStream_t stream);
EXL3_MOE_MK_DECL(3)
EXL3_MOE_MK_DECL(4)
EXL3_MOE_MK_DECL(5)
EXL3_MOE_MK_DECL(6)
EXL3_MOE_MK_DECL(7)
#undef EXL3_MOE_MK_DECL

inline bool exl3_moe_mk_gemv(int K, const MoeMkGemvArgs& a, bool prefill, bool two_projections,
                             int cfg, dim3 grid, hipStream_t stream)
{
    switch (K)
    {
        case 3: exl3_moe_mk_gemv_k3(a, prefill, two_projections, cfg, grid, stream); return true;
        case 4: exl3_moe_mk_gemv_k4(a, prefill, two_projections, cfg, grid, stream); return true;
        case 5: exl3_moe_mk_gemv_k5(a, prefill, two_projections, cfg, grid, stream); return true;
        case 6: exl3_moe_mk_gemv_k6(a, prefill, two_projections, cfg, grid, stream); return true;
        case 7: exl3_moe_mk_gemv_k7(a, prefill, two_projections, cfg, grid, stream); return true;
        default: return false;
    }
}

#endif // USE_ROCM
