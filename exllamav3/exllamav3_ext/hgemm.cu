#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#include <cstdlib>
#include <type_traits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

#if defined(USE_ROCM)

/*
Skinny fp16 GEMM for ROCm: a [m, k] @ b [k, n] -> c [m, n] fp32/fp16, tiny m (decode rows) and
tiny n. hipblaslt answers these shapes (the GDN b/a projections, k=2560 n=48, m<=8) with a
split-K solution plus a PostGSU reduce, ~60 us for a 245 KB weight that streams in ~1 us.
36 GDN layers x 2 calls per forward made that ~6% of decode device time on gfx1151.

Split-K: grid (n/32, k/128) blocks write fp32 partials to a per-device workspace, then a
one-block reduce sums them in fixed order (deterministic, no atomics). The m rows share every
b load (b is the traffic, a is L2-resident), so cost is ~flat in m. A first version with one
block per 32 columns and no k-split was latency-bound at ~85 us (2 blocks on 20 CUs, 320
dependent load rounds per lane) - slower than hipblaslt. Parallelism first, then bandwidth.
*/

namespace skinny
{

constexpr int COLS = 32;         // output columns per block (one lane each)
constexpr int WARPS = 8;         // warps per block, each a k-sub-slice of the block's k-chunk
constexpr int THREADS = WARPS * 32;
constexpr int MAX_M = 8;
constexpr int MAX_N = 128;       // above this, let hipblaslt have it
constexpr int K_CHUNK = 128;     // k rows per block -> 16 k's per warp, 20 blocks on k=2560
constexpr int MAX_KSPLIT = 64;   // k <= 8192
constexpr int NUM_DEV = MAX_DEVICES;   // from exl3_devctx.cuh

// Partials workspace [KSPLIT][M][N] fp32 per device, allocated once. Stream-ordered use only
// (the partials kernel and the reduce run back to back on the caller's stream), like the
// cuBLAS workspace this path replaces.
static float* g_partials[NUM_DEV] = {};

static float* partials(int device)
{
    if (device < 0 || device >= NUM_DEV) return nullptr;
    if (!g_partials[device])
    {
        cuda_check(cudaMalloc((void**) &g_partials[device], (size_t) MAX_KSPLIT * MAX_M * MAX_N * sizeof(float)));
    }
    return g_partials[device];
}

// grid (ceil(n / 32), ksplit). Block (x, y) covers columns [32x, 32x+32) and k rows
// [K_CHUNK*y, K_CHUNK*(y+1)); warp w takes every WARPS-th k of that chunk so the 8 warps
// stream interleaved 64-byte rows. Each lane holds M fp32 accumulators and issues its loads
// back to back (the k loop is fully unrolled: K_CHUNK / WARPS = 16 independent loads in
// flight per lane), which is what the one-block-per-32-columns version lacked.
template <int M>
__global__ __launch_bounds__(THREADS)
void skinny_partials_kernel
(
    const half* __restrict__ a,   // [M, k], row stride lda
    const half* __restrict__ b,   // [k, n], row stride n
    float* __restrict__ part,     // [ksplit, M, MAX_N]
    const int k,
    const int n,
    const int lda
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = blockIdx.x * COLS + lane;
    const bool active = col < n;
    const int kbase = blockIdx.y * K_CHUNK;

    float acc[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.0f;

    if (active)
    {
        constexpr int PER_WARP = K_CHUNK / WARPS;
        float bv[PER_WARP];
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            bv[j] = kk < k ? __half2float(b[(size_t) kk * n + col]) : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                #pragma unroll
                for (int i = 0; i < M; ++i)
                    acc[i] = fmaf(__half2float(a[(size_t) i * lda + kk]), bv[j], acc[i]);
            }
        }
    }

    __shared__ float red[WARPS][M][COLS + 1];    // 33: coprime with the 32 dword banks
    #pragma unroll
    for (int i = 0; i < M; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();

    if (warp == 0 && active)
    {
        float* out = part + ((size_t) blockIdx.y * M) * MAX_N;
        #pragma unroll
        for (int i = 0; i < M; ++i)
        {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) s += red[w][i][lane];
            out[(size_t) i * MAX_N + col] = s;
        }
    }
}

// One block, thread per (row, col): fixed-order sum over the k-splits -> deterministic.
template <typename C_T>
__global__ void skinny_reduce_kernel
(
    const float* __restrict__ part,   // [ksplit, M, MAX_N]
    C_T* __restrict__ c,
    const int m,
    const int n,
    const int ksplit,
    const int ldc
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int i = idx / n, col = idx % n;
    float s = 0.0f;
    for (int y = 0; y < ksplit; ++y)
        s += part[((size_t) y * m + i) * MAX_N + col];
    if constexpr (std::is_same_v<C_T, float>)
        c[(size_t) i * ldc + col] = s;
    else
        c[(size_t) i * ldc + col] = __float2half(s);
}

static void launch_partials(int m, const half* a, const half* b, float* part, int k, int n, int lda,
                            dim3 grid, cudaStream_t stream)
{
    #define SK_CASE(MM) case MM: skinny_partials_kernel<MM><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda); break;
    switch (m)
    {
        SK_CASE(1) SK_CASE(2) SK_CASE(3) SK_CASE(4) SK_CASE(5) SK_CASE(6) SK_CASE(7) SK_CASE(8)
        default: break;
    }
    #undef SK_CASE
}

static bool enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = getenv("EXL3_HIP_SKINNY_GEMM");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached == 1;
}

// true if handled
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    if (!enabled()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    if (b.dim() != 2 || !b.is_contiguous()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < 1 || m > MAX_M || n < 1 || n > MAX_N || k < 1) return false;
    const int ksplit = (k + K_CHUNK - 1) / K_CHUNK;
    if (ksplit > MAX_KSPLIT) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.dim() > 2 && !a.is_contiguous()) return false;
    const int lda = a.dim() == 1 ? k : (int) a.stride(-2);
    const int ldc = (int) c.stride(-2);
    float* part = partials(a.device().index());
    if (!part) return false;

    dim3 grid((n + COLS - 1) / COLS, ksplit);
    launch_partials(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream);
    const int total = m * n;
    const int rthreads = total < 256 ? ((total + 31) / 32) * 32 : 256;
    const int rblocks = (total + rthreads - 1) / rthreads;
    if (c.dtype() == at::kFloat)
        skinny_reduce_kernel<float><<<rblocks, rthreads, 0, stream>>>(part, (float*) c.data_ptr(), m, n, ksplit, ldc);
    else
        skinny_reduce_kernel<half><<<rblocks, rthreads, 0, stream>>>(part, (half*) c.data_ptr(), m, n, ksplit, ldc);
    cuda_check(cudaPeekAtLastError());
    return true;
}

} // namespace skinny

#endif // USE_ROCM

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
#if defined(USE_ROCM)
    if (skinny::try_launch(a, b, c, stream)) return;
#endif
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
