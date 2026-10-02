// Mixed-K grouped MoE GEMV, 5 bpw instances (see exl3_gemv_moe_mk.cuh)
#if defined(USE_ROCM)
#include "exl3_gemv_moe_mk_kernel.cuh"
EXL3_MOE_MK_DEFINE(5)
#endif
