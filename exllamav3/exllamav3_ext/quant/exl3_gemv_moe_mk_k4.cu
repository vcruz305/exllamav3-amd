// Mixed-K grouped MoE GEMV, 4 bpw instances (see exl3_gemv_moe_mk.cuh)
#if defined(USE_ROCM)
#include "exl3_gemv_moe_mk_kernel.cuh"
EXL3_MOE_MK_DEFINE(4)
#endif
