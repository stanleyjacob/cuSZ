/**
 * @file cuda_util.cuh
 * @author Jiannan Tian
 * @brief CUDA utility functions
 * @version 0.1.3
 * @date 2020-10-29
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

// https://stackoverflow.com/a/42309370/8740097
// usage: if (ti == 0) printf("shared memory size: %u\n", dynamic_smem_size());
__forceinline__ __device__ unsigned dynamic_smem_size()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

#endif