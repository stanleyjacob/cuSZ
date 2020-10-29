/**
 * @file cuda_utils.cuh
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

#include <cuda_runtime.h>

// https://stackoverflow.com/a/42309370/8740097
// usage: if (ti == 0) printf("shared memory size: %u\n", dynamic_smem_size());
__forceinline__ __device__ unsigned dynamic_smem_size()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

template <typename UInt>
__forceinline__ __device__ int GetBitwidth(UInt var)
{
    return static_cast<int>(*(reinterpret_cast<uint8_t*>(&var) + sizeof(UInt) - 1));
}

template <typename UInt, typename Ret>
__forceinline__ __device__ int GetBitwidth2(UInt var)
{
    return static_cast<Ret>(*(reinterpret_cast<uint8_t*>(&var) + sizeof(UInt) - 1));
}

#endif