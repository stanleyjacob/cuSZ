/**
 * @file cuda_mem.cu
 * @author Jiannan Tian
 * @brief CUDA memory operation wrappers.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-30
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "cuda_mem.cuh"
#include "type_trait.hh"

template <typename T>
inline T* mem::CreateCUDASpace(size_t l, uint8_t i)
{
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemset(d_var, i, l * sizeof(T));
    return d_var;
}

enum MemcpyDirection { h2d, d2h };

template <typename T>
void mem::CopyBetweenSpaces(T* src, T* dst, size_t l, MemcpyDirection direct)
{
    assert(src != nullptr);
    assert(dst != nullptr);
    if (direct == h2d) { cudaMemcpy(dst, src, sizeof(T) * l, cudaMemcpyHostToDevice); }
    else if (direct == d2h) {
        cudaMemcpy(dst, src, sizeof(T) * l, cudaMemcpyDeviceToHost);
    }
    else {
        // TODO log
        exit(1);
    }
}

template <typename T>
inline T* mem::CreateDeviceSpaceAndMemcpyFromHost(T* var, size_t l)
{
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemcpy(d_var, var, l * sizeof(T), cudaMemcpyHostToDevice);
    return d_var;
}
template <typename T>
inline T* mem::CreateHostSpaceAndMemcpyFromDevice(T* d_var, size_t l)
{
    auto var = new T[l];
    cudaMemcpy(var, d_var, l * sizeof(T), cudaMemcpyDeviceToHost);
    return var;
}
