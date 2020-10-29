/**
 * @file prefixsum_based.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-10-29
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#include <stdio.h>
#include "../cuda_utils.cuh"

template <typename Input, typename Dict, typename Huff, int Magnitude>
__global__ void PrefixSumBased(Input* in, size_t len, Dict* cb, Huff* out, size_t cb_len, uint32_t* hmeta)
{
    // allocated on invocation
    auto ti        = threadIdx.x;
    auto bi        = blockIdx.x;
    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;

    extern __shared__ char __buff[];

    auto __bw = reinterpret_cast<int*>(__buff);
    auto __h  = reinterpret_cast<Huff*>(__buff + chunksize * sizeof(int));

    auto stride = 1;

    Huff sym;
    int  bw = 0;
    if (bi * n_worker + ti < len) {
        auto gidx = chunksize * bi + ti;
        sym       = cb[in[gidx]];
        bw        = GetBitwidth(sym);
        __bw[ti]  = bw;
        sym       = sym << (sizeof(Huff) * 8 - bw);
    }
    else {
        __bw[ti] = 0;  // Make the "empty" spots zeros, so it won't affect the final result.
        sym      = 0;
    }

    // build sum in place up the tree
    for (auto d = n_worker >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (ti < d) {
            auto l = stride * (2 * ti + 1) - 1;
            auto r = stride * (2 * ti + 2) - 1;
            __bw[r] += __bw[l];
        }
        stride *= 2;
    }

    if (ti == 0) __bw[n_worker - 1] = 0;

    // clear the last element
    for (int d = 1; d < n_worker; d *= 2) {
        // traverse down tree & build scan
        stride >>= 1;
        __syncthreads();

        if (ti < d) {
            auto l  = stride * (2 * ti + 1) - 1;
            auto r  = stride * (2 * ti + 2) - 1;
            auto t  = __bw[l];
            __bw[l] = __bw[r];
            __bw[r] += t;
        }
        __syncthreads();
    }
    __syncthreads();

    if (bi * n_worker + ti < len) {
        auto loc       = __bw[ti];
        auto word_1    = loc / (sizeof(Huff) * 8);
        auto word_2    = (loc - 1) / (sizeof(Huff) * 8) + 1;
        auto extra_bit = loc % (sizeof(Huff) * 8);

        atomicOr(__h + word_1, sym >> extra_bit);
        if (word_2 == word_1) { atomicOr(__h + word_2, sym << (sizeof(Huff) * 8 - extra_bit)); }
        //        auto l        = __bw[ti] sym
    }
    __syncthreads();

    if (ti == n_worker - 1) __bw[n_worker - 1] += bw;

    if (ti < (__bw[n_worker - 1] - 1) / (sizeof(Huff) * 8) + 1) { out[bi * n_worker + ti] = __h[ti]; }
    __syncthreads();

    //    if (bi == 0 and ti == 0) {
    //        for (auto i = 0; i < (1 << Magnitude); i++) {
    //            printf("%u\t%d\n", i, __bw[i]);
    //        }
    //    }
}