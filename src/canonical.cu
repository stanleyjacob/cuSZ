/**
 * @file canonical.cu
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook.
 * @version 0.1
 * @date 2020-09-21
 * Created on 2020-04-10
 *
 * @deprecated Canonization of codebook is integrated in par-huffman
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <bits/stdint-uintn.h>
#include <cooperative_groups.h>
#include <stddef.h>
#include <stdint.h>
#include "canonical.cuh"

namespace cg   = cooperative_groups;
using uint8__t = uint8_t;

__device__ int max_bw = 0;

// TODO change H Q order
template <typename UInt, typename Huff>
__global__ void GPU::GetCanonicalCode(uint8_t* singleton, int cb_size)
{
    auto  type_bw   = sizeof(Huff) * 8;
    auto  codebooks = reinterpret_cast<Huff*>(singleton);
    auto  metadata  = reinterpret_cast<int*>(singleton + sizeof(Huff) * (3 * cb_size));
    auto  keys      = reinterpret_cast<UInt*>(singleton + sizeof(Huff) * (3 * cb_size) + sizeof(int) * (4 * type_bw));
    Huff* i_cb      = codebooks;
    Huff* o_cb      = codebooks + cb_size;
    Huff* canonical = codebooks + cb_size * 2;
    auto  numl      = metadata;
    auto  iter_by_  = metadata + type_bw;
    auto  first     = metadata + type_bw * 2;
    auto  entry     = metadata + type_bw * 3;

    cg::grid_group g = cg::this_grid();

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO
    auto c  = i_cb[gid];
    int  bw = *((uint8_t*)&c + (sizeof(Huff) - 1));

    if (c != ~((Huff)0x0)) {
        atomicMax(&max_bw, bw);
        atomicAdd(&numl[bw], 1);
    }
    g.sync();

    if (gid == 0) {
        // printf("\0");
        // atomicMax(&max_bw, max_bw + 0);
        memcpy(entry + 1, numl, (type_bw - 1) * sizeof(int));
        // for (int i = 1; i < type_bw; i++) entry[i] = numl[i - 1];
        for (int i = 1; i < type_bw; i++) entry[i] += entry[i - 1];
    }
    g.sync();

    if (gid < type_bw) iter_by_[gid] = entry[gid];
    __syncthreads();
    // atomicMax(&max_bw, bw);

    if (gid == 0) {  //////// first code
        for (int l = max_bw - 1; l >= 1; l--) first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
        first[0] = 0xff;  // no off-by-one error
    }
    g.sync();

    canonical[gid] = ~((Huff)0x0);
    g.sync();
    o_cb[gid] = ~((Huff)0x0);
    g.sync();

    // Reverse Codebook Generation -- TODO isolate
    if (gid == 0) {
        // no atomicRead to handle read-after-write (true dependency)
        for (int i = 0; i < cb_size; i++) {
            auto    _c  = i_cb[i];
            uint8_t _bw = *((uint8_t*)&_c + (sizeof(Huff) - 1));

            if (_c == ~((Huff)0x0)) continue;
            canonical[iter_by_[_bw]] = static_cast<Huff>(first[_bw] + iter_by_[_bw] - entry[_bw]);
            keys[iter_by_[_bw]]      = i;

            *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(Huff) - 1) = _bw;
            iter_by_[_bw]++;
        }
    }
    g.sync();

    if (canonical[gid] == ~((Huff)0x0u)) return;
    o_cb[keys[gid]] = canonical[gid];
}

template __global__ void GPU::GetCanonicalCode<uint8__t, uint32_t>(uint8_t* singleton, int cb_size);
template __global__ void GPU::GetCanonicalCode<uint16_t, uint32_t>(uint8_t* singleton, int cb_size);
template __global__ void GPU::GetCanonicalCode<uint32_t, uint32_t>(uint8_t* singleton, int cb_size);
template __global__ void GPU::GetCanonicalCode<uint8__t, uint64_t>(uint8_t* singleton, int cb_size);
template __global__ void GPU::GetCanonicalCode<uint16_t, uint64_t>(uint8_t* singleton, int cb_size);
template __global__ void GPU::GetCanonicalCode<uint32_t, uint64_t>(uint8_t* singleton, int cb_size);
