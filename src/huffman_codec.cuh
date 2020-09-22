#ifndef DEFLATE_CUH
#define DEFLATE_CUH

/**
 * @file huffman_codec.cu
 * @author Jiannan Tian
 * @brief Wrapper of Huffman codec (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-02
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <stddef.h>

template <typename Quant, typename Huff>
__global__ void EncodeFixedLen(Quant*, Huff*, size_t, Huff*);

template <typename Quant>
__global__ void Deflate(Quant*, size_t, size_t*, int);

template <typename Huff, typename T>
__device__ void InflateChunkwise(Huff*, T*, size_t, uint8_t*);

template <typename Quant, typename Huff>
__global__ void Decode(Huff*, size_t*, Quant*, size_t, int, int, uint8_t*, size_t);

#endif
