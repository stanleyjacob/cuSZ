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
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <stddef.h>

namespace lossless {
namespace wrapper {

template <typename Input, typename Huff>
__global__ void EncodeFixedLen(Input*, Huff*, size_t, Huff*, int offset=0);

template <typename Huff>
__global__ void Deflate(Huff*, size_t, size_t*, int);

template <typename Huff, typename Output>
__device__ void InflateChunkwise(Huff*, Output*, size_t, uint8_t*);

template <typename Quant, typename Huff>
__global__ void Decode(Huff*, size_t*, Quant*, size_t, int, int, uint8_t*, size_t);

}  // namespace wrapper
}  // namespace lossless

#endif
