#ifndef CANONICAL_CUH
#define CANONICAL_CUH

/**
 * @file canonical.cuh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-21
 * Created on 2020-04-10
 *
<<<<<<< HEAD
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
=======
 * @deprecated Canonization of codebook is integrated in par-huffman
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
>>>>>>> b049d80832744685ca49a9f28563391f11ba1cbb
 * See LICENSE in top-level directory
 *
 */

#include <cstdint>

namespace GPU {

__device__ int max_bw;

template <typename UInt, typename Huff>
__global__ void GetCanonicalCode(uint8_t*, int);

}  // namespace GPU
#endif
