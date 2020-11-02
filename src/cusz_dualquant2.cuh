/**
 * @file cusz_dualquant.cu
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ.
 * @version 0.1
 * @date 2020-09-21
 * Created on 19-09-23
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DUALQUANT2_CUH
#define CUSZ_DUALQUANT2_CUH

#include <cuda_runtime.h>
#include <cstddef>

#include "metadata.hh"

extern __shared__ char scratch[];

namespace cusz {
namespace predictor_quantizer {

template <int Block, typename Data, typename Quant>
__global__ void c_lorenzo_1d1l(struct Metadata<Block>* m, Data* d, Quant* q);

template <int Block, typename Data, typename Quant>
__global__ void c_lorenzo_2d1l(struct Metadata<Block>* m, Data* d, Quant* q);

template <int Block, typename Data, typename Quant>
__global__ void c_lorenzo_3d1l(struct Metadata<Block>* m, Data* d, Quant* q);

template <int Block, typename Data, typename Quant>
__global__ void x_lorenzo_1d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q);

template <int Block, typename Data, typename Quant>
__global__ void x_lorenzo_2d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q);

template <int Block, typename Data, typename Quant>
__global__ void x_lorenzo_3d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q);

}  // namespace predictor_quantizer
}  // namespace cusz

#endif
