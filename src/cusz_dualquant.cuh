/**
 * @file cusz_dualquant.cuh
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ (header).
 * @version 0.1
 * @date 2020-09-21
 * Created on 19-09-23
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DUALQUANT_CUH
#define CUSZ_DUALQUANT_CUH

#include <cuda_runtime.h>
#include <cstddef>

#include "metadata.hh"

extern __shared__ char scratch[];
// extern __shared__ float s2df[][16 + 1];  // TODO double type
// extern __shared__ float s3df[][8+ 1][8+ 1];

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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant, int B = 32>
__global__ void c_lorenzo_1d1l(Data* d, Quant* q, size_t const* dims, double const* precisions);

template <typename Data, typename Quant, int B = 16>
__global__ void c_lorenzo_2d1l(Data* d, Quant* q, size_t const* dims, double const* precisions);

template <typename Data, typename Quant, int B = 8>
__global__ void c_lorenzo_3d1l(Data* d, Quant* q, size_t const* dims, double const* precisions);

template <typename Data, typename Quant, int B = 16>
__global__ void c_lorenzo_2d1l_virtual_padding(Data* d, Quant* q, size_t const* dims, double const* precisions);

template <typename Data, typename Quant, int B = 8>
__global__ void c_lorenzo_3d1l_virtual_padding(Data* d, Quant* q, size_t const* dims, double const* precisions);

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant, int B = 32>
__global__ void x_lorenzo_1d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb);

template <typename Data, typename Quant, int B = 16>
__global__ void x_lorenzo_2d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb);

template <typename Data, typename Quant, int B = 8>
__global__ void x_lorenzo_3d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb);

}  // namespace predictor_quantizer
}  // namespace cusz

#endif
