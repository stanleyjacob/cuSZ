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

#include <cuda_runtime.h>
#include <stdio.h>  // CUDA use
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cusz_dualquant2.cuh"
#include "metadata.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z

// decoupled from dimension (redundant)
template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_1d1l(struct Metadata<Block>* m, Data* d, Quant* q)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= m->d0) return;
    // prequantization
    d[id] = round(d[id] * m->ebx2_r);  // maintain fp representation
    __syncthreads();
    // postquantization
    Data  pred        = threadIdx.x == 0 ? 0 : d[id - 1];
    Data  delta       = d[id] - pred;
    bool  quantizable = fabs(delta) < m->radius;
    Quant _code       = static_cast<Quant>(delta + m->radius);
    __syncthreads();
    d[id] = (1 - quantizable) * d[id];  // data array as outlier
    q[id] = quantizable * _code;
}

template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_2d1l(struct Metadata<Block>* m, Data* d, Quant* q)
{
    int gi1 = blockIdx.y * blockDim.y + tiy;
    int gi0 = blockIdx.x * blockDim.x + tix;

    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);
    if (gi0 >= m->d0 or gi1 >= m->d1) return;
    size_t id = gi0 + gi1 * m->stride1;  // low to high dim, inner to outer
    // prequantization
    s2df[tiy][tix] = round(d[id] * m->ebx2);  // fp representation
    __syncthreads();
    // postquantization
    auto delta = s2df[tiy][tix]                         //
                 - (tix == 0 ? 0 : s2df[tiy][tix - 1])  //
                 - (tiy == 0 ? 0 : s2df[tiy - 1][tix])  //
                 + (tix > 0 and tiy > 0 ? s2df[tiy - 1][tix - 1] : 0);
    bool quantizable = fabs(delta) < m->radius;
    auto _code       = static_cast<Quant>(delta + m->radius);
    // __syncthreads();
    d[id] = (1 - quantizable) * s2df[tiy][tix];  // data array as outlier
    q[id] = quantizable * _code;
}

template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_3d1l(struct Metadata<Block>* m, Data* d, Quant* q)
{
    int gi2 = blockIdx.z * blockDim.z + tiz;
    int gi1 = blockIdx.y * blockDim.y + tiy;
    int gi0 = blockIdx.x * blockDim.x + tix;

    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    if (gi0 >= m->d0 or gi1 >= m->d1 or gi2 >= m->d2) return;
    size_t id           = gi0 + gi1 * m->stride1 + gi2 * m->stride2;  // low to high in dim, inner to outer
    s3df[tiz][tiy][tix] = round(d[id] * m->ebx2_r);                   // prequant, fp representation
    __syncthreads();
    // postquantization
    auto delta =
        s3df[tiz][tiy][tix] - (                                                                            //
                                  (tiz > 0 and tiy > 0 and tix > 0 ? s3df[tiz - 1][tiy - 1][tix - 1] : 0)  // dist=3
                                  - (tiy > 0 and tix > 0 ? s3df[tiz][tiy - 1][tix - 1] : 0)                // dist=2
                                  - (tiz > 0 and tix > 0 ? s3df[tiz - 1][tiy][tix - 1] : 0)                //
                                  - (tiz > 0 and tiy > 0 ? s3df[tiz - 1][tiy - 1][tix] : 0)                //
                                  + (tix > 0 ? s3df[tiz][tiy][tix - 1] : 0)                                // dist=1
                                  + (tiy > 0 ? s3df[tiz][tiy - 1][tix] : 0)                                //
                                  + (tiz > 0 ? s3df[tiz - 1][tiy][tix] : 0));                              //
    bool quantizable = fabs(delta) < d->radius;
    auto _code       = static_cast<Quant>(delta + d->radius);
    d[id]            = (1 - quantizable) * s3df[tiz][tiy][tix];  // data array as outlier
    q[id]            = quantizable * _code;
}

template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::x_lorenzo_1d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
{
    auto radius = static_cast<Quant>(m->radius);

    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (b0 >= m->nb0) return;
    size_t _idx0 = b0 * Block;

    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        auto pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(m->radius);
    }
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        xd[id] *= m->ebx2;
    }
    // end of body //
}

// TODO auto->size_t
template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::x_lorenzo_2d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
{
    Data s[Block + 1][Block + 1];  // try not use shared memory first
    memset(s, 0, (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b1 >= m->nb1 or b0 >= m->nb0) return;

    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            const size_t id   = gi0 + gi1 * m->stride1;
            auto         pred = s[i1][i0 + 1] + s[i1 + 1][i0] - s[i1][i0];
            s[i1 + 1][i0 + 1] =
                q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(m->radius);
            xd[id] = s[i1 + 1][i0 + 1] * m->ebx2;
        }
    }
    // end of body //
}

template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::x_lorenzo_3d1l(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
{
    Data s[Block + 1][Block + 1][Block + 1];
    memset(s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t b2 = blockDim.z * blockIdx.z + threadIdx.z;
    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b2 >= m->nb2 or b1 >= m->nb1 or b0 >= m->nb0) return;

    size_t _idx2 = b2 * Block;
    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    for (size_t i2 = 0; i2 < Block; i2++) {
        for (size_t i1 = 0; i1 < Block; i1++) {
            for (size_t i0 = 0; i0 < Block; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= m->d2 or gi1 >= m->d1 or gi0 >= m->d0) continue;
                size_t id = gi0 + gi1 * m->stride1 + gi2 * m->stride2;

                auto pred = s[i2][i1][i0]                                                             // +, dist=3
                            - s[i2 + 1][i1][i0] - s[i2][i1 + 1][i0] - s[i2][i1][i0 + 1]               // -, dist=2
                            + s[i2 + 1][i1 + 1][i0] + s[i2 + 1][i1][i0 + 1] + s[i2][i1 + 1][i0 + 1];  // +, dist=1
                s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(m->radius);
                xd[id] = s[i2 + 1][i1 + 1][i0 + 1] * m->ebx2;
            }
        }
    }
}