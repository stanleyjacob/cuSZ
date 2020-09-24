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

#include "cusz_dualquant.cuh"
#include "metadata.hh"

using uint8__t = uint8_t;

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
// const int DIM3   = 3;
const int nBLK0 = 4;
const int nBLK1 = 5;
const int nBLK2 = 6;
// const int nBLK3  = 7;
// const int nDIM   = 8;
// const int LEN    = 12;
// const int CAP    = 13;
const int RADIUS = 14;
// const size_t EB     = 0;
// const size_t EBr    = 1;
// const size_t EBx2   = 2;
const size_t EBx2_r = 3;

// extern __constant__ int    symb_dims[16];
// extern __constant__ double symb_ebs[4];

// constexpr bool

// template <int ndim, int, Block, typename Data, typename Quant>
// typename std::enable_if<std::equa>

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
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);
    if (gi0 >= m->d0 or gi1 >= m->d1) return;
    size_t id = gi0 + gi1 * m->stride1;  // low to high dim, inner to outer
    // prequantization
    s2df[y][x] = round(d[id] * m->ebx2);  // fp representation
    __syncthreads();
    // postquantization
    auto delta = s2df[y][x]                       //
                 - (x == 0 ? 0 : s2df[y][x - 1])  //
                 - (y == 0 ? 0 : s2df[y - 1][x])  //
                 + (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0);
    bool quantizable = fabs(delta) < m->radius;
    auto _code       = static_cast<Quant>(delta + m->radius);
    // __syncthreads();
    d[id] = (1 - quantizable) * s2df[y][x];  // data array as outlier
    q[id] = quantizable * _code;
}

template <int Block, typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_3d1l(struct Metadata<Block>* m, Data* d, Quant* q)
{
    int z   = threadIdx.z;
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi2 = blockIdx.z * blockDim.z + z;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    if (gi0 >= m->d0 or gi1 >= m->d1 or gi2 >= m->d2) return;
    size_t id     = gi0 + gi1 * m->stride1 + gi2 * m->stride2;  // low to high in dim, inner to outer
    s3df[z][y][x] = round(d[id] * m->ebx2_r);                   // prequant, fp representation
    __syncthreads();
    // postquantization
    auto delta       = s3df[z][y][x] - (                                                                //
                                     (z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
                                     - (y > 0 and x > 0 ? s3df[z][y - 1][x - 1] : 0)              // dist=2
                                     - (z > 0 and x > 0 ? s3df[z - 1][y][x - 1] : 0)              //
                                     - (z > 0 and y > 0 ? s3df[z - 1][y - 1][x] : 0)              //
                                     + (x > 0 ? s3df[z][y][x - 1] : 0)                            // dist=1
                                     + (y > 0 ? s3df[z][y - 1][x] : 0)                            //
                                     + (z > 0 ? s3df[z - 1][y][x] : 0));                          //
    bool quantizable = fabs(delta) < d->radius;
    auto _code       = static_cast<Quant>(delta + d->radius);
    d[id]            = (1 - quantizable) * s3df[z][y][x];  // data array as outlier
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

////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims[DIM0]) return;
    // prequantization
    d[id] = round(d[id] * precisions[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquantization
    Data  pred        = threadIdx.x == 0 ? 0 : d[id - 1];
    Data  delta       = d[id] - pred;
    bool  quantizable = fabs(delta) < dims[RADIUS];
    Quant _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * d[id];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s2df)[B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1]>(&scratch);
    if (x < B + 1 and y < B + 1) s2df[y + 1][0] = 0, s2df[0][x + 1] = 0;
    if (x == 0 and y == 0) s2df[0][0] = 0;
    __syncthreads();
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1]) return;
    size_t id = gi0 + gi1 * dims[DIM0];  // low to high dim, inner to outer
    // prequantization
    s2df[y + 1][x + 1] = round(d[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    auto pred        = s2df[y + 1][x] + s2df[y][x + 1] - s2df[y][x];
    auto delta       = s2df[y + 1][x + 1] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * s2df[y + 1][x + 1];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant, int B>
__global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding(
    Data*         d,
    Quant*        q,
    size_t const* dims,
    double const* precisions)
{
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s2df)[B][B] = *reinterpret_cast<Data(*)[B][B]>(&scratch);
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1]) return;
    size_t id = gi0 + gi1 * dims[DIM0];  // low to high dim, inner to outer
    // __syncthreads();
    // prequantization
    s2df[y][x] = round(d[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    auto delta = s2df[y][x]                       //
                 - (x == 0 ? 0 : s2df[y][x - 1])  //
                 - (y == 0 ? 0 : s2df[y - 1][x])  //
                 + (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0);
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    // __syncthreads();
    d[id] = (1 - quantizable) * s2df[y][x];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    int z = threadIdx.z;
    int y = threadIdx.y;
    int x = threadIdx.x;

    Data(&s3df)[B + 1][B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1][B + 1]>(&scratch);

    if (x == 0) {
        s3df[z + 1][y + 1][0] = 0;
        s3df[0][z + 1][y + 1] = 0;
        s3df[y + 1][0][z + 1] = 0;
    }
    if (x == 0 and y == 0) {
        s3df[z + 1][0][0] = 0;
        s3df[0][z + 1][0] = 0;
        s3df[0][0][z + 1] = 0;
    }
    if (x == 0 and y == 0 and z == 0) s3df[0][0][0] = 0;
    __syncthreads();

    int gi2 = blockIdx.z * blockDim.z + z;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1] or gi2 >= dims[DIM2]) return;
    size_t id = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM0] * dims[DIM1];  // low to high in dim, inner to outer
    s3df[z + 1][y + 1][x + 1] = round(d[id] * precisions[EBx2_r]);       // prequant, fp representation
    __syncthreads();
    // postquantization
    auto pred = s3df[z][y][x]                                                             // dist=3
                - s3df[z + 1][y][x] - s3df[z][y + 1][x] - s3df[z][y][x + 1]               // dist=2
                + s3df[z + 1][y + 1][x] + s3df[z + 1][y][x + 1] + s3df[z][y + 1][x + 1];  // dist=1
    auto delta       = s3df[z + 1][y + 1][x + 1] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * s3df[z + 1][y + 1][x + 1];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant, int B>
__global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding(
    Data*         d,
    Quant*        q,
    size_t const* dims,
    double const* precisions)
{
    int z   = threadIdx.z;
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi2 = blockIdx.z * blockDim.z + z;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s3df)[B][B][B] = *reinterpret_cast<Data(*)[B][B][B]>(&scratch);

    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1] or gi2 >= dims[DIM2]) return;
    size_t id     = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM0] * dims[DIM1];  // low to high in dim, inner to outer
    s3df[z][y][x] = round(d[id] * precisions[EBx2_r]);                       // prequant, fp representation
    __syncthreads();
    // postquantization
    auto delta       = s3df[z][y][x] - (                                                                //
                                     (z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
                                     - (y > 0 and x > 0 ? s3df[z][y - 1][x - 1] : 0)              // dist=2
                                     - (z > 0 and x > 0 ? s3df[z - 1][y][x - 1] : 0)              //
                                     - (z > 0 and y > 0 ? s3df[z - 1][y - 1][x] : 0)              //
                                     + (x > 0 ? s3df[z][y][x - 1] : 0)                            // dist=1
                                     + (y > 0 ? s3df[z][y - 1][x] : 0)                            //
                                     + (z > 0 ? s3df[z - 1][y][x] : 0));                          //
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    d[id]            = (1 - quantizable) * s3df[z][y][x];  // data array as outlier
    q[id]            = quantizable * _code;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (b0 >= dims[nBLK0]) return;
    size_t _idx0 = b0 * B;

    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        auto pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        xd[id] *= val_2eb;
    }
    // end of body //
}

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    Data s[B + 1][B + 1];  // try not use shared memory first
    memset(s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b1 >= dims[nBLK1] or b0 >= dims[nBLK0]) return;

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            const size_t id   = gi0 + gi1 * dims[DIM0];
            auto         pred = s[i1][i0 + 1] + s[i1 + 1][i0] - s[i1][i0];
            s[i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
            xd[id]            = s[i1 + 1][i0 + 1] * val_2eb;
        }
    }
    // end of body //
}

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    Data s[B + 1][B + 1][B + 1];
    memset(s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b2 = blockDim.z * blockIdx.z + threadIdx.z;
    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b2 >= dims[nBLK2] or b1 >= dims[nBLK1] or b0 >= dims[nBLK0]) return;

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims[DIM2] or gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
                size_t id = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];

                auto pred = s[i2][i1][i0]                                                             // +, dist=3
                            - s[i2 + 1][i1][i0] - s[i2][i1 + 1][i0] - s[i2][i1][i0 + 1]               // -, dist=2
                            + s[i2 + 1][i1 + 1][i0] + s[i2 + 1][i1][i0 + 1] + s[i2][i1 + 1][i0 + 1];  // +, dist=1
                s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
                xd[id] = s[i2 + 1][i1 + 1][i0 + 1] * val_2eb;
            }
        }
    }
}

// compression
// prototype 1D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint8__t, 32>(float*, uint8__t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint16_t, 32>(float*, uint16_t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint32_t, 32>(float*, uint32_t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint8__t, 64>(float*, uint8__t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint16_t, 64>(float*, uint16_t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<float, uint32_t, 64>(float*, uint32_t*, size_t const*, double const*);
// prototype 2D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<float, uint8__t, 16>(float*, uint8__t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<float, uint16_t, 16>(float*, uint16_t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<float, uint32_t, 16>(float*, uint32_t*, size_t const*, double const*);
// prototype 3D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<float, uint8__t, 8>(float*, uint8__t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<float, uint16_t, 8>(float*, uint16_t*, size_t const*, double const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<float, uint32_t, 8>(float*, uint32_t*, size_t const*, double const*);

// using virtual padding
// prototype 2D
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<float, uint8__t, 16>(
    float*,
    uint8__t*,
    size_t const*,
    double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<float, uint16_t, 16>(
    float*,
    uint16_t*,
    size_t const*,
    double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<float, uint32_t, 16>(
    float*,
    uint32_t*,
    size_t const*,
    double const*);
// prototype 3D
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<float, uint8__t, 8>(
    float*,
    uint8__t*,
    size_t const*,
    double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<float, uint16_t, 8>(
    float*,
    uint16_t*,
    size_t const*,
    double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<float, uint32_t, 8>(
    float*,
    uint32_t*,
    size_t const*,
    double const*);

// decompression
// prototype 1D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint8__t, 32>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint16_t, 32>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint32_t, 32>(float*, float*, uint32_t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint8__t, 64>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint16_t, 64>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<float, uint32_t, 64>(float*, float*, uint32_t*, size_t const*, double);
// prototype 2D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<float, uint8__t, 16>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<float, uint16_t, 16>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<float, uint32_t, 16>(float*, float*, uint32_t*, size_t const*, double);
// prototype 3D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<float, uint8__t, 8>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<float, uint16_t, 8>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<float, uint32_t, 8>(float*, float*, uint32_t*, size_t const*, double);
