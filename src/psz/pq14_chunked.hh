#ifndef PSZ_PQ14_CHUNKED_HH
#define PSZ_PQ14_CHUNKED_HH

/**
 * @file pq14_chunked.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-02-13
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cstddef>

#include "../metadata.hh"

namespace psz {
namespace predictor_quantizer_sz14par {

template <int Block, typename Data, typename Quant>
void c_lorenzo_1d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q,
    size_t                  b0)
{
    auto   radius = static_cast<Quant>(m->radius);
    size_t _idx0  = b0 * Block;
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        Data  pred        = id < _idx0 + 1 ? 0 : d[id - 1];
        Data  err         = d[id] - pred;
        Data  dup         = d[id];
        Quant bin_count   = fabs(err) * m->eb_r + 1;
        bool  quantizable = fabs(bin_count) < m->cap;
        if (err < 0) bin_count = -bin_count;
        Quant _code = static_cast<Quant>(bin_count / 2) + radius;
        d[id]       = pred + (_code - radius) * m->ebx2;
        d[id]       = (1 - quantizable) * d[id];  // outlier
        q[id]       = quantizable * _code;
    }
}

template <int Block, typename Data, typename Quant>
void c_lorenzo_2d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q,
    size_t                  b0,
    size_t                  b1)
{
    Data __s[Block + 1][Block + 1];
    memset(__s, 0, (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;

            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            size_t id           = gi0 + gi1 * m->d0;
            __s[i1 + 1][i0 + 1] = d[id];

            Data  pred        = __s[i1 + 1][i0] + __s[i1][i0 + 1] - __s[i1][i0];
            Data  err         = __s[i1 + 1][i0 + 1] - pred;
            Quant bin_count   = fabs(err) * m->eb_r + 1;
            bool  quantizable = fabs(bin_count) < m->cap;

            if (err < 0) bin_count = -bin_count;
            auto _code          = static_cast<Quant>(bin_count / 2) + radius;
            __s[i1 + 1][i0 + 1] = pred + (_code - radius) * m->ebx2;
            d[id]               = (1 - quantizable) * __s[i1 + 1][i0 + 1];  // outlier
            q[id]               = quantizable * _code;
        }
    }
}

template <int Block, typename Data, typename Quant>
void c_lorenzo_3d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q,
    size_t                  b0,
    size_t                  b1,
    size_t                  b2)
{
    Data __s[Block + 1][Block + 1][Block + 1];
    memset(__s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

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
                size_t id                   = gi0 + gi1 * m->d0 + gi2 * m->d1 * m->d0;
                __s[i2 + 1][i1 + 1][i0 + 1] = d[id];

                Data pred = __s[i2][i1][i0]                                                                //
                            - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                            + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];
                Data  err         = __s[i2 + 1][i1 + 1][i0 + 1] - pred;
                Quant bin_count   = fabs(err) * m->eb_r + 1;
                bool  quantizable = fabs(bin_count) < m->cap;

                if (err < 0) bin_count = -bin_count;
                auto _code = static_cast<Quant>(bin_count / 2) + radius;

                __s[i2 + 1][i1 + 1][i0 + 1] = pred + (_code - radius) * m->ebx2;

                d[id] = (1 - quantizable) * __s[i2 + 1][i1 + 1][i0 + 1];  // outlier
                q[id] = quantizable * _code;
            }
        }
    }
}

template <int Block, typename Data, typename Quant>
void x_lorenzo_1d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q,
    size_t                  b0)
{
    auto   radius = static_cast<Quant>(m->radius);
    size_t _idx0  = b0 * Block;
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        Data pred = id < _idx0 + 1 ? 0 : xdata[id - 1];
        xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);
    }
}

template <int Block, typename Data, typename Quant>
void x_lorenzo_2d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q,
    size_t                  b0,
    size_t                  b1)
{
    Data __s[Block + 1][Block + 1];
    memset(__s, 0, (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            const size_t id     = gi0 + gi1 * m->d0;
            Data         pred   = __s[i1][i0 + 1] + __s[i1 + 1][i0] - __s[i1][i0];
            __s[i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);
            xdata[id]           = __s[i1 + 1][i0 + 1];
        }
    }
}

template <int Block, typename Data, typename Quant>
void x_lorenzo_3d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q,
    size_t                  b0,
    size_t                  b1,
    size_t                  b2)
{
    Data __s[Block + 1][Block + 1][Block + 1];
    memset(__s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

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
                size_t id   = gi0 + gi1 * m->d0 + gi2 * m->d1 * m->d0;
                Data   pred = __s[i2][i1][i0]                                                              //
                            - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                            + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];

                __s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);

                xdata[id] = __s[i2 + 1][i1 + 1][i0 + 1];
            }
        }
    }
}

}  // namespace predictor_quantizer_sz14par
}  // namespace psz

#endif
