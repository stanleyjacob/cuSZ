#ifndef PSZ_PQ_DUALQUANT_HH
#define PSZ_PQ_DUALQUANT_HH

/**
 * @file pq_dualquant.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-02-11
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cstddef>

#include "../metadata.hh"

namespace psz {
namespace predictor_quantizer_dualquant {

template <int Block, typename Data, typename Quant>
void c_lorenzo_1d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q,
    int                     b0)
{
    auto   radius = static_cast<Quant>(m->radius);
    size_t _idx0  = b0 * Block;
    // prequantization
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        d[id] = round(d[id] * m->ebx2_r);
    }
    // postquantization
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        Data pred        = id < _idx0 + 1 ? 0 : d[id - 1];
        Data delta       = d[id] - pred;
        bool quantizable = fabs(delta) < radius;
        auto _code       = static_cast<Quant>(delta + radius);
        d[id]            = (1 - quantizable) * d[id];  // outlier
        q[id]            = quantizable * _code;
    }
}

template <int Block, typename Data, typename Quant>
void c_lorenzo_2d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q,
    int                     b0,
    int                     b1)
{
    Data _s[Block + 1][Block + 1];  // 2D interpretation of d
    memset(_s, 0, (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    // prequantization
    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            size_t id          = gi0 + gi1 * m->d0;
            _s[i1 + 1][i0 + 1] = round(d[id] * m->ebx2_r);
        }
    }
    // postquantization
    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            size_t id          = gi0 + gi1 * m->d0;
            Data   pred        = _s[i1 + 1][i0] + _s[i1][i0 + 1] - _s[i1][i0];
            Data   delta       = _s[i1 + 1][i0 + 1] - pred;
            bool   quantizable = fabs(delta) < radius;
            auto   _code       = static_cast<Quant>(delta + radius);
            d[id]              = (1 - quantizable) * _s[i1 + 1][i0 + 1];  // outlier
            q[id]              = quantizable * _code;
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
    Data _s[Block + 1][Block + 1][Block + 1];
    memset(_s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t _idx2 = b2 * Block;
    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    // prequantization
    for (size_t i2 = 0; i2 < Block; i2++) {
        for (size_t i1 = 0; i1 < Block; i1++) {
            for (size_t i0 = 0; i0 < Block; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= m->d2 or gi1 >= m->d1 or gi0 >= m->d0) continue;
                size_t id                  = gi0 + gi1 * m->d0 + gi2 * m->d1 * m->d0;
                _s[i2 + 1][i1 + 1][i0 + 1] = round(d[id] * m->ebx2_r);
            }
        }
    }
    // postquantization
    for (size_t i2 = 0; i2 < Block; i2++) {
        for (size_t i1 = 0; i1 < Block; i1++) {
            for (size_t i0 = 0; i0 < Block; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= m->d2 or gi1 >= m->d1 or gi0 >= m->d0) continue;
                size_t id   = gi0 + gi1 * m->d0 + gi2 * m->d1 * m->d0;
                Data   pred = _s[i2][i1][i0]                                                             // +, dist=3
                            - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                            + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                Data delta       = _s[i2 + 1][i1 + 1][i0 + 1] - pred;
                bool quantizable = fabs(delta) < radius;
                auto _code       = static_cast<Quant>(delta + radius);
                d[id]            = (1 - quantizable) * _s[i2 + 1][i1 + 1][i0 + 1];  // outlier
                q[id]            = quantizable * _code;
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
        xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
    }
    for (size_t i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= m->d0) continue;
        xdata[id] = xdata[id] * m->ebx2;
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
    Data _s[Block + 1][Block + 1];
    memset(_s, 0, (Block + 1) * (Block + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(m->radius);

    size_t _idx1 = b1 * Block;
    size_t _idx0 = b0 * Block;

    for (size_t i1 = 0; i1 < Block; i1++) {
        for (size_t i0 = 0; i0 < Block; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= m->d1 or gi0 >= m->d0) continue;
            const size_t id    = gi0 + gi1 * m->d0;
            Data         pred  = _s[i1][i0 + 1] + _s[i1 + 1][i0] - _s[i1][i0];
            _s[i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
            xdata[id]          = _s[i1 + 1][i0 + 1] * m->ebx2;
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
    Data _s[Block + 1][Block + 1][Block + 1];
    memset(_s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));
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
                Data   pred = _s[i2][i1][i0]                                                             // +, dist=3
                            - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                            + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                _s[i2 + 1][i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
                xdata[id]                  = _s[i2 + 1][i1 + 1][i0 + 1] * m->ebx2;
            }
        }
    }
}

}  // namespace predictor_quantizer_dualquant
}  // namespace psz

#endif
