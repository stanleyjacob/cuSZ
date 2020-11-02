#ifndef PSZ_PQ14_HH
#define PSZ_PQ14_HH

/**
 * @file pq14.hh
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
namespace predictor_quantizer_sz14 {

template <int Block, typename Data, typename Quant>
void c_lorenzo_1d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q)
{
    auto radius = static_cast<Quant>(m->radius);
    for (ptrdiff_t id = 0; id < m->d0; id++) {
        auto  current     = d + id;
        Data  pred        = id == 0 ? 0 : d[id - 1];
        Data  err         = *current - pred;
        Quant bin_count   = fabs(err) * m->eb_r + 1;
        bool  quantizable = fabs(bin_count) < m->cap;
        if (err < 0) bin_count = -bin_count;
        Quant _code = static_cast<Quant>(bin_count / 2) + radius;
        *current    = pred + (_code - radius) * m->ebx2;
        d[id]       = (1 - quantizable) * (*current);  // outlier
        q[id]       = quantizable * _code;
    }
}

template <int Block, typename Data, typename Quant>
void c_lorenzo_2d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q)
{
    auto  radius = static_cast<Quant>(m->radius);
    Data *NW = new Data, *NE = new Data, *SW = new Data, *SE;

    for (ptrdiff_t i1 = 0; i1 < m->d1; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < m->d0; i0++) {  // SW (SE)<- to predict
            *NW       = i1 == 0 or i0 == 0 ? 0.0 : *(d + (i0 - 1) + (i1 - 1) * m->d0);
            *NE       = i1 == 0 ? 0.0 : *(d + i0 + (i1 - 1) * m->d0);
            *SW       = i0 == 0 ? 0.0 : *(d + (i0 - 1) + i1 * m->d0);
            size_t id = i0 + i1 * m->d0;
            SE        = d + id;

            Data  pred        = (*NE) + (*SW) - (*NW);
            Data  err         = (*SE) - pred;
            Quant bin_count   = fabs(err) * m->eb_r + 1;
            bool  quantizable = fabs(bin_count) < m->cap;
            if (err < 0) bin_count = -bin_count;
            auto _code = static_cast<Quant>(bin_count / 2) + radius;
            *SE        = pred + (_code - radius) * m->ebx2;
            d[id]      = (1 - quantizable) * (*SE);  // outlier
            q[id]      = quantizable * _code;
        }
    }
}

template <int Block, typename Data, typename Quant>
void c_lorenzo_3d1l(
    struct Metadata<Block>* m,  //
    Data*                   d,
    Quant*                  q)
{
    auto      radius = static_cast<Quant>(m->radius);
    Data *    NWo = new Data, *NEo = new Data, *SWo = new Data, *SEo = new Data;
    Data *    NWi = new Data, *NEi = new Data, *SWi = new Data, *SEi;
    ptrdiff_t w0 = 1, w1 = m->d0, w2 = m->d0 * m->d1;

    for (ptrdiff_t i2 = 0; i2 < m->d2; i2++) {          //  | \---> x  NWo NEo
        for (ptrdiff_t i1 = 0; i1 < m->d1; i1++) {      //  v  v       SWo SEo  NWi  NEi
            for (ptrdiff_t i0 = 0; i0 < m->d0; i0++) {  //  y   z               SWi (SEi)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0 : *(d + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(d + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(d + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(d + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(d + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(d + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(d + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = d + id;

                Data  pred        = +(*NWo) - (*NEo) - (*SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                Data  err         = (*SEi) - pred;
                Quant bin_count   = fabs(err) * m->eb_r + 1;
                bool  quantizable = fabs(bin_count) < m->cap;
                if (err < 0) bin_count = -bin_count;
                auto _code = static_cast<Quant>(bin_count / 2) + radius;
                *SEi       = pred + (_code - radius) * m->ebx2;
                d[id]      = (1 - quantizable) * (*SEi);  // outlier
                q[id]      = quantizable * _code;
            }
        }
    }
}

template <int Block, typename Data, typename Quant>
void x_lorenzo_1d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q)
{
    auto radius = static_cast<Quant>(m->radius);
    for (ptrdiff_t id = 0; id < m->d0; id++) {
        Data pred = id == 0 ? 0 : xdata[id - 1];
        xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);
    }
}
template <int Block, typename Data, typename Quant>
void x_lorenzo_2d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q)
{
    auto  radius = static_cast<Quant>(m->radius);
    Data *NW = new Data, *NE = new Data, *SW = new Data;

    for (ptrdiff_t i1 = 0; i1 < m->d1; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < m->d0; i0++) {  // SW (SE)<- to predict
            *NW = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + (i1 - 1) * m->d0);
            *NE = i1 == 0 ? 0.0 : *(xdata + i0 + (i1 - 1) * m->d0);
            *SW = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + i1 * m->d0);

            Data   pred = (*NE) + (*SW) - (*NW);
            size_t id   = i0 + i1 * m->d0;
            xdata[id]   = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);
        }
    }
}

template <int Block, typename Data, typename Quant>
void x_lorenzo_3d1l(
    struct Metadata<Block>* m,  //
    Data*                   xdata,
    Data*                   outlier,
    Quant*                  q)
{
    auto      radius = static_cast<Quant>(m->radius);
    Data *    NWo = new Data, *NEo = new Data, *SWo = new Data, *SEo = new Data;
    Data *    NWi = new Data, *NEi = new Data, *SWi = new Data, *SEi;
    ptrdiff_t w0 = 1, w1 = m->d0, w2 = m->d0 * m->d1;

    for (ptrdiff_t i2 = 0; i2 < m->d2; i2++) {          // NW  NE
        for (ptrdiff_t i1 = 0; i1 < m->d1; i1++) {      // NW  NE
            for (ptrdiff_t i0 = 0; i0 < m->d0; i0++) {  // SW (SE)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(xdata + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = xdata + id;

                Data pred = +(*NWo) - *(NEo) - *(SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * m->ebx2);
            }
        }
    }
}

}  // namespace predictor_quantizer_sz14
}  // namespace psz

#endif
