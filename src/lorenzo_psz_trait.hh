/**
 * @file lorenzo_psz_trait.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-11-01
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef LORENZO_PSZ_TRAIT_HH
#define LORENZO_PSZ_TRAIT_HH

#ifdef PSZ_OMP
#include <omp.h>
#endif

#include "metadata.hh"
#include "psz/pq14.hh"
#include "psz/pq14_chunked.hh"
#include "psz/pq_dualquant.hh"

#if __cplusplus >= 201703L
#else
#define constexpr
#endif

namespace psz {
// clang-format off
namespace pq_dualquant {
namespace zip   { template <int ndim> struct Lorenzo_nd1l; }
namespace unzip { template <int ndim> struct Lorenzo_nd1l; }
}  // namespace dualquant
namespace pq14 {
namespace zip   { template <int ndim> struct Lorenzo_nd1l; }
namespace unzip { template <int ndim> struct Lorenzo_nd1l; }
}  // namespace pq14
namespace pq14_chunked {
namespace zip   { template <int ndim> struct Lorenzo_nd1l; }
namespace unzip { template <int ndim> struct Lorenzo_nd1l; }
}  // namespace pq14_chunked
// clang-format on
}  // namespace psz

namespace pszdq    = psz::predictor_quantizer_dualquant;
namespace psz14par = psz::predictor_quantizer_sz14par;
namespace psz14    = psz::predictor_quantizer_sz14;

// chunked dualquant
template <int ndim>
struct psz::pq_dualquant::zip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        constexpr if (ndim == 1)
        {
#pragma omp parallel for
            for (auto b0 = 0; b0 < m->nb0; b0++)  //
                pszdq::c_lorenzo_1d1l<Block, Data, Quant>(m, d, q, b0);
        }
        constexpr if (ndim == 2)
        {
#pragma omp parallel for collapse(2)
            for (auto b1 = 0; b1 < m->nb1; b1++)
                for (auto b0 = 0; b0 < m->nb0; b0++)  //
                    pszdq::c_lorenzo_2d1l<Block, Data, Quant>(m, d, q, b0, b1);
        }
        constexpr if (ndim == 3)
        {
#pragma omp parallel for collapse(3)
            for (auto b2 = 0; b2 < m->nb2; b2++)
                for (auto b1 = 0; b1 < m->nb1; b1++)
                    for (auto b0 = 0; b0 < m->nb0; b0++)  //
                        pszdq::c_lorenzo_3d1l<Block, Data, Quant>(m, d, q, b0, b1, b2);
        }
    }
};

// chunked dualquant
template <int ndim>
struct psz::pq_dualquant::unzip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        constexpr if (ndim == 1)
        {
#pragma omp parallel for
            for (auto b0 = 0; b0 < m->nb0; b0++)  //
                pszdq::x_lorenzo_1d1l<Block, Data, Quant>(m, xd, outlier, q, b0);
        }
        constexpr if (ndim == 2)
        {
#pragma omp parallel for collapse(2)
            for (auto b1 = 0; b1 < m->nb1; b1++)
                for (auto b0 = 0; b0 < m->nb0; b0++)
                    pszdq::x_lorenzo_2d1l<Block, Data, Quant>(m, xd, outlier, q, b0, b1);
        }
        constexpr if (ndim == 3)
        {
#pragma omp parallel for collapse(3)
            for (auto b2 = 0; b2 < m->nb2; b2++)
                for (auto b1 = 0; b1 < m->nb1; b1++)
                    for (auto b0 = 0; b0 < m->nb0; b0++)
                        pszdq::x_lorenzo_3d1l<Block, Data, Quant>(m, xd, outlier, q, b0, b1, b2);
        }
    }
};

// chunked psz14
template <int ndim>
struct psz::pq14_chunked::zip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        constexpr if (ndim == 1)
        {
#pragma omp parallel for
            for (auto b0 = 0; b0 < m->nb0; b0++)
                psz14par::c_lorenzo_1d1l<Block, Data, Quant>  //
                    (m, d, q, b0);
        }
        constexpr if (ndim == 2)
        {
#pragma omp parallel for collapse(2)
            for (auto b1 = 0; b1 < m->nb1; b1++)
                for (auto b0 = 0; b0 < m->nb0; b0++)
                    psz14par::c_lorenzo_2d1l<Block, Data, Quant>  //
                        (m, d, q, b0, b1);
        }
        constexpr if (ndim == 3)
        {
#pragma omp parallel for collapse(3)
            for (auto b2 = 0; b2 < m->nb2; b2++)
                for (auto b1 = 0; b1 < m->nb1; b1++)
                    for (auto b0 = 0; b0 < m->nb0; b0++)
                        psz14par::c_lorenzo_3d1l<Block, Data, Quant>  //
                            (m, d, q, b0, b1, b2);
        }
    }
};

// chunked psz14
template <int ndim>
struct psz::pq14_chunked::unzip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        constexpr if (ndim == 1)
        {
#pragma omp parallel for
            for (auto b0 = 0; b0 < m->nb0; b0++)
                psz::predictor_quantizer_sz14par::x_lorenzo_1d1l<Block, Data, Quant>  //
                    (m, xd, outlier, q, b0);
        }
        constexpr if (ndim == 2)
        {
#pragma omp parallel for collapse(2)
            for (auto b1 = 0; b1 < m->nb1; b1++)
                for (auto b0 = 0; b0 < m->nb0; b0++)
                    psz::predictor_quantizer_sz14par::x_lorenzo_2d1l<Block, Data, Quant>  //
                        (m, xd, outlier, q, b0, b1);
        }
        constexpr if (ndim == 3)
        {
#pragma omp parallel for collapse(3)
            for (auto b2 = 0; b2 < m->nb2; b2++)
                for (auto b1 = 0; b1 < m->nb1; b1++)
                    for (auto b0 = 0; b0 < m->nb0; b0++)
                        psz::predictor_quantizer_sz14par::x_lorenzo_3d1l<Block, Data, Quant>  //
                            (m, xd, outlier, q, b0, b1, b2);
        }
    }
};

// singleton psz14
template <int ndim>
struct psz::pq14::zip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        constexpr if (ndim == 1) { psz14::c_lorenzo_1d1l<Block, Data, Quant>(m, d, q); }
        constexpr if (ndim == 2) { psz14::c_lorenzo_2d1l<Block, Data, Quant>(m, d, q); }
        constexpr if (ndim == 3) { psz14::c_lorenzo_3d1l<Block, Data, Quant>(m, d, q); }
    }
};

// singleton psz14
template <int ndim>
struct psz::pq14::unzip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        constexpr if (ndim == 1) { psz14::x_lorenzo_1d1l<Block, Data, Quant>(m, xd, outlier, q); }
        constexpr if (ndim == 2) { psz14::x_lorenzo_2d1l<Block, Data, Quant>(m, xd, outlier, q); }
        constexpr if (ndim == 3) { psz14::x_lorenzo_3d1l<Block, Data, Quant>(m, xd, outlier, q); }
    }
};

#endif