/**
 * @file lorenzo_trait.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "cusz_dryrun.cuh"
#include "cusz_dualquant.cuh"
#include "lorenzo_trait.cuh"

#if __cplusplus >= 201703L

template <int ndim>
struct zip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        constexpr if (n == 1) cusz::predictor_quantizer::c_lorenzo_1d1l<Block, Data, Quant>(m, d, q);
        constexpr if (n == 2) cusz::predictor_quantizer::c_lorenzo_2d1l<Block, Data, Quant>(m, d, q);
        constexpr if (n == 3) cusz::predictor_quantizer::c_lorenzo_3d1l<Block, Data, Quant>(m, d, q);
    }
};

template <int ndim>
struct unzip::Lorenzo_nd1l {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        constexpr if (n == 1) cusz::predictor_quantizer::x_lorenzo_1d1l<Block, Data, Quant>(m, xd, outlier, q);
        constexpr if (n == 2) cusz::predictor_quantizer::x_lorenzo_2d1l<Block, Data, Quant>(m, xd, outlier, q);
        constexpr if (n == 3) cusz::predictor_quantizer::x_lorenzo_3d1l<Block, Data, Quant>(m, xd, outlier, q);
    }
};

template <int ndim>
struct dryrun::Lorenzo_nd1l {
    template <int Block, typename Data>
    static void Call(struct Metadata<Block>* m, Data* d)
    {
        constexpr if (n == 1) cusz::dryrun::lorenzo_1d1l(m, d);
        constexpr if (n == 2) cusz::dryrun::lorenzo_2d1l(m, d);
        constexpr if (n == 3) cusz::dryrun::lorenzo_3d1l(m, d);
    }
};

#endif