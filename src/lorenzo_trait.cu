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

template <>
struct zip::Lorenzo_nd1l<1> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        cusz::predictor_quantizer::c_lorenzo_1d1l<Block, Data, Quant>(m, d, q);
    }
};

template <>
struct zip::Lorenzo_nd1l<2> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        cusz::predictor_quantizer::c_lorenzo_2d1l<Block, Data, Quant>(m, d, q);
    }
};

template <>
struct zip::Lorenzo_nd1l<3> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* d, Quant* q)
    {
        cusz::predictor_quantizer::c_lorenzo_3d1l<Block, Data, Quant>(m, d, q);
    }
};

template <>
struct unzip::Lorenzo_nd1l<1> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        cusz::predictor_quantizer::x_lorenzo_1d1l<Block, Data, Quant>(m, xd, outlier, q);
    }
};

template <>
struct unzip::Lorenzo_nd1l<2> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        cusz::predictor_quantizer::x_lorenzo_2d1l<Block, Data, Quant>(m, xd, outlier, q);
    }
};

template <>
struct unzip::Lorenzo_nd1l<3> {
    template <int Block, typename Data, typename Quant>
    static void Call(struct Metadata<Block>* m, Data* xd, Data* outlier, Quant* q)
    {
        cusz::predictor_quantizer::x_lorenzo_2d1l<Block, Data, Quant>(m, xd, outlier, q);
    }
};

template <>
struct dryrun::Lorenzo_nd1l<1> {
    template <int Block, typename Data>
    static void Call(struct Metadata<Block>* m, Data* d)
    {
        cusz::dryrun::lorenzo_1d1l(m, d);
    }
};

template <>
struct dryrun::Lorenzo_nd1l<2> {
    template <int Block, typename Data>
    static void Call(struct Metadata<Block>* m, Data* d)
    {
        cusz::dryrun::lorenzo_2d1l(m, d);
    }
};

template <>
struct dryrun::Lorenzo_nd1l<3> {
    template <int Block, typename Data>
    static void Call(struct Metadata<Block>* m, Data* d)
    {
        cusz::dryrun::lorenzo_3d1l(m, d);
    }
};