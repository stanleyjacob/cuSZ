#ifndef METADATA_HH
#define METADATA_HH

/**
 * @file metadata.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cstddef>
#include <string>
#include <unordered_map>

#include "argparse.hh"

template <size_t Block>
struct Metadata {
    static const size_t block_size = Block;  // extract template parameter

    size_t ndim;
    size_t len = 1;
    size_t d0, d1, d2, d3;
    size_t stride0, stride1, stride2, stride3;
    size_t nb0, nb1, nb2, nb3;  // nb3 not usable in dim
    size_t cap, radius;
    int    nnz;
    size_t total_bits, total_uint, huff_metadata_size;
    double eb, ebx2, eb_r, ebx2_r;
};

template <size_t Block>
void cuszSetDim(struct Metadata<Block>* m, size_t _ndim, size_t _d0, size_t _d1, size_t _d2, size_t _d3)
{
    auto nb_of = [&](size_t dim) { return (dim - 1) / Block + 1; };

    m->ndim = _ndim;
    m->d0 = _d0, m->d1 = _d1, m->d2 = _d2, m->d3 = _d3;
    m->len = _d0 * _d1 * _d2 * _d3;
    m->nb0 = nb_of(_d0), m->nb1 = nb_of(_d1);
    m->nb2 = nb_of(_d2), m->nb3 = nb_of(_d3);
    m->stride0 = 1;
    m->stride1 = _d0;
    m->stride2 = _d0 * _d1;
    m->stride3 = _d0 * _d1 * _d2;
};

template <size_t Block>
void cuszSetDemoDim(struct Metadata<Block>* m, std::string const& datum)
{
    // {ndim, d0, d1, d2, d3} order
    size_t const hacc_1GB[]    = {1, 280953867, 1, 1, 1};
    size_t const hacc_4GB[]    = {1, 1073726487, 1, 1, 1};
    size_t const cesm[]        = {2, 3600, 1800, 1, 1};
    size_t const hurricane[]   = {3, 500, 500, 100, 1};
    size_t const nyx[]         = {3, 512, 512, 512, 1};
    size_t const qmc[]         = {3, 288, 69, 7935, 1};
    size_t const qmc_pre[]     = {3, 69, 69, 33120, 1};
    size_t const exafel_demo[] = {2, 388, 59200, 1, 1};
    size_t const aramco[]      = {3, 235, 849, 849, 1};

    std::unordered_map<std::string, size_t const*> entries = {
        {std::string("hacc1g"), hacc_1GB}, {std::string("hacc4g"), hacc_4GB},
        {std::string("cesm"), cesm},       {std::string("hurricane"), hurricane},
        {std::string("nyx"), nyx},         {std::string("qmc"), qmc},
        {std::string("qmcpre"), qmc_pre},  {std::string("exafeldemo"), exafel_demo},
        {std::string("aramco"), aramco}};
    auto e = entries.at(datum);
    cuszSetDim(m, e[0], e[1], e[2], e[3], e[4]);
}

template <size_t Block>
void cuszSetErrorBound(struct Metadata<Block>* m, double _eb)
{
    m->eb     = _eb;
    m->ebx2   = _eb * 2;
    m->eb_r   = 1 / _eb;
    m->ebx2_r = 1 / (_eb * 2);
}

template <size_t Block>
void cuszSetQuantBinNum(struct Metadata<Block>* m, size_t _cap)
{
    m->cap = _cap;
    m->cap = _cap / 2;
}

template <size_t Block>
void cuszChangeToR2RModeMode(struct Metadata<Block>* m, double val_rng)
{
    auto _eb = m->eb * val_rng;
    cuszSetErrorBound(m, _eb);
};

#endif