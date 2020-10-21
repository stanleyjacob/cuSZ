#ifndef CUSZ_WORKFLOW2_CUH
#define CUSZ_WORKFLOW2_CUH

/**
 * @file cusz_workflow2.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <algorithm>
#include <string>
#include "argparse2_cusz.hh"
#include "metadata.hh"
#include "type_trait.hh"

using std::string;

namespace cusz {
namespace interface {

template <int ndim, typename Data, int QuantByte, int HuffByte>
void Compress2(cuszContext* ctx, typename MetadataTrait<ndim>::metadata_t* m);

template <int ndim, typename Data, int QuantByte, int HuffByte>
void Decompress2(cuszContext* ctx, typename MetadataTrait<ndim>::metadata_t* m);

}  // namespace interface

namespace impl {

template <int ndim, typename Data, int QuantByte>
void VerifyHuffman(
    cuszContext*                              ctx,
    typename MetadataTrait<ndim>::metadata_t* m,
    typename QuantTrait<QuantByte>::Quant*    xq);

}

}  // namespace cusz

#endif