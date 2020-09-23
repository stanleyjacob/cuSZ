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

using std::string;

namespace cusz {
namespace interface {

template <int ndim, int Block, typename Data, int QuantByte, int HuffByte>
void Compress2(cuszContext* ctx, struct Metadata<Block>* m);

template <int ndim, int Block, typename Data, int QuantByte, int HuffByte>
void Decompress2(cuszContext* ctx, struct Metadata<Block>* m);

}  // namespace interface

namespace impl {

template <int ndim, int Block, typename Data, typename Quant>
void VerifyHuffman(cuszContext* ctx, struct Metadata<Block>* m, Quant* xq);

}

}  // namespace cusz
