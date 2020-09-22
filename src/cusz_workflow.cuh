#ifndef CUSZ_WORKFLOW_CUH
#define CUSZ_WORKFLOW_CUH

/**
 * @file cusz_workflow.cuh
 * @author Jiannan Tian
 * @brief Workflow of cuSZ (header).
 * @version 0.1
 * @date 2020-09-21
 * Created on: 2020-02-12
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "argparse.hh"

using namespace std;

namespace cusz {
namespace interface {

template <typename Data, typename Quant, typename Huff>
void Compress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

template <typename Data, typename Quant, typename Huff>
void Decompress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

}  // namespace interface

namespace impl {

inline size_t GetEdgeOfReinterpretedSquare(size_t l) { return static_cast<size_t>(ceil(sqrt(l))); };

template <typename Data, typename Quant>
void PdQ(Data*, Quant*, size_t*, double*);

template <typename Data, typename Quant>
void ReversedPdQ(Data*, Quant*, Data*, size_t*, double);

template <typename Data, typename Quant>
void VerifyHuffman(string const&, size_t, Quant*, int, size_t*, double*);

}  // namespace impl

}  // namespace cusz

#endif
