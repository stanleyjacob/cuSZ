#ifndef GATHER_SCATTER
#define GATHER_SCATTER

#include <cuda_runtime.h>
#include <cusparse.h>

/**
 * @file gather_scatter.cu
 * @author Jiannan Tian
 * @brief Gather/scatter method to handle cuSZ prediction outlier (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-09-10
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

namespace cusz {
namespace impl {

template <typename FP>
void GatherAsCSR(FP*, size_t, size_t, size_t, size_t, int*, std::string*);

template <typename FP>
void ScatterFromCSR(FP*, size_t, size_t, size_t, size_t, int*, std::string*);

void PruneGatherAsCSR(float*, size_t, const int, const int, const int, int&, std::string*);

}  // namespace impl
}  // namespace cusz

#endif