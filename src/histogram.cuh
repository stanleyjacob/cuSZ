/**
 * @file histogram.cuh
 * @author Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp (mlhickm@g.clemson.edu)
 * @brief Fast histogramming from [Gómez-Luna et al. 2013] (header)
 * @version 0.1
 * @date 2020-09-21
 * Created on 2020-02-16
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <cuda_runtime.h>
#include <cstdio>

const static unsigned int WARP_SIZE = 32;
#define MIN(a, b) ((a) < (b)) ? (a) : (b)

namespace data_process {

namespace reduce {
__global__ void naiveHistogram(int input_data[], int output[], int N, int symbols_per_thread);

// Optimized 2013
/* Copied from J. Gomez-Luna et al */
template <typename UInt1, typename UInt2>
__global__ void p2013Histogram(UInt1* input_data, UInt2* output, size_t N, int bins, int R);

}  // namespace reduce

}  // namespace data_process

#endif
