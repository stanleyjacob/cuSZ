/**
 * @file gather_scatter.cu
 * @author Jiannan Tian
 * @brief Gather/scatter method to handle cuSZ prediction outlier.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-09-10
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cassert>
#include <iostream>
#include "argparse.hh"
using std::cout;
using std::endl;

#include "cuda_error_handling.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "io.hh"

using handle_t = cusparseHandle_t;
using stream_t = cudaStream_t;
using descr_t  = cusparseMatDescr_t;

template <typename DType>
void cusz::impl::GatherAsCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fo)
{
    uint8_t* outbin;
    size_t   lrp, lci, lv, ltotal;

    {
        handle_t handle        = nullptr;
        stream_t stream        = nullptr;
        descr_t  descr         = nullptr;
        int*     d_nnz_per_row = nullptr;
        int*     d_row_ptr     = nullptr;
        int*     d_col_ind     = nullptr;
        DType*   d_csr_val     = nullptr;

        // clang-format off
        CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream, cudaStreamNonBlocking        )); // 1. create stream
        CHECK_CUSPARSE(cusparseCreate          ( &handle                               )); // 2. create handle
        CHECK_CUSPARSE(cusparseSetStream       (  handle, stream                       )); // 3. bind stream
        CHECK_CUSPARSE(cusparseCreateMatDescr  ( &descr                                )); // 4. create descr
        CHECK_CUSPARSE(cusparseSetMatIndexBase (  descr,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
        CHECK_CUSPARSE(cusparseSetMatType      (  descr,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type

        CHECK_CUDA(cudaMalloc((void**)&d_nnz_per_row, sizeof(int) * m));

        CHECK_CUSPARSE(cusparseSnnz(
            handle, CUSPARSE_DIRECTION_ROW, // parsed by row
            m, n, descr, d_A, ldA,          // descrption of d_A
            d_nnz_per_row, nnz)             // output
        );

        lrp    = sizeof(int)   * (m + 1);
        lci    = sizeof(int)   * *nnz;
        lv     = sizeof(DType) * *nnz;
        ltotal = lrp + lci + lv;
        outbin = new uint8_t[ltotal];
        CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, lrp));
        CHECK_CUDA(cudaMalloc((void**)&d_col_ind, lci));
        CHECK_CUDA(cudaMalloc((void**)&d_csr_val, lv ));

        CHECK_CUSPARSE(cusparseSdense2csr(
            handle,                             //
            m, n, descr, d_A, ldA,              // descritpion of d_A
            d_nnz_per_row,                      // prefileld by nnz() func
            d_csr_val, d_row_ptr, d_col_ind)    // output
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(outbin,             d_row_ptr, lrp, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outbin + lrp,       d_col_ind, lci, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outbin + lrp + lci, d_csr_val, lv,  cudaMemcpyDeviceToHost));

        if (d_row_ptr)  cudaFree(d_row_ptr);
        if (d_col_ind)  cudaFree(d_col_ind);
        if (d_csr_val)  cudaFree(d_csr_val);
        if (d_nnz_per_row) cudaFree(d_nnz_per_row);
        if (handle)     cusparseDestroy(handle);
        if (stream)     cudaStreamDestroy(stream);
        if (descr)      cusparseDestroyMatDescr(descr);
        // clang-format on
    }

    // cout << log_dbg << "outlier_bin byte length:\t" << ltotal << endl;
    io::WriteArrayToBinary(*fo, outbin, ltotal);
    delete[] outbin;
};

template void
cusz::impl::GatherAsCSR<float>(float* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fo);

template <typename DType>
void cusz::impl::ScatterFromCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fi)
{
    // clang-format off
    auto lrp         = sizeof(int) * (ldA + 1);
    auto lci         = sizeof(int) * *nnz;
    auto lv          = sizeof(DType) * *nnz;
    auto l_total     = lrp + lci + lv;
    auto outlier_bin = io::ReadBinaryFile<uint8_t>(*fi, l_total);
    auto row_ptr     = reinterpret_cast<int*>(outlier_bin);
    auto col_ind     = reinterpret_cast<int*>(outlier_bin + lrp);
    auto csr_val     = reinterpret_cast<DType*>(outlier_bin + lrp + lci);  // TODO template
    // clang-format on

    {
        handle_t handle    = nullptr;
        stream_t stream    = nullptr;
        descr_t  descr     = nullptr;
        int*     d_row_ptr = nullptr;
        int*     d_col_ind = nullptr;
        DType*   d_csr_val = nullptr;

        // clang-format off
        CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream, cudaStreamNonBlocking        )); // 1. create stream
        CHECK_CUSPARSE(cusparseCreate          ( &handle                               )); // 2. create handle
        CHECK_CUSPARSE(cusparseSetStream       (  handle, stream                       )); // 3. bind stream
        CHECK_CUSPARSE(cusparseCreateMatDescr  ( &descr                                )); // 4. create descr
        CHECK_CUSPARSE(cusparseSetMatIndexBase (  descr,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
        CHECK_CUSPARSE(cusparseSetMatType      (  descr,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type

        CHECK_CUDA(cudaMalloc( (void**)&d_row_ptr,   lrp ));
        CHECK_CUDA(cudaMalloc( (void**)&d_col_ind,   lci ));
        CHECK_CUDA(cudaMalloc( (void**)&d_csr_val,      lv    ));
        CHECK_CUDA(cudaMemcpy( d_row_ptr, row_ptr, lrp, cudaMemcpyHostToDevice ));
        CHECK_CUDA(cudaMemcpy( d_col_ind, col_ind, lci, cudaMemcpyHostToDevice ));
        CHECK_CUDA(cudaMemcpy( d_csr_val, csr_val, lv,  cudaMemcpyHostToDevice ));

        CHECK_CUSPARSE(cusparseScsr2dense(handle, m, n, descr, d_csr_val, d_row_ptr, d_col_ind, d_A, ldA));
        CHECK_CUDA(cudaDeviceSynchronize());

        if (d_row_ptr) cudaFree(d_row_ptr);
        if (d_col_ind) cudaFree(d_col_ind);
        if (d_csr_val) cudaFree(d_csr_val);
        if (handle)    cusparseDestroy(handle);
        if (stream)    cudaStreamDestroy(stream);
        if (descr)     cusparseDestroyMatDescr(descr);
        // clang-format on
    }

    cout << log_info << "Extracted outlier from CSR format." << endl;

    delete[] outlier_bin;
}

template void
cusz::impl::ScatterFromCSR<float>(float* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fi);

void cusz::impl::PruneGatherAsCSR(
    float*       d_A,  //
    size_t       lenA,
    const int    lda,
    const int    m,
    const int    n,
    int&         nnzC,
    std::string* fo,
    argpack*     ap)
{
    handle_t handle       = nullptr;
    stream_t stream       = nullptr;
    descr_t  descr        = nullptr;
    int*     d_row_ptr    = nullptr;
    int*     d_col_ind    = nullptr;
    float*   d_csr_val    = nullptr;
    size_t   lworkInBytes = 0;
    char*    d_work       = nullptr;
    float    threshold    = 0;

    /*timer*/ ap->cusz_events.push_back(new Event("HOST   CONFIG\tcuSPARSE setup"));
    /*timer*/ ap->cusz_events.back()->Start();
    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream, cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate          ( &handle                               )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle, stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &descr                                )); // 4. create descr
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  descr,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  descr,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on
    /*timer*/ ap->cusz_events.back()->End();

    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, sizeof(int) * (m + 1)));

    // omit for now
    // /*timer*/ ap->cusz_events.push_back(new Event("cuSPARSE buffer-size-ext"));
    // /*timer*/ ap->cusz_events.back()->Start();
    CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(  //
        handle, m, n, d_A, lda, &threshold, descr, d_csr_val, d_row_ptr, d_col_ind, &lworkInBytes));
    // /*timer*/ ap->cusz_events.back()->End();

    // printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
    if (nullptr != d_work) cudaFree(d_work);

    CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes));

    /*timer*/ ap->cusz_events.push_back(new Event("KERNEL LOSSY\tcuSPARSE compute row_ptr and nnz"));
    /*timer*/ ap->cusz_events.back()->Start();
    /* step 4: compute row_ptrC and nnzC */
    CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
        handle, m, n, d_A, lda, &threshold, descr, d_row_ptr, &nnzC, d_work));
    CHECK_CUDA(cudaDeviceSynchronize());
    /*timer*/ ap->cusz_events.back()->End();

    if (0 == nnzC) cout << log_info << "No outlier." << endl;

    /* step 5: compute col_indC and csr_valC */
    CHECK_CUDA(cudaMalloc((void**)&d_col_ind, sizeof(int) * nnzC));
    CHECK_CUDA(cudaMalloc((void**)&d_csr_val, sizeof(float) * nnzC));

    /*timer*/ ap->cusz_events.push_back(new Event("KERNEL LOSSY\tcuSPARSE compute col_idx and csr_val"));
    /*timer*/ ap->cusz_events.back()->Start();
    CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
        handle, m, n, d_A, lda, &threshold, descr, d_csr_val, d_row_ptr, d_col_ind, d_work));
    CHECK_CUDA(cudaDeviceSynchronize());
    /*timer*/ ap->cusz_events.back()->End();

    /* step 6: output C */
    auto lrp    = sizeof(int) * (m + 1);
    auto lci    = sizeof(int) * nnzC;
    auto lv     = sizeof(float) * nnzC;
    auto ltotal = lrp + lci + lv;
    auto outbin = new uint8_t[ltotal];

    /*timer*/ ap->cusz_events.push_back(new Event("PCIe   d2h\tmemcpy csr to host"));
    /*timer*/ ap->cusz_events.back()->Start();
    // clang-format off
    CHECK_CUDA(cudaMemcpy(outbin,             d_row_ptr, lrp, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outbin + lrp,       d_col_ind, lci, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outbin + lrp + lci, d_csr_val, lv,  cudaMemcpyDeviceToHost));
    // clang-format on
    /*timer*/ ap->cusz_events.back()->End();

    /*timer*/ ap->cusz_events.push_back(new Event("HOST   I/O\twrite csr to filesystem"));
    /*timer*/ ap->cusz_events.back()->Start();
    io::WriteArrayToBinary(*fo, outbin, ltotal);
    /*timer*/ ap->cusz_events.back()->End();

    if (d_A) cudaFree(d_A);
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_ind) cudaFree(d_col_ind);
    if (d_csr_val) cudaFree(d_csr_val);
    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descr) cusparseDestroyMatDescr(descr);
    if (outbin) delete[] outbin;
}
