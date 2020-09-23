/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.1
 * @date 2020-09-21
 * Created on: 2020-02-12
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cxxabi.h>
#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "argparse.hh"
#include "argparse2_cusz.hh"
#include "constants.hh"
#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "cusz_dryrun.cuh"
#include "cusz_dualquant.cuh"
#include "cusz_workflow.cuh"
#include "filter.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "huffman_workflow.cuh"
#include "io.hh"
#include "metadata.hh"
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using uint8__t = uint8_t;

const int gpu_B_1d = 32;
const int gpu_B_2d = 16;
const int gpu_B_3d = 8;

// moved to const_device.cuh
__constant__ int    symb_dims[16];
__constant__ double symb_ebs[4];

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

/**
 * @deprecated substitute this in 0.1.1 or higher

 */
template <typename Data, typename Quant>
void cusz::impl::PdQ(Data* d_data, Quant* d_q, size_t* dims_L16, double* ebs_L4)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto  d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);
    void* args[]     = {&d_data, &d_q, &d_dims_L16, &d_ebs_L4};

    if (dims_L16[nDIM] == 1) {
        dim3 grid_dim(dims_L16[nBLK0]);
        dim3 block_dim(gpu_B_1d);
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_1d1l<Data, Quant, gpu_B_1d>,  //
            grid_dim, block_dim, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        dim3 grid_dim(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 block_dim(gpu_B_2d, gpu_B_2d);
        // old, use physical padding
        // cudaLaunchKernel(
        //     (void*)cusz::predictor_quantizer::c_lorenzo_2d1l<Data, Quant, gpu_B_2d>,  //
        //     grid_dim, block_dim, args, (gpu_B_2d + 1) * (gpu_B_2d + 1) * sizeof(Data), nullptr);
        // new, use virtual padding
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<Data, Quant, gpu_B_2d>,  //
            grid_dim, block_dim, args, (gpu_B_2d) * (gpu_B_2d) * sizeof(Data), nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        dim3 grid_dim(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 block_dim(gpu_B_3d, gpu_B_3d, gpu_B_3d);
        // old, use physical padding
        // cudaLaunchKernel(
        //     (void*)cusz::predictor_quantizer::c_lorenzo_3d1l<Data, Quant, gpu_B_3d>,  //
        //     grid_dim, block_dim, args, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(Data), nullptr);
        // new, use virtual padding
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<Data, Quant, gpu_B_3d>,  //
            grid_dim, block_dim, args, (gpu_B_3d) * (gpu_B_3d) * (gpu_B_3d) * sizeof(Data), nullptr);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
}

template void cusz::impl::PdQ<float, uint8__t>(float* d_data, uint8__t* d_q, size_t* dims_L16, double* ebs_L4);
template void cusz::impl::PdQ<float, uint16_t>(float* d_data, uint16_t* d_q, size_t* dims_L16, double* ebs_L4);
template void cusz::impl::PdQ<float, uint32_t>(float* d_data, uint32_t* d_q, size_t* dims_L16, double* ebs_L4);
// template void cusz::impl::PdQ<double, uint8__t>(double* d_data, uint8__t* d_q, size_t* dims_L16, double* ebs_L4);
// template void cusz::impl::PdQ<double, uint16_t>(double* d_data, uint16_t* d_q, size_t* dims_L16, double* ebs_L4);
// template void cusz::impl::PdQ<double, uint32_t>(double* d_data, uint32_t* d_q, size_t* dims_L16, double* ebs_L4);

/**
 * @brief
 * @deprecated substitute this in 0.1.1 or higher
 *
 * @tparam Data
 * @tparam Quant
 * @param d_xdata
 * @param d_q
 * @param d_outlier
 * @param dims_L16
 * @param _2eb
 */
template <typename Data, typename Quant>
void cusz::impl::ReversedPdQ(Data* d_xdata, Quant* d_q, Data* d_outlier, size_t* dims_L16, double _2eb)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    void* args[]     = {&d_xdata, &d_outlier, &d_q, &d_dims_L16, &_2eb};

    if (dims_L16[nDIM] == 1) {
        const static size_t p = gpu_B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1);
        cudaLaunchKernel(                                                             //
            (void*)cusz::predictor_quantizer::x_lorenzo_1d1l<Data, Quant, gpu_B_1d>,  //
            block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        const static size_t p = gpu_B_2d;

        dim3 thread_num(p, p);
        dim3 block_num(((dims_L16[nBLK0] - 1) / p + 1), ((dims_L16[nBLK1] - 1) / p + 1));
        cudaLaunchKernel(                                                             //
            (void*)cusz::predictor_quantizer::x_lorenzo_2d1l<Data, Quant, gpu_B_2d>,  //
            block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        const static size_t p = gpu_B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num(
            ((dims_L16[nBLK0] - 1) / p + 1), ((dims_L16[nBLK1] - 1) / p + 1), ((dims_L16[nBLK2] - 1) / p + 1));
        cudaLaunchKernel(                                                             //
            (void*)cusz::predictor_quantizer::x_lorenzo_3d1l<Data, Quant, gpu_B_3d>,  //
            block_num, thread_num, args, 0, nullptr);
    }
    else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();

    cudaFree(d_dims_L16);
}

/**
 * @deprecated in 0.1.1
 */
template <typename Data, typename Quant>
void cusz::impl::VerifyHuffman(
    string const& fi,
    size_t        len,
    Quant*        xq,
    int           chunk_size,
    size_t*       dims_L16,
    double*       ebs_L4)
{
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto  veri_data   = io::ReadBinaryFile<Data>(fi, len);
    Data* veri_d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto  veri_d_q    = mem::CreateCUDASpace<Quant>(len);
    PdQ(veri_d_data, veri_d_q, dims_L16, ebs_L4);

    auto veri_q = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_q, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xq[i] != veri_q[i]) count++;
    if (count != 0)
        cerr << log_err << "percentage of not being equal: " << count / (1.0 * len) << "\n";
    else
        cout << log_info << "Decoded correctly." << endl;

    if (count != 0) {
        // auto chunk_size = ap->huffman_chunk;
        auto n_chunk = (len - 1) / chunk_size + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed   = false;
            auto prev_point_printed = false;
            for (auto i = 0; i < chunk_size; i++) {
                auto idx = i + c * chunk_size;
                if (idx >= len) break;
                if (xq[idx] != xq[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xq[idx - 1] << "\t" << xq[idx - 1] << endl;
                        }
                        else {
                            cerr << "wrong at first point!" << endl;
                        }
                        prev_point_printed = true;
                    }
                    cerr << "idx:" << idx << "\tdecoded: " << xq[idx] << "\tori: " << xq[idx] << endl;
                }
            }
        }
    }

    cudaFree(veri_d_q);
    cudaFree(veri_d_data);
    delete[] veri_q;
    delete[] veri_data;
    // end of if count
}

/**
 * @deprecated soon
 */
template <typename Data, typename Quant, typename Huff>
void cusz::interface::Compress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    int    bw         = sizeof(Quant) * 8;
    string fo_cdata   = fi + ".sza";
    string fo_q       = fi + ".b" + std::to_string(bw);
    string fo_outlier = fi + ".b" + std::to_string(bw) + ".outlier";

    // TODO to use a struct
    size_t len = dims_L16[LEN];
    auto   m   = cusz::impl::GetEdgeOfReinterpretedSquare(len);  // row-major mxn matrix
    auto   mxm = m * m;

    cout << log_dbg << "original len: " << len << ", m the padded: " << m << ", mxm: " << mxm << endl;

    auto data = new Data[mxm]();
    io::ReadBinaryFile<Data>(fi, data, len);
    Data* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, mxm);

    if (ap->dry_run) {
        cout << "\n" << log_info << "Commencing dry-run..." << endl;
        DryRun(data, d_data, fi, dims_L16, ebs_L4);
        exit(0);
    }
    cout << "\n" << log_info << "Commencing compression..." << endl;

    auto d_q = mem::CreateCUDASpace<Quant>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    ::cusz::impl::PdQ(d_data, d_q, dims_L16, ebs_L4);
    ::cusz::impl::PruneGatherAsCSR(d_data, mxm, m /*lda*/, m /*m*/, m /*n*/, nnz_outlier, &fo_outlier);
    cout << log_info << "nnz.outlier:\t" << nnz_outlier << "\t(" << (nnz_outlier / 1.0 / len * 100) << "%)" << endl;

    Quant* q;
    if (ap->skip_huffman) {
        q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, len);
        io::WriteBinaryFile(q, len, &fo_q);
        cout << log_info << "Compression finished, saved quant.code (Huffman skipped).\n" << endl;
        return;
    }

    std::tie(n_bits, n_uInt, huffman_metadata_size) =
        lossless::interface::HuffmanEncode<Quant, Huff>(fo_q, d_q, len, ap->huffman_chunk, dims_L16[CAP]);

    cout << log_info << "Compression finished, saved Huffman encoded quant.code.\n" << endl;

    delete[] data;
    cudaFree(d_data);
}

/**
 * @deprecated soon
 */
template <typename Data, typename Quant, typename Huff>
void cusz::interface::Decompress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    //    string f_archive = fi + ".sza"; // TODO
    string f_extract = ap->alt_xout_name.empty() ? fi + ".szx" : ap->alt_xout_name;
    string fi_q_base, fi_q_after_huffman, fi_outlier, fi_outlier_as_cuspm;

    fi_q_base           = fi + ".b" + std::to_string(sizeof(Quant) * 8);
    fi_outlier_as_cuspm = fi_q_base + ".outlier";

    auto dict_size = dims_L16[CAP];
    auto len       = dims_L16[LEN];
    auto m         = ::cusz::impl::GetEdgeOfReinterpretedSquare(len);
    auto mxm       = m * m;

    cout << log_info << "Commencing decompression..." << endl;

    Quant* xq;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        cout << log_info << "Getting quant.code from filesystem... (Huffman encoding was skipped.)" << endl;
        xq = io::ReadBinaryFile<Quant>(fi_q_base, len);
    }
    else {
        cout << log_info << "Huffman decoding into quant.code." << endl;
        xq =
            ::lossless::interface::HuffmanDecode<Quant, Huff>(fi_q_base, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cusz::impl::VerifyHuffman<Data, Quant>(fi, len, xq, ap->huffman_chunk, dims_L16, ebs_L4);
        }
    }
    auto d_q = mem::CreateDeviceSpaceAndMemcpyFromHost(xq, len);

    auto d_outlier = mem::CreateCUDASpace<Data>(mxm);
    ::cusz::impl::ScatterFromCSR<Data>(d_outlier, mxm, m /*lda*/, m /*m*/, m /*n*/, &nnz_outlier, &fi_outlier_as_cuspm);

    // TODO merge d_outlier and d_data
    auto d_xdata = mem::CreateCUDASpace<Data>(len);
    ::cusz::impl::ReversedPdQ(d_xdata, d_q, d_outlier, dims_L16, ebs_L4[EBx2]);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    cout << log_info << "Decompression finished.\n\n";

    // TODO move CR out of VerifyData
    auto   odata        = io::ReadBinaryFile<Data>(fi, len);
    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(Huff)  // Huffman coded
                        + huffman_metadata_size;   // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Quant);
    archive_size += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

    // TODO g++ and clang++ use mangled type_id name, add macro
    // https://stackoverflow.com/a/4541470/8740097
    auto demangle = [](const char* name) {
        int   status = -4;
        char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

        const char* const demangled_name = (status == 0) ? res : name;
        string            ret_val(demangled_name);
        free(res);
        return ret_val;
    };

    if (ap->skip_huffman) {
        cout << log_info << "Data is \""          //
             << demangle(typeid(Data).name())     // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Quant).name())    // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(Data) / sizeof(Quant)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (ap->pre_binning) cout << log_info << "Because of 2x2->1 binning, extra 4x CR is added." << endl;
    if (not ap->skip_huffman) {
        cout << log_info
             << "Huffman metadata of chunking and reverse codebook size (in bytes): " << huffman_metadata_size << endl;
        cout << log_info << "Huffman coded output size: " << total_uInt * sizeof(Huff) << endl;
    }

    analysis::VerifyData(
        xdata, odata,
        len,         //
        false,       //
        ebs_L4[EB],  //
        archive_size,
        ap->pre_binning ? 4 : 1);  // suppose binning is 2x2

    if (!ap->skip_writex) {
        if (!ap->alt_xout_name.empty())
            cout << log_info << "Default decompressed data is renamed from " << string(fi + ".szx") << " to "
                 << f_extract << endl;
        io::WriteBinaryFile(xdata, len, &f_extract);
    }

    // clean up
    delete[] odata;
    delete[] xdata;
    delete[] xq;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_q);
}

template void cusz::interface::Compress<float, uint8__t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Compress<float, uint8__t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Compress<float, uint16_t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Compress<float, uint16_t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Compress<float, uint32_t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Compress<float, uint32_t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);

template void cusz::interface::Decompress<float, uint8__t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Decompress<float, uint8__t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Decompress<float, uint16_t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Decompress<float, uint16_t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Decompress<float, uint32_t, uint32_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cusz::interface::Decompress<float, uint32_t, uint64_t>  //
    (string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
