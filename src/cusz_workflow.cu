/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.1
 * @date 2020-09-20
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
#include <type_traits>
#include <typeinfo>

#include "argparse.hh"
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
#include "timer.hh"
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using uint8__t = uint8_t;

const int gpu_B_1d = 256;
const int gpu_B_2d = 16;
const int gpu_B_3d = 8;

// moved to const_device.cuh
__constant__ int    symb_dims[16];
__constant__ double symb_ebs[4];

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

template <typename T, typename Q>
void cusz::impl::PdQ(T* d_data, Q* d_bcode, size_t* dims_L16, double* ebs_L4, argpack* ap)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto  d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);
    void* args[]     = {&d_data, &d_bcode, &d_dims_L16, &d_ebs_L4};

    /*timer*/ ap->cusz_events.push_back(new Event("Dual-Quant kernel"));
    /*timer*/ ap->cusz_events.back()->Start();
    if (dims_L16[nDIM] == 1) {
        dim3 blockNum(dims_L16[nBLK0]);
        dim3 threadNum(gpu_B_1d);
        cudaLaunchKernel(
            (void*)cusz::PdQ::c_lorenzo_1d1l<T, Q, gpu_B_1d>,  //
            blockNum, threadNum, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 threadNum(gpu_B_2d, gpu_B_2d);
        cudaLaunchKernel(
            (void*)cusz::PdQ::c_lorenzo_2d1l<T, Q, gpu_B_2d>,  //
            blockNum, threadNum, args, (gpu_B_2d + 1) * (gpu_B_2d + 1) * sizeof(T), nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 threadNum(gpu_B_3d, gpu_B_3d, gpu_B_3d);
        cudaLaunchKernel(
            (void*)cusz::PdQ::c_lorenzo_3d1l<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(T), nullptr);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
    /*timer*/ ap->cusz_events.back()->End();
}

template void cusz::impl::PdQ<float, uint8__t>(float*, uint8__t*, size_t*, double*, argpack*);
template void cusz::impl::PdQ<float, uint16_t>(float*, uint16_t*, size_t*, double*, argpack*);
template void cusz::impl::PdQ<float, uint32_t>(float*, uint32_t*, size_t*, double*, argpack*);
// template void cusz::impl::PdQ<double, uint8__t>(double*, uint8__t*, size_t*, double*, argpack*);
// template void cusz::impl::PdQ<double, uint16_t>(double*, uint16_t*, size_t*, double*, argpack*);
// template void cusz::impl::PdQ<double, uint32_t>(double*, uint32_t*, size_t*, double*, argpack*);

template <typename T, typename Q>
void cusz::impl::ReversedPdQ(T* d_xdata, Q* d_bcode, T* d_outlier, size_t* dims_L16, double _2eb, argpack* ap)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    void* args[]     = {&d_xdata, &d_outlier, &d_bcode, &d_dims_L16, &_2eb};

    /*timer*/ ap->cusz_events.push_back(new Event("Reversed Dual-Quant"));
    /*timer*/ ap->cusz_events.back()->Start();
    if (dims_L16[nDIM] == 1) {
        const static size_t p = gpu_B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1);
        cudaLaunchKernel((void*)PdQ::x_lorenzo_1d1l<T, Q, gpu_B_1d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        const static size_t p = gpu_B_2d;

        dim3 thread_num(p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_2d1l<T, Q, gpu_B_2d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        const static size_t p = gpu_B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1,   //
            (dims_L16[nBLK2] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d>, block_num, thread_num, args, 0, nullptr);
        // PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16,
        // _2eb);
    }
    else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();
    /*timer*/ ap->cusz_events.back()->End();

    cudaFree(d_dims_L16);
}

template <typename T, typename Q>
void cusz::impl::VerifyHuffman(
    string const& fi,
    size_t        len,
    Q*            xbcode,
    int           chunk_size,
    size_t*       dims_L16,
    double*       ebs_L4,
    argpack*      ap)
{
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto veri_data    = io::ReadBinaryFile<T>(fi, len);
    T*   veri_d_data  = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto veri_d_bcode = mem::CreateCUDASpace<Q>(len);
    PdQ(veri_d_data, veri_d_bcode, dims_L16, ebs_L4, ap);

    auto veri_bcode = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_bcode, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xbcode[i] != veri_bcode[i]) count++;
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
                if (xbcode[idx] != xbcode[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xbcode[idx - 1] << "\t" << xbcode[idx - 1]
                                 << endl;
                        }
                        else {
                            cerr << "wrong at first point!" << endl;
                        }
                        prev_point_printed = true;
                    }
                    cerr << "idx:" << idx << "\tdecoded: " << xbcode[idx] << "\tori: " << xbcode[idx] << endl;
                }
            }
        }
    }

    cudaFree(veri_d_bcode);
    cudaFree(veri_d_data);
    delete[] veri_bcode;
    delete[] veri_data;
    // end of if count
}

template <typename T, typename Q, typename H>
void cusz::workflow::Compress(
    argpack* ap,
    size_t*  dims_L16,
    double*  ebs_L4,
    int&     nnz_outlier,
    size_t&  n_bits,
    size_t&  n_uInt,
    size_t&  huffman_metadata_size)
{
    // TODO to use a struct
    size_t len = dims_L16[LEN];
    auto   m   = cusz::impl::GetEdgeOfReinterpretedSquare(len);  // row-major mxn matrix
    auto   mxm = m * m;

    // cout << log_dbg << "original len: " << len << ", m the padded: " << m << ", mxm: " << mxm << endl;

    /*timer*/ ap->cusz_events.push_back(new Event("Read Input Data"));
    /*timer*/ ap->cusz_events.back()->Start();
    auto data = new T[mxm]();
    io::ReadBinaryFile<T>(ap->cx_path2file, data, len);
    T* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, mxm);
    /*timer*/ ap->cusz_events.back()->End();

    if (ap->to_dryrun) {
        cout << "\n" << log_info << "Commencing dry-run..." << endl;
        DryRun(data, d_data, ap->cx_path2file, dims_L16, ebs_L4);
        exit(0);
    }
    cout << "\n" << log_info << "Commencing compression..." << endl;

    auto d_bcode = mem::CreateCUDASpace<Q>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    ::cusz::impl::PdQ(d_data, d_bcode, dims_L16, ebs_L4, ap);

    /*timer*/ ap->cusz_events.push_back(new Event("Prune-Gather Outliers (end-to-end)"));
    /*timer*/ ap->cusz_events.back()->Start();
    ::cusz::impl::PruneGatherAsCSR(d_data, mxm, m /*lda*/, m /*m*/, m /*n*/, nnz_outlier, &ap->c_fo_outlier);
    /*timer*/ ap->cusz_events.back()->End();
    cout << log_info << "nnz.outlier:\t" << nnz_outlier << "\t(" << (nnz_outlier / 1.0 / len * 100) << "%)" << endl;

    Q* bcode;
    if (ap->skip_huffman) {
        bcode = mem::CreateHostSpaceAndMemcpyFromDevice(d_bcode, len);
        io::WriteArrayToBinary(ap->c_fo_q, bcode, len);
        cout << log_info << "Compression finished, saved quant.code (Huffman skipped).\n" << endl;
        return;
    }

    std::tie(n_bits, n_uInt, huffman_metadata_size) =
        HuffmanEncode<Q, H>(ap, d_bcode, len, ap->huffman_chunk, dims_L16[CAP]);

    cout << log_info << "Compression finished, saved Huffman encoded quant.code.\n\n";

    // timer summary
    for (auto& i : ap->cusz_events) i->TimeElapsed(len * 4);
    cout << endl;

    for (auto& i : ap->cusz_events) delete i;
    ap->cusz_events.clear();

    delete[] data;
    cudaFree(d_data);
}

template <typename T, typename Q, typename H>
void cusz::workflow::Decompress(
    argpack* ap,
    size_t*  dims_L16,
    double*  ebs_L4,
    int&     nnz_outlier,
    size_t&  total_bits,
    size_t&  total_uInt,
    size_t&  huffman_metadata_size)
{
    auto dict_size = dims_L16[CAP];
    auto len       = dims_L16[LEN];
    auto m         = ::cusz::impl::GetEdgeOfReinterpretedSquare(len);
    auto mxm       = m * m;

    cout << log_info << "Commencing decompression..." << endl;

    Q* xbcode;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        cout << log_info << "Getting quant.code from filesystem... (Huffman encoding was skipped.)" << endl;
        xbcode = io::ReadBinaryFile<Q>(ap->x_fi_q, len);
    }
    else {
        cout << log_info << "Huffman decoding into quant.code." << endl;
        xbcode = HuffmanDecode<Q, H>(ap, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            // TODO check in argpack
            if (ap->x_fi_origin == "") {
                cerr << log_err << "use \"--orogin /path/to/origin_data\" to specify the original dataum." << endl;
                exit(-1);
            }
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cusz::impl::VerifyHuffman<T, Q>(ap->x_fi_origin, len, xbcode, ap->huffman_chunk, dims_L16, ebs_L4, ap);
        }
    }
    auto d_bcode = mem::CreateDeviceSpaceAndMemcpyFromHost(xbcode, len);

    auto d_outlier = mem::CreateCUDASpace<T>(mxm);

    /*timer*/ ap->cusz_events.push_back(new Event("Scatter Outliers"));
    /*timer*/ ap->cusz_events.back()->Start();
    ::cusz::impl::ScatterFromCSR<T>(d_outlier, mxm, m /*lda*/, m /*m*/, m /*n*/, &nnz_outlier, &ap->x_fi_outlier);
    /*timer*/ ap->cusz_events.back()->End();

    // TODO merge d_outlier and d_data
    auto d_xdata = mem::CreateCUDASpace<T>(len);
    ::cusz::impl::ReversedPdQ(d_xdata, d_bcode, d_outlier, dims_L16, ebs_L4[EBx2], ap);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    cout << log_info << "Decompression finished.\n\n";

    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(H)    // Huffman coded
                        + huffman_metadata_size;  // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Q);
    archive_size += nnz_outlier * (sizeof(T) + sizeof(int)) + (m + 1) * sizeof(int);

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
        cout << log_info << "dtype is \""         //
             << demangle(typeid(T).name())        // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Q).name())        // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(T) / sizeof(Q)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (ap->pre_binning) cout << log_info << "Because of 2x2->1 binning, extra 4x CR is added." << endl;
    if (not ap->skip_huffman) {
        cout << log_info
             << "Huffman metadata of chunking and reverse codebook size (in bytes): " << huffman_metadata_size << endl;
        cout << log_info << "Huffman coded output size: " << total_uInt * sizeof(H) << endl;
    }

    // TODO move CR out of VerifyData
    T* odata;
    if (ap->x_fi_origin != "") {
        cout << log_info << "To compare with the original datum" << endl;
        odata = io::ReadBinaryFile<T>(ap->x_fi_origin, len);
        analysis::VerifyData(
            xdata, odata,
            len,         //
            false,       //
            ebs_L4[EB],  //
            archive_size,
            ap->pre_binning ? 4 : 1);  // TODO use template rather than 2x2
    }

    if (!ap->skip_writex)
        io::WriteArrayToBinary(ap->x_fo_xd, xdata, len);
    else {
        cout << log_info << "Skipped writing unzipped to filesystem." << endl;
    }

    cout << log_info << "Decompression finished.\n\n";

    for (auto& i : ap->cusz_events) i->TimeElapsed(len * 4);
    cout << endl;

    for (auto& i : ap->cusz_events) delete i;
    ap->cusz_events.clear();

    // clean up
    if (odata) delete[] odata;
    delete[] xdata;
    delete[] xbcode;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_bcode);
}

template void
cusz::workflow::Compress<float, uint8__t, uint32_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Compress<float, uint8__t, uint64_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Compress<float, uint16_t, uint32_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Compress<float, uint16_t, uint64_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);

template void
cusz::workflow::Decompress<float, uint8__t, uint32_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Decompress<float, uint8__t, uint64_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Decompress<float, uint16_t, uint32_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
template void
cusz::workflow::Decompress<float, uint16_t, uint64_t>(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);
