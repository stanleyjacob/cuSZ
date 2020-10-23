/**
 * @file cusz_workflow2.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <memory>
#include <string>

#include "argparse2_cusz.hh"
#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
// #include "cusz_workflow.cuh"
#include "cusz_dualquant.cuh"
#include "cusz_workflow.cuh"
#include "cusz_workflow2.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "huffman_workflow.cuh"
#include "io.hh"
#include "lorenzo_trait.cu"
// #include "lorenzo_trait.cuh"
#include "metadata.hh"
#include "type_trait.hh"

typedef cuszContext ctx_t;

template <int ndim, typename Data, int QuantByte, int HuffByte>
void cusz::interface::Compress2(ctx_t* ctx, typename MetadataTrait<ndim>::metadata_t* m)
{
    typedef typename MetadataTrait<ndim>::metadata_t metadata_t;
    typedef typename QuantTrait<QuantByte>::Quant    Quant;
    typedef typename HuffTrait<HuffByte>::Huff       Huff;

    static const int Block = MetadataTrait<ndim>::Block;

    string fo_zip     = ctx->get_fname() + ".sza";
    string fo_q       = ctx->get_fname() + ".b" + std::to_string(QuantByte * 8);
    string fo_outlier = ctx->get_fname() + ".b" + std::to_string(QuantByte * 8) + "outlier";

    auto M   = cusz::impl::GetEdgeOfReinterpretedSquare(m->len);
    auto MxM = M * M;

    // auto data = new Data[MxM]();
    auto data = std::unique_ptr<Data[]>(new Data[MxM]());

    io::ReadBinaryToArray<Data>(ctx->get_fname(), data.get(), m->len);
    auto d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data.get(), MxM);

    metadata_t* d_m;
    cudaMalloc((void**)&d_m, sizeof(metadata_t));
    cudaMemcpy(d_m, m, sizeof(metadata_t), cudaMemcpyHostToDevice);

    if (ctx->wf_dryrun) {
        void* args[] = {&d_m, &d_data};
        dim3  grid_dim(m->nb0, m->nb1, m->nb2), block_dim(m->b, m->b, m->b);
#if __cplusplus >= 201703L
        cudaLaunchKernel(
            (void*)dryrun::Lorenzo_nd1l<ndim>::Call<Block, Data>,  //
            grid_dim, block_dim, args, 0, nullptr);
#elif __cplusplus >= 201402L
        if (ndim == 1)
            cudaLaunchKernel((void*)cusz::dryrun::lorenzo_1d1l<Block, Data>, grid_dim, block_dim, args, 0, nullptr);
        else if (ndim == 2)
            cudaLaunchKernel((void*)cusz::dryrun::lorenzo_2d1l<Block, Data>, grid_dim, block_dim, args, 0, nullptr);
        else if (ndim == 3)
            cudaLaunchKernel((void*)cusz::dryrun::lorenzo_3d1l<Block, Data>, grid_dim, block_dim, args, 0, nullptr);
#endif
        goto COMPRESS_END;
    }

    if (ctx->wf_zip) {
        auto d_q = mem::CreateCUDASpace<Quant>(m->len);

        {  // Lorenzo
            void*  args[] = {&d_m, &d_data, &d_q};
            dim3   grid_dim(m->nb0, m->nb1, m->nb2), block_dim(m->b, m->b, m->b);
            size_t cache_size = Block;
            for (auto i = 0; i < ndim - 1; i++) cache_size *= Block;

#if __cplusplus >= 201703L
            cudaLaunchKernel(
                (void*)zip::Lorenzo_nd1l<ndim>::Call<Block, Data, Quant>,  //
                grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
#elif __cplusplus >= 201402L
            if (ndim == 1)  // compile time?
                cudaLaunchKernel(
                    (void*)cusz::predictor_quantizer::c_lorenzo_1d1l<Block, Data, Quant>,  //
                    grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
            if (ndim == 2)  // compile time?
                cudaLaunchKernel(
                    (void*)cusz::predictor_quantizer::c_lorenzo_2d1l<Block, Data, Quant>,  //
                    grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
            else if (ndim == 3)  // compile time?
                cudaLaunchKernel(
                    (void*)cusz::predictor_quantizer::c_lorenzo_3d1l<Block, Data, Quant>,  //
                    grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
#endif
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (ctx->skip_huff) {
            auto q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, m->len);
            io::WriteArrayToBinary(fo_q, q, m->len);
            // TODO log
            delete[] q, cudaFree(d_q), cudaFree(d_data), exit(0);
        }

        // handle outlier
        ::cusz::impl::PruneGatherAsCSR(d_data, MxM, M /*lda*/, M /*m*/, M /*n*/, m->nnz, &fo_outlier);

        // TODO handle metadata
        std::tie(m->total_bits, m->total_uint, m->huff_metadata_size) =
            lossless::interface::HuffmanEncode<Quant, Huff>(fo_q, d_q, m->len, ctx->h_chunksize, m->cap);

        cout << log_info << "Compression finished, saved Huffman encoded quant.code.\n" << endl;
    }

COMPRESS_END:
    cudaFree(d_data);
}

template <int ndim, typename Data, int QuantByte, int HuffByte>
void cusz::interface::Decompress2(ctx_t* ctx, typename MetadataTrait<ndim>::metadata_t* m)
{
    typedef typename MetadataTrait<ndim>::metadata_t metadata_t;
    typedef typename QuantTrait<QuantByte>::Quant    Quant;
    typedef typename HuffTrait<HuffByte>::Huff       Huff;

    static const int Block = MetadataTrait<ndim>::Block;

    // todo, put in context
    string fo_x       = ctx->get_fname() + ".szx";
    string fi_qbase   = ctx->get_fname() + ".b" + std::to_string(QuantByte * 8);
    string fi_outlier = fi_qbase + ".outlier";

    auto M   = ::cusz::impl::GetEdgeOfReinterpretedSquare(m->len);
    auto MxM = M * M;

    Quant* xq;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ctx->skip_huff) { xq = io::ReadBinaryToNewArray<Quant>(fi_qbase, m->len); }
    else {
        xq = ::lossless::interface::HuffmanDecode<Quant, Huff>(
            fi_qbase, m->len, ctx->h_chunksize, m->total_uint, m->cap);
        if (ctx->verify_huffman) cusz::impl::VerifyHuffman<ndim, Data, QuantByte>(ctx, m, xq);
    }
    auto d_xq      = mem::CreateDeviceSpaceAndMemcpyFromHost(xq, m->len);
    auto d_outlier = mem::CreateCUDASpace<Data>(MxM);

    cusz::impl::ScatterFromCSR(d_outlier, MxM, M, M, M, &m->nnz, &fi_outlier);

    metadata_t* d_m;
    cudaMalloc(&d_m, sizeof(metadata_t));
    cudaMemcpy(d_m, m, sizeof(metadata_t), cudaMemcpyHostToDevice);

    auto d_xdata = mem::CreateCUDASpace<Data>(m->len);

    {  // Lorenzo
        void* args[] = {&d_m, &d_xdata, &d_xq};
        dim3  grid_dim(m->nb0, m->nb1, m->nb2), block_dim(m->b, m->b, m->b);
        // size_t cache_size = Block;
        // for (auto i = 0; i < ndim - 1; i++) cache_size *= Block;

#if __cplusplus >= 201703L
        cudaLaunchKernel(
            (void*)unzip::Lorenzo_nd1l<ndim>::Call<Block, Data, Quant>,  //
            grid_dim, block_dim, args, 0, nullptr);
#elif __cplusplus >= 201402L
        if (ndim == 1)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::x_lorenzo_1d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, 0, nullptr);
        if (ndim == 2)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::x_lorenzo_2d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, 0, nullptr);
        else if (ndim == 3)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::x_lorenzo_3d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, 0, nullptr);
#endif
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, m->len);
    if (not ctx->skip_writex)
        io::WriteArrayToBinary(fo_x, xdata, m->len);
    else
        cout << log_info << "Skipped writing unzipped data to filesystem." << endl;

    delete[] xdata;
    delete[] xq;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_xq);
}

template <int ndim, typename Data, int QuantByte>
void cusz::impl::VerifyHuffman(
    ctx_t*                                    ctx,
    typename MetadataTrait<ndim>::metadata_t* m,
    typename QuantTrait<QuantByte>::Quant*    xq)
{
    typedef typename MetadataTrait<ndim>::metadata_t metadata_t;
    typedef typename QuantTrait<QuantByte>::Quant    Quant;

    static const int Block = MetadataTrait<ndim>::Block;

    // TODO error handling from invalid read
    // cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto  data   = io::ReadBinaryToNewArray<Data>(ctx->get_fname(), m->len);
    Data* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, m->len);
    auto  d_q    = mem::CreateCUDASpace<Quant>(m->len);

    metadata_t* d_m;
    cudaMalloc(&d_m, sizeof(metadata_t));
    cudaMemcpy(d_m, m, sizeof(metadata_t), cudaMemcpyHostToDevice);

    {  // Lorenzo
        void*  args[] = {&d_m, &d_data, &d_q};
        dim3   grid_dim(m->nb0, m->nb1, m->nb2), block_dim(m->b, m->b, m->b);
        size_t cache_size = Block;
        for (auto i = 0; i < m->ndim - 1; i++) cache_size *= Block;

#if __cplusplus >= 201703L
        cudaLaunchKernel(
            (void*)zip::Lorenzo_nd1l<ndim>::Call<Block, Data, Quant>,  //
            grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
#elif __cplusplus >= 201402L
        if (ndim == 1)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::c_lorenzo_1d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
        if (ndim == 2)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::c_lorenzo_2d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
        else if (ndim == 3)  // compile time?
            cudaLaunchKernel(
                (void*)cusz::predictor_quantizer::c_lorenzo_3d1l<Block, Data, Quant>,  //
                grid_dim, block_dim, args, cache_size * sizeof(Data), nullptr);
#endif
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto len    = m->len;
    auto veri_q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xq[i] != veri_q[i]) count++;
    if (count != 0)
        cerr << log_err << "percentage of not being equal: " << count / (1.0 * len) << "\n";
    else
        cout << log_info << "Decoded correctly." << endl;

    if (count != 0) {
        auto n_chunk = (len - 1) / ctx->h_chunksize + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed   = false;
            auto prev_point_printed = false;
            for (auto i = 0; i < ctx->h_chunksize; i++) {
                auto idx = i + c * ctx->h_chunksize;
                if (idx >= len) break;
                if (xq[idx] != xq[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * ctx->h_chunksize << "\tend@ " << (c + 1) * ctx->h_chunksize - 1
                             << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * ctx->h_chunksize) {  // not first point
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

    cudaFree(d_q), cudaFree(d_data);
    delete[] veri_q, delete[] data;
    // end of if count
}
