#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

#include "SDRB.hh"
#include "__io.hh"
#include "argparse.hh"
#include "cusz_workflow.cuh"
#include "types.hh"

namespace fm = cuSZ::FineMassive;
using std::string;
using std::vector;

template <typename T, int DS, int tBLK>
T* pre_binning(T* d, size_t* dims_L16) {
    auto d0      = dims_L16[DIM0];
    auto d1      = dims_L16[DIM1];
    auto d2      = dims_L16[DIM2];
    auto d3      = dims_L16[DIM3];
    auto len     = d0 * d1 * d2 * d3;
    auto new_d0  = (dims_L16[DIM0] - 1) / DS + 1;
    auto new_d1  = (dims_L16[DIM1] - 1) / DS + 1;
    auto new_d2  = (dims_L16[DIM2] - 1) / DS + 1;
    auto new_d3  = (dims_L16[DIM3] - 1) / DS + 1;
    auto new_len = new_d0 * new_d1 * new_d2 * new_d3;

    size_t new_dims[] = {new_d0, new_d1, new_d2, new_d3};
    SetDims(dims_L16, new_dims);

    auto d_d  = mem::CreateDeviceSpaceAndMemcpyFromHost(d, len);
    auto d_ds = mem::CreateCUDASpace<T>(new_len);

    dim3 blockDim(tBLK, tBLK);
    dim3 gridDim((new_d0 - 1) / tBLK + 1, (new_d1 - 1) / tBLK + 1);
    Prototype::binning2d<T, DS, tBLK><<<gridDim, blockDim>>>(d_d, d_ds, d0, d1, new_d0, new_d1);
    cudaDeviceSynchronize();

    cudaFree(d_d);
    return d_ds;
}

int main(int argc, char** argv) {
    auto ap = new argpack(argc, argv);

    auto dims_L16 = ap->use_demo ? InitializeDemoDims(ap->demo_dataset, ap->dict_size)  //
                                 : InitializeDims(ap->dict_size, ap->n_dim, ap->d0, ap->d1, ap->d2, ap->d3);

    cout << log_info << "datum file:\t" << ap->fname << endl;
    cout << log_info << "datum size:\t" << dims_L16[LEN] * sizeof(float) << endl;

    auto eb_config = new config_t(ap->dict_size, ap->mantissa, ap->exponent);
    if (ap->mode == "r2r") {  // TODO change to faster getting range
        auto valrng = GetDatumValueRange<float>(ap->fname, dims_L16[LEN]);
        eb_config->ChangeToRelativeMode(valrng);
    }
    eb_config->debug();
    auto   ebs_L4      = InitializeErrorBoundFamily(eb_config);
    size_t num_outlier = 0;
    size_t total_bits, total_uInt, huffman_metadata_size;

    cout << log_dbg << "data type:\t" << ap->dtype << endl;
    cout << log_dbg << "using uint" << ap->quant_rep << "_t as quant. rep." << endl;
    cout << log_dbg << "using uint" << ap->huffman_rep << "_t as Huffman codeword rep." << endl;

    if (ap->pre_binning) {
        auto data      = io::ReadBinaryFile<float>(ap->fname, dims_L16[LEN]);
        auto d_binning = pre_binning<float, 2, 32>(data, dims_L16);
        auto binning   = mem::CreateHostSpaceAndMemcpyFromDevice(d_binning, dims_L16[LEN]);
        ap->fname      = ap->fname + "-binning";
        io::WriteBinaryFile(binning, dims_L16[LEN], &ap->fname);

        cudaFree(d_binning);
        delete[] data;
        delete[] binning;
    }

    // TODO change to compress and decompress
    if (ap->to_archive or ap->dry_run) {  // including dry run
        if (ap->dtype == "f32") {
            if (ap->quant_rep == 8) {
                if (ap->huffman_rep == 32)
                    fm::a<float, uint8_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::a<float, uint8_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
            } else if (ap->quant_rep == 16) {
                if (ap->huffman_rep == 32)
                    fm::a<float, uint16_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::a<float, uint16_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
            } else if (ap->quant_rep == 32) {
                if (ap->huffman_rep == 32)
                    fm::a<float, uint32_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::a<float, uint32_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
            }
        }
    }

    if (ap->to_extract) {
        if (ap->dtype == "f32") {
            if (ap->quant_rep == 8) {
                if (ap->huffman_rep == 32)
                    fm::x<float, uint8_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::x<float, uint8_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);

            } else if (ap->quant_rep == 16) {
                if (ap->huffman_rep == 32)
                    fm::x<float, uint16_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::x<float, uint16_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);

            } else if (ap->quant_rep == 32) {
                if (ap->huffman_rep == 32)
                    fm::x<float, uint32_t, uint32_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
                else
                    fm::x<float, uint32_t, uint64_t>(ap->fname, dims_L16, ebs_L4, num_outlier, total_bits, total_uInt, huffman_metadata_size, ap);
            }
        }
    }

    delete[] dims_L16;
    delete[] ebs_L4;
    delete eb_config;
}
