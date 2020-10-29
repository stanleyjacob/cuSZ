//
// Created by jtian on 4/24/20.
//

#include <cuda_runtime.h>

#include <sys/stat.h>
#include <bitset>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>

#include <stdio.h>

#include "../cuda_error_handling.cuh"
#include "../cuda_mem.cuh"
#include "../cuda_utils.cuh"
#include "../huffman_codec.cuh"
#include "../huffman_workflow.cuh"
#include "../timer.hh"
#include "../types.hh"

#include "rs_merge.cuh"
#include "utils.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

uint32_t dbg_bi = 2;

using ulli = unsigned long long int;

template <typename Input, typename Huff, typename Dict, int Magnitude, int ReductionFactor, int ShuffleFactor>
std::tuple<uint32_t, Huff*, uint32_t*> Encoder(
    Input*      q,
    uint32_t    len,
    Dict*       cb,
    uint32_t    cb_len,
    uint32_t    dummy_nchunk_,
    std::string encoder,
    bool        use_dummylen = false)
{
    auto chunksize = 1 << Magnitude;
    if (use_dummylen) len = dummy_nchunk_ * chunksize;

    auto chunk_size = 1 << Magnitude;
    auto blockDim   = 1 << ShuffleFactor;
    auto gridDim    = len / chunk_size;

    auto d_q     = mem::CreateDeviceSpaceAndMemcpyFromHost(q, len);
    auto d_cb    = mem::CreateDeviceSpaceAndMemcpyFromHost(cb, cb_len);
    auto d_h     = mem::CreateCUDASpace<Huff>(len);
    auto d_hmeta = mem::CreateCUDASpace<uint32_t>(gridDim);  // chunkwise metadata

    auto buff_bytes = chunksize * (sizeof(Huff) + sizeof(int));
    // auto buff_bytes = (chunksize / 2 + chunksize / 4) * (sizeof(Huff) + sizeof(int));
    // share memory usage: 1.5 * chunksize * 4 = 6K
    // data size: sizeof(uint16_t) * chunksize: 2 * 1024
    // thread number : chunksize >> 3, 128, at max 2* 1024 / 128 = 16 threadblocks on 1 SM

    printf("len:       %d\n", (int)len);
    printf("chunksize: %d\n", (int)chunksize);
    printf("blockDim:  %d\n", (int)blockDim);
    printf("gridDim:   %d\n", (int)gridDim);
    printf("shmem bytes\t%d\t%d blocks EXPECTED on 1 SM (shmem)\n", (int)buff_bytes, int(96 * 1024 / buff_bytes));
    printf("%d should be blocks on 1 SM\n", int(1024 * 2 / blockDim));

    ReduceShuffle_fixcodebook<Input, Huff, Dict, Magnitude, ReductionFactor, ShuffleFactor>  //
        <<<gridDim, blockDim, buff_bytes>>>(d_q, len, d_cb, d_h, cb_len, d_hmeta, nullptr, dbg_bi);
    cudaDeviceSynchronize();

    auto h     = mem::CreateHostSpaceAndMemcpyFromDevice(d_h, len);
    auto hmeta = mem::CreateHostSpaceAndMemcpyFromDevice(d_hmeta, gridDim);
    cudaFree(d_q), cudaFree(d_cb), cudaFree(d_h), cudaFree(d_hmeta);

    printf("New Encoder: %s gracefully finished\n", encoder.c_str());
    return {(uint32_t)gridDim, h, hmeta};
}

template <int Magnitude, int ReductionFactor>
int submain()
{
    using Input = uint16_t;
    using Huff  = uint32_t;
    using Dict  = uint32_t;

    string   f_indata, f_cb;
    uint32_t dummy_nchunk = 0;
    uint32_t cb_len, len;

    f_indata = string("data/baryon_density.dat.b16");
    f_cb     = string("data/baryon_density.dat.dict");
    len      = 512 * 512 * 512;
    cb_len   = 1024;

    auto q  = io::ReadBinaryFile<Input>(f_indata, len);
    auto cb = io::ReadBinaryFile<Dict>(f_cb, cb_len);

    double avg_bw, entropy;

    std::tie(avg_bw, entropy) = GetEntropyAndAvgBitwidth<Input, Dict>(q, len, cb, cb_len);

    dbg_bi = 0;  // debug only

    const auto ShuffleFactor = Magnitude - ReductionFactor;
    cout << log_info << "Magnitude=" << Magnitude << "\tReductionFactor=" << ReductionFactor
         << "\tShuffleFactor=" << ShuffleFactor << endl;
    Encoder<Input, Huff, Dict, Magnitude, ReductionFactor, ShuffleFactor>                 // reduce shuffle
        (q, len, cb, cb_len, dummy_nchunk, string("reduce-shuffle"), dummy_nchunk != 0);  //

    delete[] q, cb;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "./<prog> <chunk magnitude> <reduction times>" << endl;
        exit(1);
    }
    int mag = atoi(argv[1]);
    int red = atoi(argv[2]);
    if (mag == 12) {
        if (red == 4)
            submain<12, 4>();
        else if (red == 3)
            submain<12, 3>();
        else if (red == 2)
            submain<12, 2>();
        else if (red == 1)
            submain<12, 1>();
    }
    else if (mag == 11) {
        if (red == 4)
            submain<11, 4>();
        else if (red == 3)
            submain<11, 3>();
        else if (red == 2)
            submain<11, 2>();
        else if (red == 1)
            submain<11, 1>();
    }
    else if (mag == 10) {
        if (red == 4)
            submain<10, 4>();
        else if (red == 3)
            submain<10, 3>();
        else if (red == 2)
            submain<10, 2>();
        else if (red == 1)
            submain<10, 1>();
    }

    return 0;
}
