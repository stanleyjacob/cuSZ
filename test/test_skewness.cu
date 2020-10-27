#include <cstdio>
#include <iostream>
#include "sparse_gen.hh"
// #include "io.hh"
#include "cuda_mem.cuh"
#include "sieve_outlier.cuh"

using namespace std;

int main(int argc, char** argv)
{
    // cudaDeviceReset();

    int    skewness = 0;
    int    m = 0, n = 0;
    size_t length  = 0;
    double spratio = 0.01;

    printf("2D: ./spgen [m] [n] <sp rario> <skewness>\n");
    if (argc < 4) {
        cerr << "Argument too few!" << endl;
        exit(1);
    }
    else if (argc == 5) {
        m        = atoi(argv[1]);
        n        = atoi(argv[2]);
        length   = m * n;  // atoi(argv[1]) * atoi(argv[2]);
        spratio  = atof(argv[3]);
        skewness = atoi(argv[4]);
    }
    cout << "data length:\t" << length << endl;
    printf("(m, n):\t\t(%d, %d)\n", m, n);
    cout << "sparsity:\t" << spratio << endl;
    printf(
        "skewness:\t%d (%dx change in each dimension), m=%d->%d, n=%d->%d\n",  //
        skewness, (1 << skewness), m, m << skewness, n, n >> skewness);

    m <<= skewness;
    n >>= skewness;

    auto spm = bernoulli_gen<float>(length, spratio);

    // for (auto i = 0; i < length; i++) {
    //     if (spm[i] != 0) cout << spm[i] << endl;
    // }
    /*
        float* d_spm;
        cudaError_t a = cudaMalloc((void**)&d_spm, sizeof(float) * length);
        cudaError_t b = cudaMemcpy(d_spm, spm, sizeof(float) * length, cudaMemcpyHostToDevice);
        assert(cudaSuccess == a);
        assert(cudaSuccess == b);
    */
    auto d_spm = mem::CreateDeviceSpaceAndMemcpyFromHost(spm, length);

    // printspm<<<1, 1>>>(d_spm, length);

    int*   csrRowPtrC = nullptr;
    int*   csrColIndC = nullptr;  // column major, real index
    float* csrValC    = nullptr;  // outlier values; TODO templat
    int    nnz        = 0;

    // sieve_outlier(  //
    old_sieve(        //
        d_spm,        //
        length,       //
        m,            //
        n,            //
        nnz,          //
        &csrRowPtrC,  //
        &csrColIndC,  //
        &csrValC);
}
