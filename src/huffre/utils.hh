/**
 * @file utils.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-04-24
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef UTILS_HH
#define UTILS_HH

#include <omp.h>
#include <cstdint>
#include <iostream>
#include <tuple>

using std::cout;
using std::endl;

#include "../format.hh"
#include "../huffman_workflow.cuh"

template <typename UInt>
int GetBitwidthCPU(UInt var)
{
    return static_cast<int>(*(reinterpret_cast<uint8_t*>(&var) + sizeof(UInt) - 1));
}

template <typename Input, typename Dict>
std::tuple<double, double> GetEntropyAndAvgBitwidth(Input* in, unsigned int in_len, Dict* cb, unsigned int cb_len)
{
    auto histogram = new unsigned int[cb_len]();
    {
        int i;
#pragma omp parallel for reduction(+ : histogram)
        for (i = 0; i < in_len; i++) histogram[in[i]]++;
    }

    double avg_bw = 0, entropy = 0;
    for (auto i = 0; i < cb_len; i++) {
        auto   bw = GetBitwidthCPU(cb[i]);
        auto   f  = histogram[i] * 1.0;
        double p  = f * 1.0 / in_len;
        if (bw != 255) {
            avg_bw += f * bw;
            entropy += -p * log(p);
        }
    }
    avg_bw /= in_len;

    cout << log_info << "avg. bw:\t" << avg_bw << endl;
    cout << log_info << "entropy:\t" << entropy << endl;

    return {avg_bw, entropy};
}

template <typename Input, typename Dict>
void NaiveFiltering(Input* in, uint32_t in_len, Dict* cb, uint32_t cb_len, uint32_t threshold_bw = 5)  // prepare for
                                                                                                       // extra outliers
{
    // find shortest "special" symbol
    unsigned int shortest = 255;
    unsigned int count    = 0;
    Input        special;
    Dict         special_code;
    for (auto i = 0; i < cb_len; i++) {
        //        cout << i << "\t" << GetBitwidthCPU(cb[i]) << "\t" << bitset<32>(cb[i]) << endl;

        auto sym_len = GetBitwidthCPU(cb[i]);
        if (sym_len < shortest) {
            shortest     = sym_len;
            special      = i;
            special_code = cb[i];
        }
    }
    cout << log_dbg << "shortest codeword in_len\t" << shortest << "\tcodeword\t" << bitset<32>(special_code) << endl;
    cout << log_dbg << "filtering threshold bw\t" << threshold_bw << endl;

    for (auto i = 0; i < in_len; i++) {
        auto sym     = cb[in[i]];
        auto sym_len = GetBitwidthCPU(sym);
        if (sym_len > threshold_bw) {
            in[i] = special;
            count++;
        }
    }
    cout << log_info << count << " are changed, " << (count * 100.0 / in_len) << "%" << endl;
}

#endif