#ifndef HUFFMAN_WORKFLOW
#define HUFFMAN_WORKFLOW

/**
 * @file huffman_workflow.cuh
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding (header).
 * @version 0.1
 * @date 2020-09-21
 * Created on 2020-04-24
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
//#include <sys/stat.h>

#include <cstdint>
#include <string>
#include <tuple>

using std::string;

// const int GB_unit = 1073741824;  // 1024^3

const int tBLK_ENCODE    = 256;
const int tBLK_DEFLATE   = 128;
const int tBLK_CANONICAL = 128;

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
// inline bool exists_test2(const std::string& name) {
//    return (access(name.c_str(), F_OK) != -1);
//}

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

namespace lossless {

namespace interface {

template <typename UInt, typename Huff, typename DATA = float>
tuple3ul HuffmanEncode(string& f_in, UInt* d_in, size_t len, int chunk_size, int dict_size = 1024);

template <typename UInt, typename Huff, typename DATA = float>
UInt* HuffmanDecode(std::string& f_in_base, size_t len, int chunk_size, int total_uInts, int dict_size = 1024);

}  // namespace interface

namespace wrap {

template <typename UInt>
void GetFrequency(UInt*, size_t, unsigned int*, int);

template <typename Huff>
void Deflate(Huff*, size_t, int, int, size_t*);

}  // namespace wrap

namespace util {
template <typename Huff>
void PrintChunkHuffmanCoding(
    size_t* dH_bit_meta,
    size_t* dH_uInt_meta,
    size_t  len,
    int     chunk_size,
    size_t  total_bits,
    size_t  total_uInts);
}

}  // namespace lossless

#endif
