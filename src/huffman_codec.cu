/**
 * @file huffman_codec.cu
 * @author Jiannan Tian
 * @brief Wrapper of Huffman codec.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-02
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <limits>
#include "huffman_codec.cuh"

using uint8__t = uint8_t;

template <typename Q, typename H>
__global__ void EncodeFixedLen(Q* data, H* hcoded, size_t data_len, H* codebook)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= data_len) return;
    hcoded[gid] = codebook[data[gid]];  // try to exploit cache?
    __syncthreads();
}

template <typename Huff, typename TempSizeT>
__global__ void Deflate(
    Huff*      hcoded,  //
    size_t     len,
    TempSizeT* densely_meta,
    int        PART_SIZE)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (len - 1) / PART_SIZE + 1) return;
    uint8_t bitwidth;
    size_t  densely_coded_lsb_pos = sizeof(Huff) * 8, total_bitwidth = 0;
    size_t  ending = (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
    //    if ((gid + 1) * PART_SIZE > len) printf("\n\ngid %lu\tending %lu\n\n", gid, ending);
    Huff  msb_bw_word_lsb, _1, _2;
    Huff* current = hcoded + gid * PART_SIZE;
    for (size_t i = 0; i < ending; i++) {
        msb_bw_word_lsb = hcoded[gid * PART_SIZE + i];
        bitwidth        = *((uint8_t*)&msb_bw_word_lsb + (sizeof(Huff) - 1));

        *((uint8_t*)&msb_bw_word_lsb + sizeof(Huff) - 1) = 0x0;
        if (densely_coded_lsb_pos == sizeof(Huff) * 8) *current = 0x0;  // a new unit of data type
        if (bitwidth <= densely_coded_lsb_pos) {
            densely_coded_lsb_pos -= bitwidth;
            *current |= msb_bw_word_lsb << densely_coded_lsb_pos;
            if (densely_coded_lsb_pos == 0) {
                densely_coded_lsb_pos = sizeof(Huff) * 8;
                ++current;
            }
        }
        else {
            // example: we have 5-bit code 11111 but 3 bits left for (*current)
            // we put first 3 bits of 11111 to the last 3 bits of (*current)
            // and put last 2 bits from MSB of (*(++current))
            // the comment continues with the example
            _1 = msb_bw_word_lsb >> (bitwidth - densely_coded_lsb_pos);
            _2 = msb_bw_word_lsb << (sizeof(Huff) * 8 - (bitwidth - densely_coded_lsb_pos));
            *current |= _1;
            *(++current) = 0x0;
            *current |= _2;
            densely_coded_lsb_pos = sizeof(Huff) * 8 - (bitwidth - densely_coded_lsb_pos);
        }
        total_bitwidth += bitwidth;
    }
    *(densely_meta + gid) = total_bitwidth;
}

template <typename Huff, typename Quant>
__device__ void InflateChunkwise(Huff* in_huff, Quant* out_quant, size_t total_bw, uint8_t* singleton)
{
    uint8_t next_bit;
    size_t  idx_bit;
    size_t  idx_byte   = 0;
    size_t  idx_bcoded = 0;
    auto    first      = reinterpret_cast<Huff*>(singleton);
    auto    entry      = first + sizeof(Huff) * 8;
    auto    keys       = reinterpret_cast<Quant*>(singleton + sizeof(Huff) * (2 * sizeof(Huff) * 8));
    Huff    v          = (in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1)) & 0x1;  // get the first bit
    size_t  l          = 1;
    size_t  i          = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / (sizeof(Huff) * 8);
            idx_bit  = i % (sizeof(Huff) * 8);
            next_bit = ((in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            ++l;
        }
        out_quant[idx_bcoded++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / (sizeof(Huff) * 8);
            idx_bit  = i % (sizeof(Huff) * 8);
            next_bit = ((in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

template <typename Q, typename H>
__global__ void Decode(
    H*       densely,     //
    size_t*  dH_meta,     //
    Q*       bcode,       //
    size_t   len,         //
    int      chunk_size,  //
    int      n_chunk,
    uint8_t* singleton,
    size_t   singleton_size)
{
    extern __shared__ uint8_t _s_singleton[];
    if (threadIdx.x == 0) memcpy(_s_singleton, singleton, singleton_size);
    __syncthreads();

    auto dH_bit_meta   = dH_meta;
    auto dH_uInt_entry = dH_meta + n_chunk;

    size_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (chunk_id == 0) printf("n_chunk: %lu\n", n_chunk);
    if (chunk_id >= n_chunk) return;

    InflateChunkwise(                       //
        densely + dH_uInt_entry[chunk_id],  //
        bcode + chunk_size * chunk_id,      //
        dH_bit_meta[chunk_id],              //
        _s_singleton);
    __syncthreads();
};

using ulli = unsigned long long int;

template __global__ void EncodeFixedLen<uint8__t, uint32_t>(uint8__t*, uint32_t*, size_t, uint32_t*);
template __global__ void EncodeFixedLen<uint8__t, uint64_t>(uint8__t*, uint64_t*, size_t, uint64_t*);
template __global__ void EncodeFixedLen<uint16_t, uint32_t>(uint16_t*, uint32_t*, size_t, uint32_t*);
template __global__ void EncodeFixedLen<uint16_t, uint64_t>(uint16_t*, uint64_t*, size_t, uint64_t*);
template __global__ void EncodeFixedLen<uint8__t, ulli>(uint8__t*, ulli*, size_t, ulli*);
template __global__ void EncodeFixedLen<uint16_t, ulli>(uint16_t*, ulli*, size_t, ulli*);
// template __global__ void EncodeFixedLen<uint32_t, uint32_t>(uint32_t*, uint32_t*, size_t, uint32_t*);
// template __global__ void EncodeFixedLen<uint32_t, uint64_t>(uint32_t*, uint64_t*, size_t, uint64_t*);

template __global__ void Deflate<uint32_t, size_t>(uint32_t* hcoded, size_t len, size_t* densely_meta, int PART_SIZE);
template __global__ void Deflate<uint64_t, size_t>(uint64_t* hcoded, size_t len, size_t* densely_meta, int PART_SIZE);
template __global__ void Deflate<ulli, size_t>(ulli* hcoded, size_t len, size_t* densely_meta, int PART_SIZE);

template __global__ void Deflate<uint32_t, uint32_t>(uint32_t*, size_t, uint32_t*, int);
template __global__ void Deflate<uint64_t, uint32_t>(uint64_t*, size_t, uint32_t*, int);
template __global__ void Deflate<ulli, uint32_t>(ulli*, size_t, uint32_t*, int);

// H for Huffman, uint{32,64}_t
// T for quant code, uint{8,16,32}_t
template __device__ void InflateChunkwise<uint32_t, uint8__t>(uint32_t*, uint8__t*, size_t, uint8__t*);
template __device__ void InflateChunkwise<uint32_t, uint16_t>(uint32_t*, uint16_t*, size_t, uint8__t*);
template __device__ void InflateChunkwise<uint64_t, uint8__t>(uint64_t*, uint8__t*, size_t, uint8__t*);
template __device__ void InflateChunkwise<uint64_t, uint16_t>(uint64_t*, uint16_t*, size_t, uint8__t*);
template __device__ void InflateChunkwise<ulli, uint8__t>(ulli*, uint8__t*, size_t, uint8__t*);
template __device__ void InflateChunkwise<ulli, uint16_t>(ulli*, uint16_t*, size_t, uint8__t*);
// template __device__ void InflateChunkwise<uint32_t, uint32_t>(uint32_t*, uint32_t*, size_t, uint8__t*);
// template __device__ void InflateChunkwise<uint64_t, uint32_t>(uint64_t*, uint32_t*, size_t, uint8__t*);

template __global__ void Decode<uint8__t, uint32_t>(uint32_t*, size_t*, uint8__t*, size_t, int, int, uint8__t*, size_t);
template __global__ void Decode<uint8__t, uint64_t>(uint64_t*, size_t*, uint8__t*, size_t, int, int, uint8__t*, size_t);
template __global__ void Decode<uint16_t, uint32_t>(uint32_t*, size_t*, uint16_t*, size_t, int, int, uint8__t*, size_t);
template __global__ void Decode<uint16_t, uint64_t>(uint64_t*, size_t*, uint16_t*, size_t, int, int, uint8__t*, size_t);
template __global__ void Decode<uint8__t, ulli>(ulli*, size_t*, uint8__t*, size_t, int, int, uint8__t*, size_t);
template __global__ void Decode<uint16_t, ulli>(ulli*, size_t*, uint16_t*, size_t, int, int, uint8__t*, size_t);
// template __global__ void Decode<uint32_t, uint32_t>(uint32_t*, size_t*, uint32_t*, size_t, int, int, uint8__t*,
// size_t); template __global__ void Decode<uint32_t, uint64_t>(uint64_t*, size_t*, uint32_t*, size_t, int, int,
// uint8__t*, size_t);
