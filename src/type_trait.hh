#ifndef TYPE_TRAIT_HH
#define TYPE_TRAIT_HH

/**
 * @file type_trait.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "metadata.hh"

template <int ndim>
struct MetadataTrait;

template <>
struct MetadataTrait<1> {
    typedef struct Metadata<256> metadata_t;
    static const int             Block = 256;
};

// template <>
// struct MetadataTrait<11> {
//     typedef struct Metadata<64> metadata_t;
//     static const int            Block = 64;
// };

template <>
struct MetadataTrait<2> {
    typedef struct Metadata<16> metadata_t;
    static const int            Block = 16;
};

template <>
struct MetadataTrait<3> {
    typedef struct Metadata<8> metadata_t;
    static const int           Block = 8;
};

// clang-format off
template <int QuantByte> struct QuantTrait;
template <> struct QuantTrait<1> {typedef unsigned char Quant;};
template <> struct QuantTrait<2> {typedef unsigned short Quant;};

template <int SymbolByte> struct CodebookTrait;
template <> struct CodebookTrait<4> {typedef unsigned long Huff;};
template <> struct CodebookTrait<8> {typedef unsigned long long Huff;};

// TODO there is an issue about difference betwen ull and uint64_t
template <int HuffByte> struct HuffTrait;
template <> struct HuffTrait<4> {typedef unsigned long Huff;};
template <> struct HuffTrait<8> {typedef unsigned long long Huff;};
// clang-format on

typedef MetadataTrait<1>::metadata_t m1_t;
typedef MetadataTrait<2>::metadata_t m2_t;
typedef MetadataTrait<3>::metadata_t m3_t;

typedef QuantTrait<1>::Quant q1_t;
typedef QuantTrait<2>::Quant q2_t;

// alias style 1
typedef int8_t  int8__t;
typedef uint8_t uint8__t;
typedef float   fp32;
typedef double  fp64;
// alias style 2
typedef int8_t             I1;
typedef int16_t            I2;
typedef int32_t            I4;
typedef long long          I8;
typedef int64_t            I8_2;
typedef uint8_t            UI1;
typedef uint16_t           UI2;
typedef uint32_t           UI4;
typedef unsigned long long UI8;
typedef uint64_t           UI8_2;
typedef float              F4;
typedef double             F8;

#endif
