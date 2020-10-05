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
    typedef struct Metadata<32> metadata_t;
};

template <>
struct MetadataTrait<11> {
    typedef struct Metadata<64> metadata_t;
};

template <>
struct MetadataTrait<21> {
    typedef struct Metadata<64> metadata_t;
};

template <>
struct MetadataTrait<31> {
    typedef struct Metadata<128> metadata_t;
};

template <>
struct MetadataTrait<2> {
    typedef struct Metadata<16> metadata_t;
};

template <>
struct MetadataTrait<3> {
    typedef struct Metadata<8> metadata_t;
};

template <int QuantByte>
struct QuantTrait;

template <>
struct QuantTrait<1> {
    typedef unsigned char Quant;
};

template <>
struct QuantTrait<2> {
    typedef unsigned short Quant;
};

template <int HuffByte>
struct HuffTrait;

template <>
struct HuffTrait<4> {
    typedef unsigned long Huff;
};

template <>
struct HuffTrait<8> {
    // TODO there is an issue about difference betwen ull and uint64_t
    typedef unsigned long long Huff;
};
