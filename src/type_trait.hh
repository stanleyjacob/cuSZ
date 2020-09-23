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
struct MetadataTrait<2> {
    typedef struct Metadata<16> metadata_t;
};

template <>
struct MetadataTrait<3> {
    typedef struct Metadata<8> metadata_t;
};

template <int quant_cap>
struct QuantTrait;

template <>
struct QuantTrait<32> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<64> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<128> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<256> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<512> {
    typedef unsigned short Quant;
};
template <>
struct QuantTrait<1024> {
    typedef unsigned short Quant;
};
template <>
struct QuantTrait<2048> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<4096> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<8192> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<16384> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<32768> {
    typedef unsigned char Quant;
};
template <>
struct QuantTrait<65536> {
    typedef unsigned char Quant;
};