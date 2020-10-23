#include "cusz_dualquant.cu"
#include "type_trait.hh"

// compression
// prototype 1D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint8__t, 32>(fp32*, uint8__t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint16_t, 32>(fp32*, uint16_t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint32_t, 32>(fp32*, uint32_t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint8__t, 64>(fp32*, uint8__t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint16_t, 64>(fp32*, uint16_t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l<fp32, uint32_t, 64>(fp32*, uint32_t*, size_t const*, fp64 const*);
// prototype 2D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<fp32, uint8__t, 16>(fp32*, uint8__t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<fp32, uint16_t, 16>(fp32*, uint16_t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l<fp32, uint32_t, 16>(fp32*, uint32_t*, size_t const*, fp64 const*);
// prototype 3D
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<fp32, uint8__t, 8>(fp32*, uint8__t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<fp32, uint16_t, 8>(fp32*, uint16_t*, size_t const*, fp64 const*);
template __global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l<fp32, uint32_t, 8>(fp32*, uint32_t*, size_t const*, fp64 const*);

// using virtual padding
// prototype 2D
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<fp32, uint8__t, 16>(
    fp32*,
    uint8__t*,
    size_t const*,
    fp64 const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<fp32, uint16_t, 16>(
    fp32*,
    uint16_t*,
    size_t const*,
    fp64 const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<fp32, uint32_t, 16>(
    fp32*,
    uint32_t*,
    size_t const*,
    fp64 const*);
// prototype 3D
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<fp32, uint8__t, 8>(
    fp32*,
    uint8__t*,
    size_t const*,
    fp64 const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<fp32, uint16_t, 8>(
    fp32*,
    uint16_t*,
    size_t const*,
    fp64 const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<fp32, uint32_t, 8>(
    fp32*,
    uint32_t*,
    size_t const*,
    fp64 const*);

// decompression
// prototype 1D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint8__t, 32>(fp32*, fp32*, uint8__t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint16_t, 32>(fp32*, fp32*, uint16_t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint32_t, 32>(fp32*, fp32*, uint32_t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint8__t, 64>(fp32*, fp32*, uint8__t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint16_t, 64>(fp32*, fp32*, uint16_t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l<fp32, uint32_t, 64>(fp32*, fp32*, uint32_t*, size_t const*, fp64);
// prototype 2D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<fp32, uint8__t, 16>(fp32*, fp32*, uint8__t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<fp32, uint16_t, 16>(fp32*, fp32*, uint16_t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l<fp32, uint32_t, 16>(fp32*, fp32*, uint32_t*, size_t const*, fp64);
// prototype 3D
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<fp32, uint8__t, 8>(fp32*, fp32*, uint8__t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<fp32, uint16_t, 8>(fp32*, fp32*, uint16_t*, size_t const*, fp64);
template __global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l<fp32, uint32_t, 8>(fp32*, fp32*, uint32_t*, size_t const*, fp64);
