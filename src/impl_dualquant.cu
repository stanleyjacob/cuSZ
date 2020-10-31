#include "cusz_dualquant.cu"
#include "type_trait.hh"

// clang-format off
// compression prototypes
template __global__ void cusz::predictor_quantizer::c_lorenzo_1d1l<F4, UI1, MetadataTrait<1>::Block>(F4*, UI1*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_1d1l<F4, UI2, MetadataTrait<1>::Block>(F4*, UI2*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_1d1l<F4, UI4, MetadataTrait<1>::Block>(F4*, UI4*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l<F4, UI1, MetadataTrait<2>::Block>(F4*, UI1*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l<F4, UI2, MetadataTrait<2>::Block>(F4*, UI2*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_2d1l<F4, UI4, MetadataTrait<2>::Block>(F4*, UI4*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l<F4, UI1, MetadataTrait<3>::Block>(F4*, UI1*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l<F4, UI2, MetadataTrait<3>::Block>(F4*, UI2*, size_t const*, double const*);
template __global__ void cusz::predictor_quantizer::c_lorenzo_3d1l<F4, UI4, MetadataTrait<3>::Block>(F4*, UI4*, size_t const*, double const*);
// decompression prototypes
template __global__ void cusz::predictor_quantizer::x_lorenzo_1d1l<F4, UI1, MetadataTrait<1>::Block>(F4*, F4*, UI1*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_1d1l<F4, UI2, MetadataTrait<1>::Block>(F4*, F4*, UI2*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_1d1l<F4, UI4, MetadataTrait<1>::Block>(F4*, F4*, UI4*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_2d1l<F4, UI1, MetadataTrait<2>::Block>(F4*, F4*, UI1*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_2d1l<F4, UI2, MetadataTrait<2>::Block>(F4*, F4*, UI2*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_2d1l<F4, UI4, MetadataTrait<2>::Block>(F4*, F4*, UI4*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_3d1l<F4, UI1, MetadataTrait<3>::Block>(F4*, F4*, UI1*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_3d1l<F4, UI2, MetadataTrait<3>::Block>(F4*, F4*, UI2*, size_t const*, double);
template __global__ void cusz::predictor_quantizer::x_lorenzo_3d1l<F4, UI4, MetadataTrait<3>::Block>(F4*, F4*, UI4*, size_t const*, double);

// using virtual padding
// prototype 2D
// deprecated
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<F4, UI1, 16>(F4*, UI1*, size_t const*, double const**);
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<F4, UI2, 16>(F4*, UI2*, size_t const*, double const**);
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_2d1l_virtual_padding<F4, UI4, 16>(F4*, UI4*, size_t const*, double const**);
// // prototype 3D
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<F4, UI1, 8>(F4*, UI1*, size_t const*, double const**);
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<F4, UI2, 8>(F4*, UI2*, size_t const*, double const**);
// template __global__ void
// cusz::predictor_quantizer::c_lorenzo_3d1l_virtual_padding<F4, UI4, 8>(F4*, UI4*, size_t const*, double const**);
