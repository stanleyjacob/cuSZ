#include "cusz_workflow2.cu"

#if __cplusplus >= 201402L  // used for editor

// template <ndim, Data, QuantByte, HuffByte>
template void cusz::interface::Compress2<1, F4, 1, 4>(ctx_t*, m1_t*);
template void cusz::interface::Compress2<1, F4, 2, 4>(ctx_t*, m1_t*);
template void cusz::interface::Compress2<2, F4, 1, 4>(ctx_t*, m2_t*);
template void cusz::interface::Compress2<2, F4, 2, 4>(ctx_t*, m2_t*);
template void cusz::interface::Compress2<3, F4, 1, 4>(ctx_t*, m3_t*);
template void cusz::interface::Compress2<3, F4, 2, 4>(ctx_t*, m3_t*);

// template <int ndim, typename Data, int QuantByte, int HuffByte>
template void cusz::interface::Decompress2<1, F4, 1, 4>(ctx_t*, m1_t*);
template void cusz::interface::Decompress2<1, F4, 2, 4>(ctx_t*, m1_t*);
template void cusz::interface::Decompress2<2, F4, 1, 4>(ctx_t*, m2_t*);
template void cusz::interface::Decompress2<2, F4, 2, 4>(ctx_t*, m2_t*);
template void cusz::interface::Decompress2<3, F4, 1, 4>(ctx_t*, m3_t*);
template void cusz::interface::Decompress2<3, F4, 2, 4>(ctx_t*, m3_t*);

// template <int ndim, typename Data, int QuantByte>
template void cusz::impl::VerifyHuffman<1, F4, 1>(ctx_t*, m1_t*, q1_t*);
template void cusz::impl::VerifyHuffman<1, F4, 2>(ctx_t*, m1_t*, q2_t*);
template void cusz::impl::VerifyHuffman<2, F4, 1>(ctx_t*, m2_t*, q1_t*);
template void cusz::impl::VerifyHuffman<2, F4, 2>(ctx_t*, m2_t*, q2_t*);
template void cusz::impl::VerifyHuffman<3, F4, 1>(ctx_t*, m3_t*, q1_t*);
template void cusz::impl::VerifyHuffman<3, F4, 2>(ctx_t*, m3_t*, q2_t*);

#endif