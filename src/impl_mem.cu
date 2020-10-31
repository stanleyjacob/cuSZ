#include "cuda_mem.cu"
#include "type_trait.hh"

template UI1* mem::CreateCUDASpace<UI1>(size_t, uint8_t);
template UI2* mem::CreateCUDASpace<UI2>(size_t, uint8_t);
template UI4* mem::CreateCUDASpace<UI4>(size_t, uint8_t);
template UI8* mem::CreateCUDASpace<UI8>(size_t, uint8_t);
template I1*  mem::CreateCUDASpace<I1>(size_t, uint8_t);
template I2*  mem::CreateCUDASpace<I2>(size_t, uint8_t);
template I4*  mem::CreateCUDASpace<I4>(size_t, uint8_t);
template I8*  mem::CreateCUDASpace<I8>(size_t, uint8_t);
template F4*  mem::CreateCUDASpace<F4>(size_t, uint8_t);
template F8*  mem::CreateCUDASpace<F8>(size_t, uint8_t);

template UI1* mem::CreateDeviceSpaceAndMemcpyFromHost(UI1*, size_t);
template UI2* mem::CreateDeviceSpaceAndMemcpyFromHost(UI2*, size_t);
template UI4* mem::CreateDeviceSpaceAndMemcpyFromHost(UI4*, size_t);
template UI8* mem::CreateDeviceSpaceAndMemcpyFromHost(UI8*, size_t);
template I1*  mem::CreateDeviceSpaceAndMemcpyFromHost(I1*, size_t);
template I2*  mem::CreateDeviceSpaceAndMemcpyFromHost(I2*, size_t);
template I4*  mem::CreateDeviceSpaceAndMemcpyFromHost(I4*, size_t);
template I8*  mem::CreateDeviceSpaceAndMemcpyFromHost(I8*, size_t);
template F4*  mem::CreateDeviceSpaceAndMemcpyFromHost(F4*, size_t);
template F8*  mem::CreateDeviceSpaceAndMemcpyFromHost(F8*, size_t);

template UI1* mem::CreateHostSpaceAndMemcpyFromDevice(UI1*, size_t);
template UI2* mem::CreateHostSpaceAndMemcpyFromDevice(UI2*, size_t);
template UI4* mem::CreateHostSpaceAndMemcpyFromDevice(UI4*, size_t);
template UI8* mem::CreateHostSpaceAndMemcpyFromDevice(UI8*, size_t);
template I1*  mem::CreateHostSpaceAndMemcpyFromDevice(I1*, size_t);
template I2*  mem::CreateHostSpaceAndMemcpyFromDevice(I2*, size_t);
template I4*  mem::CreateHostSpaceAndMemcpyFromDevice(I4*, size_t);
template I8*  mem::CreateHostSpaceAndMemcpyFromDevice(I8*, size_t);
template F4*  mem::CreateHostSpaceAndMemcpyFromDevice(F4*, size_t);
template F8*  mem::CreateHostSpaceAndMemcpyFromDevice(F8*, size_t);