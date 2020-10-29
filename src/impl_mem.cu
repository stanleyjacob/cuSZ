#include "cuda_mem.cu"
#include "type_trait.hh"

template uint8__t* mem::CreateCUDASpace<uint8__t>(size_t l, uint8_t i);
template uint16_t* mem::CreateCUDASpace<uint16_t>(size_t l, uint8_t i);
template uint32_t* mem::CreateCUDASpace<uint32_t>(size_t l, uint8_t i);
template uint64_t* mem::CreateCUDASpace<uint64_t>(size_t l, uint8_t i);
template int8__t*  mem::CreateCUDASpace<int8__t>(size_t l, uint8_t i);
template int16_t*  mem::CreateCUDASpace<int16_t>(size_t l, uint8_t i);
template int32_t*  mem::CreateCUDASpace<int32_t>(size_t l, uint8_t i);
template int64_t*  mem::CreateCUDASpace<int64_t>(size_t l, uint8_t i);
template fp32*     mem::CreateCUDASpace<fp32>(size_t l, uint8_t i);
template fp64*     mem::CreateCUDASpace<fp64>(size_t l, uint8_t i);

template uint8__t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint8__t* var, size_t l);
template uint16_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint16_t* var, size_t l);
template uint32_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint32_t* var, size_t l);
template uint64_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint64_t* var, size_t l);
template int8__t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int8__t* var, size_t l);
template int16_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int16_t* var, size_t l);
template int32_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int32_t* var, size_t l);
template int64_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int64_t* var, size_t l);
template fp32*     mem::CreateDeviceSpaceAndMemcpyFromHost(fp32* var, size_t l);
template fp64*     mem::CreateDeviceSpaceAndMemcpyFromHost(fp64* var, size_t l);

template uint8__t* mem::CreateHostSpaceAndMemcpyFromDevice(uint8__t* d_var, size_t l);
template uint16_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint16_t* d_var, size_t l);
template uint32_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint32_t* d_var, size_t l);
template uint64_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint64_t* d_var, size_t l);
template int8__t*  mem::CreateHostSpaceAndMemcpyFromDevice(int8__t* d_var, size_t l);
template int16_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int16_t* d_var, size_t l);
template int32_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int32_t* d_var, size_t l);
template int64_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int64_t* d_var, size_t l);
template fp32*     mem::CreateHostSpaceAndMemcpyFromDevice(fp32* d_var, size_t l);
template fp64*     mem::CreateHostSpaceAndMemcpyFromDevice(fp64* d_var, size_t l);