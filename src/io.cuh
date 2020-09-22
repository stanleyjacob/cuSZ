//
// Created by JianNan Tian on 2020-08-23.
//

#ifndef IO_CUH
#define IO_CUH

#include <fstream>
#include <iostream>
#include "cuda_error_handling.cuh"

namespace io {
template <typename T>
T* ReadBinaryFileToPinnedMemory(const std::string& __name, size_t __len)
{
    std::ifstream ifs(__name.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << __name << std::endl;
        exit(1);
        // return;
    }
    T* __a;
    cudaMallocHost((void**)&__a, sizeof(T));
    ifs.read(reinterpret_cast<char*>(__a), std::streamsize(__len * sizeof(T)));
    ifs.close();
    return __a;
}
}  // namespace io

#endif