/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

/**
 * @brief Get the Device Property object
 * modified from `cuda-samples/Samples/deviceQuery/deviceQuery.cpp`
 */
void GetDeviceProperty()
{
    int         num_dev  = 0;
    cudaError_t error_id = cudaGetDeviceCount(&num_dev);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }
    if (num_dev == 0) { printf("NO CUDA device detected.\n"); }
    int dev, driver_ver = 0, runtime_ver = 0;

    for (dev = 0; dev < num_dev; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, dev);
        printf("device #%d, %s: \n", dev, dev_prop.name);

        cudaDriverGetVersion(&driver_ver);
        cudaRuntimeGetVersion(&runtime_ver);
        printf(
            "  driver/runtime\t%d.%d/%d.%d\n", driver_ver / 1000, (driver_ver % 100) / 10, runtime_ver / 1000,
            (runtime_ver % 100) / 10);
        printf("  compute capability:\t%d.%d\n", dev_prop.major, dev_prop.minor);
        printf("  global memory:\t%.0f MiB\n", static_cast<float>(dev_prop.totalGlobalMem / 1048576.0f));
        printf("  constant memory:\t%zu bytes\n", dev_prop.totalConstMem);
        printf("  shared mem per block:\t%zu bytes\n", dev_prop.sharedMemPerBlock);
        printf("  shared mem per SM:\t%zu bytes\n", dev_prop.sharedMemPerMultiprocessor);
        printf("  registers per block:\t%d\n", dev_prop.regsPerBlock);
    }
    printf("\n");
}