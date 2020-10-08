/**
 * @file timer.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.2
 * @date 2020-10-08
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#ifndef TIMER_CUH
#define TIMER_CUH

#include <stdio.h>
#include <string>

using std::string;

using stream_t = cudaStream_t;

typedef struct DeviceEvent {
    std::string event_name;
    cudaEvent_t start, end;
    float       ms;
    stream_t*   stream_ptr = nullptr;

    /**** version 1
        DeviceEvent(string s)
        {
            event_name = std::move(s);
            cudaEventCreate(&start);
            cudaEventCreate(&end);
        }

        ~DeviceEvent()
        {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        inline void Start() { cudaEventRecord(start); }  // non-streaming

        inline void End() { cudaEventRecord(end); }  // non-streaming

        double TimeElapsed()
        {
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, end);
            return (double)ms;
        }
     */
    DeviceEvent(string s)
    {
        event_name = std::move(s);
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);  // non-streaming
    }

    ~DeviceEvent()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    double End()
    {
        cudaEventRecord(end);  // non-streaming
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        return (double)ms;
    }

} DeviceEvent;

#endif