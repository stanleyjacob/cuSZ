

#ifndef TIMER_HH
#define TIMER_HH

/**
 * @file timer.hh
 * @author Jiannan Tian
 * @brief High-resolution timer wrapper from <chrono>
 * @version 0.1
 * @date 2020-09-20
 * Created on 2019-08-26
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::string;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double>  duration_t;
typedef std::chrono::time_point<hires> hires_clock_t;

enum class DefinedEvent { kStart, kEnd, kDuration };

class HostEvent {
    hires_clock_t start, end;
    // double        time_elapsed;

   public:
    string event_name;
    HostEvent(string s)
    {
        event_name = std::move(s);
        start      = hires::now();
    }

    double End()
    {
        end = hires::now();
        if (start >= end) cerr << "start >= end" << endl;
        return static_cast<duration_t>(end - start).count() * 1000;  // in ms
    }
};

// extern std::vector<Event*> cusz_events;

#endif  // TIMER_HH
