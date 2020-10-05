

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

class Event {
    string        event_name;
    hires_clock_t start, end;
    double        time_elapsed;

   public:
    Event(string s) { event_name = std::move(s); }

    void Start() { start = hires::now(); }
    void End() { end = hires::now(); }
    // double TimeElapsed(size_t len, bool single_unit = true);

    double TimeElapsed(size_t bytelen, bool single_unit = true)
    {
        // if (not start) cerr << "not started" << endl;
        // if (not end) cerr << "not ended" << endl;
        if (start >= end) cerr << "start >= end" << endl;
        time_elapsed = static_cast<duration_t>(end - start).count();

        if (single_unit) {
            printf(
                "time elapsed (us):\t%14.4lf\t|\tthroughput (GB/s):\t%10.3lf\t|\t%s\n",  //
                time_elapsed * 1e6,                                                      //
                bytelen / time_elapsed / (1024 * 1024 * 1024),                           //
                event_name.c_str());
        }
        else {
            if (time_elapsed * 1e3 > 1) {  // > 1 ms
                printf(
                    "time elapsed (ms):\t%14.4lf\t|\tthroughput (GB/s):\t%10.3lf\t|\t%s\n",  //
                    time_elapsed * 1e3,                                                      //
                    bytelen / time_elapsed / (1024 * 1024 * 1024),                           //
                    event_name.c_str());
            }
            else {
                printf(
                    "time elapsed (us):\t%14.4lf\t|\tthroughput (GB/s):\t%10.3lf\t|\t%s\n",  //
                    time_elapsed * 1e6,                                                      //
                    bytelen / time_elapsed / (1024 * 1024 * 1024),                           //
                    event_name.c_str());
            }
        }

        return time_elapsed;
    }
};

// extern std::vector<Event*> cusz_events;

#endif  // TIMER_HH
