

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
#include <cstdlib>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double>                               duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> hires_clock_t;

enum class EventDefinition { kSTART, kEND, kDURATION };

class cuszEvent {
    hires_clock_t* const start = nullptr;
    hires_clock_t* const end   = nullptr;

   public:
    void Record(EventDefinition e)
    {
        if (e == EventDefinition::kSTART)
            *start = hires::now();
        else if (e == EventDefinition::kEND) {
            *end = hires::now();
        }
    }

    double Export()
    {
        if (not start) cerr << "timer not started" << endl;
        if (not end) cerr << "timer not ended" << endl;
        return static_cast<duration_t>(*end - *start).count();
    }
};

#endif  // TIMER_HH
