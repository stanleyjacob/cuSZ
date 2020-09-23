/**
 * @file format.hh
 * @author Jiannan Tian
 * @brief Formatting for log print (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-27
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef FORMAT_HH
#define FORMAT_HH

#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using std::regex;
using std::string;

extern const string log_null;
extern const string log_err;
extern const string log_dbg;
extern const string log_info;
extern const string log_warn;

extern const string fmt_b;
extern const string fmt_0;

extern const regex  bful;
extern const string bful_text;
extern const regex  bf;
extern const string bf_text;
extern const regex  ul;
extern const string ul_text;
extern const regex  red;
extern const string red_text;
namespace cusz {
namespace log {

// https://stackoverflow.com/a/26080768/8740097
template <typename T>
void build(std::ostream& o, T t);

template <typename T, typename... Args>
void build(std::ostream& o, T t, Args... args);

template <typename... Args>
void print(string log_head, Args... args);

}  // namespace log
}  // namespace cusz

#endif  // FORMAT_HH
