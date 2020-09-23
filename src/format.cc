/**
 * @file format.cc
 * @author Jiannan Tian
 * @brief Formatting for log print.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-27
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "format.hh"
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using std::regex;
using std::string;

const string log_null = "       ";  // width = 7
const string log_err  = "\e[31m[ERR]\e[0m  ";
const string log_dbg  = "\e[34m[dbg]\e[0m  ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_warn = "\e[31m[WARN]\e[0m ";

const string fmt_b("\e[1m");
const string fmt_0("\e[0m");

const regex  bful("@(.*?)@");
const string bful_text("\e[1m\e[4m$1\e[0m");
const regex  bf("\\*(.*?)\\*");
const string bf_text("\e[1m$1\e[0m");
const regex  ul(R"(_((\w|-|\d|\.)+?)_)");
const string ul_text("\e[4m$1\e[0m");
const regex  red(R"(\^\^(.*?)\^\^)");
const string red_text("\e[31m$1\e[0m");

// https://stackoverflow.com/a/26080768/8740097
template <typename T>
void cusz::log::build(std::ostream& o, T t)
{
    o << t << std::endl;
}

// template void cusz::log::build<int>(std::ostream& o, int t);
// template void cusz::log::build<T>(std::ostream& o, T t);

template <typename T, typename... Args>
void cusz::log::build(std::ostream& o, T t, Args... args)  // recursive variadic function
{
    cusz::log::build(o, t);
    cusz::log::build(o, args...);
}

template <typename... Args>
void cusz::log::print(string log_head, Args... args)
{
    std::ostringstream oss;
    cusz::log::build(oss, args...);
    std::cout << log_head << oss.str();
}
