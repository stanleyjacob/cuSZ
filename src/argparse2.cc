/**
 * @file argparse2.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "argparse2.hh"
#include "format.hh"

void ArgParser::print_err(int i, char** argv, const char* notif_prefix)
{
    char* notif;
    int   size = asprintf(&notif, "%d: %s", i, argv[i]);
    cerr << log_err << notif_prefix << fmt_b << notif << fmt_0 << "\n";
    cerr << string(log_null.length() + strlen(notif_prefix), ' ');
    cerr << fmt_b;
    cerr << string(strlen(notif), '~');
    cerr << fmt_0 << "\n";
    trap(-1);
}

std::string ArgParser::format(const std::string& s)
{
    // std::regex  bful("@(.*?)@");
    // std::string bful_text("\e[1m\e[4m$1\e[0m");
    // std::regex  bf("\\*(.*?)\\*");
    // std::string bf_text("\e[1m$1\e[0m");
    // std::regex  ul(R"(_((\w|-|\d|\.)+?)_)");
    // std::string ul_text("\e[4m$1\e[0m");
    // std::regex  red(R"(\^\^(.*?)\^\^)");
    // std::string red_text("\e[31m$1\e[0m");
    auto a = std::regex_replace(s, bful, bful_text);
    auto b = std::regex_replace(a, bf, bf_text);
    auto c = std::regex_replace(b, ul, ul_text);
    auto d = std::regex_replace(c, red, red_text);
    return d;
}

int ArgParser::str2int(const char* s)
{
    char* end;
    auto  res = std::strtol(s, &end, 10);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << fmt_b << end << fmt_0 << endl;
        cerr << string(log_null.length() + strlen(notif), ' ') << fmt_b  //
             << string(strlen(end), '~')                                 //
             << fmt_0 << endl;
        trap(-1);
        return 0;  // just a placeholder
    }
    return (int)res;
}

int ArgParser::str2fp(const char* s)
{
    char* end;
    auto  res = std::strtod(s, &end);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << fmt_b << end << fmt_0 << endl;
        cerr << string(log_null.length() + strlen(notif), ' ') << fmt_b  //
             << string(strlen(end), '~')                                 //
             << fmt_0 << endl;
        trap(-1);
        return 0;  // just a placeholder
    }
    return (int)res;
}
