#ifndef ARGPARSE2_HH
#define ARGPARSE2_HH

/**
 * @file argparse2.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <iostream>
#include <string>

#include "format.hh"
#include "metadata.hh"

using std::cerr;
using std::cout;
using std::endl;

using std::string;

enum class LoggingLevel { kINFO, kDBG, kVERBOSE };

enum class CompressMode {
    kABS,
    kR2R,
    kPWREL,
};

class ArgParser {
   private:
   protected:
    virtual void parse(int argc, char** argv) = 0;

    void print_err(int i, char** argv, const char* notif_prefix);

    int ap_status;

    int ndim;
    int d0, d1, d2, d3, len;
    int cb_len;

    int rep_d;  // lossless rep. data
    int rep_h;  // lossless rep. Huff

    std::string fname;

    bool use_demo;
    bool wf_dryrun;

    static std::string format(const std::string& s);

    void trap(int status) { this->ap_status = status; }

   public:
    ArgParser() {}
    ~ArgParser() {}

    int get_ndim() const { return this->ndim; };  // TODO redundant

    int str2int(const char* s);

    int str2fp(const char* s);

    virtual void PrintShortDoc() = 0;
    virtual void PrintFullDoc()  = 0;
    virtual void CheckArgs()     = 0;
};

#endif