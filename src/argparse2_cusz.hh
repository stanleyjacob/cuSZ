#ifndef ARGPARSE2_CUSZ_HH
#define ARGPARSE2_CUSZ_HH

/**
 * @file argparse2_cusz.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "argparse2.hh"
#include "metadata.hh"

class cuszContext : ArgParser {
   private:
    std::string mode_str;

    std::string log_level_str;
    std::string dtype;
    std::string demo_dataset;

    double mantissa;
    double exponent;

    virtual void parse(int argc, char** argv);

   public:
    bool   wf_zip;
    bool   wf_unzip;
    bool   skip_huff;
    bool   skip_writex;
    bool   verify_huffman;
    int    rep_q;  // quant. code
    int    rep_h;  // Huffman
    int    h_chunksize;
    bool   pre_binning;
    double val_rng;

    CompressMode mode;
    LoggingLevel log_level;  // TODO update doc

    cuszContext(int argc, char** argv);
    ~cuszContext(){};

    using ArgParser::get_fname;
    using ArgParser::get_ndim;
    using ArgParser::wf_dryrun;

    virtual void CheckArgs();
    virtual void PrintShortDoc();
    virtual void PrintFullDoc();

    template <size_t Block>
    void Export(struct Metadata<Block>* const m);
};

#endif