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

class cuszContext : public ArgParser {
   private:
    std::string mode_str;
    std::string kernel_str;

    std::string log_level_str;
    std::string dtype;
    std::string demo_dataset;

    double mantissa;
    double exponent;

    virtual void parse(int argc, char** argv);

   public:
    bool DO_zip;  // TODO, DO -> conduct, perform??
    bool DO_unzip;
    bool do_verify_huffman;
    bool do_cpu_gzip;  // TODO update this in driver program
    bool skip_huff;
    bool skip_writex;

    int quant_byte;  // quant. code
    int huff_byte;   // Huffman
    int h_chunksize;

    bool   pre_binning;
    double val_rng;

    CompressMode mode;
    LoggingLevel log_level;  // TODO update doc

    cuszContext(int argc, char** argv);
    ~cuszContext(){};

    virtual void CheckArgs();
    virtual void PrintShortDoc();
    virtual void PrintFullDoc();

    template <size_t Block>
    void Export(struct Metadata<Block>* const m);
};

#endif