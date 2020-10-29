#ifndef ARGPARSE2_HUFF_HH
#define ARGPARSE2_HUFF_HH

/**
 * @file argparse2_huff.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "argparse2.hh"

class HuffmanArgParser : public ArgParser {
   private:
    bool DO_enc;
    bool DO_dec;
    int  chunk;

    virtual void parse(int argc, char** argv);

   public:
    HuffmanArgParser(int argc, char** argv);
    ~HuffmanArgParser(){};
    virtual void CheckArgs();
    virtual void PrintShortDoc();
    virtual void PrintFullDoc();
};

#endif