#ifndef ARGPARSE_HH
#define ARGPARSE_HH

/**
 * @file argparse.hh
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include "format.hh"

using std::string;

extern const char* version_text;
extern const int   version;
extern const int   compatibility;

typedef struct ArgPack {
    int read_args_status;

    string cx_path2file;
    string c_huff_base, c_fo_q, c_fo_outlier, c_fo_yamp;
    string x_fi_q, x_fi_outlier, x_fi_yamp, x_fo_xd;
    string x_fi_origin;

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;
    int    dict_size;
    int    quant_rep;
    int    input_rep;        // for standalone huffman
    int    huffman_datalen;  // for standalone huffman
    int    huffman_rep;
    int    huffman_chunk;
<<<<<<< HEAD
    int    n_dim, d0, d1, d2, d3;
    double mantissa, exponent;
    bool   to_archive, to_extract, to_dryrun;
    bool   autotune_huffman_chunk;
=======
    int    n_dim;
    int    d0;
    int    d1;
    int    d2;
    int    d3;
    double mantissa;
    double exponent;
    bool   to_archive;
    bool   to_extract;
    bool   to_encode;    // for standalone huffman
    bool   to_decode;    // for standalone huffman
    bool   get_entropy;  // for standalone huffman (not in use)
>>>>>>> add "Huffman (re)"
    bool   use_demo;
    bool   verbose;
    bool   to_verify;
    bool   verify_huffman;
    bool   skip_huffman, skip_writex;
    bool   pre_binning;

    int  input_rep;        // for standalone huffman
    int  huffman_datalen;  // for standalone huffman
    bool to_encode;        // for standalone huffman
    bool to_decode;        // for standalone huffman
    bool get_entropy;      // for standalone huffman (not in use)
    bool to_gzip;          // wenyu: whether to do a gzip lossless compression on encoded data

    static string format(const string& s);

    int trap(int _status);

    void CheckArgs();

    void HuffmanCheckArgs();

    static void cuszDoc();

    static void HuffmanDoc();

    static void cuszFullDoc();

    ArgPack(int argc, char** argv);

    ArgPack(int argc, char** argv, bool standalone_huffman);

<<<<<<< HEAD
    void SortOutFilenames();

=======
>>>>>>> add "Huffman (re)"
} argpack;

#endif  // ARGPARSE_HH
