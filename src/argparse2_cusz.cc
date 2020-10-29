/**
 * @file argparse2_cusz.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "argparse2_cusz.hh"
#include "argparse2_doc.hh"
#include "metadata.hh"

void cuszContext::parse(int argc, char** argv)
{
    int i = 1;
    while (i < argc) {
        auto s = string(argv[i]);
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                // more readable args
                case '-':
                    // clang-format off
                    if (s == "--help")      goto HELP;
                    if (s == "--tutorial" or
                        s == "--doc")       goto TUTORIAL;
                    if (s == "--mode")      goto MODE;
                    if (s == "--kernel")    goto KERNEL;
                    if (s == "--input")     goto INPUT_DATUM;
                    if (s == "--type")      goto TYPE;
                    if (s == "--demo")      goto DEMO;
                    if (s == "--logging")   goto LOGGING;
                    if (s == "--quant")     goto QUANT;
                    if (s == "--huff")      goto HUFFMAN;
                    if (s == "--huff-chunk")goto HUFF_CHUNK;
                    if (s == "--verify")    goto VERIFY;
                    if (s == "--dict-size") goto DICT;
                    if (s == "--error-bound" or
                        s == "--eb")        goto ERROR_BOUND;
                    if (s == "--compress" or //
                        s == "--zip")       goto WF_Z;
                    if (s == "--decompress" or //
                        s == "--extract" or //
                        s == "--unzip")     goto WF_X;
                    if (s == "--dry-run")   goto WF_DRYRUN;
                    if (s == "--exclude" or //
                        s == "--skip")      goto EXCLUDE;
                    if (s == "--pre")       goto PRE;
                    if (s == "--cpu")       goto CPU;
                    // clang-format on
                // work
                case 'z':
                WF_Z:
                    DO_zip = true;
                    break;
                case 'x':
                WF_X:
                    DO_unzip = true;
                    break;
                case 'r':
                WF_DRYRUN:
                    DO_dryrun = true;
                    break;
                case 'm':  // mode
                MODE:
                    if (i + 1 <= argc) mode_str = string(argv[++i]);
                    if (mode_str == "r2r")
                        mode = CompressMode::kR2R;
                    else if (mode_str == "abs")
                        mode = CompressMode::kABS;
                    else if (mode_str == "pwrel")
                        mode = CompressMode::kPWREL;
                    break;
                case 'k':
                KERNEL:
                    if (i + 1 <= argc) kernel_str = string(argv[++i]);
                    if (mode_str == "lorenzo")
                        cout << "kernel selection not implemented" << endl;
                    else if (mode_str == "spline")
                        cout << "kernel selection not implemented" << endl;
                    break;
                // skip
                case 'X':
                case 'S':
                EXCLUDE:
                    if (i + 1 <= argc) {
                        string exclude(argv[++i]);
                        if (exclude.find("huffman") != std::string::npos) skip_huff = true;
                        if (exclude.find("write.x") != std::string::npos) skip_writex = true;
                    }
                    break;

                // input dimensionality
                case '1':
                    ndim = 1;
                    if (i + 1 <= argc) {
                        d0  = str2int(argv[++i]);
                        len = d0;
                    }
                    break;
                case '2':
                    ndim = 2;
                    if (i + 2 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        len = d0 * d1;
                    }
                    break;
                case '3':
                    ndim = 3;
                    if (i + 3 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]), d2 = str2int(argv[++i]);
                        len = d0 * d1 * d2;
                    }
                    break;
                case '4':
                    ndim = 4;
                    if (i + 4 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]), d3 = str2int(argv[++i]);
                        len = d0 * d1 * d2 * d3;
                    }
                    break;
                // ************************************************** //
                // help document
                // ************************************************** //
                case 'h':
                HELP:
                    PrintFullDoc();
                    exit(0);
                    break;
                case 'T':
                TUTORIAL:
                    // TODO add more entry
                    if (i + 1 <= argc) {
                        string tutorial_str(argv[++i]);
                        if (tutorial_str.find("what") != std::string::npos) {  //
                            cout << "dim : how data dimension is interpreted.\n";
                            exit(0);
                        }
                        if (tutorial_str.find("dim") != std::string::npos) cout << doc_dim_order << endl;
                        exit(0);
                    }
                // ************************************************** //
                // input datum file
                // ************************************************** //
                case 'i':
                INPUT_DATUM:
                    if (i + 1 <= argc) fname = string(argv[++i]);
                    break;
                // ************************************************** //
                // specify data type
                // ************************************************** //
                case 't':
                TYPE:
                    if (i + 1 <= argc) dtype = string(argv[++i]);
                    break;
                // preprocess
                case 'p':
                PRE:
                    if (i + 1 <= argc) {
                        string pre(argv[++i]);
                        if (pre.find("binning") != std::string::npos) pre_binning = true;
                        // if (pre.find("<name>") != std::string::npos) pre_<name> = true;
                    }
                    break;

                // demo datasets
                case 'D':
                DEMO:
                    if (i + 1 <= argc) {
                        use_demo     = true;  // for skipping checking dimension args
                        demo_dataset = string(argv[++i]);
                    }
                    break;
                case 'L':
                LOGGING:
                    if (i + 1 <= argc) log_level_str = std::string(argv[++i]);
                    if (log_level_str == "dbg" or log_level_str == "debug")
                        log_level = LoggingLevel::kDBG;
                    else if (log_level_str == "verbose")  // TODO --verbose, -vv something
                        log_level = LoggingLevel::kVERBOSE;
                    else
                        log_level = LoggingLevel::kINFO;
                    break;
                // internal representation and size
                case 'Q':
                QUANT:
                    if (i + 1 <= argc) quant_byte = str2int(argv[++i]);
                    break;
                case 'H':
                HUFFMAN:
                    if (i + 1 <= argc) huff_byte = str2int(argv[++i]);
                    break;
                case 'C':
                HUFF_CHUNK:
                    if (i + 1 <= argc) h_chunksize = str2int(argv[++i]);
                    break;
                // error bound
                case 'e':
                ERROR_BOUND:
                    if (i + 1 <= argc) {
                        string eb(argv[++i]);
                        if (eb.find('e') != std::string::npos) {
                            string dlm = "e";
                            mantissa   = ::atof(eb.substr(0, eb.find(dlm)).c_str());
                            exponent   = ::atof(eb.substr(eb.find(dlm) + dlm.length(), eb.length()).c_str());
                        }
                        else {
                            mantissa = ::atof(eb.c_str());
                            exponent = 0.0;
                        }
                    }
                    break;
                case 'U':
                CPU:
                    if (i + 1 <= argc) {
                        string cpu_do(argv[++i]);
                        if (cpu_do.find("gzip") != std::string::npos) do_cpu_gzip = true;
                    }
                    break;
                case 'y':
                VERIFY:  // TODO verify data quanlity
                    if (i + 1 <= argc) {
                        string veri(argv[++i]);
                        if (veri.find("huffman") != std::string::npos) do_verify_huffman = true;
                        // TODO verify data quality
                    }
                    break;

                case 'd':
                DICT:
                    if (i + 1 <= argc) cb_len = str2int(argv[++i]);
                    break;
                default:
                    cout << "aaaaaaaaaaa" << endl;
                    const char* notif_prefix = "invalid option at position ";  //
                    print_err(i, argv, notif_prefix);
                    // break;
            }
        }
        else {
            const char* notif_prefix = "invalid argument at position ";
            cout << "ddddddddd" << endl;
            print_err(i, argv, notif_prefix);
            // break;
        }
        i++;
    }
}

cuszContext::cuszContext(int argc, char** argv)
{
    if (argc == 1) {
        PrintShortDoc();
        exit(0);
    }
    // default values
    cb_len      = 1024;
    quant_byte  = 2;
    huff_byte   = 4;
    h_chunksize = 512;
    ndim = -1, d0 = 1, d1 = 1, d2 = 1, d3 = 1;
    mantissa = 1.0, exponent = -4.0;
    DO_zip    = false;
    DO_unzip  = false;
    DO_dryrun = false;

    use_demo          = false;
    do_verify_huffman = false;
    skip_huff         = false;
    skip_writex       = false;
    pre_binning       = false;

    parse(argc, argv);

    // phase 1: check grammar
    if (ap_status != 0) {
        // cout << log_info << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    // phase 2: check if meaningful
    CheckArgs();
}

void cuszContext::PrintShortDoc()
{
    cout << "cusz, " << cusz_build_str << endl;
    cout << cusz_short_doc << endl;
}

void cuszContext::PrintFullDoc() { cout << format(cusz_full_doc) << endl; }

void cuszContext::CheckArgs()
{
    bool to_abort = false;
    if (fname.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (d0 * d1 * d2 * d3 == 1 and not use_demo) {
        cerr << log_err << "Wrong input size(s)!" << endl;
        to_abort = true;
    }
    if (!DO_zip and !DO_unzip and !DO_dryrun) {
        cerr << log_err << "Select compress (-a), decompress (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    // if (dtype != "f32" and dtype != "f64") {
    if (not(dtype == "int8" or dtype == "int16" or dtype == "int32" or dtype == "int64"  //
            or dtype == "fp32" or dtype == "fp64")) {
        cout << dtype << endl;
        cerr << log_err << "Not specifying data type!" << endl;
        to_abort = true;
    }

    if (quant_byte == 1) {  // TODO
        assert(cb_len <= 256);
    }
    else if (quant_byte == 2) {
        assert(cb_len <= 65536);
    }

    if (DO_dryrun and DO_zip and DO_unzip) {
        cerr << log_warn << "No need to dry-run, compress and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        DO_zip   = false;
        DO_unzip = false;
    }
    else if (DO_dryrun and DO_zip) {
        cerr << log_warn << "No need to dry-run and compress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        DO_zip = false;
    }
    else if (DO_dryrun and DO_unzip) {
        cerr << log_warn << "No need to dry-run and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        DO_unzip = false;
    }

    if (to_abort) {
        PrintShortDoc();
        exit(-1);
    }
}

template <size_t Block>
void cuszContext::Export(struct Metadata<Block>* const m)
{
    // specify dimensions
    if (use_demo)
        cuszSetDemoDim(m, demo_dataset);
    else
        cuszSetDim(m, ndim, d0, d1, d1, d3);

    // quant. setup
    cuszSetQuantBinNum(m, cb_len);

    // error bound set
    auto eb = mantissa * std::pow(10, exponent);  // TODO for now, only base-10
    cuszSetErrorBound(m, eb);
}