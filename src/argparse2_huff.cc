/**
 * @file argparse2_huff.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cassert>
#include <iostream>

#include "argparse2_doc.hh"
#include "argparse2_huff.hh"

using std::cout, std::cerr, std::endl;

void HuffmanArgParser::parse(int argc, char** argv)
{
    int i = 1;
    while (i < argc) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                // more readable args
                case '-':
                    if (string(argv[i]) == "--help") goto HELP;
                    if (string(argv[i]) == "--enc" or string(argv[i]) == "--encode") goto WF_ENC;
                    if (string(argv[i]) == "--dec" or string(argv[i]) == "--decode") goto WF_DEC;
                    if (string(argv[i]) == "--dry-run") goto WF_DRYRUN;
                    if (string(argv[i]) == "--input") goto INPUT;
                    if (string(argv[i]) == "--rep-d" or string(argv[i]) == "--interpret") goto REP_D;
                    if (string(argv[i]) == "--rep-h") goto REP_H;
                    if (string(argv[i]) == "--chunk") goto CHUNKSIZE;
                    if (string(argv[i]) == "--dict-size") goto DICT;

                    // workflow
                case 'e':
                WF_ENC:
                    wf_enc = true;
                    break;
                case 'd':
                WF_DEC:
                    wf_dec = true;
                    break;
                case 'r':
                WF_DRYRUN:
                    wf_dryrun = true;
                    break;

                    // dimension
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
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]), d2 = str2int(argv[++i]),
                        d3  = str2int(argv[++i]);
                        len = d0 * d1 * d2 * d3;
                    }
                    break;

                // help document
                case 'h':
                HELP:
                    PrintShortDoc();
                    exit(0);
                    break;

                // input datum file
                case 'i':
                INPUT:
                    if (i + 1 <= argc) { fname = string(argv[++i]); }
                    break;
                case 'R':
                REP_D:
                    if (i + 1 <= argc) rep_d = str2int(argv[++i]);
                    break;
                case 'H':
                REP_H:
                    if (i + 1 <= argc) rep_h = str2int(argv[++i]);
                    break;
                case 'C':
                CHUNKSIZE:
                    if (i + 1 <= argc) chunk = str2int(argv[++i]);
                    break;
                case 'c':
                DICT:
                    if (i + 1 <= argc) cb_len = str2int(argv[++i]);
                    break;
                default:
                    const char* notif_prefix = "invalid option at position ";
                    char*       notif;
                    int         size = asprintf(&notif, "%d: %s", i, argv[i]);
                    cerr << log_err << notif_prefix << fmt_b << notif << fmt_0 << "\n";
                    cerr << string(log_null.length() + strlen(notif_prefix), ' ');
                    cerr << fmt_b;
                    cerr << string(strlen(notif), '~');
                    cerr << fmt_0 << "\n";
                    trap(-1);
            }
        }
        else {
            const char* notif_prefix = "invalid argument at position ";
            char*       notif;
            int         size = asprintf(&notif, "%d: %s", i, argv[i]);
            cerr << log_err << notif_prefix << fmt_b << notif << fmt_0 << "\n";
            cerr << string(log_null.length() + strlen(notif_prefix), ' ');
            cerr << fmt_b;
            cerr << string(strlen(notif), '~');
            cerr << fmt_0 << "\n";
            trap(-1);
        }
        i++;
    }
}

HuffmanArgParser::HuffmanArgParser(int argc, char** argv)
{
    if (argc == 1) {
        PrintShortDoc();
        exit(0);
    }
    // default values
    cb_len = 1024;
    rep_d  = 16;
    len    = -1;  // TODO argcheck
    rep_h  = 32;
    chunk  = 512;

    ndim = -1, d0 = 1, d1 = 1, d2 = 1, d3 = 1;

    wf_enc    = false;
    wf_dec    = false;
    wf_dryrun = false;  // TODO dry run is meaningful differently for cuSZ and Huffman

    parse(argc, argv);

    // phase 1: check grammar
    if (ap_status != 0) {
        cout << log_info << "Exiting..." << endl;
        exit(-1);
    }

    // phase 2: check if meaningful
    CheckArgs();
}

void HuffmanArgParser::PrintShortDoc() { cout << huff_re_short_doc << endl; }

void HuffmanArgParser::PrintFullDoc() {}

void HuffmanArgParser::CheckArgs()
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
    if (!wf_enc and !wf_dec and !wf_dryrun) {
        cerr << log_err << "Select encode (-a), decode (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }

    if (rep_d == 8) {  // TODO
        assert(cb_len <= 256);
    }
    else if (rep_d == 16) {
        assert(cb_len <= 65536);
    }

    if (wf_dryrun and wf_enc and wf_dec) {
        cerr << log_warn << "No need to dry-run, encode and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        wf_enc = false;
        wf_dec = false;
    }
    else if (wf_dryrun and wf_enc) {
        cerr << log_warn << "No need to dry-run and encode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        wf_enc = false;
    }
    else if (wf_dryrun and wf_dec) {
        cerr << log_warn << "No need to dry-run and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        wf_dec = false;
    }

    if (to_abort) {
        PrintShortDoc();
        exit(-1);
    }
}