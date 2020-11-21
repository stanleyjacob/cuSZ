/**
 * @file document.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#ifndef ARGUMENT_PARSER_DOCUMENT_HH
#define ARGUMENT_PARSER_DOCUMENT_HH

#include <regex>
#include <string>
#include "../format.hh"

using std::regex;
using std::string;

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

string  //
Format(const std::string& s)
{
    auto a = std::regex_replace(s, bful, bful_text);
    auto b = std::regex_replace(a, bf, bf_text);
    auto c = std::regex_replace(b, ul, ul_text);
    auto d = std::regex_replace(c, red, red_text);
    return d;
}

static const char cusz_short_doc[] =
    // "cusz, version [placeholder]\n"
    "\n"
    "usage: cusz [-zxrhV] [-i file] [-t dtype] [-m mode] [-e eb] [-1|2|3 ...] ...\n"
    "\n"
    "  z : zip/compress\n"
    "  x : unzip/decompress\n"
    "  r : dryrun\n"
    "  h : print full-length help document\n"
    "  V : print machine information\n"
    "\n"
    "  i file  : path to input datum\n"
    "  t dtype : f[32|64] or fp[4|8], i[8|16|32|64] or int[1|2|3|4]\n"
    "  m mode  : compresssion mode; abs, r2r\n"
    "  e eb    : error bound; default 1e-4\n"
    "  1 x     : 1D interpretion of datum\n"
    "  2 x y   : 2D interpretion of datum\n"
    "  3 x y z : 3D interpretion of datum\n"
    "  D name  : use demo dataset, skip interpretation\n"
    "              (1D) hacc1g  hacc4g  (2D) cesm  exafel\n"
    "              (3D) hurricane  nyx  qmc  qmcpre  aramco  parihaka\n"
    "  d num   : specify codebook size\n"
    "  T name  : tutorials; \"what\" to list details.\n"
    "              dim todo todo todo todo\n"
    "\n"
    "example:\n"
    "  zip 1: ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z\n"
    "  zip 2: ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z\n"
    "  unzip: ./bin/cusz -i ./data/ex-cesm-CLDHGH.sz -x\n"
    "\n"
    "\"cusz -h\" for details.\n";

static const char cusz_full_doc[] =
    "*NAME*\n"
    "        cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific Data\n"
    "        Lowercased \"*cusz*\" is the command."
    "\n"
    "*SYNOPSIS*\n"
    "        The basic use is listed below,\n"
    "        *cusz* *-t* f32 *-m* r2r *-e* 1.0e-4.0 *-i* ./data/ex-cesm-CLDHGH *-2* 3600 1800 *-z* \n"
    //       cusz -t f32 -m r2r -e 1.0e-4.0 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z \n
    "             ------ ------ ----------- ------------------------ ------------  | \n"
    "              dtype  mode  error bound      input datum file    low-to-high  zip \n"
    "\n"
    "        *cusz* *-i* ./data/ex--cesm-CLDHGH.sz *-x* \n"
    //       cusz -i ./data/ex--cesm-CLDHGH.sz -x \n"
    "             ----------------------------  | \n"
    "                       sz archive         unzip \n"
    "\n"
    "        *cusz* *-t* f32|64 *-m* [eb mode] *-e* [eb] *-i* [datum file] *-D* [demo dataset] *-z*\n"
    "        *cusz* *-t* f32|64 *-m* [eb mode] *-e* [eb] *-i* [datum file] *-1*|*-2*|*-3* [nx [ny [nz]] *-z*\n"
    "        *cusz* *-i* [datum basename].sz *-x*\n"
    "\n"
    "*OPTIONS*\n"
    "    *Mandatory* (zip and dryrun)\n"
    "        *-z* or *--compress* or *--*@z@*ip*\n"
    "        *-r* or *--dry-*@r@*un*\n"
    "                No lossless Huffman codec. Only to get data quality summary.\n"
    "                In addition, quant. rep. and dict. size are retained\n"
    "\n"
    "        *-m* or *--*@m@*ode* <abs|r2r>\n"
    "                Specify error-controling mode. Supported modes include:\n"
    "                _abs_: absolute mode, eb = input eb\n"
    "                _r2r_: relative-to-value-range mode, eb = input eb x value range\n"
    "\n"
    "        *-e* or *--eb* or *--error-bound* [num]\n"
    "                Specify error bound. e.g., _1.23_, _1e-4_, _1.23e-4.56_\n"
    "\n"
    "        *-i* or *--*@i@*nput* [datum file]\n"
    "\n"
    "        *-d* or *--dict-size* [256|512|1024|...]\n"
    "                Specify dictionary size/quantization bin number.\n"
    "                Should be a power-of-2.\n"
    "\n"
    "        *-1* [x]       Specify 1D datum/field size.\n"
    "        *-2* [x] [y]   Specify 2D datum/field sizes, with dimensions from low to high.\n"
    "        *-3* [x] [y] [z]   Specify 3D datum/field sizes, with dimensions from low to high.\n"
    "\n"
    "    *Mandatory* (unzip)\n"
    "        *-x* or *--e*@x@*tract* or *--decompress* or *--unzip*\n"
    "\n"
    "        *-i* or *--*@i@*nput* [corresponding datum basename (w/o extension)]\n"
    "\n"
    "    *Additional I/O*\n"
    "        *--origin* /path/to/origin-datum\n"
    "                For verification & get data quality evaluation.\n"
    "        *--opath*  /path/to\n"
    "                Specify alternative output path.\n"
    "\n"
    "    *Modules*\n"
    "        *-S* or *--e*@x@*clude* or *--*@s@*kip* _module-1_,_module-2_,...,_module-n_,\n"
    "                Disable functionality modules. Supported module(s) include:\n"
    "                _huffman_  Huffman codec after prediction+quantization (p+q) and before reveresed p+q.\n"
    "                _write.x_  Skip write decompression data.\n"
    "\n"
    "        *-p* or *--pre* _method-1_,_method-2_,...,_method-n_\n"
    "                Enable preprocessing. Supported preproessing method(s) include:\n"
    "                _binning_  Downsampling datum by 2x2 to 1.\n"
    "\n"
    "    *Demonstration*\n"
    "        *-h* or *--help*\n"
    "                Get help documentation.\n"
    "\n"
    "        *-V* or *--verbose*\n"
    "                Print host and device information for diagnostics.\n"
    "\n"
    "        *-M* or *--meta*\n"
    "                Get archive metadata. (TODO)\n"
    "\n"
    "        *-D* or *--demo* [demo-dataset]\n"
    "                Use demo dataset, will omit given dimension(s). Supported datasets include:\n"
    "                1D: _hacc_  _hacc1g_  _hacc4g_\n"
    "                2D: _cesm_  _exafeldemo_\n"
    "                3D: _hurricane_  _nyx_  _qmc_  _qmcpre_  _aramco_  _parihaka_\n"
    "\n"
    "    *Internal* (will be automated with configuration when going public)\n"
    "        *-Q* or *--*@q@*uant-byte* <1|2>\n"
    "                Specify bincode/quantization code representation.\n"
    "                Options _1_, _2_ are for *1-* and *2-*byte, respectively. (default: 2)\n"
    "                ^^Manually specifying this may not result in optimal memory footprint.^^\n"
    "\n"
    "        *-H* or *--*@h@*uff-byte* <4|8>\n"
    "                Specify Huffman codeword representation.\n"
    "                Options _4_, _8_ are for *4-* and *8-*byte, respectively. (default: 4)\n"
    "                ^^Manually specifying this may not result in optimal memory footprint.^^\n"
    "\n"
    "        *-C* or *--huff-*@c@*hunk* [256|512|1024|...]\n"
    "                Manually specify chunk size for Huffman codec, overriding autotuning.\n"
    "                Should be a power-of-2 that is sufficiently large.\n"
    "                ^^This affects Huffman decoding performance significantly.^^\n"
    "\n"
    "*EXAMPLES*\n"
    "    *Demo Datasets*\n"
    "        *CESM* example:\n"
    "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z\n"
    "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -r\n"
    "        ./bin/cusz -i ./data/ex-cesm-CLDHGH.sz -x\n"
    "\n"
    "        *CESM* example with specified output path:\n"
    "        makdir data2 data3\n"
    "            # zip, output to `data2`\n"
    "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z --opath data2\n"
    "            # unzip, in situ\n"
    "        ./bin/cusz -i ./data2/ex-cesm-CLDHGH.sz -x && ls data2\n"
    "            # unzip, output to `data3`\n"
    "        ./bin/cusz -i ./data2/ex-cesm-CLDHGH.sz -x --opath data3 && ls data3\n"
    "            # unzip, output to `data3`, compare to the original datum\n"
    "        ./bin/cusz -i ./data2/ex-cesm-CLDHGH.sz -x --opath data3 --origin ./data/ex-cesm-CLDHGH && ls "
    "data3\n"
    "        ## Please create directory by hand before specifying (considering access permission control).\n"
    "\n"
    "        *Hurricane Isabel* example:\n"
    "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-hurr-CLOUDf48 -D hurricane -z\n"
    "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-hurr-CLOUDf48 -D hurricane -r\n"
    "        ./bin/cusz -i ./data/ex-hurr-CLOUDf48.sz -x\n"
    "\n";
// TODO
// "        *EXAFEL* example:\n"
// "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-exafel-59200x388 -D exafeldemo -z -x --pre binning\n"
// "        ./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-exafel-59200x388 -D exafeldemo -z -x --pre binning "
// "--skip huffman\n"
// "        ./bin/cusz -i ./data/ex-exafel-59200x388.BN.sz -x\n";

static const char huff_re_short_doc[] =
    "\n"
    "OVERVIEW: Huffman submodule as standalone program\n"  // TODO from this line on
    "\n"
    "USAGE:\n"
    "  The basic use with demo datum is listed below,\n"
    "    ./huff --encode --decode --verify --input ./baryon_density.dat.b16 \\\n"
    "        -3 512 512 512 --input-rep 16 --huffman-rep 32 --huffman-chunk 2048 --dict-size 1024\n"
    "  or shorter\n"
    "    ./huff -e -d -V -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -H 32 -C 2048 -c 1024\n"
    "            ^  ^  ^ --------------------------- -------------- ----- ----- ------- -------\n"
    "            |  |  |       input datum file         dimension   input Huff. Huff.   codebook\n"
    "          enc dec verify                                       rep.  rep.  chunk   size\n"
    "\n"
    "EXAMPLES\n"
    "  Essential:\n"
    "    ./bin/huff -e -d -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -c 1024\n"
    "    have to input dimension, and higher dimension for a multiplication of each dim.,\n"
    "    as default values input-rep=16 (bits), huff-rep=32 (bits), codebokk-size=1024 (symbols)\n"
    "\n";

static const char doc_dim_order[] =
    "\n"
    "  Input dimension follows low-to-high (e.g., x-y-z) order.\n"
    "  Taking 2D CESM-ATM as an example, \n"
    "\n"
    "  |<------------------------- x 3600 --------------------------->|    \n"
    "  +--------------------------------------------------------------+  - \n"
    "  |                                                              |  ^ \n"
    "  |                                                              |  | \n"
    "  |              CESM-ATM:    1800x3600 (y-x order)              |  | \n"
    "  |              datum name:  <field>_1800_3600                  |  y \n"
    "  |                                                              | 1800 \n"
    "  |              input:       -2 3600 1800                       |  | \n"
    "  |              input order: -2 [x] [y]                         |  | \n"
    "  |                                                              |  | \n"
    "  |                                                              |  v \n"
    "  +--------------------------------------------------------------+  - \n"
    "\n"
    "  Taking 3D Hurricane as another example, whose dimensions are\n"
    "  100x500x500, the input is \"-3 500 500 100\".\n";

#endif