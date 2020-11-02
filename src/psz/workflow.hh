#ifndef PSZ_WORKFLOW_HH
#define PSZ_WORKFLOW_HH

/**
 * @file workflow.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-02-11
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "../analysis.hh"
#include "../io.hh"
#include "../lorenzo_psz_trait.hh"
#include "../type_trait.hh"
#include "../verify.hh"

#ifdef PSZ_OMP
#include <omp.h>
#endif

#include "pq14.hh"
#include "pq14_chunked.hh"
#include "pq_dualquant.hh"

namespace AnalysisNamespace = analysis;

namespace psz {

template <int ndim, typename Data, typename Quant>
void cx_sim(
    typename MetadataTrait<ndim>::metadata_t* m,
    std::string&                              finame,  //
    size_t&                                   num_outlier,
    bool                                      fine_massive = false,
    bool                                      blocked      = false,
    bool                                      show_histo   = false)
{
    using Block = typename MetadataTrait<ndim>::Block;
    size_t len  = m->len;

    auto data     = io::ReadBinaryToNewArray<Data>(finame, len);
    auto data_cmp = io::ReadBinaryToNewArray<Data>(finame, len);

    Data* pred_err = nullptr;
    Data* comp_err = nullptr;
#ifdef PRED_COMP_ERR
    pred_err = new Data[len]();
    comp_err = new Data[len]();
#endif

    auto xdata   = new Data[len]();
    auto outlier = new Data[len]();
    auto code    = new Quant[len]();

    if (fine_massive)
        cout << "\e[46musing (blocked) dualquant\e[0m" << endl;
    else {
        cout << (blocked == true ? "\e[46musing blocked sz14\e[0m" : "\e[46musing non-blocked sz14\e[0m") << endl;
    }

    // ------------------------------------------------------------
    // start of compression
    // ------------------------------------------------------------
    if (blocked) {
        if (fine_massive)
            psz::pq_dualquant::zip::Lorenzo_nd1l<ndim>::Call(m, data, code);
        else
            psz::pq14_chunked::zip::Lorenzo_nd1l<ndim>::Call(m, data, code);
    }
    else {
        psz::pq14::zip::Lorenzo_nd1l<ndim>::Call(m, data, code);
    }
    // ------------------------------------------------------------
    // end of compression
    // ------------------------------------------------------------

    //    io::write_binary_file(code, len, new string("/Users/jtian/WorkSpace/cuSZ/src/CLDMED.bincode"));

    if (show_histo) { Analysis::histogram<int>(std::string("bincode/quant.code"), code, len, 8); }
    Analysis::getEntropy(code, len, 1024);
#ifdef PRED_COMP_ERR
    Analysis::histogram<Data>(std::string("pred.error"), pred_err, len, 8);  // TODO when changing to 8, seg fault
    Analysis::histogram<Data>(std::string("comp.error"), comp_err, len, 16);
#endif

    for_each(outlier, outlier + len, [&](Data& n) { num_outlier += n == 0 ? 0 : 1; });

    // ------------------------------------------------------------
    // start of decompression
    // ------------------------------------------------------------
    if (blocked) {
        if (fine_massive)
            psz::pq_dualquant::unzip::Lorenzo_nd1l<ndim>::Call(m, data, code);
        else
            psz::pq14_chunked::unzip::Lorenzo_nd1l<ndim>::Call(m, data, code);
    }
    else {
        psz::pq14::unzip::Lorenzo_nd1l<ndim>::Call(m, data, code);
    }
    // ------------------------------------------------------------
    // end of decompression
    // ------------------------------------------------------------

    if (show_histo) {
        Analysis::histogram(std::string("original datum"), data_cmp, len, 16);
        Analysis::histogram(std::string("reconstructed datum"), xdata, len, 16);
    }

    cout << "\e[46mnum.outlier:\t" << num_outlier << "\e[0m" << endl;
    cout << std::setprecision(5) << "error bound: " << m->eb << endl;

    if (fine_massive) {  //
        io::WriteArrayToBinary(xdata, len, new string(finame + ".psz.cusz.out"));
    }
    else if (blocked == true and fine_massive == false) {
        io::WriteArrayToBinary(xdata, len, new string(finame + ".psz.sz14blocked.out"));
    }
    else if (blocked == false and fine_massive == false) {
        io::WriteArrayToBinary(xdata, len, new string(finame + ".psz.sz14.out"));
        // io::WriteArrayToBinary(pred_err, len, new string(finame + ".psz.sz14.prederr"));
        // io::WriteArrayToBinary(comp_err, len, new string(finame + ".psz.sz14.xerr"));
    }
    AnalysisNamespace::VerifyData(xdata, data_cmp, len, 1);
}

}  // namespace psz

#endif
