/**
 * @file cusz.cu
 * @author Jiannan Tian
 * @brief Driver program of cuSZ.
 * @version 0.1.1
 * @date 2020-09-23
 * Created on 2019-12-30
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "argparse2.hh"
#include "argparse2_cusz.hh"
#include "cusz_workflow2.cuh"
#include "metadata.hh"
#include "type_trait.hh"

typedef cuszContext context_t;

template <int ndim, typename Data, int QuantByte, int HuffByte>
void main_alt(cuszContext* ctx, unsigned char* archive_bin, unsigned char* metadata_bin)
{
    typedef typename MetadataTrait<ndim>::metadata_t metadata_t;
    // auto ctx = new cuszContext(argc, argv);
    auto m = new metadata_t;

    ctx->Export(m);
    if (ctx->mode == CompressMode::kR2R) {
        auto val_rng = GetDatumValueRange<Data>(ctx->get_fname(), m->len);
        cuszChangeToR2RModeMode(m, val_rng);
    }

    if (ctx->wf_zip or ctx->wf_dryrun) cusz::interface::Compress2<Data, QuantByte, HuffByte>(ctx, m);
    if (ctx->wf_unzip) cusz::interface::Decompress2<Data, QuantByte, HuffByte>(ctx, m);
}
