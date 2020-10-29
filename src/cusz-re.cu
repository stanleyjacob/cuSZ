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

typedef cuszContext ctx_t;

template <int ndim, typename Data, int QuantByte, int HuffByte>
void main_alt(ctx_t* ctx, unsigned char* archive_bin, unsigned char* metadata_bin)
{
    typedef typename MetadataTrait<ndim>::metadata_t metadata_t;

    auto m = new metadata_t;

    ctx->Export(m);
    if (ctx->mode == CompressMode::kR2R)  //
        cuszChangeToR2RModeMode(m, GetDatumValueRange<Data>(ctx->get_fname(), m->len));

    if (ctx->DO_zip or ctx->DO_dryrun)  //
        cusz::interface::Compress2<ndim, Data, QuantByte, HuffByte>(ctx, m);
    if (ctx->DO_unzip)  //
        cusz::interface::Decompress2<ndim, Data, QuantByte, HuffByte>(ctx, m);
}
