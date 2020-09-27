// jtian 20-05-30
/**
 * @file reduce_move_merge.cuh
 * @author Jiannan Tian
 * @brief Reduction and merging based Huffman encoding
 * @version 0.1.3
 * @date 2020-10-29
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */
#include <stdio.h>
#include "../cuda_util.cuh"
#include "auxiliary.cuh"

template <typename UInt>
__forceinline__ __host__ __device__ int get_bw(UInt var)
{
    return (int)*((uint8_t*)&var + sizeof(UInt) - 1);
}

template <typename UInt>
int GetBitwidth(UInt var)
{
    return static_cast<int>(*(reinterpret_cast<uint8_t*>(&var) + sizeof(UInt) - 1));
}

template <typename Input, typename Huff, int Magnitude>
__global__ void PrefixSumBased(Input* in, size_t len, Huff* cb, Huff* out, size_t cb_len, uint32_t* hmeta)
{
    // allocated on invocation
    auto ti        = threadIdx.x;
    auto bi        = blockIdx.x;
    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;

    extern __shared__ char __buff[];

    auto __bw = reinterpret_cast<int*>(__buff);
    auto __h  = reinterpret_cast<Huff*>(__buff + chunksize * sizeof(int));

    auto stride = 1;

    Huff sym;
    int  bw = 0;
    if (bi * n_worker + ti < len) {
        auto gidx = chunksize * bi + ti;
        sym       = cb[in[gidx]];
        bw        = GetBitwidth(sym);
        __bw[ti]  = bw;
        sym       = sym << (sizeof(Huff) * 8 - bw);
    }
    else {
        __bw[ti] = 0;  // Make the "empty" spots zeros, so it won't affect the final result.
        sym      = 0;
    }

    // build sum in place up the tree
    for (auto d = n_worker >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (ti < d) {
            auto l = stride * (2 * ti + 1) - 1;
            auto r = stride * (2 * ti + 2) - 1;
            __bw[r] += __bw[l];
        }
        stride *= 2;
    }

    if (ti == 0) __bw[n_worker - 1] = 0;

    // clear the last element
    for (int d = 1; d < n_worker; d *= 2) {
        // traverse down tree & build scan
        stride >>= 1;
        __syncthreads();

        if (ti < d) {
            auto l  = stride * (2 * ti + 1) - 1;
            auto r  = stride * (2 * ti + 2) - 1;
            auto t  = __bw[l];
            __bw[l] = __bw[r];
            __bw[r] += t;
        }
        __syncthreads();
    }
    __syncthreads();

    if (bi * n_worker + ti < len) {
        auto loc       = __bw[ti];
        auto word_1    = loc / (sizeof(Huff) * 8);
        auto word_2    = (loc - 1) / (sizeof(Huff) * 8) + 1;
        auto extra_bit = loc % (sizeof(Huff) * 8);

        atomicOr(__h + word_1, sym >> extra_bit);
        if (word_2 == word_1) { atomicOr(__h + word_2, sym << (sizeof(Huff) * 8 - extra_bit)); }
        //        auto l        = __bw[ti] sym
    }
    __syncthreads();

    if (ti == n_worker - 1) __bw[n_worker - 1] += bw;

    if (ti < (__bw[n_worker - 1] - 1) / (sizeof(Huff) * 8) + 1) { out[bi * n_worker + ti] = __h[ti]; }
    __syncthreads();

    //    if (bi == 0 and ti == 0) {
    //        for (auto i = 0; i < (1 << Magnitude); i++) {
    //            printf("%u\t%d\n", i, __bw[i]);
    //        }
    //    }
}

/*
 * Input* in: quantization code
 * Huff* cb: Huffman codebook
 * TODO (msb) bitwdith (space) Huffman code (lsb)
 * TODO -> ReduceShuffleMerge
 * TODO -> chunksize == 2 * blockDim.x
 */
// template <typename Input, typename Huff, int Magnitude, int ReductionFactor, int ShuffleFactor>
//__global__ void ReduceShuffle_Butterfly(Input* in, size_t len, Huff* cb, Huff* out, size_t cb_len, uint32_t* hmeta,
// char* rich_dbg = nullptr, uint32_t dbg_bi = 3)
//{
//    static_assert(Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction)
//    factor"); static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1"); static_assert((2 <<
//    Magnitude) < 98304, "Shared memory used too much.");
//
//    extern __shared__ char __buff[];
//
//    auto n_worker  = blockDim.x;
//    auto chunksize = 1 << Magnitude;
//    auto __data    = reinterpret_cast<Huff*>(__buff);
//    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(Huff));
//    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
//    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
//    auto data_exc  = data_src;                // swap zone
//    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)
//
//    //    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
//    //    auto bw_exc    = bw_src;                  // swap zone
//
//    auto ti = threadIdx.x;
//    auto bi = blockIdx.x;
//
//    //// Reduction I: detach metadata and code; merge as much as possible
//    //// butterfly
//    ////////////////////////////////////////////////////////////////////////////////
//    auto first_stride = 1 << (Magnitude - 1);
//    // the same as non-butterfly
//    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
//        auto lidx = ti + n_worker * r;                         //
//        auto gidx = chunksize * bi + lidx;                     // to load from global memory
//        auto lsym = cb[in[gidx]];
//        auto rsym = cb[in[gidx + first_stride]];  // the next one
//        __syncthreads();
//
//        auto lbw = GetBitwidth(lsym);
//        auto rbw = GetBitwidth(rsym);
//        lsym <<= sizeof(Huff) * 8 - lbw;  // left aligned
//        rsym <<= sizeof(Huff) * 8 - rbw;  //
//
//        data_src[lidx] = lsym | (rsym >> lbw);  //
//        bw_src[lidx]   = lbw + rbw;             // sum bitwidths
//        //        if (bi == dbg_bi) printf("reduce=1; idx=(%u,%u);\tbw=(%u,%u);\n", lidx, lidx + first_stride, lbw,
//        rbw);
//    }
//    __syncthreads();
//
//#ifdef REDUCE1TIME
//    return;
//#endif
//
//    //// Reduction II: merge as much as possible
//    //// butterfly
//    ////////////////////////////////////////////////////////////////////////////////
//    auto stride = 1 << (Magnitude - 2);
//    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
//        auto repeat = 1 << rf;
//        for (auto r = 0; r < repeat; r++) {
//            auto lidx = ti + n_worker * r;
//            auto lsym = data_src[lidx];
//            auto rsym = data_src[lidx + stride];
//            auto lbw  = bw_src[lidx];
//            auto rbw  = bw_src[lidx + stride];
//
//            data_src[lidx] = lsym | (rsym >> lbw);
//            bw_src[lidx]   = lbw + rbw;  // sum bitwidths
//            __syncthreads();
//            //            if (bi == dbg_bi) printf("rf=%u; idx=(%u,%u)\tbw=(%u,%u)\n", rf, lidx, lidx + stride, lbw,
//            rbw);
//        }
//        stride /= 2;
//        __syncthreads();
//    }
//    __syncthreads();
//
//#ifdef REDUCE12TIME
//    return;
//#endif
//
//    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
//    //// butterfly
//    ////////////////////////////////////////////////////////////////////////////////
//    auto curr_team_halfsize = 1;
//    auto curr_league_size   = 1 << ShuffleFactor;
//    auto next_league_size   = curr_league_size / 2;
//
//    for (auto sf = ShuffleFactor; sf > 0; sf--, curr_team_halfsize *= 2, curr_league_size /= 2, next_league_size /= 2)
//    {
//        auto rank        = ti;
//        auto iter        = ShuffleFactor + 1 - sf;
//        auto next_rank   = NextButterflyIdx<ShuffleFactor>(rank, iter);
//        auto team_size   = 1 << (iter - 1);
//        auto league_size = 1 << (Magnitude - iter + 1);
//        auto league_rank = rank / team_size;
//
//        auto l_bw = bw_src[league_rank];
//        auto r_bw = bw_src[league_rank + league_size / 2];
//
//        unsigned int dtype_ofst  = l_bw / (sizeof(Huff) * 8);
//        unsigned int used_bits   = l_bw % (sizeof(Huff) * 8);
//        unsigned int unused_bits = sizeof(Huff) * 8 - used_bits;
//
//        // next team rank can be disrupted from previous snippet
//        auto leader = next_rank / (team_size * 2) * (team_size * 2);  // team leader
//        auto r      = leader + team_size;
//        auto l_end  = data_dst + leader + dtype_ofst;
//
//        auto this_point = data_src[rank];
//        /* experimental */ __syncthreads();
//
//        atomicAnd(data_src + rank, 0x0);  // meaning: data_src[ti] = 0;
//
//        if (next_rank >= leader and next_rank < leader + team_size) {
//            atomicOr(data_dst + next_rank, this_point);
//        }
//        if (next_rank >= r and next_rank < r + team_size) {  // whole right subgroup
//            auto _1st = this_point >> used_bits;
//            auto _2nd = this_point << unused_bits;
//            atomicOr(l_end + (next_rank - r) + 0, _1st);  // meaning: *(l_end+(next_rank-r)+0)=_1st;
//            atomicOr(l_end + (next_rank - r) + 1, _2nd);  // meaning: *(l_end+(next_rank-rank-r)+0)=_1st;
//        }
//
//        if (next_rank == leader) bw_src[league_rank] = l_bw + r_bw;  // very imbalanced
//
//        data_exc = data_src, data_src = data_dst, data_dst = data_exc;
//        /* necessary */ __syncthreads();
//    }
//    /* necessary */ __syncthreads();
//
//    //// end of reduce-shuffle
//    ////////////////////////////////////////////////////////////////////////////////
//    auto final_bw = bw_src[0];
//    if (ti == 0) hmeta[bi] = final_bw;
//    __syncthreads();
//
//#ifdef ALLMERGETIME
//    return;
//#endif
//
//    auto multiple_of_128B = (final_bw - 1) / (sizeof(Huff) * 8) + 1;
//    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
//    if (ti < multiple_of_128B) out[chunksize * bi + ti] = data_src[ti];
//    __syncthreads();
//}

template <typename Input, typename Dict, typename Huff, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void ReduceShuffle_PrefixSum(
    Input*    in,
    size_t    len,
    Dict*     cb,
    Huff*     out,
    size_t    cb_len,
    uint32_t* hmeta,
    char*     rich_dbg = nullptr,
    uint32_t  dbg_bi   = 3)
{
    static_assert(
        Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    auto __data    = reinterpret_cast<Huff*>(__buff);
    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(Huff));
    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    auto data_exc  = data_src;                // swap zone
    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)
    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
    auto bw_exc    = bw_src;                  // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = 2 * (ti + n_worker * r);                   // every two
        auto gidx = chunksize * bi + lidx;                     // to load from global memory

        auto lsym = cb[in[gidx]];
        auto rsym = cb[in[gidx + 1]];  // the next one
        __syncthreads();

        auto lbw = GetBitwidth(lsym);
        auto rbw = GetBitwidth(rsym);
        lsym <<= sizeof(Huff) * 8 - lbw;  // left aligned
        rsym <<= sizeof(Huff) * 8 - rbw;  //

        data_src[lidx >> 1] = lsym | (rsym >> lbw);  //
        bw_src[lidx >> 1]   = lbw + rbw;             // sum bitwidths
    }
    __syncthreads();

    //// Reduction II: merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = 2 * (ti + n_worker * r);
            auto lsym = data_src[lidx];
            auto rsym = data_src[lidx + 1];
            auto lbw  = bw_src[lidx];
            auto rbw  = bw_src[lidx + 1];

            data_dst[lidx >> 1] = lsym | (rsym >> lbw);
            bw_dst[lidx >> 1]   = lbw + rbw;  // sum bitwidths
            __syncthreads();
        }
        __syncthreads();
        data_exc = data_src, bw_exc = bw_src;
        data_src = data_dst, bw_src = bw_dst;
        data_dst = data_exc, bw_dst = bw_exc;
        __syncthreads();
    }
    __syncthreads();

    //    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
    //    ////////////////////////////////////////////////////////////////////////////////
    //    auto stride = 1;
    //    for (auto sf = ShuffleFactor; sf > 0; sf--, stride *= 2) {
    //        auto         l           = ti / (stride << 1) * (stride << 1);
    //        auto         r           = l + stride;
    //        auto         l_bw        = bw_src[l];
    //        unsigned int dtype_ofst  = l_bw / (sizeof(Huff) * 8);
    //        unsigned int used_bits   = l_bw % (sizeof(Huff) * 8);
    //        unsigned int unused_bits = sizeof(Huff) * 8 - used_bits;
    //        auto         l_end       = data_src + l + dtype_ofst;
    //
    //        auto this_point = data_src[ti];
    //        auto _1st       = this_point >> used_bits;
    //        auto _2nd       = this_point << unused_bits;
    //
    //        // ti in [r, r+stride) or ((r ..< r+stride )), whole right subgroup
    //        // &data_src[ ((r ..< r+stride)) ] have conflicts with (l_end+ ((0 ..< stride)) + 0/1)
    //        // because the whole shuffle subprocedure compresses at a factor of < 2
    //        if (ti >= r and ti < r + stride) {
    //            atomicAnd(data_src + ti, 0x0);  // meaning: data_src[ti] = 0;
    //        }
    //        /* experimental */ __syncthreads();
    //        if (ti >= r and ti < r + stride) {         // whole right subgroup
    //            atomicOr(l_end + (ti - r) + 0, _1st);  // meaning: *(l_end + (ti -r) + 0) = _1st;
    //            atomicOr(l_end + (ti - r) + 1, _2nd);  // meaning: *(l_end + (ti -r) + 0) = _1st;
    //        }
    //        ///* optional */ __syncthreads();
    //
    //        if (ti == l) bw_src[l] += bw_src[l + stride];  // very imbalanced
    //        /* necessary */ __syncthreads();
    //    }
    //    /* necessary */ __syncthreads();

    auto stride   = 1;
    auto __bw_alt = bw_src;
    auto __h      = data_src;

    Huff sym;
    int  bw = 0;
    //    if (bi * n_worker + ti < len) {
    sym = __h[ti];
    bw  = __bw_alt[ti];
    sym = sym << (sizeof(Huff) * 8 - bw);
    //    }
    //    else {
    //        __bw_alt[ti] = 0;  // Make the "empty" spots zeros, so it won't affect the final result.
    //        sym      = 0;
    //    }

    // build sum in place up the tree
    for (auto d = n_worker >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (ti < d) {
            auto l = stride * (2 * ti + 1) - 1;
            auto r = stride * (2 * ti + 2) - 1;
            __bw_alt[r] += __bw_alt[l];
        }
        stride *= 2;
    }

    if (ti == 0) __bw_alt[n_worker - 1] = 0;

    // clear the last element
    for (int d = 1; d < n_worker; d *= 2) {
        // traverse down tree & build scan
        stride >>= 1;
        __syncthreads();
        if (ti < d) {
            auto l      = stride * (2 * ti + 1) - 1;
            auto r      = stride * (2 * ti + 2) - 1;
            auto t      = __bw_alt[l];
            __bw_alt[l] = __bw_alt[r];
            __bw_alt[r] += t;
        }
        __syncthreads();
    }
    __syncthreads();

    auto loc       = __bw_alt[ti];
    auto word_1    = loc / (sizeof(Huff) * 8);
    auto word_2    = (loc - 1) / (sizeof(Huff) * 8) + 1;
    auto extra_bit = loc % (sizeof(Huff) * 8);

    atomicOr(__h + word_1, sym >> extra_bit);
    if (word_2 == word_1) { atomicOr(__h + word_2, sym << (sizeof(Huff) * 8 - extra_bit)); }
    //        auto l        = __bw_alt[ti] sym
    //    }
    __syncthreads();

    if (ti == n_worker - 1) __bw_alt[n_worker - 1] += bw;

    if (ti < (__bw_alt[n_worker - 1] - 1) / (sizeof(Huff) + 8) + 1) { out[bi * n_worker + ti] = __h[ti]; }
    __syncthreads();

    //// end of reduce-shuffle
    ////////////////////////////////////////////////////////////////////////////////
    auto final_bw = bw_src[0];
    if (ti == 0) hmeta[bi] = final_bw;
    __syncthreads();

    auto multiple_of_128B = (final_bw - 1) / (sizeof(Huff) * 8) + 1;
    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
    if (ti < multiple_of_128B) out[chunksize * bi + ti] = data_src[ti];
    __syncthreads();
}

/*
 * dry run to figure out the violating
 * thus we don't have to overkill by marking too many data and covert them to shorter avatars
 */
template <typename Input, typename Dict, typename Huff, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void TrackViolating(Input* in, size_t len, Dict* cb, int* outlier_num = nullptr)
{
    static_assert(
        Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    // auto __data    = reinterpret_cast<Huff*>(__buff);

    auto __bw = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(Huff));
    // auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    // auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    // auto data_exc  = data_src;                // swap zone
    auto bw_src = __bw;                  // 1st bw zone of (chunksize/2)
    auto bw_dst = __bw + chunksize / 2;  // 2nd bw zone of (chunksize/4)
    auto bw_exc = bw_src;                // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = 2 * (ti + n_worker * r);                   // every two
        auto gidx = chunksize * bi + lidx;                     // to load from global memory

        auto lsym = cb[in[gidx]];
        auto rsym = cb[in[gidx + 1]];  // the next one
        __syncthreads();

        auto lbw = GetBitwidth(lsym);
        auto rbw = GetBitwidth(rsym);
        // lsym <<= sizeof(Huff) * 8 - lbw;  // left aligned
        // rsym <<= sizeof(Huff) * 8 - rbw;  //
        // data_src[lidx >> 1] = lsym | (rsym >> lbw);  //
        bw_src[lidx >> 1] = lbw + rbw;  // sum bitwidths
    }
    __syncthreads();

    //// Reduction II: merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = 2 * (ti + n_worker * r);
            // auto lsym = data_src[lidx];
            // auto rsym = data_src[lidx + 1];
            auto lbw = bw_src[lidx];
            auto rbw = bw_src[lidx + 1];
            // data_dst[lidx >> 1] = lsym | (rsym >> lbw);
            bw_dst[lidx >> 1] = lbw + rbw;  // sum bitwidths
            __syncthreads();
        }
        __syncthreads();

        bw_exc = bw_src;
        bw_src = bw_dst;
        bw_dst = bw_exc;
        __syncthreads();
    }
    __syncthreads();

    if (bw_src[ti] > 32) atomicAdd(outlier_num, 1 << ReductionFactor);
}

__global__ void ReadViolating(int* outlier_num, size_t len)
{
    printf("the violating number is %d, %lf %\n", outlier_num[0], outlier_num[0] * 1.0 / len);
}

/*
 * Input* in: quantization code
 * Huff* cb: Huffman codebook
 * TODO (msb) bitwdith (space) Huffman code (lsb)
 * TODO -> ReduceShuffleMerge
 * TODO -> chunksize == 2 * blockDim.x
 */

/**
 * @brief Reduction for ReductionFactor times, and shuffle-merge for ShuffleFactor times
      input
 * @tparam Input
 * @tparam Dict
 * @tparam Huff
 * @tparam Magnitude
 * @tparam ReductionFactor
 * @tparam ShuffleFactor
 * @param in
 * @param len
 * @param cb
 * @param out
 * @param cb_len
 * @param hmeta
 * @param rich_dbg
 * @param dbg_bi
 * @return __global__
 */
template <typename Input, typename Dict, typename Huff, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void ReduceShuffle(
    Input*    in,
    size_t    len,
    Dict*     cb,
    Huff*     out,
    size_t    cb_len,
    uint32_t* hmeta,
    char*     rich_dbg = nullptr,
    uint32_t  dbg_bi   = 3)
{
    static_assert(
        Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    auto __data    = reinterpret_cast<Huff*>(__buff);
    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(Huff));
    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    auto data_exc  = data_src;                // swap zone
    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)
    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
    auto bw_exc    = bw_src;                  // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = 2 * (ti + n_worker * r);                   // every two
        auto gidx = chunksize * bi + lidx;                     // to load from global memory

        auto lsym = cb[in[gidx]];
        auto rsym = cb[in[gidx + 1]];  // the next one
        __syncthreads();

        auto lbw = GetBitwidth(lsym);
        auto rbw = GetBitwidth(rsym);
        lsym <<= sizeof(Dict) * 8 - lbw;  // left aligned
        rsym <<= sizeof(Dict) * 8 - rbw;  //

        data_src[lidx >> 1] = lsym | (rsym >> lbw);  //
        bw_src[lidx >> 1]   = lbw + rbw;             // sum bitwidths

#ifdef DBG
        if (bi == dbg_bi) {
            printf(
                "blkid=%u; idx=%4u; first_reduce=%u; bw=(%2u, %2u)=%u; sym=(%u, %u); merged=(%u)\n",  //
                dbg_bi, _1st_idx, r, _1st_bw, _2nd_bw, (_1st_bw + _2nd_bw), _1st, _2nd, (_1st | (_2nd >> _1st_bw)));
        }
        __syncthreads();
#endif
    }
    __syncthreads();

#ifdef REDUCE1TIME
    return;
#endif

    //// Reduction II: merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = 2 * (ti + n_worker * r);
            auto lsym = data_src[lidx];
            auto rsym = data_src[lidx + 1];
            auto lbw  = bw_src[lidx];
            auto rbw  = bw_src[lidx + 1];
#ifdef DBG
            if (bi == dbg_bi)
                printf(
                    "blkid=%u; idx=%4u; loop=%u; repeat=%u; bw=(%2u, %2u)=%2u; sym=(%u, %u); merged=(%u)\n",  //
                    dbg_bi, lidx, rf, r, lbw, rbw, (lbw + rbw), lsym, rsym, (lsym | (rsym >> lbw)));
#endif

            data_dst[lidx >> 1] = lsym | (rsym >> lbw);
            bw_dst[lidx >> 1]   = lbw + rbw;  // sum bitwidths
            __syncthreads();
        }
        __syncthreads();
        data_exc = data_src, bw_exc = bw_src;
        data_src = data_dst, bw_src = bw_dst;
        data_dst = data_exc, bw_dst = bw_exc;
        __syncthreads();
    }
    __syncthreads();

#ifdef REDUCE12TIME
    return;
#endif

#ifdef DBG
    if (bi == dbg_bi and ti == 0) {
        for (auto ii = 0; ii < (1 << ShuffleFactor); ii++)
            if (data_src[ii] != 0x0)
                printf("idx=%3u;\tbw=%3u; code=0b", ii, bw_src[ii]), echo_bitset<Huff, 32>(data_src[ii]);
        printf("\n");
    }
    __syncthreads();
#endif

    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
    ////////////////////////////////////////////////////////////////////////////////
    auto stride = 1;
    for (auto sf = ShuffleFactor; sf > 0; sf--, stride *= 2) {
        auto         l           = ti / (stride << 1) * (stride << 1);
        auto         r           = l + stride;
        auto         l_bw        = bw_src[l];
        unsigned int dtype_ofst  = l_bw / (sizeof(Huff) * 8);
        unsigned int used_bits   = l_bw % (sizeof(Huff) * 8);
        unsigned int unused_bits = sizeof(Huff) * 8 - used_bits;
        auto         l_end       = data_src + l + dtype_ofst;

        auto this_point = data_src[ti];
        auto _1st       = this_point >> used_bits;
        auto _2nd       = this_point << unused_bits;

#ifdef dgb
        if (bi == dbg_bi and ti == 0)
            for (auto ii = 0; ii < (1 << ShuffleFactor); ii++) {
                printf("sf_iter=%u; data_idx=%u\t", sf, ii);
                if (data_src[ii] != 0x0)
                    printf("bw=%3u; code=", bw_src[ii]), echo_bitset<Huff, 32>(data_src[ii]);
                else
                    printf("bw=nan; code="), printf("0x0                             \n");
            }
        __syncthreads();
#endif

        // ti in [r, r+stride) or ((r ..< r+stride )), whole right subgroup
        // &data_src[ ((r ..< r+stride)) ] have conflicts with (l_end+ ((0 ..< stride)) + 0/1)
        // because the whole shuffle subprocedure compresses at a factor of < 2
        if (ti >= r and ti < r + stride) {
            atomicAnd(data_src + ti, 0x0);  // meaning: data_src[ti] = 0;
        }
        /* experimental */ __syncthreads();
        if (ti >= r and ti < r + stride) {         // whole right subgroup
            atomicOr(l_end + (ti - r) + 0, _1st);  // meaning: *(l_end + (ti -r) + 0) = _1st;
            atomicOr(l_end + (ti - r) + 1, _2nd);  // meaning: *(l_end + (ti -r) + 0) = _1st;
        }
        ///* optional */ __syncthreads();

        if (ti == l) bw_src[l] += bw_src[l + stride];  // very imbalanced
        /* necessary */ __syncthreads();
    }
    /* necessary */ __syncthreads();

    //// end of reduce-shuffle
    ////////////////////////////////////////////////////////////////////////////////
    auto final_bw = bw_src[0];  // create private var
    if (ti == 0) hmeta[bi] = final_bw;
    __syncthreads();

#ifdef ALLMERGETIME
    return;
#endif

    auto multiple_of_128B = (final_bw - 1) / (sizeof(Huff) * 8) + 1;
    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
    if (ti < multiple_of_128B) out[chunksize * bi + ti] = data_src[ti];
    __syncthreads();

    /*
    if (bi == dbg_bi and ti == 0) {
        printf("dbg blk id:%u (printing)\n", dbg_bi);
        for (auto i = 0; i < multiple_of_128B; i++) echo_bitset<uint32_t, 32>(out[chunksize * bi + i]);
    }
    */

    // we can
    // g.sync();
    // prefix sum

    // copy dense Huffman codes to global memory
    // auto n_of_128 = (final_bw / 8 - 1) / 128 + 1; // if we go warp ops
}

// |_0___________||_1___________||_2___________|
// |xxxxx________||xxxxxxxx_____||xxxxxx_______|
//  ^              ^
//  |              |
//  |    ^
//  | #0 | #1
//  2044 (4)       3070 (2)       2040 (8)
//  16             23.7 -> 24     15.7
// dense_h array: storing huffman bitstream

/*
 * Input* q: quantization code
 * Huff* cb: Huffman codebook
 * TODO (msb) bitwdith (space) Huffman code (lsb)
 * TODO -> ReduceShuffleMerge
 * TODO -> chunksize == 2 * blockDim.x
 */
template <typename Input, typename Huff, typename Dict, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void ReduceShuffle_fixcodebook(
    Input*    q,
    size_t    len,
    Dict*     cb,
    Huff*     h,
    size_t    cb_len,
    uint32_t* hmeta,
    char*     rich_dbg = nullptr,
    uint32_t  dbg_bi   = 3)
{
    static_assert(
        Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    auto __data    = reinterpret_cast<Huff*>(__buff);
    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(Huff));
    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    auto data_exc  = data_src;                // swap zone
    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)
    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
    auto bw_exc    = bw_src;                  // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = 2 * (ti + n_worker * r);                   // every two
        auto gidx = chunksize * bi + lidx;                     // to load from global memory

        auto lsym = cb[q[gidx]];
        auto rsym = cb[q[gidx + 1]];  // the next one
        __syncthreads();

        auto lbw = get_bw(lsym);
        auto rbw = get_bw(rsym);
        lsym <<= sizeof(Huff) * 8 - lbw;  // left aligned
        rsym <<= sizeof(Huff) * 8 - rbw;  //

        data_src[lidx >> 1] = lsym | (rsym >> lbw);  //
        bw_src[lidx >> 1]   = lbw + rbw;             // sum bitwidths

#ifdef DBG
        if (bi == dbg_bi) {
            printf(
                "blkid=%u; idx=%4u; first_reduce=%u; bw=(%2u, %2u)=%u; sym=(%u, %u); merged=(%u)\n",  //
                dbg_bi, _1st_idx, r, _1st_bw, _2nd_bw, (_1st_bw + _2nd_bw), _1st, _2nd, (_1st | (_2nd >> _1st_bw)));
        }
        __syncthreads();
#endif
    }
    __syncthreads();

#ifdef REDUCE1TIME
    return;
#endif

    //// Reduction II: merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = 2 * (ti + n_worker * r);
            auto lsym = data_src[lidx];
            auto rsym = data_src[lidx + 1];
            auto lbw  = bw_src[lidx];
            auto rbw  = bw_src[lidx + 1];
#ifdef DBG
            if (bi == dbg_bi)
                printf(
                    "blkid=%u; idx=%4u; loop=%u; repeat=%u; bw=(%2u, %2u)=%2u; sym=(%u, %u); merged=(%u)\n",  //
                    dbg_bi, lidx, rf, r, lbw, rbw, (lbw + rbw), lsym, rsym, (lsym | (rsym >> lbw)));
#endif

            data_dst[lidx >> 1] = lsym | (rsym >> lbw);
            bw_dst[lidx >> 1]   = lbw + rbw;  // sum bitwidths
            __syncthreads();
        }
        __syncthreads();
        data_exc = data_src, bw_exc = bw_src;
        data_src = data_dst, bw_src = bw_dst;
        data_dst = data_exc, bw_dst = bw_exc;
        __syncthreads();
    }
    __syncthreads();

#ifdef REDUCE12TIME
    return;
#endif

#ifdef DBG
    if (bi == dbg_bi and ti == 0) {
        for (auto ii = 0; ii < (1 << ShuffleFactor); ii++)
            if (data_src[ii] != 0x0)
                printf("idx=%3u;\tbw=%3u; code=0b", ii, bw_src[ii]), echo_bitset<Huff, 32>(data_src[ii]);
        printf("\n");
    }
    __syncthreads();
#endif

    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
    ////////////////////////////////////////////////////////////////////////////////
    auto stride = 1;
    for (auto sf = ShuffleFactor; sf > 0; sf--, stride *= 2) {
        auto         l           = ti / (stride << 1) * (stride << 1);
        auto         r           = l + stride;
        auto         l_bw        = bw_src[l];
        unsigned int dtype_ofst  = l_bw / (sizeof(Huff) * 8);
        unsigned int used_bits   = l_bw % (sizeof(Huff) * 8);
        unsigned int unused_bits = sizeof(Huff) * 8 - used_bits;
        auto         l_end       = data_src + l + dtype_ofst;

        auto this_point = data_src[ti];
        auto _1st       = this_point >> used_bits;
        auto _2nd       = this_point << unused_bits;

#ifdef dgb
        if (bi == dbg_bi and ti == 0)
            for (auto ii = 0; ii < (1 << ShuffleFactor); ii++) {
                printf("sf_iter=%u; data_idx=%u\t", sf, ii);
                if (data_src[ii] != 0x0)
                    printf("bw=%3u; code=", bw_src[ii]), echo_bitset<Huff, 32>(data_src[ii]);
                else
                    printf("bw=nan; code="), printf("0x0                             \n");
            }
        __syncthreads();
#endif

        // ti in [r, r+stride) or ((r ..< r+stride )), whole right subgroup
        // &data_src[ ((r ..< r+stride)) ] have conflicts with (l_end+ ((0 ..< stride)) + 0/1)
        // because the whole shuffle subprocedure compresses at a factor of < 2
        if (ti >= r and ti < r + stride) {
            atomicAnd(data_src + ti, 0x0);  // meaning: data_src[ti] = 0;
        }
        /* experimental */ __syncthreads();
        if (ti >= r and ti < r + stride) {         // whole right subgroup
            atomicOr(l_end + (ti - r) + 0, _1st);  // meaning: *(l_end + (ti -r) + 0) = _1st;
            atomicOr(l_end + (ti - r) + 1, _2nd);  // meaning: *(l_end + (ti -r) + 0) = _1st;
        }
        ///* optional */ __syncthreads();

        if (ti == l) bw_src[l] += bw_src[l + stride];  // very imbalanced
        /* necessary */ __syncthreads();
    }
    /* necessary */ __syncthreads();

    //// end of reduce-shuffle
    ////////////////////////////////////////////////////////////////////////////////
    auto final_bw = bw_src[0];
    if (ti == 0) hmeta[bi] = final_bw;
    __syncthreads();

#ifdef ALLMERGETIME
    return;
#endif

    auto multiple_of_128B = (final_bw - 1) / (sizeof(Huff) * 8) + 1;
    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
    if (ti < multiple_of_128B) h[chunksize * bi + ti] = data_src[ti];
    __syncthreads();

    /*
    if (bi == dbg_bi and ti == 0) {
        printf("dbg blk id:%u (printing)\n", dbg_bi);
        for (auto i = 0; i < multiple_of_128B; i++) echo_bitset<uint32_t, 32>(h[chunksize * bi + i]);
    }
    */

    // we can
    // g.sync();
    // prefix sum

    // copy dense Huffman codes to global memory
    // auto n_of_128 = (final_bw / 8 - 1) / 128 + 1; // if we go warp ops
}
