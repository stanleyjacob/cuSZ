/**
 * @file butterfly.cuh
 * @author Jiannan Tian
 * @brief Butterfly index
 * @version 0.1.3
 * @date 2020-10-29
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef BUTTERFLY_CUH
#define BUTTERFLY_CUH

template <int Magnitude>
__forceinline__ __device__ unsigned int ReverseButterflyIdx(unsigned int idx)
{
    auto reversed_idx = 0x0u;
    for (auto i = 0; i < Magnitude; i++) {
        auto tmp = idx & (0x1u << i);
        tmp      = (tmp >> i) << (Magnitude - 1 - i);
        reversed_idx |= tmp;
    }
    return reversed_idx;
}

template <int Magnitude>
__forceinline__ __device__ unsigned int NextButterflyIdx(unsigned int idx, unsigned int iter)
{
    auto team_size        = 1 << (iter - 1);
    auto next_team_size   = 2 * team_size;
    auto league_size      = 1 << (Magnitude - iter + 1);
    auto next_league_size = league_size / 2;

    auto team_rank           = idx % team_size;
    auto league_rank         = idx / team_size;
    auto next_subleague_rank = league_rank / next_league_size;
    auto next_league_rank    = league_rank % next_league_size;
    auto next_rank           = next_league_rank * next_team_size + next_subleague_rank * team_size + team_rank;

    return next_rank;
}

#endif