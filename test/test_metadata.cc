/**
 * @file test_metadata.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-10-05
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "../src/metadata.hh"

const int _2d = 16;
const int _3d = 8;

struct Metadata<_2d>* m2d;
struct Metadata<_3d>* m3d;

TEST(Metadata, BlockSize)
{
    ASSERT_EQ(134217728, m3d->len);
    ASSERT_EQ(1, m3d->stride0);
    ASSERT_EQ(512, m3d->stride1);
    ASSERT_EQ(512 * 512, m3d->stride2);
    ASSERT_EQ(512 * 512 * 512, m3d->stride3);
    ASSERT_EQ(512 / _3d, m3d->nb0);
    ASSERT_EQ(512 / _3d, m3d->nb1);
    ASSERT_EQ(512 / _3d, m3d->nb2);
    ASSERT_EQ(1, m3d->nb3);
    // ASSERT_EQ(134217728, m3d->len);
    // ASSERT_EQ(134217728, m3d->len);
}

// TEST(MetaData, BlockSize)
// {
//     ASSERT_EQ(3, sub(5, 2));
//     ASSERT_EQ(-10, sub(5, 15));
// }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    m3d = new struct Metadata<_3d>();
    cuszSetDim(m3d, 3, 512, 512, 512, 1);

    return RUN_ALL_TESTS();
}