/**
 * @file autotune.hh
 * @author Jiannan Tian
 * @brief called as if a plain CPU function
 * @version 0.1.3
 * @date 2020-11-03
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef AUTOTUNE_H
#define AUTOTUNE_H

#include <stddef.h>

namespace cusz {
namespace tune {
size_t GetCUDACoreNum();
}
}  // namespace cusz

#endif