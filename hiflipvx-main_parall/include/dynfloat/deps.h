#pragma once

// STL
#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <ratio>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

#if defined(__GNUC__) || defined(__clang__)
#define DYNFLOAT_FORCE_INLINE __attribute((always_inline)) inline
#elif defined(_MSC_VER)
#define DYNFLOAT_FORCE_INLINE __forceinline
#else
#define DYNFLOAT_FORCE_INLINE inline
#endif

#if defined(__SYNTHESIS__)
#define DYNFLOAT_XILINX_SYNTHESIS
#include <hls_math.h>
#endif

namespace dynfloat
{
namespace xilinx
{
static constexpr std::int64_t dsp_input_sizes[] = {18};        // NOLINT
static constexpr std::int64_t bram_sizes[]      = {9, 18, 36}; // NOLINT
} // namespace xilinx
} // namespace dynfloat

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127) // warning C4127: conditional expression is constant
#endif
