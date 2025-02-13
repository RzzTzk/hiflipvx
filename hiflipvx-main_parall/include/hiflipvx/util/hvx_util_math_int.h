/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_util_math_int.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_MATH_INT_H_
#define HVX_UTIL_MATH_INT_H_

#include "hvx_util_math_flt.h"

namespace hvx {
namespace util {

/******************************************************************************************************************************************/

/*!
 * @brief Cast flt (to int type) with underflow policy
 */
template<typename dst_type_,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         typename src_type_,
         std::enable_if_t<std::is_integral<dst_type_>::value, bool>       = true,
         std::enable_if_t<std::is_floating_point<src_type_>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
CastFlt(const src_type_ src) noexcept -> dst_type_ {
    HVX_INLINE_TOP();

#if defined(HVX_SYNTHESIS_ACTIVE)
    switch (underflow_type_) {
        case hvx::util::underflow_e::kCeil:
            return static_cast<dst_type_>(hls::ceilf(src));
        case hvx::util::underflow_e::kFloor:
            return static_cast<dst_type_>(hls::floorf(src));
        case hvx::util::underflow_e::kRound:
            return static_cast<dst_type_>(hls::roundf(src));
        case hvx::util::underflow_e::kTrunc:
            return static_cast<dst_type_>(hls::truncf(src));
        default:
            return static_cast<dst_type_>(src);
    }
#else
    switch (underflow_type_) {
        case hvx::util::underflow_e::kCeil:
            return static_cast<dst_type_>(ceilf(src));
        case hvx::util::underflow_e::kFloor:
            return static_cast<dst_type_>(floorf(src));
        case hvx::util::underflow_e::kRound:
            return static_cast<dst_type_>(roundf(src));
        case hvx::util::underflow_e::kTrunc:
            return static_cast<dst_type_>(truncf(src));
        default:
            return static_cast<dst_type_>(src);
    }
#endif
}

/*!
 * @brief Cast flt (to flt type)
 */
template<typename dst_type_,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         typename src_type_,
         std::enable_if_t<std::is_floating_point<dst_type_>::value, bool> = true,
         std::enable_if_t<std::is_floating_point<src_type_>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
CastFlt(const src_type_ src) noexcept -> dst_type_ {
    HVX_INLINE_TOP();
    return static_cast<dst_type_>(src);
}

/*!
 * @brief Cast int to int type with overflow policy
 */
template<typename dst_type_,
         hvx::util::overflow_e overflow_type_ = hvx::util::overflow_e::kWrap,
         typename src_type_,
         std::enable_if_t<std::is_integral<dst_type_>::value, bool> = true,
         std::enable_if_t<std::is_integral<src_type_>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
CastIntToInt(const src_type_ src) noexcept -> dst_type_ {
    HVX_INLINE_TOP();
    return static_cast<dst_type_>(
        (overflow_type_ == hvx::util::overflow_e::kSaturate)
            ? (hvx::util::Clamp(static_cast<int64_t>(src), static_cast<int64_t>(std::numeric_limits<dst_type_>::lowest()),
                                static_cast<int64_t>(std::numeric_limits<dst_type_>::max())))
            : (src));
}

/*!
 * @brief Computes square root: Rounding to floor or to nearest integer
 * @tparam src_type_      The data type of the input
 * @tparam dst_type_      The data type of the output
 * @tparam check_max_     Is true, if the maximum value should be checked (needed for rounding to nearest even)
 * @tparam underflow_type_  Rounding policy (to zero & nearest even)
 * @tparam dst_bit_width_ bit width of the output
 * @param src             The input value
 * @param dst             The square root of the input value
 */
template<typename src_type_, typename dst_type_, bool check_max_, hvx::util::underflow_e underflow_type_, uint8_t dst_bit_width_>
HVX_FORCE_INLINE constexpr auto
IntSqrt(const src_type_ src, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    constexpr dst_type_ MAX_VAL = std::numeric_limits<dst_type_>::max();

    // number of stages (latency)
    constexpr auto N = dst_bit_width_; // sizeof(DstType) * 8;

    // variables
    auto A1 = dst_type_{}; // A^1 Intermediate result
    auto A2 = src_type_{}; // A^2 Square of the intermediate result

    // each stage computes 1 bit of the resulting vector
    for (auto n = N - 1u; n < N; --n) {
        HVX_UNROLL();

        // Add new bit of position n and compute (A1 + B1)^2
        const auto B1      = static_cast<dst_type_>(1) << n;
        const auto B2      = static_cast<src_type_>(B1) << n;
        const auto AB      = static_cast<src_type_>(A1) << n;
        const auto A2_next = A2 + B2 + (AB << 1); // A*A + B*B + 2*A*B

        // store if tmp does not exceed input
        if (A2_next <= src) {
            A1 |= B1;
            A2 = A2_next;
        }
    }

    // round to the nearest integer and check for overflow
    if (underflow_type_ == hvx::util::underflow_e::kRound) {
        if (check_max_) {
            if (src - A2 > static_cast<src_type_>(A1) && A1 != MAX_VAL)
                ++A1;
        } else {
            if (src - A2 > static_cast<src_type_>(A1))
                ++A1;
        }
    }

    // return result
    dst = A1;
}

/*!
 * @brief   computes the ATAN2(x,y) using the cordic algorithm
 * @param x x-vector
 * @param y y_vector
 * @return  atan2(x,y)
 */
HVX_FORCE_INLINE constexpr auto
IntAtan2(int64_t x, int64_t y) noexcept -> int64_t {
    HVX_INLINE_TOP();

    // table of fixed angles
    constexpr std::array<int64_t, 32> i_atantab = {1073741824, 536870912, 316933406, 167458907, 85004756, 42667331, 21354465, 10679838,
                                                   5340245,    2670163,   1335087,   667544,    333772,   166886,   83443,    41722,
                                                   20861,      10430,     5215,      2608,      1304,     652,      326,      163,
                                                   81,         41,        20,        10,        5,        3,        1};

    // Initalization
    int64_t sgn   = (y >= 0) ? -1 : 1;
    int64_t xh    = -sgn * y;
    int64_t yh    = +sgn * x;
    int64_t angle = sgn * i_atantab[0];

    x = xh;
    y = yh;

    // Iteration maximum: i<31
    for (uint16_t i = 1, k = 0; i <= 24; i++, k++) {
        HVX_UNROLL();

        sgn = y >= 0 ? -1 : 1;
        angle += sgn * i_atantab.at(i);
        xh = x - sgn * (y >> k);
        yh = y + sgn * (x >> k);
        x  = xh;
        y  = yh;
        if (y == 0)
            break;
    }

    return -angle;
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_MATH_INT_H_
