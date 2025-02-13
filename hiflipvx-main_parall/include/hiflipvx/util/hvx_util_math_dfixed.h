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
 * @file    hvx_util_math_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_MATH_DFIXED_H_
#define HVX_UTIL_MATH_DFIXED_H_

#include "hvx_util_math_int.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief Fixed-point square root function (with underflow)
 */
template<typename src_type_,
         int64_t acc_shift,
         hvx::util::underflow_e underflow_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>                      = true,
         std::enable_if_t<std::is_integral<typename src_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
DfixedSqrt(uint32_t val) noexcept -> uint32_t {
    HVX_INLINE_TOP();

    constexpr auto sqrt_out_digits = static_cast<uint8_t>((acc_shift + src_type_::digits) / 2);
    uint32_t res                   = 0;
#if defined(__SYNTHESIS__) // TODO: extract vendor/tool dependent code
    res = hls::sqrt(val);
#else
    hvx::util::IntSqrt<uint32_t, uint32_t, false, underflow_, sqrt_out_digits>(val, res);
#endif
    return res;
}

/*!
 * @brief Casts fixed-point to fixed-point (with underflow)
 */
template<int64_t src_frac_bits_, int64_t dst_frac_bits_, hvx::util::underflow_e underflow_>
HVX_FORCE_INLINE constexpr auto
CastFixedToFixed(const int64_t src) noexcept -> int64_t {
    HVX_INLINE_TOP();

    constexpr int64_t res_shift = hvx::util::Abs(src_frac_bits_ - dst_frac_bits_);
    constexpr int64_t round     = num_e::k1 << (hvx::util::Max(res_shift, static_cast<int64_t>(1)) - num_e::k1);
    return (src_frac_bits_ > dst_frac_bits_)
             ? ((underflow_ == hvx::util::underflow_e::kTrunc) ? (src >> res_shift) : ((src + round) >> res_shift))
             : (src << res_shift);
}

/*!
 * @brief Casts the smaller fixed-point value of two fixed-point values to the bigger one
 */
template<int64_t param1_frac_bits_, int64_t param2_frac_bits_>
HVX_FORCE_INLINE constexpr auto
Cast2FixedToBigger(int64_t& param1, int64_t& param2) noexcept -> void {
    HVX_INLINE_TOP();

    constexpr int64_t shift = hvx::util::Abs(param1_frac_bits_ - param2_frac_bits_);
    if (param1_frac_bits_ > param2_frac_bits_) {
        param2 = param2 << shift;
    } else {
        param1 = param1 << shift;
    }
}

/*!
 * @brief Casts the smaller fixed-point value of two fixed-point values to the bigger one
 */
template<int64_t param1_frac_bits_, int64_t param2_frac_bits_>
HVX_FORCE_INLINE constexpr auto
Cast3FixedToBigger(int64_t& param1, int64_t& param2a, int64_t& param2b) noexcept -> void {
    HVX_INLINE_TOP();

    constexpr int64_t shift = hvx::util::Abs(param1_frac_bits_ - param2_frac_bits_);
    if (param1_frac_bits_ > param2_frac_bits_) {
        param2a = param2a << shift;
        param2b = param2b << shift;
    } else {
        param1 = param1 << shift;
    }
}

/*!
 * @brief Casts from floating-point to fixed-point (with underflow)
 */
// template<typename dst_type_, hvx::util::underflow_e underflow_type_, std::enable_if_t<std::is_integral<dst_type_>::value, bool> = true>
// HVX_FORCE_INLINE constexpr auto
// CastFltToFixed(const float src, const float shift) noexcept -> dst_type_ {
//     HVX_INLINE_TOP();
//
//     // convert flt to fixed
//     const float data = src * shift;
//
//     // cast with underflow check
//     return hvx::util::CastFlt<dst_type_, underflow_type_>(data);
// }

template<typename src_type_, std::enable_if_t<std::is_integral<typename src_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
CastDfixedToFlt(src_type_ src) noexcept -> float {
    HVX_INLINE_TOP();
    return hvx::util::FltRightShift<src_type_::frac_bits>(static_cast<float>(src.data));
}

template<typename dst_type_,
         int64_t frac_bits_,
         hvx::util::underflow_e underflow_type_,
         std::enable_if_t<std::is_integral<dst_type_>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
CastFltToFixed(const float src) noexcept -> dst_type_ {
    HVX_INLINE_TOP();

    // convert flt to fixed
    const float data = hvx::util::FltLeftShift<frac_bits_>(src);

    // cast with underflow check
    return hvx::util::CastFlt<dst_type_, underflow_type_>(data);
}

/*!
 * @brief Checks if the fraction bit width is correct (when integer data type)
 */
template<typename type_, uint8_t fraction_bit_width_>
HVX_FORCE_INLINE constexpr auto
CheckFracBitWidth() noexcept -> bool {
    HVX_INLINE_TOP();
    return fraction_bit_width_ <= 32 && (fraction_bit_width_ <= std::numeric_limits<type_>::digits || !std::is_integral<type_>::value);
}

/*!
 * @brief Round a fixed number, which needs to be shifted
 */
template<uint8_t shift_bits_, hvx::util::underflow_e round_policy_, bool is_signed_ = true>
HVX_FORCE_INLINE constexpr auto
FixedUnderflow(const int64_t input) noexcept -> int64_t {
    HVX_INLINE_TOP();

    // computes 0.5
    constexpr int64_t round = static_cast<int64_t>(1) << (hvx::util::Max(shift_bits_, static_cast<uint8_t>(1)) - static_cast<uint8_t>(1));

    // check if value can get negative
    const bool is_negative = is_signed_ && (input < 0);

    // calculate the remainder
    const bool remainder = input & ((1 << shift_bits_) - 1);

    // stores the final result
    int64_t result = 0;

    // shift and apply the underflow policy
    switch (round_policy_) {
        case hvx::util::underflow_e::kTrunc: {
            result = input >> shift_bits_;
            if (is_negative)
                result += 1;
            break;
        }
        case hvx::util::underflow_e::kCeil: {
            result = input >> shift_bits_;
            if (remainder != 0)
                result += 1;
            break;
        }
        case hvx::util::underflow_e::kFloor: {
            result = input >> shift_bits_;
            break;
        }
        case hvx::util::underflow_e::kRound: {
            result = (input + round) >> shift_bits_;
            break;
        }
        default:
            break;
    }

    return result;
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_MATH_DFIXED_H_
