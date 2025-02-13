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
 * @file    hvx_nn_conv_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_CONV_DFIXED_H_
#define HVX_NN_CONV_DFIXED_H_

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace nn {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief verifies the dimensions of all src and dst
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>       = true,
         std::enable_if_t<hvx::util::is_dfixed_v<wgts_type_>, bool>      = true,
         std::enable_if_t<hvx::util::is_dfixed_v<bias_type_>, bool>      = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool>       = true,
         std::enable_if_t<src_type_::is_flt == wgts_type_::is_flt, bool> = true,
         std::enable_if_t<src_type_::is_flt == bias_type_::is_flt, bool> = true,
         std::enable_if_t<src_type_::is_flt == dst_type_::is_flt, bool>  = true>
HVX_FORCE_INLINE constexpr auto
ConvVerifyType() noexcept -> void {
    HVX_INLINE_TOP();

    // compile time assertions
    static_assert(
        hvx::util::CompareDataType<typename src_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>() &&
            hvx::util::CompareDataType<typename wgts_type_::data_type, uint8_t, uint16_t, uint32_t, int8_t, int16_t, int32_t, float>() &&
            hvx::util::CompareDataType<typename bias_type_::data_type, uint8_t, uint16_t, uint32_t, int8_t, int16_t, int32_t, float>() &&
            hvx::util::CompareDataType<typename dst_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>(),
        "Data type is not supported for dfixed!");
    static_assert(hvx::util::CheckFracBitWidth<typename src_type_::data_type, src_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename wgts_type_::data_type, wgts_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename bias_type_::data_type, bias_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename dst_type_::data_type, dst_type_::frac_bits>(),
                  "Fraction size out of scope!");
}

/*!
 * @brief comp global sum, add bias and apply overflow/underflow policies (using fixed-point numbers)
 */
template<typename param_,
         std::enable_if_t<param_::src_type::is_int, bool>  = true,
         std::enable_if_t<param_::wgts_type::is_int, bool> = true,
         std::enable_if_t<param_::bias_type::is_int, bool> = true,
         std::enable_if_t<param_::dst_type::is_int, bool>  = true>
HVX_FORCE_INLINE constexpr auto
ConvAddBias(int64_t src_chnl_v,
            int64_t sum_local,
            typename param_::comp_type& sum_global,
            typename param_::bias_type::data_type bias) noexcept -> int64_t {
    HVX_INLINE_TOP();

    // fraction sizes
    constexpr int32_t bias_frac_bits = param_::bias_type::frac_bits;
    constexpr int32_t dst_frac_bits  = param_::dst_type::frac_bits;
    constexpr int32_t sum_frac_bits  = param_::src_type::frac_bits + param_::wgts_type::frac_bits;
    constexpr int32_t res_frac_bits  = hvx::util::Max(sum_frac_bits, bias_frac_bits);
    constexpr int32_t dst_shift      = hvx::util::Abs(res_frac_bits - dst_frac_bits);
    constexpr int32_t res_shift      = hvx::util::Abs(sum_frac_bits - bias_frac_bits);

    // get bias and update global summation
    const auto bias_t       = static_cast<int64_t>(bias);
    const auto sum_global_t = (src_chnl_v == 0) ? (static_cast<int64_t>(sum_local)) : (sum_global.data + static_cast<int64_t>(sum_local));
    sum_global.data         = sum_global_t;

    // sum conv_t and bias_t (shift to value with bigger fraction size)
    auto res = (sum_frac_bits > bias_frac_bits) ? (sum_global_t + (bias_t << res_shift)) : (bias_t + (sum_global_t << res_shift));

    // Rounding (shift to dst fraction size)
    if (param_::underflow_type == hvx::util::underflow_e::kRound)
        res += static_cast<int64_t>(1) << hvx::util::Max(res_frac_bits - 1, 0);
    res = (res_frac_bits > dst_frac_bits) ? (res >> dst_shift) : (res << dst_shift);

    // Check for overflow
    if (param_::overflow_type == hvx::util::overflow_e::kSaturate) {
        res = hvx::util::Max(res, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::lowest()));
        res = hvx::util::Min(res, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::max()));
    }

    // write the result back
    return res;
}

/*!
 * @brief comp global sum, add bias and apply overflow/underflow policies (using floating-point numbers)
 */
template<typename param_,
         std::enable_if_t<param_::src_type::is_flt, bool>  = true,
         std::enable_if_t<param_::wgts_type::is_flt, bool> = true,
         std::enable_if_t<param_::bias_type::is_flt, bool> = true,
         std::enable_if_t<param_::dst_type::is_flt, bool>  = true>
HVX_FORCE_INLINE constexpr auto
ConvAddBias(int64_t src_chnl_v,
            float sum_local,
            typename param_::comp_type& sum_global,
            typename param_::bias_type::data_type bias) noexcept -> float {
    HVX_INLINE_TOP();

    // Some hardcoded bit widths
    constexpr int32_t frac_bits = 32;
    constexpr int32_t int_bits  = 16;
    static_assert(int_bits >= 1, "Minimum 1 integer bit needed!");

    // compile time constants
    constexpr auto one       = static_cast<int64_t>(1);
    constexpr auto shift     = static_cast<float>(one << frac_bits);
    constexpr auto shift_inv = static_cast<float>(1.0f / shift);
    constexpr auto int48_max = static_cast<float>(one << (frac_bits + int_bits)) - 1.0f;
    constexpr auto int48_min = -1.0f * static_cast<float>(one << (frac_bits + int_bits));

    // verifiy if overflow is still possible
    static_assert((one << (63 - frac_bits - int_bits)) > param_::fm_vec_elms, "Possible number overflow! To many dst_chnls!");

    // convert to integer for floating point summation, since it does not meet pipeline intervall of 1
    auto sum_local_int = static_cast<int64_t>(hvx::util::Clamp(sum_local * shift + 0.5f, int48_min, int48_max));

    // update global summation
    const int64_t sum_global_int = (src_chnl_v == 0) ? (sum_local_int) : (sum_global.data + sum_local_int);
    sum_global.data              = sum_global_int;

    // convert back and add bias
    float res = static_cast<float>(sum_global_int) * shift_inv + static_cast<float>(bias);

    // write the result back
    return res;
}

/*!
 * @brief comp an vector of dst chnls
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type>, bool>      = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::wgts_type>, bool>     = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::bias_type>, bool>     = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type>, bool>      = true,
         std::enable_if_t<param_::src_type::is_flt == param_::wgts_type::is_flt, bool>  = true,
         std::enable_if_t<param_::bias_type::is_flt == param_::wgts_type::is_flt, bool> = true,
         std::enable_if_t<param_::dst_type::is_flt == param_::wgts_type::is_flt, bool>  = true>
HVX_FORCE_INLINE constexpr auto
ConvComp(int64_t src_chnl_v,
         typename param_::comp_type& sum_global,
         hvx::util::vector<typename param_::src_type, param_::sum_elms>& win_vec,
         hvx::util::vector<typename param_::wgts_type, param_::sum_elms>& wgts_vec,
         typename param_::bias_type& bias,
         typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // use float32 or int64 for internal computation (type dependent)
    using comp_type = std::conditional_t<std::is_integral<typename param_::src_type::data_type>::value, int64_t, float>;

    // variables
    comp_type sum_local{};

    //
    for (int64_t src_chnl_p = 0; src_chnl_p < param_::chnl_vec_size; ++src_chnl_p) {
        for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
            for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                const int64_t win_ptr  = src_chnl_p * param_::knl_elms + knl_row * param_::knl_cols + knl_col;
                const int64_t knl_ptr  = (param_::knl_rows - 1 - knl_row) * param_::knl_cols + (param_::knl_cols - 1 - knl_col);
                const int64_t wgts_ptr = (src_chnl_p * param_::knl_elms) + knl_ptr;
                const auto src         = static_cast<comp_type>(win_vec.Get(win_ptr).data);
                const auto wgt         = static_cast<comp_type>(wgts_vec.Get(wgts_ptr).data);
                sum_local += src * wgt;
            }
        }
    }

    // comp global sum, add bias and apply overflow/underflow policies
    const auto res = hvx::nn::impl::ConvAddBias<param_>(src_chnl_v, sum_local, sum_global, bias.data);

    // convert to dst type
    dst.data = static_cast<typename param_::dst_type::data_type>(res);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_CONV_DFIXED_H_
