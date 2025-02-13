/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ï¿½AS ISï¿½, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_nn_depthwise_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_DEPTHWISE_DFIXED_H_
#define HVX_NN_DEPTHWISE_DFIXED_H_

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace nn {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief Verify the dfixed data type
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>                                                  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<wgts_type_>, bool>                                                 = true,
         std::enable_if_t<hvx::util::is_dfixed_v<bias_type_>, bool>                                                 = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool>                                                  = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename wgts_type_::data_type>::value, bool> = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename bias_type_::data_type>::value, bool> = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool>  = true>
HVX_FORCE_INLINE constexpr auto
DepthwiseVerifyTypes() noexcept -> void {
    HVX_INLINE_TOP();

    // compile time assertions
    static_assert(hvx::util::CompareDataType<typename src_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>() &&
                      hvx::util::CompareDataType<typename wgts_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>() &&
                      hvx::util::CompareDataType<typename bias_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<typename dst_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>(),
                  "Data type is not supported for dfixed!");
    static_assert(hvx::util::CheckFracBitWidth<typename src_type_::data_type, src_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename wgts_type_::data_type, wgts_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename bias_type_::data_type, bias_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename dst_type_::data_type, dst_type_::frac_bits>(),
                  "Fraction size out of scope!");
}

/*!
 * @brief comp a single channel (using floating-point numbers)
 */
template<typename param_,
         typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         std::enable_if_t<src_type_::is_flt, bool>  = true,
         std::enable_if_t<wgts_type_::is_flt, bool> = true,
         std::enable_if_t<bias_type_::is_flt, bool> = true,
         std::enable_if_t<dst_type_::is_flt, bool>  = true>
HVX_FORCE_INLINE constexpr auto
DepthwiseAddBias(float sum, typename bias_type_::data_type bias) noexcept -> float {
    HVX_INLINE_TOP();
    return sum + bias;
}

/*!
 * @brief comp a single channel (using fixed-point numbers)
 */
template<typename param_,
         typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         std::enable_if_t<src_type_::is_int, bool>  = true,
         std::enable_if_t<wgts_type_::is_int, bool> = true,
         std::enable_if_t<bias_type_::is_int, bool> = true,
         std::enable_if_t<dst_type_::is_int, bool>  = true>
HVX_FORCE_INLINE constexpr auto
DepthwiseAddBias(int64_t sum, typename bias_type_::data_type bias) noexcept -> int64_t {
    HVX_INLINE_TOP();

    // fraction sizes
    constexpr int32_t bias_frac_bits = bias_type_::frac_bits;
    constexpr int32_t dst_frac_bits  = dst_type_::frac_bits;
    constexpr int32_t sum_frac_bits  = src_type_::frac_bits + wgts_type_::frac_bits;
    constexpr int32_t res_frac_bits  = hvx::util::Max(sum_frac_bits, bias_frac_bits);
    constexpr int32_t dst_shift      = hvx::util::Abs(res_frac_bits - dst_frac_bits);
    constexpr int32_t res_shift      = hvx::util::Abs(sum_frac_bits - bias_frac_bits);

    // get bias
    const auto bias_t = static_cast<int64_t>(bias);

    // sum conv_t and bias_t (shift to value with bigger fraction size)
    auto res = (sum_frac_bits > bias_frac_bits) ? (sum + (bias_t << res_shift)) : (bias_t + (sum << res_shift));

    // Rounding (shift to dst fraction size)
    if (param_::underflow_type == hvx::util::underflow_e::kRound)
        res += static_cast<int64_t>(1) << hvx::util::Max(res_frac_bits - 1, 0);
    res = (res_frac_bits > dst_frac_bits) ? (res >> dst_shift) : (res << dst_shift);

    // Check for overflow
    if (param_::overflow_type == hvx::util::overflow_e::kSaturate) {
        res = hvx::util::Max(res, static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::lowest()));
        res = hvx::util::Min(res, static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::max()));
    }

    // write the result back
    return res;
}

/*!
 * @brief comp a single dst element
 */
template<typename wgts_dim_,
         typename param_,
         typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         int64_t knl_cols_                                          = hvx::util::TensorGetDimElms<wgts_dim_, 0>(),
         int64_t knl_rows_                                          = hvx::util::TensorGetDimElms<wgts_dim_, 1>(),
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<wgts_type_>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<bias_type_>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool>  = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename wgts_type_::data_type>::value, bool> = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename bias_type_::data_type>::value, bool> = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool>  = true>
HVX_FORCE_INLINE constexpr auto
DepthwiseComp(hvx::util::vector<src_type_, knl_cols_ * knl_rows_>& win_vec,
              hvx::util::vector<wgts_type_, knl_cols_ * knl_rows_>& wgts_vec,
              bias_type_& bias,
              dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // use float32 or int64 for internal computation
    using comp_type = std::conditional_t<std::is_integral<typename src_type_::data_type>::value, int64_t, float>;

    // variables
    comp_type sum = 0;

    //
    for (int64_t knl_row = 0; knl_row < knl_rows_; ++knl_row) {
        for (int64_t knl_col = 0; knl_col < knl_cols_; ++knl_col) {
            HVX_UNROLL();
            const int64_t knl_ptr = ((knl_rows_ - 1 - knl_row) * knl_cols_) + (knl_cols_ - 1 - knl_col);
            const int64_t win_ptr = knl_row * knl_cols_ + knl_col;
            const auto src        = static_cast<comp_type>(win_vec.Get(win_ptr).data);
            const auto wgt        = static_cast<comp_type>(wgts_vec.Get(knl_ptr).data);
            sum += src * wgt;
        }
    }

    // comp global sum, add bias and apply overflow/underflow policies
    const auto res = hvx::nn::impl::DepthwiseAddBias<param_, src_type_, wgts_type_, bias_type_, dst_type_>(sum, bias.data);
    dst.data       = static_cast<typename dst_type_::data_type>(res);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_DEPTHWISE_DFIXED_H_
