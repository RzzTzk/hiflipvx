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
 * @file    hvx_nn_layernorm_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_LAYERNORM_DFIXED_H
#define HVX_NN_LAYERNORM_DFIXED_H

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace nn {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief verifies the dimensions
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::wgts_type>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::bias_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
LayernormVerifyType() noexcept -> void {
    HVX_INLINE_TOP();

    using src_data_type  = typename param_::src_type::data_type;
    using dst_data_type  = typename param_::dst_type::data_type;
    using wgts_data_type = typename param_::wgts_type::data_type;
    using bias_data_type = typename param_::bias_type::data_type;

    // compile time assertions
    static_assert(hvx::util::CompareDataType<src_data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<dst_data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<wgts_data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<bias_data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>(),
                  "Data type is not supported!");
    static_assert(hvx::util::CheckFracBitWidth<src_data_type, param_::src_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<dst_data_type, param_::dst_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<wgts_data_type, param_::wgts_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<bias_data_type, param_::bias_type_::frac_bits>(),
                  "Fraction size out of scope!");
}

/*!
 * @brief Computes the layer normalization for floating point
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type> && param_::src_type::is_flt, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_flt, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::wgts_type> && param_::wgts_type::is_flt, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::bias_type> && param_::bias_type::is_flt, bool> = true>
HVX_FORCE_INLINE constexpr auto
LayernormComp(typename param_::src_type src,
              typename param_::wgts_type wgt,
              typename param_::bias_type bias,
              typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();
    dst.data = static_cast<typename param_::dst_type::data_type>(src.data * wgt.data + bias.data);
}

/*!
 * @brief Computes the layer normalization for fixed point
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type> && param_::src_type::is_int, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_int, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::wgts_type> && param_::wgts_type::is_int, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::bias_type> && param_::bias_type::is_int, bool> = true>
HVX_FORCE_INLINE constexpr auto
LayernormComp(typename param_::src_type src,
              typename param_::wgts_type wgt,
              typename param_::bias_type bias,
              typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // constants
    constexpr int64_t mul_frac_bits = param_::src_type::frac_bits + param_::wgts_type::frac_bits;
    constexpr int64_t mul_shift     = hvx::util::Abs(mul_frac_bits - param_::bias_type::frac_bits);
    constexpr int64_t res_shift     = hvx::util::Abs(mul_frac_bits - param_::dst_type::frac_bits);

    // layer normalization (TODO: underflow)
    const auto bias_val = static_cast<int64_t>(bias.data);
    auto res            = static_cast<int64_t>(src.data) * static_cast<int64_t>(wgt.data);
    res = (mul_frac_bits > param_::bias_type::frac_bits) ? (res + (bias_val << mul_shift)) : ((res << mul_shift) + bias_val);
    res = (mul_frac_bits > param_::dst_type::frac_bits) ? (res >> res_shift) : (res << res_shift);

    // checking overflow
    if (param_::overflow_type == hvx::util::overflow_e::kSaturate) {
        res = hvx::util::Max(res, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::lowest()));
        res = hvx::util::Min(res, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::max()));
    } else if (param_::overflow_type == hvx::util::overflow_e::kClip) {
        res = hvx::util::Max(res, static_cast<int64_t>(param_::clip_min));
        res = hvx::util::Min(res, static_cast<int64_t>(param_::clip_max));
    }

    // convert to dst type
    dst.data = static_cast<typename param_::dst_type::data_type>(res);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_LAYERNORM_DFIXED_H
