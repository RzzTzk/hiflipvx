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
 * @file    hvx_nn_pool_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_POOL_DFIXED_H_
#define HVX_NN_POOL_DFIXED_H_

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace nn {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief verifies the dimensions of all src and dst
 */
template<typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>                                                 = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool>                                                 = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolVerifyType() noexcept -> void {
    HVX_INLINE_TOP();

    // compile time assertions
    static_assert(hvx::util::CompareDataType<typename src_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>() &&
                      hvx::util::CompareDataType<typename dst_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, float>(),
                  "Data type is not supported!");
    static_assert(hvx::util::CheckFracBitWidth<typename src_type_::data_type, src_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename dst_type_::data_type, dst_type_::frac_bits>(),
                  "Fraction size out of scope!");
}

/*!
 * @brief Max pool for float values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_> && src_type_::is_flt, bool>                            = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_> && dst_type_::is_flt, bool>                            = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolMax(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // get maximum value of the src values
    auto res = std::numeric_limits<typename src_type_::data_type>::lowest();
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        res = hvx::util::Max(res, win_vec.Get(knl_pix).data);
    }

    // convert to destination type and store result
    dst.data = static_cast<typename dst_type_::data_type>(res);
}

/*!
 * @brief Max pool for integer values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_> && src_type_::is_int, bool>                            = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_> && dst_type_::is_int, bool>                            = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolMax(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // constants
    constexpr auto shift_val = static_cast<uint32_t>(hvx::util::Abs(src_type_::frac_bits - dst_type_::frac_bits));
    constexpr auto half      = static_cast<int64_t>(1) << hvx::util::Max(0, static_cast<int32_t>(shift_val) - 1);
    constexpr auto dst_max   = static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::max());
    constexpr auto dst_min   = static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::lowest());

    // get maximum value of the src values
    int64_t res = std::numeric_limits<typename src_type_::data_type>::lowest();
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        res = hvx::util::Max(res, static_cast<int64_t>(win_vec.Get(knl_pix).data));
    }

    // destination integer size is smaller than source (check overflow policy)
    if (src_type_::frac_bits < dst_type_::frac_bits) {
        res = res << shift_val;
        if (param_::overflow_type == hvx::util::overflow_e::kSaturate)
            res = hvx::util::Clamp(res, dst_min, dst_max);
    }

    // destination fraction size is smaller than source (check underflow policy)
    else if (src_type_::frac_bits > dst_type_::frac_bits) {
        if (param_::underflow_type == hvx::util::underflow_e::kRound)
            res = res + half;
        res = res >> shift_val;
    }

    // convert to destination type and store result
    dst.data = static_cast<typename dst_type_::data_type>(res);
}

/*!
 * @brief Average pool for floating point values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_> && src_type_::is_flt, bool>                            = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_> && dst_type_::is_flt, bool>                            = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolAvg(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // normalization
    constexpr auto norm_flt = 1.0f / static_cast<float>(param_::knl_elms);

    // sum up the values of the src win and comp the average
    float res = 0.0f;
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        res += static_cast<float>(win_vec.Get(knl_pix).data);
    }
    res = res * norm_flt;

    // convert to destination type and store result
    dst.data = static_cast<typename dst_type_::data_type>(res);
}

/*!
 * @brief Average pool for integer values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_> && src_type_::is_int, bool>                            = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_> && dst_type_::is_int, bool>                            = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolAvg(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // normalization
    constexpr auto dst_frac_shift = static_cast<float>(static_cast<int64_t>(1) << dst_type_::frac_bits);
    constexpr auto norm_flt       = 1.0f / static_cast<float>(param_::knl_elms);
    constexpr auto norm_int       = static_cast<int64_t>(norm_flt * dst_frac_shift); // TODO: no sophisticated rounding at compile time?
    constexpr auto dst_max        = static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::max());
    constexpr auto dst_min        = static_cast<int64_t>(std::numeric_limits<typename dst_type_::data_type>::lowest());

    // sum up the values of the src win and comp the average
    int64_t res = 0;
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        res += static_cast<int64_t>(win_vec.Get(knl_pix).data);
    }
    res = res * norm_int;

    // shift to destination fraction size and check underflow policy
    if (param_::underflow_type == hvx::util::underflow_e::kRound)
        res = res + src_type_::half;
    res = res >> src_type_::frac_bits;

    // check overflow policy
    if ((src_type_::frac_bits < dst_type_::frac_bits) && (param_::overflow_type == hvx::util::overflow_e::kSaturate))
        res = hvx::util::Clamp(res, dst_min, dst_max);

    // convert to destination type and store result
    dst.data = static_cast<typename dst_type_::data_type>(res);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_POOL_DFIXED_H_
