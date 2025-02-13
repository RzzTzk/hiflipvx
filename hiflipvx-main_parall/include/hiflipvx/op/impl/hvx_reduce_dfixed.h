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
 * @file    hvx_reduce_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_REDUCE_DFIXED_H_
#define HVX_REDUCE_DFIXED_H_

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace red {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief Verify the dfixed data type
 */
template<typename src_type_,
         typename dst_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
VerifyDataType() noexcept -> void {
    HVX_INLINE_TOP();

    // compile time assertions
    static_assert(hvx::util::CompareDataType<typename src_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<typename dst_type_::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>(),
                  "Data type is not supported!");
    static_assert(hvx::util::CheckFracBitWidth<typename src_type_::data_type, src_type_::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename dst_type_::data_type, dst_type_::frac_bits>(),
                  "Fraction size out of scope!");
}

/******************************************************************************************************************************************/

/*!
 * @brief Gets the number of elements that need to be reduced (its zero if nothing is reduced)
 */
template<typename src_dim_, typename reduce_>
constexpr auto
ReduceElmsCalc(int64_t i) noexcept -> int64_t {
    return reduce_::dims[i] ? src_dim_::dims[i] : 0; // NOLINT
}

/*!
 * @brief Creates an array that contains the number of elements that need to be reduced in each dimension (its zero if nothing is reduced)
 */
template<typename src_dim_, typename reduce_, std::size_t... idx_>
constexpr auto
ReduceElmsArray(std::index_sequence<idx_...>) {
    return std::array<int64_t, sizeof...(idx_)>{{hvx::red::impl::ReduceElmsCalc<src_dim_, reduce_>(idx_)...}};
}

/*!
 * @brief Calculates the extra bits needed for the mean operation without running into an overflow
 */
template<typename array_t>
constexpr auto
MeanBitsCalc(const array_t& arr, std::size_t index = 0) noexcept -> int64_t {
    return (index < arr.size()) ? std::max(hvx::util::Log2Ceil(arr[index]), hvx::red::impl::MeanBitsCalc(arr, index + 1)) : 0; // NOLINT
}

/*!
 * @brief calculates the extra bits needed for the sum operation without running into an overflow
 */
template<typename array_t>
constexpr auto
SumBitsCalc(const array_t& arr, std::size_t index = 0) noexcept -> int64_t {
    return (index < arr.size()) ? (hvx::util::Log2Ceil(arr[index]) + hvx::red::impl::SumBitsCalc(arr, index + 1)) : 0; // NOLINT
}

/*!
 * @brief calculates the extra bits needed for a specific operations without running into an overflow
 */
template<typename src_dim_, typename reduce_, hvx::util::reduce_e op_type_, int64_t dim_num_>
constexpr auto
ExtraBitsCalculate() noexcept -> int64_t {
    constexpr auto reduce_elms = hvx::red::impl::ReduceElmsArray<src_dim_, reduce_>(std::make_index_sequence<dim_num_>{});
    constexpr auto mean_bits   = hvx::red::impl::MeanBitsCalc(reduce_elms);
    constexpr auto sum_bits    = hvx::red::impl::SumBitsCalc(reduce_elms);
    return (op_type_ == hvx::util::reduce_e::Mean) ? mean_bits : (op_type_ == hvx::util::reduce_e::Sum) ? sum_bits : 0;
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceRead(typename param_::src_vec src, typename param_::buf_type& dst) noexcept -> void {
    dst = static_cast<typename param_::buf_type>(src.Get(0).data);
}

/*!
 * @brief
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceWrite(typename param_::buf_type src, typename param_::dst_vec& dst) noexcept -> void {
    dst.Get(0).data = static_cast<typename param_::dst_type::data_type>(src);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_,
         int64_t dim_elms_,
         typename buf_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type> && param_::src_type::is_int, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_int, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceUpdate(buf_type_ data_curr, buf_type_ data_next, int64_t data_ptr, buf_type_ buf_data) noexcept -> buf_type_ {
    static_assert((param_::extra_bits + param_::src_type::digits) < 63, "Not enough bits available!");

    // converts division into multiply and shift
    constexpr int64_t shift = hvx::util::Min(static_cast<int64_t>(24), 63 - (param_::extra_bits + param_::src_type::digits));
    constexpr int64_t mul   = (1 << shift) / dim_elms_;

    // checks if buf type is signed
    constexpr bool is_signed = std::is_signed<buf_type_>::value;

    // checks if it is possible to have overflow
    constexpr bool signed_to_unsigned = param_::src_type::is_signed && !param_::dst_type::is_signed;
    constexpr bool src_bigger_dst_all = param_::src_type::digits > param_::dst_type::digits;
    constexpr bool src_bigger_dst_sum =
        (param_::op_type == hvx::util::reduce_e::Sum) && ((param_::src_type::digits + param_::extra_bits) > param_::dst_type::digits);
    constexpr bool can_have_overflow = signed_to_unsigned || src_bigger_dst_all || src_bigger_dst_sum;

    // gets the input
    auto src = static_cast<int64_t>(data_curr);
    auto dst = static_cast<int64_t>(data_next);
    auto buf = static_cast<int64_t>(buf_data);

    // computes reduce operator
    switch (param_::op_type) {
        // computes maximum of all values inside of 1 dimension
        case hvx::util::reduce_e::Max: {
            dst = (data_ptr == 0) ? src : hvx::util::Max(src, buf);
            break;
        }

        // computes mean of all values inside 1 dimension (Sum / N)
        case hvx::util::reduce_e::Mean: {
            if (data_ptr == 0)
                dst = src;
            else if (data_ptr < (dim_elms_ - 1))
                dst = (src + buf);
            else if (data_ptr == (dim_elms_ - 1))
                dst = hvx::util::FixedUnderflow<shift, param_::underflow_type, is_signed>((src + buf) * mul);
            break;
        }

        // computes minimum of all values inside of 1 dimension
        case hvx::util::reduce_e::Min: {
            dst = (data_ptr == 0) ? src : hvx::util::Min(src, buf);
            break;
        }

        // computes sum of all values inside of 1 dimension
        case hvx::util::reduce_e::Sum: {
            dst = (data_ptr == 0) ? src : (src + buf);
            break;
        }
        default:
            break;
    }

    // checks for overflow
    if (can_have_overflow && (param_::overflow_type == hvx::util::overflow_e::kSaturate)) {
        dst = hvx::util::Min(dst, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::max()));
        dst = hvx::util::Max(dst, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::lowest()));
    }

    // writes back result
    return static_cast<typename param_::buf_type>(dst);
}

/*!
 * @brief
 */
template<typename param_,
         int64_t dim_elms_,
         typename buf_type_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src_type> && param_::src_type::is_flt, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_flt, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceUpdate(buf_type_ data_curr, buf_type_ data_next, int64_t data_ptr, buf_type_ buf_data) noexcept -> buf_type_ {
    // converts division into multiply
    constexpr float mul = 1.0f / dim_elms_;

    // gets the input
    auto src = static_cast<float>(data_curr);
    auto dst = static_cast<float>(data_next);
    auto buf = static_cast<float>(buf_data);

    // computes reduce operator
    switch (param_::op_type) {
        case hvx::util::reduce_e::Max: {
            dst = (data_ptr == 0) ? src : hvx::util::Max(src, buf);
            break;
        }
        case hvx::util::reduce_e::Mean: {
            if (data_ptr == 0)
                dst = src;
            else if (data_ptr < (dim_elms_ - 1))
                dst = (src + buf);
            else if (data_ptr == (dim_elms_ - 1))
                dst = (src + buf) * mul;
            break;
        }
        case hvx::util::reduce_e::Min: {
            dst = (data_ptr == 0) ? src : hvx::util::Min(src, buf);
            break;
        }
        case hvx::util::reduce_e::Sum: {
            dst = (data_ptr == 0) ? src : (src + buf);
            break;
        }
        default:
            break;
    }

    // writes back result
    return static_cast<typename param_::buf_type>(dst);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace red
} // namespace hvx

#endif // HVX_REDUCE_DFIXED_H_
