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
 * @file    hvx_nn_softmax_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_SOFTMAX_DFIXED_H_
#define HVX_NN_SOFTMAX_DFIXED_H_

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
SoftmaxVerifyType() noexcept -> void {
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
 * @brief Calculates: n(i) = exp(src(i)) | N: sum of all n
 */
template<typename dim_,
         typename param_,
         typename src_type_,
         typename comp_type_,
         typename buf_type_,
         int64_t chnls_                                                                       = hvx::util::TensorGetDimElms<dim_, 0>(),
         int64_t chnl_vec_size_                                                               = hvx::util::TensorGetDimVecSize<dim_, 0>(),
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>                            = true,
         std::enable_if_t<hvx::util::is_dfixed_v<comp_type_>, bool>                           = true,
         std::enable_if_t<hvx::util::is_dfixed_v<buf_type_>, bool>                            = true,
         std::enable_if_t<std::is_same<typename comp_type_::data_type, int64_t>::value, bool> = true,
         std::enable_if_t<std::is_same<typename buf_type_::data_type, float>::value, bool>    = true>
HVX_FORCE_INLINE constexpr auto
SoftmaxStage1(int64_t chnl_p, int64_t chnl_v, comp_type_& sum_local, comp_type_& sum_global, src_type_& src, buf_type_& dst) noexcept
    -> void {
    HVX_INLINE_TOP();

    // fixed-point parameters
    constexpr auto src_frac_bits_      = static_cast<int64_t>(src_type_::frac_bits);
    constexpr auto src_frac_shift_     = static_cast<float>(1 << src_frac_bits_);
    constexpr auto src_frac_shift_inv_ = static_cast<float>(1.0f / src_frac_shift_);

    // summation hack parameters
    constexpr int32_t sum_frac_bits = 32;
    constexpr int32_t sum_int_bits  = 16;
    constexpr auto one              = static_cast<int64_t>(1);
    constexpr auto sum_max          = static_cast<float>(one << (sum_frac_bits + sum_int_bits)) - 1.0f;
    constexpr auto sum_min          = -1.0f * static_cast<float>(one << (sum_frac_bits + sum_int_bits));
    static_assert((one << (63 - sum_frac_bits - sum_int_bits)) > chnls_, "Possible number overflow! To many chnls!");

    // convert to floating point if needed
    const float var = (src_type_::is_int == true) ? (static_cast<float>(src.data) * src_frac_shift_inv_) : (src.data);

    // comp exponential of src value
    float exponential = std::exp(var);

    // convert the sum to float (data = ldexpf(exponential, FIXED_POINT_POSITION))
    float res = exponential * src_frac_shift_;
    res       = hvx::util::Clamp(res, sum_min, sum_max);

    // add to the sum of all exponentials
    sum_local.data += static_cast<int32_t>(res);

    // buffer exponential result
    dst.data = exponential;

    // update global sum
    if (chnl_p == chnl_vec_size_ - 1)
        sum_global.data = (chnl_v == 0) ? (sum_local.data) : (sum_global.data + sum_local.data);
}

/*!
 * @brief Calculates: m(i) = n(i) / N
 */
template<typename dim_,
         typename param_,
         typename src_type_,
         typename dst_type_,
         typename comp_type_,
         typename buf_type_,
         int64_t chnl_vec_size_                                     = hvx::util::TensorGetDimVecSize<dim_, 0>(),
         std::enable_if_t<hvx::util::is_dfixed_v<src_type_>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type_>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<comp_type_>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<buf_type_>, bool>  = true,
         std::enable_if_t<std::is_same<typename src_type_::data_type, typename dst_type_::data_type>::value, bool> = true,
         std::enable_if_t<std::is_same<typename buf_type_::data_type, float>::value, bool>                         = true>
HVX_FORCE_INLINE constexpr auto
SoftmaxStage2(int64_t chnl_p,
              comp_type_& sum_global,
              hvx::util::vector<buf_type_, chnl_vec_size_>& src_vec,
              hvx::util::vector<dst_type_, chnl_vec_size_>& dst_vec) noexcept -> void {
    HVX_INLINE_TOP();

    // fixed-point parameters
    constexpr auto src_frac_bits_  = static_cast<int64_t>(src_type_::frac_bits);
    constexpr auto src_frac_shift_ = static_cast<float>(1 << src_frac_bits_);

    // replace division by multiplication
    const auto sum_inv = (src_frac_shift_ / static_cast<float>(sum_global.data));

    // comp result
    const auto res = src_vec.Get(chnl_p).data * sum_inv;

    // convert back to fixed if needed
    dst_vec.Get(chnl_p) = static_cast<dst_type_>(res);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_SOFTMAX_DFIXED_H_
