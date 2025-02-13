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
 * @file    hvx_nn_softmax_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_SOFTMAX_DFLOAT_H_
#define HVX_NN_SOFTMAX_DFLOAT_H_

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
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
SoftmaxVerifyType() noexcept -> void {
    HVX_INLINE_TOP();
}

/*!
 * @brief Calculates: n(i) = exp(src(i)) | N: sum of all n
 */
template<typename dim_,
         typename param_,
         typename src_type_,
         typename comp_type_,
         typename buf_type_,
         int64_t chnls_                                            = hvx::util::TensorGetDimElms<dim_, 0>(),
         int64_t chnl_vec_size_                                    = hvx::util::TensorGetDimVecSize<dim_, 0>(),
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<comp_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<buf_type_>, bool>  = true>
HVX_FORCE_INLINE constexpr auto
SoftmaxStage1(int64_t chnl_p, int64_t chnl_v, comp_type_& sum_local, comp_type_& sum_global, src_type_& src, buf_type_& dst) noexcept
    -> void {
    HVX_INLINE_TOP();

    //
    constexpr auto execution      = hvx::util::ToDfloatExecution(param_::exec_type);
    constexpr auto round          = hvx::util::ToDfloatUnderflow(param_::underflow_type); // param_::underflow_type
    constexpr auto special_values = hvx::util::ToDfloatOverflow(param_::overflow_type);   // param_::overflow_type
    using df_execution            = dynfloat::execution<execution, round, special_values>;

    // comp exponential of src value
    const auto exp_t = dynfloat::exp<df_execution>(src);

    // add to the sum of all exponentials
    sum_local = dynfloat::add<df_execution>(static_cast<comp_type_>(exp_t), sum_local);

    // buffer exponential result
    dst = static_cast<buf_type_>(exp_t);

    // update global sum
    if (chnl_p == chnl_vec_size_ - 1)
        sum_global = (chnl_v == 0) ? (sum_local) : dynfloat::add<df_execution>(sum_global, sum_local);
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
         int64_t chnl_vec_size_                                    = hvx::util::TensorGetDimVecSize<dim_, 0>(),
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<comp_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<buf_type_>, bool>  = true>
HVX_FORCE_INLINE constexpr auto
SoftmaxStage2(int64_t chnl_p,
              comp_type_& sum_global,
              hvx::util::vector<buf_type_, chnl_vec_size_>& src_vec,
              hvx::util::vector<dst_type_, chnl_vec_size_>& dst_vec) noexcept -> void {
    HVX_INLINE_TOP();

    //
    constexpr auto execution      = hvx::util::ToDfloatExecution(param_::exec_type);
    constexpr auto round          = hvx::util::ToDfloatUnderflow(param_::underflow_type); // param_::underflow_type
    constexpr auto special_values = hvx::util::ToDfloatOverflow(param_::overflow_type);   // param_::overflow_type
    using df_execution            = dynfloat::execution<execution, round, special_values>;

    //
    const auto inv_sum  = dynfloat::reciprocal<df_execution>(sum_global);
    dst_vec.Get(chnl_p) = dynfloat::mixed_mul<dst_type_, df_execution>(src_vec.Get(chnl_p), inv_sum); // buf_type * comp_type
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_SOFTMAX_DFLOAT_H_
