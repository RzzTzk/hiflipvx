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
 * @file    hvx_nn_conv_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_CONV_DFLOAT_H_
#define HVX_NN_CONV_DFLOAT_H_

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
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<wgts_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<bias_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ConvVerifyType() noexcept -> void {
    HVX_INLINE_TOP();
}

/*!
 * @brief comp an vector of dst chnls
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::wgts_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::bias_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ConvComp(int64_t chnl_v,
         typename param_::comp_type& sum_global,
         hvx::util::vector<typename param_::src_type, param_::sum_elms>& win_vec,
         hvx::util::vector<typename param_::wgts_type, param_::sum_elms>& wgts_vec,
         typename param_::bias_type& bias,
         typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    //
    constexpr auto execution      = hvx::util::ToDfloatExecution(param_::exec_type);
    constexpr auto round          = hvx::util::ToDfloatUnderflow(param_::underflow_type); // param_::underflow_type
    constexpr auto special_values = hvx::util::ToDfloatOverflow(param_::overflow_type);   // param_::overflow_type
    using df_execution            = dynfloat::execution<execution, round, special_values>;

    // create type to add bias based on the comp type and bias type
    constexpr auto add_exp_bits = hvx::util::Max(param_::comp_type::exp_bits, param_::bias_type::exp_bits);
    constexpr auto add_man_bits = hvx::util::Max(param_::comp_type::man_bits, param_::bias_type::man_bits);
    using add_type              = dynfloat::dfloat<add_exp_bits, add_man_bits>;

    // variables
    typename param_::comp_type sum_local{};

    // compute local sum
    for (int64_t src_chnl_p = 0; src_chnl_p < param_::chnl_vec_size; ++src_chnl_p) {
        for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
            for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                const int64_t win_ptr  = src_chnl_p * param_::knl_elms + knl_row * param_::knl_cols + knl_col;
                const int64_t knl_ptr  = (param_::knl_rows - 1 - knl_row) * param_::knl_cols + (param_::knl_cols - 1 - knl_col);
                const int64_t wgts_ptr = (src_chnl_p * param_::knl_elms) + knl_ptr;

                //
                const auto mul =
                    dynfloat::mixed_mul<typename param_::comp_type, df_execution>(wgts_vec.Get(wgts_ptr), win_vec.Get(win_ptr));
                sum_local = dynfloat::add<df_execution>(sum_local, mul);
            }
        }
    }

    auto add_t = dynfloat::add<df_execution>(sum_global, sum_local);
    auto sum_global_part = (chnl_v == 0) ? (sum_local) : (add_t);
    sum_global           = sum_global_part;

    // add bias
    const auto result = dynfloat::add<df_execution>(static_cast<add_type>(sum_global_part), static_cast<add_type>(bias));

    // store result
    dst = static_cast<typename param_::dst_type>(result);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_CONV_DFLOAT_H_
