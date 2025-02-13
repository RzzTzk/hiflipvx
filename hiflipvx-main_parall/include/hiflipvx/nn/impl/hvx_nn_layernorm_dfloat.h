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
 * @file    hvx_nn_layernorm_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_LAYERNORM_DFLOAT_H
#define HVX_NN_LAYERNORM_DFLOAT_H

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace nn {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief verifies the dimensions
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::wgts_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::bias_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
LayernormVerifyType() noexcept -> void {
    HVX_INLINE_TOP();
}

/*!
 * @brief Computes the layer normalization for floating point
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::wgts_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::bias_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
LayernormComp(typename param_::src_type src,
              typename param_::wgts_type wgt,
              typename param_::bias_type bias,
              typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    //
    constexpr auto execution      = hvx::util::ToDfloatExecution(param_::exec_type);
    constexpr auto round          = hvx::util::ToDfloatUnderflow(param_::underflow_type); // param_::underflow_type
    constexpr auto special_values = hvx::util::ToDfloatOverflow(param_::overflow_type);   // param_::overflow_type
    using df_execution            = dynfloat::execution<execution, round, special_values>;

    //
    constexpr auto mul_exp_bits = hvx::util::Max(param_::src_type::exp_bits, param_::wgts_type::exp_bits);
    constexpr auto mul_man_bits = hvx::util::Max(param_::src_type::man_bits, param_::wgts_type::man_bits);
    using mul_type              = dynfloat::dfloat<mul_exp_bits, mul_man_bits>;
    constexpr auto add_exp_bits = hvx::util::Max(mul_type::exp_bits, param_::bias_type::exp_bits);
    constexpr auto add_man_bits = hvx::util::Max(mul_type::man_bits, param_::bias_type::man_bits);
    using add_type              = dynfloat::dfloat<add_exp_bits, add_man_bits>;

    //
    const auto mul_t  = dynfloat::mixed_mul<mul_type, df_execution>(src, wgt);
    const auto result = dynfloat::add<df_execution>(static_cast<add_type>(mul_t), static_cast<add_type>(bias));
    dst               = static_cast<typename param_::dst_type>(result);
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_LAYERNORM_DFLOAT_H
