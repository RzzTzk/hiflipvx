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
 * @file    hvx_nn_pool_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_POOL_DFLOAT_H_
#define HVX_NN_POOL_DFLOAT_H_

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
PoolVerifyType() noexcept -> void {
    HVX_INLINE_TOP();
}

/*!
 * @brief Max pool for float values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolMax(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    auto res = std::numeric_limits<src_type_>::lowest();
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        res = hvx::util::Max(res, win_vec.Get(knl_pix));
    }
    dst = static_cast<dst_type_>(res);
}

/*!
 * @brief Average pool for floating point values
 */
template<typename param_,
         typename src_type_,
         typename dst_type_,
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
PoolAvg(hvx::util::vector<src_type_, param_::knl_elms>& win_vec, dst_type_& dst) noexcept -> void {
    HVX_INLINE_TOP();

    //
    constexpr auto execution      = hvx::util::ToDfloatExecution(param_::exec_type);
    constexpr auto round          = hvx::util::ToDfloatUnderflow(param_::underflow_type);
    constexpr auto special_values = hvx::util::ToDfloatOverflow(param_::overflow_type);
    using df_execution            = dynfloat::execution<execution, round, special_values>;

    // sum up the values of the input window
    src_type_ sum{};
    for (int64_t knl_pix = 0; knl_pix < (param_::knl_elms); ++knl_pix) {
        HVX_UNROLL();
        sum = dynfloat::add<df_execution>(sum, win_vec.Get(knl_pix));
    }

    // normalization
    constexpr auto dnm      = param_::knl_rows * param_::knl_cols;
    constexpr auto pow2_dnm = ((dnm - 1) & dnm) == 0;
    if (pow2_dnm) {
        constexpr auto lg2_dnm = dynfloat::utils::lg2(dnm);
        dst                    = sum >> lg2_dnm;
    } else {
        constexpr auto norm_flt = static_cast<src_type_>(1.0 / static_cast<double>(dnm));
        const auto mul_res      = dynfloat::mul<df_execution>(sum, norm_flt);
        dst                     = static_cast<dst_type_>(mul_res);
    }
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace nn
} // namespace hvx

#endif // HVX_NN_POOL_DFLOAT_H_
