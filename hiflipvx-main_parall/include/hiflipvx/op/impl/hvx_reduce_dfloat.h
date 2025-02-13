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
 * @file    hvx_reduce_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_REDUCE_DFLOAT_H_
#define HVX_REDUCE_DFLOAT_H_

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
         std::enable_if_t<dynfloat::is_dfloat_v<src_type_>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type_>, bool> = true>
HVX_FORCE_INLINE constexpr auto
VerifyDataType() noexcept -> void {
    HVX_INLINE_TOP();
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceRead(typename param_::src_vec src, typename param_::buf_type& dst) noexcept -> void {
    dst = static_cast<typename param_::buf_type>(src.Get(0));
}

/*!
 * @brief
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceWrite(typename param_::buf_type src, typename param_::dst_vec& dst) noexcept -> void {
    dst.Get(0) = static_cast<typename param_::dst_type>(src);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_,
         int64_t dim_elms_,
         typename buf_type_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool> = true>
HVX_FORCE_INLINE constexpr auto
ReduceUpdate(buf_type_ data_curr, buf_type_ data_next, int64_t data_ptr, buf_type_ buf_data) noexcept -> buf_type_ {
    // converts division into multiplication
    constexpr auto mul = static_cast<dynfloat::pltfrm_dsp_opt>(1.0f / dim_elms_);

    //
    using execution_t = dynfloat::execution<hvx::util::ToDfloatExecution(param_::exec_type),      //
                                            hvx::util::ToDfloatUnderflow(param_::underflow_type), //
                                            hvx::util::ToDfloatOverflow(param_::overflow_type)>;  //

    // gets the input
    auto src = data_curr;
    auto dst = data_next;
    auto buf = buf_data;

    // computes reduce operator
    switch (param_::op_type) {
        case hvx::util::reduce_e::Max: {
            dst = (data_ptr == 0) ? src : std::max(src, buf);
            break;
        }
        case hvx::util::reduce_e::Mean: { // TODO: latency
            if (data_ptr == 0)
                dst = src;
            else if (data_ptr < (dim_elms_ - 1))
                dst = dynfloat::add<execution_t>(src, buf);
            else if (data_ptr == (dim_elms_ - 1))
                dst = dynfloat::mixed_mul<buf_type_, execution_t>(dynfloat::add<execution_t>(src, buf), mul);
            break;
        }
        case hvx::util::reduce_e::Min: {
            dst = (data_ptr == 0) ? src : std::min(src, buf);
            break;
        }
        case hvx::util::reduce_e::Sum: {
            dst = (data_ptr == 0) ? src : dynfloat::add<execution_t>(src, buf);
            break;
        }
        default:
            break;
    }

    // writes back result
    return dst;
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace red
} // namespace hvx

#endif // HVX_REDUCE_DFLOAT_H_
