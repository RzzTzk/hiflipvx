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
 * @file    hvx_ew_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_EW_DFLOAT_H_
#define HVX_EW_DFLOAT_H_

#include "../../util/hvx_util_helper.h"

namespace hvx {
namespace ew {
namespace impl {
/******************************************************************************************************************************************/

/*!
 * @brief Verify the dfixed data type
 */
template<typename src1_type,
         typename src2_type,
         typename dst_type,
         typename arg_type,
         std::enable_if_t<dynfloat::is_dfloat_v<src1_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<src2_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<dst_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<arg_type>, bool>  = true>
HVX_FORCE_INLINE constexpr auto
VerifyDataType() noexcept -> void {
    HVX_INLINE_TOP();
}

/*!
 * @brief Computes the elementwise function for floating point data type
 */
template<typename param_,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src1_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::src2_type>, bool> = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::dst_type>, bool>  = true,
         std::enable_if_t<dynfloat::is_dfloat_v<typename param_::arg_type>, bool>  = true>
HVX_FORCE_INLINE constexpr auto
ElementwiseComp(typename param_::src1_type src1,
                typename param_::src2_type src2,
                typename param_::arg_type arg1,
                typename param_::arg_type arg2,
                typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // casts data type for operation to the maximum exponent and mantissa of both operators
    using op1_type = dynfloat::dfloat<hvx::util::Max(param_::src1_type::exp_bits, param_::src2_type::exp_bits),
                                      hvx::util::Max(param_::src1_type::man_bits, param_::src2_type::man_bits)>;
    using op2_type = dynfloat::dfloat<hvx::util::Max(param_::src1_type::exp_bits, param_::arg_type::exp_bits),
                                      hvx::util::Max(param_::src1_type::man_bits, param_::arg_type::man_bits)>;
    using dst_type = typename param_::dst_type;

    //
    using execution_t = dynfloat::execution<hvx::util::ToDfloatExecution(param_::exec_type),      //
                                            hvx::util::ToDfloatUnderflow(param_::underflow_type), //
                                            hvx::util::ToDfloatOverflow(param_::overflow_type)>;  //

    //
    switch (param_::op_type) {
        case hvx::util::elmwise_e::Abs: {
            dst = static_cast<dst_type>(std::abs(src1));
            break;
        }
        case hvx::util::elmwise_e::Add: {
            dst = static_cast<dst_type>(dynfloat::add<execution_t>(static_cast<op1_type>(src1), static_cast<op1_type>(src2)));
            break;
        }
        case hvx::util::elmwise_e::AddConst: {
            dst = static_cast<dst_type>(dynfloat::add<execution_t>(static_cast<op2_type>(src1), static_cast<op2_type>(arg1)));
            break;
        }
        case hvx::util::elmwise_e::Clip: {
            dst = std::min(std::max(static_cast<op2_type>(src1), static_cast<op2_type>(arg1)), static_cast<op2_type>(arg2));
            break;
        }
        case hvx::util::elmwise_e::Max: {
            dst = static_cast<dst_type>(std::max(static_cast<op1_type>(src1), static_cast<op1_type>(src2)));
            break;
        }
        case hvx::util::elmwise_e::MaxConst: {
            dst = static_cast<dst_type>(std::max(static_cast<op2_type>(src1), static_cast<op2_type>(arg1)));
            break;
        }
        case hvx::util::elmwise_e::Min: {
            dst = static_cast<dst_type>(std::min(static_cast<op1_type>(src1), static_cast<op1_type>(src2)));
            break;
        }
        case hvx::util::elmwise_e::MinConst: {
            dst = static_cast<dst_type>(std::min(static_cast<op2_type>(src1), static_cast<op2_type>(arg1)));
            break;
        }
        case hvx::util::elmwise_e::Mul: {
            dst = dynfloat::mixed_mul<dst_type, execution_t>(src1, src2);
            break;
        }
        case hvx::util::elmwise_e::MulConst: {
            dst = dynfloat::mixed_mul<dst_type, execution_t>(src1, arg1);
            break;
        }
        case hvx::util::elmwise_e::Sigmoid: {
            const auto exp_t = dynfloat::exp<execution_t>(-src1);
            const auto add_t = dynfloat::add<execution_t>(static_cast<typename param_::src1_type>(1), exp_t);
            dst              = static_cast<dst_type>(dynfloat::reciprocal<execution_t>(add_t));
            break;
        }
        case hvx::util::elmwise_e::Sub: {
            dst = static_cast<dst_type>(dynfloat::sub<execution_t>(static_cast<op1_type>(src1), static_cast<op1_type>(src2)));
            break;
        }
        case hvx::util::elmwise_e::Tanh: {
            dst = static_cast<dst_type>(dynfloat::tanh<execution_t>(src1));
        }
        default: {
            static_cast<dst_type>(42);
            break;
        }
    }

    (void)arg2;
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace ew
} // namespace hvx

#endif // HVX_EW_DFLOAT_H_
