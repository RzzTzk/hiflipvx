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
 * @file    hvx_ew_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_EW_DFIXED_H_
#define HVX_EW_DFIXED_H_

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
         std::enable_if_t<hvx::util::is_dfixed_v<src1_type>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<src2_type>, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<dst_type>, bool>  = true,
         std::enable_if_t<hvx::util::is_dfixed_v<arg_type>, bool>  = true>
HVX_FORCE_INLINE constexpr auto
VerifyDataType() noexcept -> void {
    HVX_INLINE_TOP();

    // compile time assertions
    static_assert(hvx::util::CompareDataType<typename src1_type::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<typename src2_type::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<typename dst_type::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>() &&
                      hvx::util::CompareDataType<typename arg_type::data_type, uint8_t, uint16_t, int8_t, int16_t, int32_t, float>(),
                  "Data type is not supported!");
    static_assert(hvx::util::CheckFracBitWidth<typename src1_type::data_type, src1_type::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename src2_type::data_type, src2_type::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename dst_type::data_type, dst_type::frac_bits>() &&
                      hvx::util::CheckFracBitWidth<typename arg_type::data_type, arg_type::frac_bits>(),
                  "Fraction size out of scope!");
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes the number of fraction bits after the operation
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
OpFracBits() noexcept -> int64_t {
    switch (param_::op_type) {
        case hvx::util::elmwise_e::Add:
        case hvx::util::elmwise_e::Max:
        case hvx::util::elmwise_e::Min:
        case hvx::util::elmwise_e::Sub:
            return hvx::util::Max(param_::src1_type::frac_bits, param_::src2_type::frac_bits);
        case hvx::util::elmwise_e::Clip:
        case hvx::util::elmwise_e::AddConst:
        case hvx::util::elmwise_e::MaxConst:
        case hvx::util::elmwise_e::MinConst:
            return hvx::util::Max(param_::src1_type::frac_bits, param_::arg_type::frac_bits);
        case hvx::util::elmwise_e::Abs:
        case hvx::util::elmwise_e::Sigmoid:
        case hvx::util::elmwise_e::Tanh:
            return param_::src1_type::frac_bits;
        case hvx::util::elmwise_e::Mul:
            return param_::src1_type::frac_bits + param_::src2_type::frac_bits;
        case hvx::util::elmwise_e::MulConst:
            return param_::src1_type::frac_bits + param_::arg_type::frac_bits;
        default:
            return 0;
    }
}

/*
 * @brief Applies the different policies like overflow and underflow
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
IntApplyPolicies(int64_t& value) noexcept -> typename param_::dst_type::data_type {
    // checks if data needs to be shifted to left/right direction
    constexpr int64_t op_frac_bits = hvx::ew::impl::OpFracBits<param_>();
    constexpr auto shift_bits      = static_cast<uint32_t>(hvx::util::Abs(op_frac_bits - param_::dst_type::frac_bits));
    constexpr bool shift_right     = (op_frac_bits > param_::dst_type::frac_bits);

    // shift input value to output fraction size and check for underflow
    if (shift_right == true)
        value = hvx::util::FixedUnderflow<shift_bits, param_::underflow_type, true>(value);
    else
        value = static_cast<int64_t>(static_cast<uint64_t>(value) << shift_bits);

    // check input for overflow and cast to output data type (TODO: operation dependent overflow)
    if (param_::overflow_type == hvx::util::overflow_e::kSaturate) {
        value = hvx::util::Min(value, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::max()));
        value = hvx::util::Max(value, static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::lowest()));
    }

    // convert to destination data type
    return static_cast<typename param_::dst_type::data_type>(value);
}

/******************************************************************************************************************************************/

/*!
 * @brief ADD: a + b
 */
template<typename opa_type_, typename opb_type_>
HVX_FORCE_INLINE constexpr auto
Add(opa_type_& opa, opb_type_& opb) noexcept -> int64_t {
    auto opa_data = static_cast<int64_t>(opa.data);
    auto opb_data = static_cast<int64_t>(opb.data);
    hvx::util::Cast2FixedToBigger<opa_type_::frac_bits, opb_type_::frac_bits>(opa_data, opb_data);
    return opa_data + opb_data;
}

/*!
 * @brief CLIP(a, low, high)
 */
template<typename opa_type_, typename arg_type>
HVX_FORCE_INLINE constexpr auto
Clip(opa_type_& opa, arg_type& arg1, arg_type& arg2) noexcept -> int64_t {
    auto opa_data  = static_cast<int64_t>(opa.data);
    auto arg1_data = static_cast<int64_t>(arg1.data);
    auto arg2_data = static_cast<int64_t>(arg2.data);
    hvx::util::Cast3FixedToBigger<opa_type_::frac_bits, arg_type::frac_bits>(opa_data, arg1_data, arg2_data);
    return hvx::util::Clamp(opa_data, arg1_data, arg2_data);
}

/*!
 * @brief MAX(a, b)
 */
template<typename opa_type_, typename opb_type_>
HVX_FORCE_INLINE constexpr auto
Max(opa_type_& opa, opb_type_& opb) noexcept -> int64_t {
    auto opa_data = static_cast<int64_t>(opa.data);
    auto opb_data = static_cast<int64_t>(opb.data);
    hvx::util::Cast2FixedToBigger<opa_type_::frac_bits, opb_type_::frac_bits>(opa_data, opb_data);
    return hvx::util::Max(opa_data, opb_data);
}

/*!
 * @brief MIN(a, b)
 */
template<typename opa_type_, typename opb_type_>
HVX_FORCE_INLINE constexpr auto
Min(opa_type_& opa, opb_type_& opb) noexcept -> int64_t {
    auto opa_data = static_cast<int64_t>(opa.data);
    auto opb_data = static_cast<int64_t>(opb.data);
    hvx::util::Cast2FixedToBigger<opa_type_::frac_bits, opb_type_::frac_bits>(opa_data, opb_data);
    return hvx::util::Min(opa_data, opb_data);
}

/*!
 * @brief MUL: a * b
 */
template<typename opa_type_, typename opb_type_>
HVX_FORCE_INLINE constexpr auto
Mul(opa_type_& opa, opb_type_& opb) noexcept -> int64_t {
    return static_cast<int64_t>(opa.data) * static_cast<int64_t>(opb.data);
}

/*!
 * @brief SIGMOID: 1 / (1 + EXP(-a)) - TODO: fixed-point
 */
template<typename param_, typename opa_type_>
HVX_FORCE_INLINE constexpr auto
Sigmoid(opa_type_ opa) noexcept -> int64_t {
    const float data = hvx::util::CastDfixedToFlt<opa_type_>(opa);                                // to floating-point
    const float res  = hvx::util::FltSigmoid(data);                                               // calculate
    return hvx::util::CastFltToFixed<int64_t, opa_type_::frac_bits, param_::underflow_type>(res); // to fixed-point
}

/*!
 * @brief SUB: a - b
 */
template<typename opa_type_, typename opb_type_>
HVX_FORCE_INLINE constexpr auto
Sub(opa_type_& opa, opb_type_& opb) noexcept -> int64_t {
    auto opa_data = static_cast<int64_t>(opa.data);
    auto opb_data = static_cast<int64_t>(opb.data);
    hvx::util::Cast2FixedToBigger<opa_type_::frac_bits, opb_type_::frac_bits>(opa_data, opb_data);
    return opa_data - opb_data;
}

/*!
 * @brief TANH(a) - TODO: fixed-point
 */
template<typename param_, typename opa_type_>
HVX_FORCE_INLINE constexpr auto
Tanh(opa_type_ opa) noexcept -> int64_t {
    const float data = hvx::util::CastDfixedToFlt<opa_type_>(opa);                                // to floating-point
    const float res  = hvx::util::FltTanh(data);                                                  // calculate
    return hvx::util::CastFltToFixed<int64_t, opa_type_::frac_bits, param_::underflow_type>(res); // to fixed-point
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes the elementwise function for integer fixed point data type
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src1_type> && param_::src1_type::is_int, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src2_type> && param_::src2_type::is_int, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_int, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::arg_type> && param_::arg_type::is_int, bool>   = true>
HVX_FORCE_INLINE constexpr auto
ElementwiseComp(typename param_::src1_type src1,
                typename param_::src2_type src2,
                typename param_::arg_type arg1,
                typename param_::arg_type arg2,
                typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // input/output variables
    int64_t dst_data = 0;

    switch (param_::op_type) {
        case hvx::util::elmwise_e::Abs:
            dst_data = hvx::util::Abs(src1.data);
            break;
        case hvx::util::elmwise_e::Add:
            dst_data = hvx::ew::impl::Add(src1, src2);
            break;
        case hvx::util::elmwise_e::AddConst:
            dst_data = hvx::ew::impl::Add(src1, arg1);
            break;
        case hvx::util::elmwise_e::Clip:
            dst_data = hvx::ew::impl::Clip(src1, arg1, arg2);
            break;
        case hvx::util::elmwise_e::Max:
            dst_data = hvx::ew::impl::Max(src1, src2);
            break;
        case hvx::util::elmwise_e::MaxConst:
            dst_data = hvx::ew::impl::Max(src1, arg1);
            break;
        case hvx::util::elmwise_e::Min:
            dst_data = hvx::ew::impl::Min(src1, src2);
            break;
        case hvx::util::elmwise_e::MinConst:
            dst_data = hvx::ew::impl::Min(src1, arg1);
            break;
        case hvx::util::elmwise_e::Mul:
            dst_data = hvx::ew::impl::Mul(src1, src2);
            break;
        case hvx::util::elmwise_e::MulConst:
            dst_data = hvx::ew::impl::Mul(src1, arg1);
            break;
        case hvx::util::elmwise_e::Sigmoid:
            dst_data = hvx::ew::impl::Sigmoid<param_>(src1);
            break;
        case hvx::util::elmwise_e::Sub:
            dst_data = hvx::ew::impl::Sub(src1, src2);
            break;
        case hvx::util::elmwise_e::Tanh:
            dst_data = hvx::ew::impl::Tanh<param_>(src1);
            break;
        default:
            dst_data = 42;
            break;
    }

    // applcy underflow/overflow policies on output
    dst.data = hvx::ew::impl::IntApplyPolicies<param_>(dst_data);
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes the elementwise function for floating point data type
 */
template<typename param_,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src1_type> && param_::src1_type::is_flt, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::src2_type> && param_::src2_type::is_flt, bool> = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::dst_type> && param_::dst_type::is_flt, bool>   = true,
         std::enable_if_t<hvx::util::is_dfixed_v<typename param_::arg_type> && param_::arg_type::is_flt, bool>   = true>
HVX_FORCE_INLINE constexpr auto
ElementwiseComp(typename param_::src1_type src1,
                typename param_::src2_type src2,
                typename param_::arg_type arg1,
                typename param_::arg_type arg2,
                typename param_::dst_type& dst) noexcept -> void {
    HVX_INLINE_TOP();

    switch (param_::op_type) {
        case hvx::util::elmwise_e::Abs:
        case hvx::util::elmwise_e::Cast:
        case hvx::util::elmwise_e::Exp:
        case hvx::util::elmwise_e::Log:
        case hvx::util::elmwise_e::Neg:
        case hvx::util::elmwise_e::Sigmoid:
        case hvx::util::elmwise_e::Sqrt:
        case hvx::util::elmwise_e::Tanh:
            dst.data = hvx::util::FltOp<param_::op_type>(src1.data);
            break;
        case hvx::util::elmwise_e::Add:
        case hvx::util::elmwise_e::Div:
        case hvx::util::elmwise_e::Max:
        case hvx::util::elmwise_e::Min:
        case hvx::util::elmwise_e::Mod:
        case hvx::util::elmwise_e::Mul:
        case hvx::util::elmwise_e::Pow:
        case hvx::util::elmwise_e::Sub:
            dst.data = hvx::util::FltOp<param_::op_type>(src1.data, src2.data);
            break;
        case hvx::util::elmwise_e::AddConst:
        case hvx::util::elmwise_e::DivVarConst:
        case hvx::util::elmwise_e::MaxConst:
        case hvx::util::elmwise_e::MinConst:
        case hvx::util::elmwise_e::ModVarConst:
        case hvx::util::elmwise_e::MulConst:
        case hvx::util::elmwise_e::PowVarConst:
            dst.data = hvx::util::FltOp<param_::op_type>(src1.data, arg1.data);
            break;
        case hvx::util::elmwise_e::DivConstVar:
        case hvx::util::elmwise_e::ModConstVar:
        case hvx::util::elmwise_e::PowConstVar:
            dst.data = hvx::util::FltOp<param_::op_type>(arg1.data, src1.data);
            break;
        case hvx::util::elmwise_e::Clip:
            dst.data = hvx::util::FltOp<param_::op_type>(src1.data, arg1.data, arg2.data);
            break;
        case hvx::util::elmwise_e::None:
        default:
            dst.data = 42;
            break;
    }
}

/******************************************************************************************************************************************/
} // namespace impl
} // namespace ew
} // namespace hvx

#endif // HVX_EW_DFIXED_H_

///*!
// * @brief
// */
// template<typename param_>
// HVX_FORCE_INLINE constexpr auto
// IntApplyPolicies(int64_t& value) noexcept -> typename param_::dst_type::data_type {
//    // maximum and minimum values
//    constexpr auto dst_max = static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::max());
//    constexpr auto dst_min = static_cast<int64_t>(std::numeric_limits<typename param_::dst_type::data_type>::lowest());
//
//    // the maximum/minimum values after a sepcific operation
//    constexpr int64_t op_max       = hvx::ew::impl::OpMax<param_>();
//    constexpr int64_t op_min       = hvx::ew::impl::OpMin<param_>();
//    constexpr int64_t op_frac_bits = hvx::ew::impl::OpFracBits<param_>();
//
//    // checks if data needs to be shifted to left/right direction (TODO: op_res_frac_bits not always equal src_type_frac_bits)
//    constexpr auto shift_val   = static_cast<uint32_t>(hvx::util::Abs(op_frac_bits - param_::dst_type::frac_bits));
//    constexpr bool shift_right = (op_frac_bits > param_::dst_type::frac_bits);
//
//    // checks if positive/negative overflow is possible
//    constexpr int64_t negate   = ((op_min < 0) ? (-1) : (1));
//    constexpr bool pos_overflow = ((shift_right) ? (op_max >> shift_val) : (op_max << shift_val)) > dst_max;
//    constexpr bool neg_overflow =
//        (((shift_right) ? (hvx::util::Abs(op_min) >> shift_val) : (hvx::util::Abs(op_min) << shift_val)) * negate) < dst_min;
//
//    // shift input value to output fraction size and check for underflow (TODO: rounding methods)
//    if (shift_right == true) {
//        if (param_::underflow_type == hvx::util::underflow_e::kRound)
//            value = (value + param_::dst_type::half) >> shift_val;
//        else
//            value = value >> shift_val;
//    } else {
//        value = value << shift_val;
//    }
//
//    // check input for overflow and cast to output data type
//    if (param_::overflow_type == hvx::util::overflow_e::kSaturate) {
//        if (pos_overflow == true)
//            value = hvx::util::Min(value, static_cast<int64_t>(dst_max));
//        if (neg_overflow == true)
//            value = hvx::util::Max(value, static_cast<int64_t>(dst_min));
//    }
//    return static_cast<typename param_::dst_type::data_type>(value);
//}

///*!
// * @brief
// */
// template<typename param_>
// HVX_FORCE_INLINE constexpr auto
// OpMax() noexcept -> int64_t {
//    constexpr auto src1_max        = static_cast<int64_t>(std::numeric_limits<typename param_::src1_type::data_type>::max());
//    constexpr auto src2_max        = static_cast<int64_t>(std::numeric_limits<typename param_::src2_type::data_type>::max());
//    constexpr auto arg1_max        = static_cast<int64_t>(std::numeric_limits<typename param_::arg_type::data_type>::max());
//    constexpr auto zero            = static_cast<int64_t>(0);
//    constexpr auto src1_src2_shift = hvx::util::Max(param_::src1_type::frac_bits - param_::src2_type::frac_bits, zero);
//    constexpr auto src2_src1_shift = hvx::util::Max(param_::src2_type::frac_bits - param_::src1_type::frac_bits, zero);
//    constexpr auto src1_arg1_shift = hvx::util::Max(param_::src1_type::frac_bits - param_::arg_type::frac_bits, zero);
//    constexpr auto arg1_src1_shift = hvx::util::Max(param_::arg_type::frac_bits - param_::src1_type::frac_bits, zero);
//
//    switch (param_::op_type) {
//        case hvx::util::elmwise_e::Add:
//            return (src1_max << src1_src2_shift) + (src2_max << src2_src1_shift);
//        case hvx::util::elmwise_e::AddConst:
//            return (src1_max << src1_arg1_shift) + (src2_max << arg1_src1_shift);
//        case hvx::util::elmwise_e::Max:
//            return hvx::util::Max((src1_max << src1_src2_shift), (arg1_max << src2_src1_shift));
//        case hvx::util::elmwise_e::MaxConst:
//            return hvx::util::Max((src1_max << src1_arg1_shift), (arg1_max << arg1_src1_shift));
//        case hvx::util::elmwise_e::Sigmoid:
//            return 1 << param_::src1_type::frac_bits;
//        default:
//            return 0;
//    }
//}
//
///*!
// * @brief
// */
// template<typename param_>
// HVX_FORCE_INLINE constexpr auto
// OpMin() noexcept -> int64_t {
//    constexpr auto src1_min        = static_cast<int64_t>(std::numeric_limits<typename param_::src1_type::data_type>::min());
//    constexpr auto src2_min        = static_cast<int64_t>(std::numeric_limits<typename param_::src2_type::data_type>::min());
//    constexpr auto arg1_min        = static_cast<int64_t>(std::numeric_limits<typename param_::arg_type::data_type>::min());
//    constexpr auto zero            = static_cast<int64_t>(0);
//    constexpr auto src1_src2_shift = hvx::util::Max(param_::src1_type::frac_bits - param_::src2_type::frac_bits, zero);
//    constexpr auto src2_src1_shift = hvx::util::Max(param_::src2_type::frac_bits - param_::src1_type::frac_bits, zero);
//    constexpr auto src1_arg1_shift = hvx::util::Max(param_::src1_type::frac_bits - param_::arg_type::frac_bits, zero);
//    constexpr auto arg1_src1_shift = hvx::util::Max(param_::arg_type::frac_bits - param_::src1_type::frac_bits, zero);
//
//    switch (param_::op_type) {
//        case hvx::util::elmwise_e::Add:
//            return (src1_min << src1_src2_shift) - (src2_min << src2_src1_shift);
//        case hvx::util::elmwise_e::AddConst:
//            return (src1_min << src1_arg1_shift) - (src2_min << arg1_src1_shift);
//        case hvx::util::elmwise_e::Max:
//            return hvx::util::Min((src1_min << src1_src2_shift), (arg1_min << src2_src1_shift));
//        case hvx::util::elmwise_e::MaxConst:
//            return hvx::util::Min((src1_min << src1_arg1_shift), (arg1_min << arg1_src1_shift));
//        case hvx::util::elmwise_e::Sigmoid:
//        default:
//            return 0;
//    }
//}
