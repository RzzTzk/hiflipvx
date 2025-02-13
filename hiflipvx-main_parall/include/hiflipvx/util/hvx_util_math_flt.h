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
 * @file    hvx_util_math_flt.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_MATH_FLT_H_
#define HVX_UTIL_MATH_FLT_H_

#include "hvx_util_tensor.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

// Function for right shifting a floating-point variable by shifts positions
template<int32_t shift_>
auto
FltRightShift(float value) noexcept -> float {
    static_assert(shift_ >= 0, "Right shift function cannot accept negative shifts.");

    int32_t exponent = 0;
    float mantissa   = std::frexp(value, &exponent); // Get mantissa and exponent
    exponent -= shift_;                              // Increase exponent
    return std::ldexp(mantissa, exponent);           // Construct the shifted value
}

// Function for left shifting a floating-point variable by shifts positions
template<int32_t shift_>
auto
FltLeftShift(float value) noexcept -> float {
    static_assert(shift_ >= 0, "Right shift function cannot accept negative shifts.");

    int32_t exponent = 0;
    float mantissa   = std::frexp(value, &exponent); // Get mantissa and exponent
    exponent += shift_;                              // Decrease exponent (shifts is positive for left shift)
    return std::ldexp(mantissa, exponent);           // Construct the shifted value
}

/******************************************************************************************************************************************/

template<hvx::util::elmwise_e op_type_, int64_t shift_ = 0>
HVX_FORCE_INLINE auto
FltOp(const float src1, const float src2 = 0.0f, const float src3 = 0.0f) -> float {
    float res = 0.0f;

    switch (op_type_) {
        case hvx::util::elmwise_e::Abs:
            res = hvx::util::Abs(src1);
            break;
        case hvx::util::elmwise_e::Add:
        case hvx::util::elmwise_e::AddConst:
            res = src1 + src2;
            break;
        case hvx::util::elmwise_e::Clip:
            res = hvx::util::Clamp(src1, src2, src3);
            break;
        case hvx::util::elmwise_e::Max:
        case hvx::util::elmwise_e::MaxConst:
            res = hvx::util::Max(src1, src2);
            break;
        case hvx::util::elmwise_e::Min:
        case hvx::util::elmwise_e::MinConst:
            res = hvx::util::Min(src1, src2);
            break;
        case hvx::util::elmwise_e::Mul:
        case hvx::util::elmwise_e::MulConst:
            res = src1 * src2;
            break;
        case hvx::util::elmwise_e::Sigmoid:
            res = 1.0f / (1.0f + std::exp(-src1));
            break;
        case hvx::util::elmwise_e::Sub:
            res = src1 - src2;
            break;
        case hvx::util::elmwise_e::Tanh: {
            res = hvx::util::FltLeftShift<1>(src1);
            res = 1.0f - 2.0f / (1.0f + std::exp(res));
            break;
        }
        case hvx::util::elmwise_e::Sqrt:
            res = std::sqrt(src1);
            break;
        case hvx::util::elmwise_e::Neg:
            res = -src1;
            break;
        case hvx::util::elmwise_e::Cast:
            res = src1;
            break;
        case hvx::util::elmwise_e::Exp:
            res = std::exp(src1);
            break;
        case hvx::util::elmwise_e::Log:
            res = std::log(src1);
            break;
        case hvx::util::elmwise_e::Div:
        case hvx::util::elmwise_e::DivConstVar:
        case hvx::util::elmwise_e::DivVarConst:
            res = src1 / src2;
            break;
        case hvx::util::elmwise_e::Mod:
        case hvx::util::elmwise_e::ModConstVar:
        case hvx::util::elmwise_e::ModVarConst:
            res = std::fmod(src1, src2);
            break;
        case hvx::util::elmwise_e::Pow:
        case hvx::util::elmwise_e::PowConstVar:
        case hvx::util::elmwise_e::PowVarConst:
            res = std::pow(src1, src2);
            break;
        case hvx::util::elmwise_e::None:
        default:
            res = 42.0f;
            break;
    }

    return res;
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltAdd(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
    return src1 + src2;
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltSub(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
    return src1 - src2;
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltMul(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
    return src1 * src2;
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltAbs(const float src) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::fabs(src);
#else
    return hvx::util::Abs(src);
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltNeg(const float src) -> float {
    HVX_INLINE_TOP();
    return -src;
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltExp(const float src) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::exp(src);
#else
    return std::exp(src);
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltLog(const float src) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::log(src);
#else
    return std::log(src);
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltDiv(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::divide(src1, src2);
#else
    return src1 / src2;
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltMax(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::fmax(src1, src2);
#else
    return hvx::util::Max(src1, src2);
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltMin(const float src1, const float src2) -> float {
    HVX_INLINE_TOP();
#if defined(HVX_SYNTHESIS_ACTIVE)
    return hls::fmin(src1, src2);
#else
    return hvx::util::Min(src1, src2);
#endif
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltClamp(const float src, const float low, const float high) -> float {
    HVX_INLINE_TOP();
    return hvx::util::Clamp(src, low, high);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltSigmoid(const float src) -> float {
    HVX_INLINE_TOP();
    float res = FltNeg(src);
    res       = FltExp(res);
    res       = FltAdd(1.0f, res);
    return FltDiv(1.0f, res);
}

/*!
 * @brief
 */
HVX_FORCE_INLINE auto
FltTanh(const float src) -> float {
    HVX_INLINE_TOP();
    float res = FltLeftShift<1>(src);
    res       = FltExp(res);
    res       = FltAdd(1.0f, res);
    res       = FltDiv(2.0f, res);
    return FltSub(1.0f, res);
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_MATH_FLT_H_
