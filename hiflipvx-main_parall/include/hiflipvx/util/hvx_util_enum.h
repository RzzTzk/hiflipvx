/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ï¿½AS ISï¿½, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_util_enum.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_ENUM_H_
#define HVX_UTIL_ENUM_H_

#include "hvx_util_macro.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief A list of different reduction operations
 */
enum class reduce_e : int8_t {
    Max,
    Mean,
    Min,
    Sum,
    // L1,
    // L2,
    // LogSum,
    // LogSumExp,
    // Prod,
    // SumSquare,
    None
};

/*!
 * @brief A list of different layer
 */
enum class layer_e : int8_t{
    Conv,
    Pool,
    Depthwise,
    None,
};

/*!
 * @brief A list of different layer
 */
enum class reorder_e : int8_t{
    Positive,
    Negative,
    None,
};

/*!
 * @brief A list of different elementwise operations
 */
enum class elmwise_e : int8_t {
    Abs,            // abs(src1)
    Add,            // add(src1,src2)
    AddConst,       // add(src1,arg1)
    Cast,           // cast(src1)
    Clip,           // clamp(src1, arg1, arg2)
    Div,            // div(src1,src2)
    DivConstVar,    // div(arg1,src1)
    DivVarConst,    // div(src1,arg1)
    Exp,            // exp(src1)
    Log,            // log(src1)
    Max,            // max(src1,src2)
    MaxConst,       // max(src1,arg1)
    Min,            // min(src1,src2)
    MinConst,       // min(src1,arg1)
    Mod,            // Mod(src1,src2)
    ModConstVar,    // Mod(arg1,src1)
    ModVarConst,    // Mod(src1,arg1)
    Mul,            // mul(src1,src2)
    MulConst,       // mul(src1,arg1)
    Neg,            // neg(src1)
    Pow,            // Pow(src1,src2)
    PowConstVar,    // Pow(arg1,src1)
    PowVarConst,    // Pow(src1,arg1)
    Sigmoid,        // sigmoid(src1)
    Sqrt,           // sqrt(src1)
    Sub,            // sub(src1,src2)
    Tanh,           // sigmoid(src1)



    //BitShiftLeft,   //  src1 << arg1
    //BitShiftReight, // src1 >> arg1
    //
    // Affine,     //
    // Softplus,   //
    // ScaledTanh, //
    // Sign,       //
    //
    // BitwiseAnd,
    // BitwiseNot,
    // BitwiseOr,
    // BitwiseXor,
    // LogicalAnd,
    // LogicalEqual,
    // LogicalGreater,
    // LogicalGreaterEqual,
    // LogicalLess,
    // LogicalLessEqual,
    // LogicalNot,
    // LogicalOr,
    // LogicalXor,
    None,
};

/*!
 * @brief for approximate computing
 */
enum class execution_e : int8_t {
    kFast,
    kRefinedFast,
    kRefined,
    kRefinedExact,
    kExact,
};

/*!
 * @brief for type of threshold
 */
enum class threshold_e : int8_t {
    kBinary,
    kRange
};

/*!
 * @brief for overflow
 */
enum class overflow_e : int8_t {
    kWrap,
    kSaturate,
    kClip
};

/*!
 * @brief for underflow
 */
enum class underflow_e : int8_t {
    kCeil,
    kFloor,
    kRound,
    kTrunc,
};

/*!
 * @brief for pooling type
 */
enum class pooling_e : int8_t {
    kAvg,
    kMax
};

/*!
 * @brief for axis extra signals
 */
enum axis_e : int64_t {
#if defined(HVX_SYNTHESIS_ACTIVE)
    kSof = AXIS_ENABLE_USER,
    kEof = AXIS_ENABLE_LAST,
#else
    kSof = 0, // start of frame
    kEof = 0, // end of frame
#endif
};

/*!
 * @brief all limitations
 */
enum limits_e : int64_t {
    kTensorDimMax = 6, // maximum number of tensor dimensions
};

/*!
 * @brief constant values
 */
enum num_e : int64_t {
    k0 = 0,
    k1 = 1,
};

/******************************************************************************************************************************************/

/*!
 * @brief An overload of the conventionally named function "enum_name()" (should be found by ADL) to to convert enum to const char*.
 */
HVX_FORCE_INLINE constexpr auto
enum_name(const execution_e e) noexcept -> const char* {
    switch (e) {
        case execution_e::kFast:
            return "fast";
        case execution_e::kRefinedFast:
            return "refined_fast";
        case execution_e::kRefined:
            return "refined";
        case execution_e::kRefinedExact:
            return "refined_exact";
        case execution_e::kExact:
            return "exact";
    }
    return "";
}

/*! \brief An overload of the conventionally named function "enum_name()" (should be found by ADL) to to convert enum to const char*.
 * \ingroup group_cnn
 */
constexpr auto
enum_name(const hvx::util::pooling_e e) noexcept -> const char* {
    switch (e) {
        case hvx::util::pooling_e::kMax:
            return "PoolingMax";
        case hvx::util::pooling_e::kAvg:
            return "PoolingAvg";
    } // Do not declare default case to get warning by static analyzer if a case is missing
    return "";
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_ENUM_H_
