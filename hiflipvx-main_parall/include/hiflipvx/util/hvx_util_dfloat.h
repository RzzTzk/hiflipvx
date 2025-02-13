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
 * @file    hvx_util_dfloat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_DFLOAT_H_
#define HVX_UTIL_DFLOAT_H_

#include "hvx_util_tensor.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

HVX_FORCE_INLINE constexpr auto
ToDfloatExecution(const execution_e e) noexcept -> dynfloat::strategy {
    switch (e) {
        case execution_e::kFast:
            return dynfloat::strategy::fast;
        case execution_e::kRefinedFast:
            return dynfloat::strategy::refined_fast;
        case execution_e::kRefined:
            return dynfloat::strategy::refined;
        case execution_e::kRefinedExact:
            return dynfloat::strategy::refined_exact;
        case execution_e::kExact:
            return dynfloat::strategy::exact;
    }
    return dynfloat::strategy::exact;
}

HVX_FORCE_INLINE constexpr auto // TODO
ToDfloatUnderflow(const hvx::util::underflow_e p) {
    switch (p) {
        case hvx::util::underflow_e::kTrunc:
            return dynfloat::rounding_behavior::to_zero;
        case hvx::util::underflow_e::kRound:
            return dynfloat::rounding_behavior::nearest_even;
        default:
            return dynfloat::rounding_behavior::to_zero;
    }
    return dynfloat::rounding_behavior::to_zero;
}

HVX_FORCE_INLINE constexpr auto
ToDfloatOverflow(const hvx::util::overflow_e p) {
    switch (p) {
        case hvx::util::overflow_e::kWrap:
            return dynfloat::special_values::zero;
        case hvx::util::overflow_e::kSaturate:
            return dynfloat::special_values::zero_and_saturation;
        default:
            return dynfloat::special_values::zero;
    }
    return dynfloat::special_values::zero_and_saturation;
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_DFLOAT_H_
