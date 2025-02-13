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
 * @file    hvx_util_math_utils.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_MATH_UTILS_H_
#define HVX_UTIL_MATH_UTILS_H_

#include "hvx_util_type_traits.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief get maximum of 2 values
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Max(const type_& a, const type_& b) noexcept -> type_ {
    return (a > b) ? (a) : (b);
}

/*!
 * @brief get minimum of 2 values
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Min(const type_& a, const type_& b) noexcept -> type_ {
    return (a < b) ? (a) : (b);
}

/*!
 * @brief get absolute value
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Abs(const type_& a) noexcept -> type_ {
    return (a < 0) ? (-a) : (a);
}

/*!
 * @brief clamp value with upper and lower limit
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Clamp(const type_& a, const type_& low, const type_& high) noexcept -> type_ {
    assert(high >= low);
    return (a < low) ? (low) : ((a > high) ? high : a);
}

/*!
 * @brief computes the greatest common divider
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Gcd(const type_ a, const type_ b) noexcept -> type_ {
    return (b == type_(0) ? a : hvx::util::Gcd(b, a % b));
}

/*!
 * @brief computes the least common multiple
 */
template<typename type_>
HVX_FORCE_INLINE constexpr auto
Lcm(const type_ a, const type_ b) noexcept -> type_ {
    return hvx::util::Abs(a * (b / hvx::util::Gcd(a, b)));
}

/*!
 * @brief computes floor(log2(n)) using constexpr
 */
HVX_FORCE_INLINE constexpr auto
Log2Floor(const int64_t n) noexcept -> int64_t {
    return (n <= 1) ? 0 : 1 + Log2Floor(n / 2);
}

/*!
 * @brief computes ceil(log2(n)) using constexpr
 */
HVX_FORCE_INLINE constexpr auto
Log2Ceil(int64_t n) noexcept -> int64_t {
    return (n <= 1) ? 0 : Log2Floor(n - 1) + 1;
}

/******************************************************************************************************************************************/

/*!
 * @brief Checks if type is contained in a list of types
 */
template<typename>
HVX_FORCE_INLINE constexpr auto
CompareDataType() noexcept -> bool {
    HVX_INLINE_TOP();
    return false;
}

/*!
 * @brief Checks if type is contained in a list of types
 */
template<typename type_, typename first_type_, typename... rest_type_>
HVX_FORCE_INLINE constexpr auto
CompareDataType() noexcept -> bool {
    HVX_INLINE_TOP();
    using stripped_type1 = std::remove_reference_t<std::remove_volatile_t<std::remove_const_t<type_>>>;
    using stripped_type2 = std::remove_reference_t<std::remove_volatile_t<std::remove_const_t<first_type_>>>;
    return std::is_same<stripped_type1, stripped_type2>::value || hvx::util::CompareDataType<type_, rest_type_...>();
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_MATH_UTILS_H_
