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
 * @file    hvx_util_dfixed.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_DFIXED_H_
#define HVX_UTIL_DFIXED_H_

#include "hvx_util_math_utils.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief definition of a fixed-point data type (TODO: also add integer bit width)
 */
template<typename type_, int64_t frac_bits_ = 0>
struct dfixed {
    // compile time elements (type traits)
    static constexpr bool is_signed = std::is_signed<type_>::value;
    static constexpr bool is_int    = std::is_integral<type_>::value;
    static constexpr bool is_flt    = std::is_floating_point<type_>::value;
    //static constexpr int64_t size   = sizeof(type_);
    static constexpr int64_t digits = std::numeric_limits<type_>::digits;
    static constexpr int64_t half   = hvx::util::num_e::k1 << hvx::util::Max(0, static_cast<int32_t>(frac_bits_) - 1);
    static constexpr auto frac_bits = frac_bits_;
    using data_type                 = type_;

    // stores value
    type_ data;

    /*!
     * @brief
     */
    constexpr dfixed() noexcept = default;

    /*!
     * @brief
     */
    constexpr dfixed(float value) {
        if (std::is_integral<type_>::value == true)
            value = value * static_cast<float>(static_cast<int64_t>(1) << frac_bits_);

        data = static_cast<type_>(value);
    }

    /*!
     * @brief
     */
    operator float() const {
        constexpr float shift =
            (std::is_integral<type_>::value == false) ? (1.0f) : (1.0f / static_cast<float>(static_cast<int64_t>(1) << frac_bits_));
        return static_cast<float>(data) * shift;
    }

    /*!
     * @brief
     */
    auto operator=(float value) -> dfixed<type_, frac_bits_>& {
        if (std::is_integral<type_>::value == true)
            value = value * static_cast<float>(static_cast<int64_t>(1) << frac_bits_);

        data = static_cast<type_>(value);
        return *this;
    }

    /*!
     * @brief
     */
    static constexpr auto lowest() noexcept -> type_ {
        return std::numeric_limits<type_>::lowest();
    }

    /*!
     * @brief
     */
    static constexpr auto max() noexcept -> type_ {
        return std::numeric_limits<type_>::max();
    }
};

/*!
 * @brief
 */
namespace details_is_dfixed {
template<typename>
struct is_dfixed: std::false_type {};

/*!
 * @brief
 */
template<typename type_, int64_t frac_bits_>
struct is_dfixed<dfixed<type_, frac_bits_>>: std::true_type {};
} // namespace details_is_dfixed

/*!
 * @brief
 */
template<typename type_>
struct is_dfixed: details_is_dfixed::is_dfixed<std::remove_volatile_t<std::remove_const_t<type_>>> {};

/*!
 * @brief
 */
template<typename type_>
static constexpr auto is_dfixed_v = static_cast<bool>(is_dfixed<type_>::value);

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_DFIXED_H_
