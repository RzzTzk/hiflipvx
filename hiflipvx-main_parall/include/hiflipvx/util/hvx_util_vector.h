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
 * @file    hvx_util_vector.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_VECTORR_H_
#define HVX_UTIL_VECTORR_H_

#include "hvx_util_def_types.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief a simple vector data type
 */
template<typename type_, int64_t size_>
struct vector {
    // stores values
    using type = type_;
    type_ data[size_]; // NOLINT

    /*!
     * @brief gets an element (call by reference)
     */
    HVX_FORCE_INLINE constexpr auto Get(const int64_t ptr) noexcept -> type_& {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(ptr < size_);
#endif
        return data[ptr]; // NOLINT
    }

    /*!
     * @brief stores an element
     */
    HVX_FORCE_INLINE constexpr auto Set(type_& value, const int64_t ptr) noexcept -> void {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(ptr < size_);
#endif
        data[ptr] = value; // NOLINT
    }
};

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_VECTORR_H_
