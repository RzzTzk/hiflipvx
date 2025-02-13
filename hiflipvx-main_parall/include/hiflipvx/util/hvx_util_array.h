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
 * @file    hvx_util_array.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_ARRAY_H_
#define HVX_UTIL_ARRAY_H_

#include "hvx_util_vector.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief a simple array data type
 */
template<typename type_, int64_t cols_>
struct array1d {
    // stores the values
#ifdef HVX_SYNTHESIS_ACTIVE
    type_ data[cols_]; // NOLINT
#else
    type_* data = new type_[cols_];
#endif
    using data_type = type_;

    /*!
     * @brief gets an element (call by reference)
     */
    HVX_FORCE_INLINE constexpr auto Get(const int64_t col) noexcept -> type_& {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(col < cols_);
#endif
        return data[col]; // NOLINT
    }

    /*!
     * @brief stores an element
     */
    HVX_FORCE_INLINE constexpr auto Set(type_ value, const int64_t col) noexcept -> void {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(col < cols_);
#endif
        data[col] = value; // NOLINT
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief a simple array data type
 */
template<typename type_, int64_t cols_, int64_t rows_>
struct array2d {
    // stores the values
#ifdef HVX_SYNTHESIS_ACTIVE
    type_ data[rows_][cols_]; // NOLINT
#else
    type_* data = new type_[cols_ * rows_];
#endif
    using data_type = type_;

    /*!
     * @brief gets an element (call by reference)
     */
    HVX_FORCE_INLINE constexpr auto Get(const int64_t col, const int64_t row) noexcept -> type_& {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(col < cols_);
        assert(row < rows_);
#endif
#ifdef HVX_SYNTHESIS_ACTIVE
        return data[row][col]; // NOLINT
#else
        return data[row * cols_ + col];  // NOLINT
#endif
    }

    /*!
     * @brief stores an element
     */
    HVX_FORCE_INLINE constexpr auto Set(type_& value, const int64_t col, const int64_t row) noexcept -> void {
        HVX_INLINE_TOP();
#if !defined(HVX_SYNTHESIS_ACTIVE)
        assert(col < cols_);
        assert(row < rows_);
#endif
#ifdef HVX_SYNTHESIS_ACTIVE
        data[row][col] = value; // NOLINT
#else
        data[row * cols_ + col] = value; // NOLINT
#endif
    }
};

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_ARRAY_H_
