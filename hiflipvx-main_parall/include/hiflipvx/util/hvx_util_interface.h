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
 * @file    hvx_util_interface.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_INTERFACE_H_
#define HVX_UTIL_INTERFACE_H_

#include "hvx_util_tensor.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief Stream data type. Stores a vector of elements.
 */
template<typename type_, int64_t vec_size_>
struct stream_data {
    // data vector
    type_ data[vec_size_]; // NOLINT
    using data_type = type_;

    /*!
     * @brief gets an element (call by reference)
     */
    HVX_FORCE_INLINE constexpr auto Get(const int64_t col) noexcept -> type_& {
        HVX_INLINE_TOP();
        return data[col]; // NOLINT
    }

    /*!
     * @brief stores an element
     */
    HVX_FORCE_INLINE constexpr auto Set(type_& value, const int64_t col) noexcept -> void {
        HVX_INLINE_TOP();
        data[col] = value; // NOLINT
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief verify if read/write pointers and last signal are correct
 */
template<typename src_dim_, typename dst_dim_>
HVX_FORCE_INLINE constexpr auto
StreamSignalVerify(int64_t ptr_src, int64_t ptr_dst) noexcept -> void {
#if !defined(HVX_SYNTHESIS_ACTIVE)
    assert(ptr_src == src_dim_::vec_elms);
    assert(ptr_dst == dst_dim_::vec_elms);
#endif
    (void)ptr_src;
    (void)ptr_dst;
}

/*!
 * @brief Reads a vector from the input if a condition is met
 */
template<typename src_port, typename src_vec>
HVX_FORCE_INLINE constexpr auto
StreamReadData(src_port* src, src_vec& src_data, int64_t& ptr_src, bool cond) noexcept -> void {
    if (cond == true) {
        src_data = src[ptr_src]; // NOLINT
        ++ptr_src;
    }
}

/*!
 * @brief Writes a vector ta the input if a condition is met
 */
template<typename dst_port, typename dst_vec>
HVX_FORCE_INLINE constexpr auto
StreamWriteData(dst_port* dst, dst_vec& dst_data, int64_t& ptr_dst, bool cond) noexcept -> void {
    if (cond == true) {
        dst[ptr_dst] = dst_data; // NOLINT
        ++ptr_dst;
    }
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_INTERFACE_H_
