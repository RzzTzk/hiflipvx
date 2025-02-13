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
 * @file    hvx_util_def_types.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_DEF_TYPES_H_
#define HVX_UTIL_DEF_TYPES_H_

#include "hvx_util_dfixed.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

template<int64_t bits_>
struct select_int {
    static_assert(bits_ <= 64, "Unsupported bit width");
    using type =
        std::conditional_t<bits_ <= 8, int8_t, std::conditional_t<bits_ <= 16, int16_t, std::conditional_t<bits_ <= 32, int32_t, int64_t>>>;
};

template<int64_t bits_>
struct select_uint {
    static_assert(bits_ <= 64, "Unsupported bit width");
    using type = std::
        conditional_t<bits_ <= 8, uint8_t, std::conditional_t<bits_ <= 16, uint16_t, std::conditional_t<bits_ <= 32, uint32_t, int64_t>>>;
};

/******************************************************************************************************************************************/

/*!
 * @brief definition of a default integer type
 */
template<typename, typename, typename = void, typename = void>
struct def_int_type {
    using type = int64_t;
};

/*!
 * @brief definition of a default integer type (when using dfixed)
 */
template<typename src_type_, typename wgts_type_>
struct def_int_type<src_type_,
                    wgts_type_,
                    std::enable_if_t<hvx::util::is_dfixed_v<src_type_>>,
                    std::enable_if_t<hvx::util::is_dfixed_v<wgts_type_>>> {
    using type = hvx::util::dfixed<int64_t, 0>;
};

#if defined(HIFLIPVX_DYNFLOAT_ACTIVE)
/*!
 * @brief definition of a default integer type (when using dynfloat)
 */
template<typename src_type_, typename wgts_type_>
struct def_int_type<src_type_,
                    wgts_type_,
                    std::enable_if_t<dynfloat::is_dfloat_v<src_type_>>,
                    std::enable_if_t<dynfloat::is_dfloat_v<wgts_type_>>> {
    using type = dynfloat::dfloat<hvx::util::Max(src_type_::exp_bits, wgts_type_::exp_bits),
                                  hvx::util::Max(src_type_::man_bits, wgts_type_::man_bits)>;
};
#endif

/*!
 * @brief definition of a default integer type (incl. dynfloat/dfixed)
 */
template<typename src_type_, typename wgts_type_>
using def_int_type_t = typename def_int_type<src_type_, wgts_type_>::type;

/******************************************************************************************************************************************/

/*!
 * @brief definition of a default flaoting-point type
 */
template<typename, typename = void>
struct def_flt_type {
    using type = float;
};

/*!
 * @brief definition of a default flaoting-point type (when using dfixed)
 */
template<typename src_type_>
struct def_flt_type<src_type_, std::enable_if_t<hvx::util::is_dfixed_v<src_type_>>> {
    using type = hvx::util::dfixed<float, 0>;
};

#if defined(HIFLIPVX_DYNFLOAT_ACTIVE)
/*!
 * @brief definition of a default flaoting-point type (when using dynfloat)
 */
template<typename src_type_>
struct def_flt_type<src_type_, std::enable_if_t<dynfloat::is_dfloat_v<src_type_>>> {
    using type = src_type_;
};
#endif

/*!
 * @brief definition of a default flaoting-point type (incl. dynfloat/dfixed)
 */
template<typename src_type_>
using def_flt_type_t = typename def_flt_type<src_type_>::type;

/******************************************************************************************************************************************/

template<typename src_type, int64_t bits_, typename = void>
struct def_buf_type {
    using type = src_type;
};

template<typename src_type, int64_t extra_bits_>
struct def_buf_type<src_type, extra_bits_, std::enable_if_t<hvx::util::is_dfixed_v<src_type>>> {
    using int_type = std::conditional_t<src_type::is_signed,
                                        typename hvx::util::select_int<sizeof(typename src_type::data_type) * 8 + extra_bits_>::type,
                                        typename hvx::util::select_uint<sizeof(typename src_type::data_type) * 8 + extra_bits_>::type>;
    using type     = std::conditional_t<src_type::is_flt, float, int_type>;
};

template<typename src_type, int64_t extra_bits_>
struct def_buf_type<src_type, extra_bits_, std::enable_if_t<dynfloat::is_dfloat_v<src_type>>> {
    using type = src_type;
};

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_DEF_TYPES_H_
