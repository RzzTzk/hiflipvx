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
 * @file    hvx_sw_test_convert.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_CONVERT_H
#define HVX_SW_TEST_CONVERT_H

#include "hvx_sw_test_helper.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief SW transpose function
 */
template<typename param_>
constexpr auto
SwTranspose(typename param_::type* src, typename param_::type* dst) noexcept -> void {
    // gets pointers for each dimension of the tensor
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> src_base = {
        (param_::src_dim::dim_num > param_::perm::dim0) ? (param_::src_base.at(0)) : (0),
        (param_::src_dim::dim_num > param_::perm::dim1) ? (param_::src_base.at(1)) : (0),
        (param_::src_dim::dim_num > param_::perm::dim2) ? (param_::src_base.at(2)) : (0),
        (param_::src_dim::dim_num > param_::perm::dim3) ? (param_::src_base.at(3)) : (0),
        (param_::src_dim::dim_num > param_::perm::dim4) ? (param_::src_base.at(4)) : (0),
        (param_::src_dim::dim_num > param_::perm::dim5) ? (param_::src_base.at(5)) : (0)};

    // The XILINX gcc compiler for C-simulation does not like if elements inside an array inside a type trait are accessed
    constexpr auto dst_dims = param_::dst_dims;

    // transposes tensor
    int64_t dst_ptr = 0;
    for (int64_t i5 = 0; i5 < dst_dims[5]; ++i5) {
        for (int64_t i4 = 0; i4 < dst_dims[4]; ++i4) {
            for (int64_t i3 = 0; i3 < dst_dims[3]; ++i3) {
                for (int64_t i2 = 0; i2 < dst_dims[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < dst_dims[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < dst_dims[0]; ++i0) {
                            const int64_t src_ptr = i5 * src_base.at(5) + i4 * src_base.at(4) + i3 * src_base.at(3) + i2 * src_base.at(2) +
                                                    i1 * src_base.at(1) + i0 * src_base.at(0);
                            dst[dst_ptr] = src[src_ptr]; // NOLINT
                            ++dst_ptr;
                        }
                    }
                }
            }
        }
    }
}

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_CONVERT_H
