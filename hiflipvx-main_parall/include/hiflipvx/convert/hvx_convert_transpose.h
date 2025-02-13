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
 * @file    hvx_convert_transpose.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_TRANSPOSE_H
#define HVX_CONVERT_TRANSPOSE_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the transpose function
 */
template<typename type_             = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_          = hvx::util::TensorParam<2, hvx::util::VectorParam<1, 1>, hvx::util::VectorParam<1, 1>>,
         typename perm_             = hvx::util::TransposePerm<1, 0>,
         int64_t dst_dim0_vec_size_ = 1>
struct TransposeParam {
    // dimensions
    using perm    = perm_;
    using src_dim = src_dim_;
    using dst_dim = hvx::util::TensorParam<src_dim::dim_num,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim0], dst_dim0_vec_size_>,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim1], 1>,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim2], 1>,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim3], 1>,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim4], 1>,
                                           hvx::util::VectorParam<src_dim::dims[perm::dim5], 1>>;

    // types
    using type     = type_;
    using src_vec  = hvx::util::vector<type, src_dim::vec_size>;
    using dst_vec  = hvx::util::vector<type, dst_dim::vec_size>;
    using src_port = src_vec;
    using dst_port = dst_vec;

    // buffer dimensions
    static constexpr int64_t buf_dims =
        (perm::dim5 != 5)
            ? (6)
            : ((perm::dim4 != 4)
                   ? (5)
                   : ((perm::dim3 != 3) ? (4) : ((perm::dim2 != 2) ? (3) : ((perm::dim1 != 1) ? (2) : ((perm::dim0 != 0) ? (1) : (0))))));
    static constexpr int64_t buf_elms         = hvx::util::TensorGetRangeElms<src_dim, buf_dims>();
    static constexpr int64_t src_buf_vec_elms = buf_elms / src_dim::vec_size;
    static constexpr int64_t dst_buf_vec_elms = buf_elms / dst_dim::vec_size;

    // pointer for source and destination tensor
    static constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> src_base = {
        hvx::util::TensorGetRangeElms<src_dim, perm::dim0>(), hvx::util::TensorGetRangeElms<src_dim, perm::dim1>(),
        hvx::util::TensorGetRangeElms<src_dim, perm::dim2>(), hvx::util::TensorGetRangeElms<src_dim, perm::dim3>(),
        hvx::util::TensorGetRangeElms<src_dim, perm::dim4>(), hvx::util::TensorGetRangeElms<src_dim, perm::dim5>()};
    static constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> dst_base = {
        hvx::util::TensorGetRangeElms<dst_dim, 0>(), hvx::util::TensorGetRangeElms<dst_dim, 1>(),
        hvx::util::TensorGetRangeElms<dst_dim, 2>(), hvx::util::TensorGetRangeElms<dst_dim, 3>(),
        hvx::util::TensorGetRangeElms<dst_dim, 4>(), hvx::util::TensorGetRangeElms<dst_dim, 5>()};
    static constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> dst_dims = {
        dst_dim::dims[0], dst_dim::dims[1], dst_dim::dims[2], dst_dim::dims[3], dst_dim::dims[4], dst_dim::dims[5]};

    // assertions
    constexpr TransposeParam() {
        static_assert(buf_dims > 0, "You need to transpose at least 1 dimension!");
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, true, true, true, true, true>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Top HW transpose function
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwTransposeTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffers
    static hvx::util::array1d<typename param_::type, param_::buf_elms> buffer;
    typename param_::src_vec src_data{};
    typename param_::dst_vec dst_data{};

    // The XILINX gcc compiler for C-simulation does not like if elements inside an array inside a type trait are accessed
    constexpr auto dst_base = param_::dst_base;
    constexpr auto src_base = param_::src_base;
    constexpr auto dst_dims = param_::dst_dims;

    //
    for (int64_t i = 0, ptr_src = 0, ptr_dst = 0; i < (param_::src_dim::vec_elms + param_::dst_dim::vec_elms); ++i) {
        HVX_PIPELINE_ON(1, frp);

        // pointers
        const int64_t ptr_src_buf_off = i % (param_::src_buf_vec_elms + param_::dst_buf_vec_elms);
        const int64_t ptr_dst_buf_off = ptr_src_buf_off - param_::src_buf_vec_elms;

        // Read input
        if (ptr_src_buf_off < param_::src_buf_vec_elms) {
            // read next src vector
            hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

            // write data to buffer
            for (int64_t p = 0; p < param_::src_dim::vec_size; ++p) {
                const int64_t ptr_src_buf = (ptr_src_buf_off * param_::src_dim::vec_size) + p;
                buffer.Set(src_data.Get(p), ptr_src_buf);
            }
        }

        // Write output
        else {
            for (int64_t p = 0; p < param_::dst_dim::vec_size; ++p) {
                const int64_t ptr_dst_buf_off_t = (ptr_dst_buf_off * param_::dst_dim::vec_size + p);

                // create pointer
                int64_t ptr_dst_buf = 0;
                if ((param_::buf_dims > 0) && (param_::src_dim::dim_num > param_::perm::dim0))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(0)) % dst_dims[0]) * src_base.at(0);
                if ((param_::buf_dims > 1) && (param_::src_dim::dim_num > param_::perm::dim1))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(1)) % dst_dims[1]) * src_base.at(1);
                if ((param_::buf_dims > 2) && (param_::src_dim::dim_num > param_::perm::dim2))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(2)) % dst_dims[2]) * src_base.at(2);
                if ((param_::buf_dims > 3) && (param_::src_dim::dim_num > param_::perm::dim3))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(3)) % dst_dims[3]) * src_base.at(3);
                if ((param_::buf_dims > 4) && (param_::src_dim::dim_num > param_::perm::dim4))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(4)) % dst_dims[4]) * src_base.at(4);
                if ((param_::buf_dims > 5) && (param_::src_dim::dim_num > param_::perm::dim5))
                    ptr_dst_buf += ((ptr_dst_buf_off_t / dst_base.at(5)) % dst_dims[5]) * src_base.at(5);

                // read data from buffer
                dst_data.Set(buffer.Get(ptr_dst_buf), p);
            }

            // write next dst vector
            hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
        }
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_TRANSPOSE_H
