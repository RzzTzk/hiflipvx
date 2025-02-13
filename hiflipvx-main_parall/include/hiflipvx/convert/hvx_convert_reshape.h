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
 * @file    hvx_convert_reshape.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_RESHAPE_H
#define HVX_CONVERT_RESHAPE_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the reshape function
 */
template<typename type_    = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_ = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename dst_dim_ = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>>
struct ReshapeParam {
    // dimensions
    using src_dim = src_dim_;
    using dst_dim = dst_dim_;

    // types
    using type     = type_;
    using src_vec  = hvx::util::vector<type_, src_dim::vec_size>;
    using dst_vec  = hvx::util::vector<type_, dst_dim::vec_size>;
    using src_port = src_vec;
    using dst_port = dst_vec;

    // parameters
    static constexpr auto lcm_vec_size    = hvx::util::Lcm<int64_t>(dst_dim::vec_size, src_dim::vec_size);
    static constexpr int64_t dst_multiple = lcm_vec_size / dst_dim::vec_size;
    static constexpr int64_t src_multiple = lcm_vec_size / src_dim::vec_size;

    // assertions
    constexpr ReshapeParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim_, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim_, false, true, true, true, true, true>();
        static_assert(src_dim_::elms == dst_dim_::elms, "Input and output need the same number of elements!");
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Top function of the Reshape function, if (input vector size > output vector size)
 */
template<typename param_, std::enable_if_t<(param_::src_dim::vec_size > param_::dst_dim::vec_size), bool> = true>
HVX_FORCE_INLINE constexpr auto
HwReshapeTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffer to overcome the vector size missmatch
    hvx::util::vector<typename param_::type, param_::lcm_vec_size> buffer{};
    HVX_DATAPACK(buffer.data);

    // copy elements
    for (int64_t i = 0, ptr_src = 0, ptr_dst = 0; i < param_::dst_dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer input/output vector
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};

        // flattening loop to improve latency
        const int64_t ptr_dst_multiple = i % param_::dst_multiple;

        // Read data from input and write to buffer
        if (ptr_dst_multiple < param_::src_multiple) {
            // read next src vector
            hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

            // conversion to buffer
            for (int64_t p = 0; p < param_::src_dim::vec_size; ++p)
                buffer.Set(src_data.Get(p), ptr_dst_multiple * param_::src_dim::vec_size + p);
        }

        // conversion from buffer
        for (int64_t p = 0; p < param_::dst_dim::vec_size; ++p)
            dst_data.Set(buffer.Get(ptr_dst_multiple * param_::dst_dim::vec_size + p), p);

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
    }
}

/*!
 * @brief Top function of the Reshape function, if (input vector size <= output vector size)
 */
template<typename param_, std::enable_if_t<(param_::src_dim::vec_size <= param_::dst_dim::vec_size), bool> = true>
HVX_FORCE_INLINE constexpr auto
HwReshapeTop(typename param_::src_vec* src, typename param_::dst_vec* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffer to overcome the vector size missmatch
    hvx::util::vector<typename param_::type, param_::lcm_vec_size> buffer{};
    HVX_DATAPACK(buffer.data);

    // copy elements
    for (int64_t i = 0, ptr_src = 0, ptr_dst = 0; i < param_::dst_dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer input/output vector
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};

        // flattening loop to improve latency
        const int64_t ptr_src_multiple = (i % param_::src_multiple);
        const int64_t ptr_dst_multiple = ptr_src_multiple - (param_::src_multiple - param_::dst_multiple);

        // read next src vector
        hvx::util::StreamReadData<typename param_::src_vec>(src, src_data, ptr_src, true);

        // conversion to buffer
        for (int64_t p = 0; p < param_::src_dim::vec_size; ++p)
            buffer.Set(src_data.Get(p), ptr_src_multiple * param_::src_dim::vec_size + p);

        if (ptr_dst_multiple >= 0) {
            // conversion from buffer
            for (int64_t p = 0; p < param_::dst_dim::vec_size; ++p)
                dst_data.Set(buffer.Get(ptr_dst_multiple * param_::dst_dim::vec_size + p), p);

            // write next dst vector
            hvx::util::StreamWriteData<typename param_::dst_vec>(dst, dst_data, ptr_dst, true);
        }
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_RESHAPE_H
