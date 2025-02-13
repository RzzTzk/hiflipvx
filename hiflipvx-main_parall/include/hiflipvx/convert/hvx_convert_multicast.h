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
 * @file    hvx_convert_multicast.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_MULTICAST_H
#define HVX_CONVERT_MULTICAST_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the multicast function
 */
template<typename type_ = hvx::util::dfixed<int16_t, 15>, typename dim_ = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>>
struct MulticastParam {
    // dimensions
    using dim = dim_;

    // types
    using type     = type_;
    using vec      = hvx::util::vector<type_, dim::vec_size>;
    using src_port = vec;
    using dst_port = vec;

    // assertions
    constexpr MulticastParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<dim, false, false, false, false, false, false>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Writes to output
 */
template<int64_t dst_num_, typename param_>
HVX_FORCE_INLINE constexpr auto
HwMulticastDst(hvx::util::vector<int64_t, dst_num_>& ptr_dst, typename param_::vec& src) noexcept -> void {
    HVX_INLINE_TOP();
    (void)ptr_dst;
    (void)src;
}

/*!
 * @brief Writes to output
 */
template<int64_t dst_num_, typename param_, typename dst_dim_, typename... dst_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwMulticastDst(hvx::util::vector<int64_t, dst_num_>& ptr_dst,
               typename param_::vec& vec,
               hvx::util::vector<typename param_::type, dst_dim_::vec_size>* dst,
               hvx::util::vector<typename param_::type, dst_dim_rest_::vec_size>*... dst_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // write to output
    hvx::util::StreamWriteData<typename param_::vec>(dst, vec, ptr_dst.Get(sizeof...(dst_dim_rest_)), true);

    // write to next output if available
    hvx::convert::HwMulticastDst<dst_num_, param_, dst_dim_rest_...>(ptr_dst, vec, dst_rest...);
}

/*!
 * @brief Top HW multicast function
 */
template<typename param_, typename dst_dim_, typename... dst_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwMulticastTop(typename param_::vec* src,
               hvx::util::vector<typename param_::type, dst_dim_::vec_size>* dst,
               hvx::util::vector<typename param_::type, dst_dim_rest_::vec_size>*... dst_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // number of output and dimensions
    constexpr int64_t dst_num = sizeof...(dst_dim_rest_) + 1;

    // pointer to inputs/outputs
    hvx::util::vector<int64_t, dst_num> ptr_dst{};
    HVX_ARRAY_PARTITION_COMPLETE(ptr_dst.data, 0);

    // Computes operations (pipelined)
    for (int64_t i = 0, ptr_src = 0; i < param_::dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // Read input
        typename param_::vec src_data{};
        hvx::util::StreamReadData<typename param_::vec>(src, src_data, ptr_src, true);

        // Write outputs
        hvx::convert::HwMulticastDst<dst_num, param_, dst_dim_, dst_dim_rest_...>(ptr_dst, src_data, dst, dst_rest...);
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_MULTICAST_H
