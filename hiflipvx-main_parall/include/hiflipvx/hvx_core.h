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
 * @file    hvx_core.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CORE_H_
#define HVX_CORE_H_

#include "hvx_defs.h"

namespace hvx {
/******************************************************************************************************************************************/

/*!
 * @brief Computes absolute values of tensor elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwAbs(typename param_::src1_port* src1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg, arg);
}

/******************************************************************************************************************************************/

/*!
 * @brief Adds two tensors elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwAdd(typename param_::src1_port* src1, typename param_::src2_port* src2, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, src2, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src2, dst, arg, arg);
}

/*!
 * @brief Adds a constant to all elements of a tensor
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwAddConst(typename param_::src1_port* src1, const float arg1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const auto arg_fixed = static_cast<typename param_::arg_type>(arg1);
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg_fixed, arg_fixed);
}

/******************************************************************************************************************************************/

/*!
 * @brief Clips a tensor elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwClip(typename param_::src1_port* src1, const float low, const float high, typename param_::dst_port* dst) {
    HVX_DATAPACK_TOP(src1, dst);
    const auto arg1_fixed = static_cast<typename param_::arg_type>(low);
    const auto arg2_fixed = static_cast<typename param_::arg_type>(high);
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg1_fixed, arg2_fixed);
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes max values of two tensors elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMax(typename param_::src1_port* src1, typename param_::src2_port* src2, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, src2, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src2, dst, arg, arg);
}

/*!
 * @brief Computes max value of a tensor with a constant elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMaxConst(typename param_::src1_port* src1, const float arg1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const auto arg_fixed = static_cast<typename param_::arg_type>(arg1);
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg_fixed, arg_fixed);
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes min values of two tensors elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMin(typename param_::src1_port* src1, typename param_::src2_port* src2, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, src2, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src2, dst, arg, arg);
}

/*!
 * @brief Computes min value of a tensor with a constant elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMinConst(typename param_::src1_port* src1, const float arg1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const auto arg_fixed = static_cast<typename param_::arg_type>(arg1);
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg_fixed, arg_fixed);
}

/******************************************************************************************************************************************/

/*!
 * @brief Multiplies two tensors elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMul(typename param_::src1_port* src1, typename param_::src2_port* src2, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, src2, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src2, dst, arg, arg);
}

/*!
 * @brief Multiplies a constant with all elements of a tensor
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMulConst(typename param_::src1_port* src1, const float arg1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);

    // const typename param_::arg_type arg_fixed{};
    // arg_fixed = static_cast<typename param_::arg_type>(arg1);
    const auto arg_fixed = static_cast<typename param_::arg_type>(arg1);
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg_fixed, arg_fixed);
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes the Sigmoid function on all elements of a tensor
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSigmoid(typename param_::src1_port* src1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg, arg);
}

/******************************************************************************************************************************************/

/*!
 * @brief Subtracts two tensors elementwise
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSub(typename param_::src1_port* src1, typename param_::src2_port* src2, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, src2, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src2, dst, arg, arg);
}

/******************************************************************************************************************************************/

/*!
 * @brief Computes the Tanh function on all elements of a tensor
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwTanh(typename param_::src1_port* src1, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src1, dst);
    const typename param_::arg_type arg{};
    hvx::ew::ElementwiseTop<param_>(src1, src1, dst, arg, arg);
}

/******************************************************************************************************************************************/

/*!
 * @brief 
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwReduceMax(typename param_::src_port* src, typename param_::dst_port* dst) {
    HVX_DATAPACK_TOP(src, dst);
    hvx::red::ReduceTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwReduceMean(typename param_::src_port* src, typename param_::dst_port* dst) {
    HVX_DATAPACK_TOP(src, dst);
    hvx::red::ReduceTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwReduceMin(typename param_::src_port* src, typename param_::dst_port* dst) {
    HVX_DATAPACK_TOP(src, dst);
    hvx::red::ReduceTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwReduceSum(typename param_::src_port* src, typename param_::dst_port* dst) {
    HVX_DATAPACK_TOP(src, dst);
    hvx::red::ReduceTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Layer Normalization layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwLayernorm(typename param_::src_port* src,
            typename param_::wgts_vec* wgts,
            typename param_::bias_vec* bias,
            typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, wgts, bias, dst);
    hvx::nn::LayernormTop<param_>(src, wgts, bias, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Softmax layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSoftmax(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst);
    hvx::nn::SoftmaxTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Average pooling layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwPoolAvg(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst);
    hvx::nn::PoolTop<param_, hvx::util::pooling_e::kAvg>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Max pooling layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwPoolMax(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst);
    hvx::nn::PoolTop<param_, hvx::util::pooling_e::kMax>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Dense layer (with Bias)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwDense(typename param_::src_port* src,
        typename param_::wgts_vec* wgts,
        typename param_::bias_vec* bias,
        typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, wgts, bias, dst);
    hvx::nn::DenseTop<param_>(src, wgts, bias, dst);
}

/*!
 * @brief Dense layer (without Bias)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwDense(typename param_::src_port* src, typename param_::wgts_vec* wgts, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, wgts, dst);
    hvx::nn::DenseTop<param_>(src, wgts, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Depthwise convolution layer (with Bias)
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwDepthwise(typename param_::src_port* src,
            typename param_::wgts_vec* wgts,
            typename param_::bias_vec* bias,
            typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, wgts, bias, dst);
    hvx::nn::DepthwiseTop<param_, true>(src, wgts, bias, dst);
}

/*!
 * @brief Depthwise convolution layer (without Bias)
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwDepthwise(typename param_::src_port* src, typename param_::wgts_vec* wgts, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, wgts, dst);
    hvx::nn::DepthwiseTop<param_, false>(src, wgts, nullptr, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Convolution layer (with Bias)
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwConv(typename param_::src_port* src,
       typename param_::wgts_vec* wgts,
       typename param_::bias_vec* bias,
       typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, bias, dst); // wgts,
    hvx::nn::ConvTop<param_, true>(src, wgts, bias, dst);
}

/*!
 * @brief Convolution layer (without Bias)
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwConv(typename param_::src_port* src, typename param_::wgts_vec* wgts, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst); // wgts,
    hvx::nn::ConvTop<param_, false>(src, wgts, nullptr, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Transpose layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwTranspose(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst);
    hvx::convert::HwTransposeTop<param_>(src, dst);
}

/******************************************************************************************************************************************/

/*!
 * @brief Reshape layer
 */
template<typename param_>
HVX_FORCE_INLINE auto
HwReshape(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst);
    hvx::convert::HwReshapeTop<param_>(src, dst);
}

/******************************************************************************************************************************************/
/*!
 * @brief Multicast layer (2 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMulticast(typename param_::vec* src, typename param_::vec* dst0, typename param_::vec* dst1) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1);
    hvx::convert::HwMulticastTop<param_, typename param_::dim, typename param_::dim>(src, dst0, dst1);
}

/*!
 * @brief Multicast layer (3 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMulticast(typename param_::vec* src, typename param_::vec* dst0, typename param_::vec* dst1, typename param_::vec* dst2) noexcept
    -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1, dst2);
    hvx::convert::HwMulticastTop<param_, typename param_::dim, typename param_::dim, typename param_::dim>(src, dst0, dst1, dst2);
}

/*!
 * @brief Multicast layer (4 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMulticast(typename param_::vec* src,
            typename param_::vec* dst0,
            typename param_::vec* dst1,
            typename param_::vec* dst2,
            typename param_::vec* dst3) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1, dst2, dst3);
    hvx::convert::HwMulticastTop<param_, typename param_::dim, typename param_::dim, typename param_::dim, typename param_::dim>(
        src, dst0, dst1, dst2, dst3);
}

/******************************************************************************************************************************************/

/*!
 * @brief Concat layer (2 inputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwConcat(typename param_::split0_vec* src0, typename param_::split1_vec* src1, typename param_::vec* dst) noexcept -> void {
    HVX_DATAPACK_TOP(dst, src0, src1);
    hvx::convert::HwConcatTop<param_, typename param_::split0::dim, typename param_::split1::dim>(dst, src0, src1);
}

/*!
 * @brief Concat layer (3 inputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwConcat(typename param_::split0_vec* src0,
         typename param_::split1_vec* src1,
         typename param_::split2_vec* src2,
         typename param_::vec* dst) noexcept -> void {
    HVX_DATAPACK_TOP(dst, src0, src1, src2);
    hvx::convert::HwConcatTop<param_, typename param_::split0::dim, typename param_::split1::dim, typename param_::split2::dim>(dst, src0,
                                                                                                                                src1, src2);
}

/*!
 * @brief Concat layer (4 inputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwConcat(typename param_::split0_vec* src0,
         typename param_::split1_vec* src1,
         typename param_::split2_vec* src2,
         typename param_::split3_vec* src3,
         typename param_::vec* dst) noexcept -> void {
    HVX_DATAPACK_TOP(dst, src0, src1, src2, src3);
    hvx::convert::HwConcatTop<param_, typename param_::split0::dim, typename param_::split1::dim, typename param_::split2::dim,
                              typename param_::split3::dim>(dst, src0, src1, src2, src3);
}

/******************************************************************************************************************************************/

/*!
 * @brief Split layer (2 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSplit(typename param_::vec* src, typename param_::split0_vec* dst0, typename param_::split1_vec* dst1) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1);
    hvx::convert::HwSplitTop<param_, typename param_::split0::dim, typename param_::split1::dim>(src, dst0, dst1);
}

/*!
 * @brief Split layer (3 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSplit(typename param_::vec* src,
        typename param_::split0_vec* dst0,
        typename param_::split1_vec* dst1,
        typename param_::split2_vec* dst2) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1, dst2);
    hvx::convert::HwSplitTop<param_, typename param_::split0::dim, typename param_::split1::dim, typename param_::split2::dim>(src, dst0,
                                                                                                                               dst1, dst2);
}

/*!
 * @brief Split layer (4 outputs)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwSplit(typename param_::vec* src,
        typename param_::split0_vec* dst0,
        typename param_::split1_vec* dst1,
        typename param_::split2_vec* dst2,
        typename param_::split3_vec* dst3) noexcept -> void {
    HVX_DATAPACK_TOP(src, dst0, dst1, dst2, dst3);
    hvx::convert::HwSplitTop<param_, typename param_::split0::dim, typename param_::split1::dim, typename param_::split2::dim,
                             typename param_::split3::dim>(src, dst0, dst1, dst2, dst3);
}

/******************************************************************************************************************************************/
//xwq
// /*!
//  * @brief FiFo to be played between two layers
//  */
// template<typename type_, int64_t original_buf_size_, int64_t fifo_size_>
// HVX_FORCE_INLINE auto
// HwFifo() noexcept -> hvx::util::array1d<type_, original_buf_size_> {
//     HVX_INLINE_TOP();
//     return hvx::convert::StreamFifo<type_, original_buf_size_, fifo_size_>();
// }


// /*!
//  * @brief Converts from HLS stream to HVX type
//  */
// template<typename param_>
// HVX_FORCE_INLINE auto
// HwStreamToHvx(typename param_::port& src, typename param_::vec* dst) {
//     HVX_DATAPACK_TOP(src, dst);
//     hvx::convert::StreamToHvx<typename param_::vec, typename param_::dim, param_::flags>(src, dst);
// }

// /*!
//  * @brief Converts from HVX to HLS-stream type
//  */
// template<typename param_>
// HVX_FORCE_INLINE auto
// HwHvxToStream(typename param_::vec* src, typename param_::port& dst) noexcept -> void {
//     HVX_DATAPACK_TOP(src, dst);
//     hvx::convert::HvxToStream<typename param_::vec, typename param_::dim, param_::flags>(src, dst);
// }

// /*!
//  * @brief Sends the elements from src array to the output (for testbenches)
//  */
// template<typename type_, int64_t latency_, int64_t delay_, int64_t vec_elms_>
// HVX_FORCE_INLINE auto
// HwSrcGenerator(type_* src, type_* dst) noexcept -> void {
//     HVX_INLINE_TOP();
//     hvx::convert::SrcGenerator<type_, latency_, delay_, vec_elms_>(src, dst);
// }

// /*!
//  * @brief Sends random elements to the output (for testbenches)
//  */
// template<typename type_, int64_t latency_, int64_t delay_, int64_t vec_elms_>
// HVX_FORCE_INLINE auto
// HwSrcGenerator(type_* dst) noexcept -> void {
//     HVX_INLINE_TOP();
//     hvx::convert::SrcGenerator<type_, latency_, delay_, vec_elms_>(dst);
// }

/******************************************************************************************************************************************/

} // namespace hvx

#endif // HVX_CORE_H_
