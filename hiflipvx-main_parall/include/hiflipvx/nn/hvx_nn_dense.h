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
 * @file    hvx_nn_dense.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_DENSE_H_
#define HVX_NN_DENSE_H_

#include "hvx_nn_conv.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the dense (fully connected) function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>, // data type for the inputs
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>, // data type for the outputs
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>, // data type for the weights
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>, // data type for the bias
         typename batch_v                       = hvx::util::VectorParam<1, 1>,   // batch size
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,   // number of channels (equal to input channels)
         typename fms_v                         = hvx::util::VectorParam<1, 1>,   // number of feature maps (equal to output channels)
         int64_t buf_wgts_                      = false,                          // if weights should be internally buffered on first read
         int64_t buf_bias_                      = false,                          // if bias should be buffered internally on first read
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
struct DenseParam {
    // tensor parameters
    using src_dim  = hvx::util::TensorParam<2, chnls_v, batch_v>;
    using dst_dim  = hvx::util::TensorParam<2, fms_v, batch_v>;
    using wgts_dim = hvx::util::TensorParam<2, chnls_v, fms_v>;
    using bias_dim = hvx::util::TensorParam<1, fms_v>;

    // dimensions
    static constexpr auto batch         = batch_v::elms;
    static constexpr auto chnls         = chnls_v::elms;
    static constexpr auto chnl_vec_size = chnls_v::vec_size;
    static constexpr auto chnl_vec_elms = chnls_v::elms / chnls_v::vec_size;
    static constexpr auto fms           = fms_v::elms;
    static constexpr auto fm_vec_size   = fms_v::vec_size;
    static constexpr auto fm_vec_elms   = fms_v::elms / fms_v::vec_size;

    // data types
    using src_type  = src_type_;
    using dst_type  = dst_type_;
    using wgts_type = wgts_type_;
    using bias_type = bias_type_;
    using src_vec   = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec   = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using wgts_vec  = hvx::util::vector<wgts_type, wgts_dim::vec_size>;
    using bias_vec  = hvx::util::vector<bias_type, bias_dim::vec_size>;
    using src_port  = src_vec;
    using dst_port  = dst_vec;
    using wgts_port = wgts_vec;
    using bias_port = bias_vec;

    // buffer parameters
    static constexpr auto buffer_wgts = buf_wgts_;
    static constexpr auto buffer_bias = buf_bias_;

    // numerical stability
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;

    // constructor (verifies the dimensions and types)
    constexpr DenseParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, true, true, true, true, true>();
        hvx::nn::impl::ConvVerifyType<src_type, wgts_type, bias_type, dst_type>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief top function of the dense layer (with bias)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
DenseTop(typename param_::src_vec* src,
         typename param_::wgts_vec* wgts,
         typename param_::bias_vec* bias,
         typename param_::dst_vec* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // converts dense parameters to convoltuion parameters
    using dense = hvx::nn::ConvParam<typename param_::src_type, typename param_::dst_type, typename param_::wgts_type,
                                     typename param_::bias_type, typename param_::src_dim::dim1, hvx::util::VectorParam<1, 1>,
                                     hvx::util::VectorParam<1, 1>, typename param_::wgts_dim::dim0, typename param_::wgts_dim::dim1,
                                     hvx::util::VectorParam<1, 1>, hvx::util::VectorParam<1, 1>, hvx::util::Array2dParam<0, 0>,
                                     hvx::util::Array2dParam<0, 0>, hvx::util::Array2dParam<1, 1>, param_::buffer_wgts, param_::buffer_bias,
                                     param_::overflow_type, param_::underflow_type, param_::exec_type>;

    // calls the convolution function
    hvx::nn::ConvTop<dense, true>(src, wgts, bias, dst);
}

/*!
 * @brief top function of the dense layer (without bias)
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
DenseTop(typename param_::src_vec* src, typename param_::wgts_vec* wgts, typename param_::dst_vec* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // converts dense parameters to convoltuion parameters
    using dense = hvx::nn::ConvParam<typename param_::src_type, typename param_::dst_type, typename param_::wgts_type,
                                     typename param_::bias_type, typename param_::src_dim::dim1, hvx::util::VectorParam<1, 1>,
                                     hvx::util::VectorParam<1, 1>, typename param_::wgts_dim::dim0, typename param_::wgts_dim::dim1,
                                     hvx::util::VectorParam<1, 1>, hvx::util::VectorParam<1, 1>, hvx::util::Array2dParam<0, 0>,
                                     hvx::util::Array2dParam<0, 0>, hvx::util::Array2dParam<1, 1>, param_::buffer_wgts, param_::buffer_bias,
                                     param_::overflow_type, param_::underflow_type, param_::exec_type>;

    // calls the convolution function
    hvx::nn::ConvTop<dense, false>(src, wgts, nullptr, dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_DENSE_H_
