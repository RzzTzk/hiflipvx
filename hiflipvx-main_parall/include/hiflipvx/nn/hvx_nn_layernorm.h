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
 * @file    hvx_nn_layernorm.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_LAYERNORM_H_
#define HVX_NN_LAYERNORM_H_

#include "impl/hvx_nn_layernorm_dfixed.h"
#include "impl/hvx_nn_layernorm_dfloat.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the layer normalization function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         int64_t buf_wgts_                      = false,
         int64_t buf_bias_                      = false,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         typename clip_                         = hvx::util::ClipParam<1, 0>>
struct LayernormParam {
    // tensor parameters
    using src_dim  = hvx::util::TensorParam<4, chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim  = src_dim;
    using wgts_dim = hvx::util::TensorParam<1, chnls_v>;
    using bias_dim = hvx::util::TensorParam<1, chnls_v>;

    // dimensions
    static constexpr auto batch         = batch_v::elms;
    static constexpr auto src_rows      = src_rows_v::elms;
    static constexpr auto src_cols      = src_cols_v::elms;
    static constexpr auto chnls         = chnls_v::elms;
    static constexpr auto chnl_vec_size = chnls_v::vec_size;
    static constexpr auto chnl_vec_elms = chnls_v::vec_elms;

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
    static constexpr auto clip_max       = clip_::max;
    static constexpr auto clip_min       = clip_::min;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;

    // latency
    static constexpr auto lat = batch * src_rows * src_cols * chnl_vec_elms;

    // constructor (verifies the dimensions and data types)
    constexpr LayernormParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<wgts_dim, false, false, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<bias_dim, false, false, true, true, true, true>();
        hvx::nn::impl::LayernormVerifyType<src_type_, dst_type_, wgts_type_, bias_type_>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief reads weights and bias
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
LayernormWeightsBias(bool cond,
                     int64_t chnl_v,
                     typename param_::wgts_vec* wgts,
                     typename param_::bias_vec* bias,
                     hvx::util::array1d<typename param_::wgts_vec, param_::chnl_vec_elms>& wgts_buf,
                     hvx::util::array1d<typename param_::bias_vec, param_::chnl_vec_elms>& bias_buf,
                     typename param_::wgts_vec& wgts_data,
                     typename param_::bias_vec& bias_data) noexcept -> void {
    HVX_INLINE_TOP();

    // load/store weights
    if (param_::buffer_wgts == true) {
        if (cond == true) {
            wgts_data = wgts[chnl_v]; // NOLINT
            wgts_buf.Set(wgts_data, chnl_v);
        } else {
            wgts_data = wgts_buf.Get(chnl_v);
        }
    } else {
        wgts_data = wgts[chnl_v]; // NOLINT
    }

    // load/store bias
    if (param_::buffer_bias == true) {
        if (cond == true) {
            bias_data = bias[chnl_v]; // NOLINT
            bias_buf.Set(bias_data, chnl_v);
        } else {
            bias_data = bias_buf.Get(chnl_v);
        }
    } else {
        bias_data = bias[chnl_v]; // NOLINT
    }
}

/*!
 * @brief applies normalization on an src vector
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
LayernormComp(typename param_::src_vec& src_data,
              typename param_::wgts_vec& wgts_data,
              typename param_::bias_vec& bias_data,
              typename param_::dst_vec& dst_data) noexcept -> void {
    HVX_INLINE_TOP();
    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
        HVX_UNROLL();
        hvx::nn::impl::LayernormComp<param_>(src_data.Get(chnl_p), wgts_data.Get(chnl_p), bias_data.Get(chnl_p), dst_data.Get(chnl_p));
    }
}

/*!
 * @brief top function of the normalization layer
 */
template<typename param_>
HVX_FORCE_INLINE auto
LayernormTop(typename param_::src_port* src,
             typename param_::wgts_vec* wgts,
             typename param_::bias_vec* bias,
             typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffers normalization wgts and bias [dont initialize]
    static hvx::util::array1d<typename param_::wgts_vec, param_::chnl_vec_elms> wgts_buf;
    static hvx::util::array1d<typename param_::bias_vec, param_::chnl_vec_elms> bias_buf;
    HVX_DATAPACK(wgts_buf.data, bias_buf.data);

    // iterates through the tensor vector by vector
    int64_t ptr_src = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::lat; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer the src and dst vectors
        typename param_::src_vec src_data{};
        typename param_::wgts_vec wgts_data{};
        typename param_::bias_vec bias_data{};
        typename param_::dst_vec dst_data{};

        // flattening loop for stride optimization
        const bool cond_wgts = (i < param_::chnl_vec_elms);
        const int64_t chnl_v = (i % param_::chnl_vec_elms);

        // read next src vector
        hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

        // read wgts and bias
        hvx::nn::LayernormWeightsBias<param_>(cond_wgts, chnl_v, wgts, bias, wgts_buf, bias_buf, wgts_data, bias_data);

        // applies normalization on a vector
        hvx::nn::LayernormComp<param_>(src_data, wgts_data, bias_data, dst_data);

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
    }
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(ptr_src, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_LAYERNORM_H_
