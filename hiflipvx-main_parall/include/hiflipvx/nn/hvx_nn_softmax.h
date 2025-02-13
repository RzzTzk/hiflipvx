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
 * @file    hvx_nn_softmax.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_SOFTMAX_H_
#define HVX_NN_SOFTMAX_H_

#include "impl/hvx_nn_softmax_dfixed.h"
#include "impl/hvx_nn_softmax_dfloat.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the softmax function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
struct SoftmaxParam {
    // tensor parameters
    using src_dim = hvx::util::TensorParam<4, chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim = src_dim;

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
    using comp_type = hvx::util::def_int_type_t<src_type_, src_type_>;
    using buf_type  = hvx::util::def_flt_type_t<src_type_>;
    using src_vec   = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec   = hvx::util::vector<dst_type, src_dim::vec_size>;
    using buf_vec   = hvx::util::vector<buf_type, src_dim::vec_size>;
    using src_port  = src_vec;
    using dst_port  = dst_vec;

    // numerical stability
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;

    // latency
    static constexpr auto lat = 2 * batch * src_rows * src_cols * chnl_vec_elms;

    // block processing (bp) to prevent loop dependency by summation
    // hvx::util::Min(chnl_vec_elms_, static_cast<int64_t>((hvx::util::is_dfixed_v<src_type_>) ? (1) : (8)));
    static constexpr auto bp_width = 1;

    // constructor (verifies the dimensions and data types)
    constexpr SoftmaxParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::nn::impl::SoftmaxVerifyType<src_type, dst_type>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief applies softmax stage 1 on an src vector
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
SoftmaxCompStage1(int64_t chnl_v,
                  typename param_::comp_type& sum_global,
                  typename param_::src_vec& src_data,
                  hvx::util::array1d<typename param_::buf_vec, param_::chnl_vec_elms>& wgts_buf) noexcept -> void {
    HVX_INLINE_TOP();

    // buffers expenontial results of one vector
    typename param_::buf_vec dst_vec{};

    // comp the some of all exponentials in a vector
    typename param_::comp_type sum_local{0};

    // calculates: n(i) = exp(src(i)) | N: sum of all n
    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
        HVX_UNROLL(); // (TODO: delete template parameters except param_)
        hvx::nn::impl::SoftmaxStage1<typename param_::src_dim, param_>(chnl_p, chnl_v, sum_local, sum_global, src_data.Get(chnl_p),
                                                                       dst_vec.Get(chnl_p));
    }

    // buffer exponential result
    wgts_buf.Set(dst_vec, chnl_v);
}

/*!
 * @brief  applies softmax stage 2 on an insrcput vector
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
SoftmaxCompStage2(int64_t chnl_v,
                  typename param_::comp_type& sum_global,
                  typename param_::dst_vec& dst_vec,
                  hvx::util::array1d<typename param_::buf_vec, param_::chnl_vec_elms>& wgts_buf) noexcept -> void {
    HVX_INLINE_TOP();

    // To buffer the complete src/dst vector
    typename param_::buf_vec src_vec{};

    // read a vector exponential src values
    src_vec = wgts_buf.Get(chnl_v);

    // calculates: m(i) = n(i) / N
    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
        HVX_UNROLL(); // (TODO: delete template parameters except param_)
        hvx::nn::impl::SoftmaxStage2<typename param_::src_dim, param_, typename param_::src_type>(chnl_p, sum_global, src_vec, dst_vec);
    }
}

/*!
 * @brief
 */
template<typename param_>
HVX_FORCE_INLINE auto
SoftmaxTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffers the exponential of all incoming values [dont initialize]
    static hvx::util::array1d<typename param_::buf_vec, param_::chnl_vec_elms> wgts_buf;
    HVX_DATAPACK(wgts_buf.data);

    // buffers the global sum  [dont initialize]
    static hvx::util::array1d<typename param_::comp_type, param_::bp_width> sum_global;
    HVX_ARRAY_PARTITION_COMPLETE(sum_global.data, 0);

    // Softmax Computation
    int64_t ptr_src = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::lat; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // flattening loop to improve latency
        const int64_t chnl_v = i % param_::chnl_vec_elms;

        // buffer the src and dst vectors
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};

        // two stage computation
        if (((i / param_::chnl_vec_elms) & 1) == 0) {
            // read next src vector
            hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

            // calculates: n(i) = exp(src(i)) | N: sum of all n
            hvx::nn::SoftmaxCompStage1<param_>(chnl_v, sum_global.Get(chnl_v % param_::bp_width), src_data, wgts_buf);

        } else {
            // calculates: m(i) = n(i) / N
            hvx::nn::SoftmaxCompStage2<param_>(chnl_v, sum_global.Get(chnl_v % param_::bp_width), dst_data, wgts_buf);

            // write next dst vector
            hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
        }
    }
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(ptr_src, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_SOFTMAX_H_
