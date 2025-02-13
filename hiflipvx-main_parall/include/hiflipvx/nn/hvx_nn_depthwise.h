/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ï¿½AS ISï¿½, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_nn_depthwise.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_DEPTHWISE_H_
#define HVX_NN_DEPTHWISE_H_

#include "impl/hvx_nn_depthwise_dfixed.h"
#include "impl/hvx_nn_depthwise_dfloat.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the dense (fully connected) function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         typename knl_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename knl_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename pad_                          = hvx::util::Array2dParam<0, 0>,
         typename dil_                          = hvx::util::Array2dParam<0, 0>,
         typename str_                          = hvx::util::Array2dParam<1, 1>,
         int64_t buf_wgts_                      = false,
         int64_t buf_bias_                      = false,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
struct DepthwiseParam {
    // destination rows/cols
    using dst_rows_v = decltype(hvx::util::WinDstVecParams<src_rows_v, knl_rows_v, pad_::rows, dil_::rows, str_::rows>());
    using dst_cols_v = decltype(hvx::util::WinDstVecParams<src_cols_v, knl_cols_v, pad_::cols, dil_::cols, str_::cols>());

    // tensor parameters
    using src_dim  = hvx::util::TensorParam<4, chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim  = hvx::util::TensorParam<4, chnls_v, dst_cols_v, dst_rows_v, batch_v>;
    using wgts_dim = hvx::util::TensorParam<3, knl_cols_v, knl_rows_v, chnls_v>;
    using bias_dim = hvx::util::TensorParam<1, chnls_v>; // <3, chnls_v, dst_cols_v, dst_rows_v>

    // dimensions
    static constexpr auto batch            = batch_v::elms;
    static constexpr auto src_rows         = src_rows_v::elms;
    static constexpr auto src_row_vec_size = src_rows_v::vec_size;
    static constexpr auto src_row_vec_elms = src_rows_v::vec_elms;
    static constexpr auto src_cols         = src_cols_v::elms;
    static constexpr auto src_col_vec_size = src_cols_v::vec_size;
    static constexpr auto src_col_vec_elms = src_cols_v::vec_elms;
    static constexpr auto dst_rows         = dst_rows_v::elms;
    static constexpr auto dst_row_vec_size = dst_rows_v::vec_size;
    static constexpr auto dst_row_vec_elms = dst_rows_v::vec_elms;
    static constexpr auto dst_cols         = dst_cols_v::elms;
    static constexpr auto dst_col_vec_size = dst_cols_v::vec_size;
    static constexpr auto dst_col_vec_elms = dst_cols_v::vec_elms;
    static constexpr auto chnls            = chnls_v::elms;
    static constexpr auto chnl_vec_size    = chnls_v::vec_size;
    static constexpr auto chnl_vec_elms    = chnls_v::elms / chnls_v::vec_size;
    static constexpr auto wgts_vec_size    = wgts_dim::vec_size;
    static constexpr auto wgts_vec_elms    = wgts_dim::vec_elms;
    static constexpr auto bias_vec_size    = bias_dim::vec_size;
    static constexpr auto bias_vec_elms    = bias_dim::vec_elms;

    // data types
    using src_type  = src_type_;
    using dst_type  = dst_type_;
    using wgts_type = wgts_type_;
    using bias_type = bias_type_;
    //using comp_type = hvx::util::def_int_type_t<src_type, wgts_type_>;
    using src_vec   = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec   = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using wgts_vec  = hvx::util::vector<wgts_type, wgts_dim::vec_size>;
    using bias_vec  = hvx::util::vector<bias_type, bias_dim::vec_size>;
    using src_port  = src_vec;
    using dst_port  = dst_vec;
    using wgts_port = wgts_vec;
    using bias_port = bias_vec;

    // window (kernel) parameters
    static constexpr auto knl_rows          = knl_rows_v::elms;
    static constexpr auto knl_rows_vec_size = knl_rows_v::vec_size;
    static constexpr auto knl_cols          = knl_cols_v::elms;
    static constexpr auto knl_cols_vec_size = knl_cols_v::vec_size;
    static constexpr auto knl_elms          = knl_rows * knl_cols;
    static constexpr auto pad_rows          = pad_::rows;
    static constexpr auto pad_cols          = pad_::cols;
    static constexpr auto dil_rows          = dil_::rows;
    static constexpr auto dil_cols          = dil_::cols;
    static constexpr auto knl_dil_rows      = hvx::util::WinKnlDilLen<knl_rows_v::elms, dil_::rows>();
    static constexpr auto knl_dil_cols      = hvx::util::WinKnlDilLen<knl_cols_v::elms, dil_::cols>();
    static constexpr auto knl_dil_elms      = knl_dil_rows * knl_dil_cols;
    static constexpr auto str_rows          = str_::rows;
    static constexpr auto str_cols          = str_::cols;

    // buffer parameters
    static constexpr auto row_buf_elms = src_cols * chnl_vec_elms;
    static constexpr auto row_buf_num  = hvx::util::Max(knl_dil_rows - 1, static_cast<int64_t>(1));
    static constexpr auto win_buf_elms = chnl_vec_elms;
    static constexpr auto win_buf_num  = hvx::util::Max(knl_dil_cols - 1, static_cast<int64_t>(1)) * knl_dil_rows;
    static constexpr auto src_buf_elms = chnl_vec_elms;
    static constexpr auto src_buf_num  = 1;
    static constexpr auto win_elms     = knl_elms;
    static constexpr auto win_dil_elms = knl_dil_elms;
    static constexpr auto buffer_wgts  = buf_wgts_;
    static constexpr auto buffer_bias  = buf_bias_;

    // numerical stability
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;

    // latency
    static constexpr auto ohd_rows  = pad_rows;
    static constexpr auto ohd_cols  = pad_cols;
    static constexpr auto lat_rows  = src_row_vec_elms + ohd_rows;
    static constexpr auto lat_cols  = src_col_vec_elms + ohd_cols;
    static constexpr auto lat_chnls = chnl_vec_elms;
    static constexpr auto lat       = batch * lat_rows * lat_cols * lat_chnls;

    // constructor (verifies the dimension)
    constexpr DepthwiseParam() {
        static_assert(knl_rows == knl_rows_vec_size, "Knl rows are not fully vectorized!");
        static_assert(knl_cols == knl_cols_vec_size, "Knl cols are not fully vectorized!");
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifySameDims<src_dim, dst_dim, 4, true, false, false, true, true, true>();
        hvx::util::BiasVerifyDim<bias_dim, dst_rows, dst_cols, chnls, chnl_vec_size>();
        hvx::util::WinVerifyDim<src_rows, src_cols, knl_rows, knl_cols, pad_rows, pad_cols, dil_rows, dil_cols>();
        hvx::nn::impl::DepthwiseVerifyTypes<src_type, wgts_type, bias_type, dst_type>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief applies depthwise function on an src vector
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
DepthwiseComp(hvx::util::array1d<typename param_::src_vec, param_::knl_elms>& win,
              typename param_::wgts_vec& wgts_data,
              typename param_::bias_vec& bias_data,
              typename param_::dst_vec& dst_data) noexcept -> void {
    HVX_INLINE_TOP();
    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
        HVX_UNROLL();

        // buffers needed win and weights to comp one dst element
        hvx::util::vector<typename param_::wgts_type, param_::knl_elms> wgts_tmp{};
        hvx::util::vector<typename param_::src_type, param_::knl_elms> win_tmp{};

        // get needed win and weights
        for (int64_t knl_pix = 0; knl_pix < param_::knl_elms; ++knl_pix) {
            win_tmp.Set(win.Get(knl_pix).Get(chnl_p), knl_pix);
            wgts_tmp.Set(wgts_data.Get(chnl_p * param_::knl_elms + knl_pix), knl_pix);
        }

        // applies depthwise function on a single element (TODO: delete template parameters except param_)
        hvx::nn::impl::DepthwiseComp<typename param_::wgts_dim, param_>(win_tmp, wgts_tmp, bias_data.Get(chnl_p), dst_data.Get(chnl_p));
    }
}

/*!
 * @brief top function of the conv layer
 */
template<typename param_, bool with_bias_ = false>
HVX_FORCE_INLINE auto
DepthwiseTop(typename param_::src_port* src,
             typename param_::wgts_vec* wgts,
             typename param_::bias_vec* bias,
             typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffer the weights and bias (if needed) does not read when IP executes multiple times [dont initialize]
    static hvx::util::array1d<typename param_::wgts_vec, param_::wgts_vec_elms> wgts_buf;
    static hvx::util::array1d<typename param_::bias_vec, param_::bias_vec_elms> bias_buf;
    static bool wgts_buffered_ = false, bias_buffered_ = false;

    // buffers needed src elements for window to not read same element twice from global memory [dont initialize]
    static hvx::util::array2d<typename param_::src_vec, param_::row_buf_elms, param_::row_buf_num> row_buf;
    static hvx::util::array2d<typename param_::src_vec, param_::win_buf_elms, param_::win_buf_num> win_buf;
    static hvx::util::array2d<typename param_::src_vec, param_::src_buf_elms, param_::src_buf_num> src_buf;
    static hvx::util::array1d<typename param_::src_vec, param_::win_elms> win;
    static hvx::util::array1d<typename param_::src_vec, param_::win_dil_elms> win_dil;

    // directives for buffers and windows
    HVX_DATAPACK(wgts_buf.data, bias_buf.data, row_buf.data, win_buf.data, src_buf.data, win.data, win_dil.data);
    HVX_ARRAY_PARTITION_COMPLETE(row_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(src_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_dil.data, 1);

    // iterates through the tensor vector by vector
    int64_t ptr_src = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::lat; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer the src, dst, wgts and bias vectors
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};
        typename param_::wgts_vec wgts_data{};
        typename param_::bias_vec bias_data{};

        // flattening loop to improve latency (TODO: loop inefficient for stride >= 2)
        const int64_t src_row = (i / (param_::lat_chnls * param_::lat_cols)) % (param_::lat_rows);
        const int64_t src_col = (i / (param_::lat_chnls)) % (param_::lat_cols);
        const int64_t chnl_v  = (i % (param_::lat_chnls));

        // comp conditions for src and dst (TODO: delete template parameters except param_)
        const auto cond = hvx::util::WinCompCond<param_::src_rows, param_::src_cols, param_::dst_rows, param_::dst_cols, param_::knl_rows,
                                                 param_::knl_cols, param_::pad_rows, param_::pad_cols, param_::dil_rows, param_::dil_cols>(
            src_col, src_row);
        const bool cond_wgts = (cond.dst_row && cond.dst_col);
        const bool cond_bias = (with_bias_ && cond.dst_row && cond.dst_col);

        // read next src vector
        hvx::util::StreamReadData<>(src, src_data, ptr_src, (cond.src_row && cond.src_col));

        // updates the window and its buffers (TODO: delete template parameters except param_)
        hvx::util::WinUpdate<typename param_::src_type, typename param_::src_dim, param_::knl_rows, param_::knl_cols, param_::dil_rows,
                             param_::dil_cols>(src_row, src_col, chnl_v, 0, src_data, row_buf, src_buf, win_buf, win_dil, win);

        // read weights src vector (TODO: delete template parameters except param_)
        hvx::util::WeightsUpdate<typename param_::wgts_type, param_::wgts_vec_size, 1, param_::chnl_vec_elms, param_::buffer_wgts>(
            0, chnl_v, ptr_dst, wgts_buffered_, cond_wgts, wgts, wgts_buf, wgts_data);

        // read bias src vector (TODO: delete template parameters except param_)
        hvx::util::BiasUpdate<typename param_::bias_type, param_::chnl_vec_size, param_::bias_vec_elms, param_::buffer_bias>(
            ptr_dst, bias_buffered_, cond_bias, bias, bias_buf, bias_data);

        // applies depthwise function on an src vector
        hvx::nn::DepthwiseComp<param_>(win, wgts_data, bias_data, dst_data);

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, (cond.dst_row && cond.dst_col));
    }
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(ptr_src, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_DEPTHWISE_H_
