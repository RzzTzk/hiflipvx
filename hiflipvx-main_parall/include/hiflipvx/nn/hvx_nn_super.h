/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * �Software�), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_nn_super.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_SUPER_H_
#define HVX_NN_SUPER_H_

#include "../hvx_defs.h"
#include "impl/hvx_nn_conv_dfixed.h"
#include "impl/hvx_nn_conv_dfloat.h"
#include "impl/hvx_nn_depthwise_dfixed.h"
#include "impl/hvx_nn_depthwise_dfloat.h"
#include "impl/hvx_nn_pool_dfixed.h"
#include "impl/hvx_nn_pool_dfloat.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief Computes the elementwise function for integer fixed point data type
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>, // data type for the inputs
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>, // data type for the outputs
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>, // data type for the weights
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>, // data type for the bias
         typename batch_v                       = hvx::util::VectorParam<1, 1>,   // batch size
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,   // number of rows in the input tensor
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,   // number of columns in the input tensor
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,   // number of channels (equal to input channels)
         typename fms_v                         = hvx::util::VectorParam<1, 1>,   // number of feature maps (equal to output channels)
         typename knl_rows_v                    = hvx::util::VectorParam<1, 1>,   // number of rows in the kernel
         typename knl_cols_v                    = hvx::util::VectorParam<1, 1>,   // number of columns in the kernel
         typename pad_rows                      = hvx::util::Array2dParam<0, 0>,  // number of zeros added on Y-axis on up and down sides
         typename pad_cols                      = hvx::util::Array2dParam<0, 0>,  // number of zeros added on x-axis on left and right sides
         typename dil_                          = hvx::util::Array2dParam<0, 0>,  // gap between two kernel elements in (Y/X)-direction
         typename str_                          = hvx::util::Array2dParam<1, 1>,  // number of elements a window moves in (Y/X)-direction
         int64_t buf_wgts_                      = false,                          // if weights should be internally buffered on first read
         int64_t buf_bias_                      = false,                          // if bias should be buffered internally on first read
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         hvx::util::layer_e layer_type_         = hvx::util::layer_e::None,
         hvx::util::pooling_e pool_type_        = hvx::util::pooling_e::kMax>
struct SuperParam {
    // destination rows/cols
    using dst_rows_v =
        decltype(hvx::util::WinDstVecParams<src_rows_v, knl_rows_v, pad_rows::rows, pad_rows::cols, dil_::rows, str_::rows>());
    using dst_cols_v =
        decltype(hvx::util::WinDstVecParams<src_cols_v, knl_cols_v, pad_cols::rows, pad_cols::cols, dil_::cols, str_::cols>());
    // tensor parameters

    static constexpr int64_t group_vec_elms = chnls_v::elms - ((chnls_v::elms - 1) * (layer_type_ == hvx::util::layer_e::Conv));
    static constexpr int64_t group_vec_size = chnls_v::vec_size - ((chnls_v::vec_size - 1) * (layer_type_ == hvx::util::layer_e::Conv));
    using src_chnls_v                       = hvx::util::VectorParam<chnls_v::elms, chnls_v::vec_size>;
    using dst_chnls_v                       = hvx::util::VectorParam<group_vec_elms * fms_v::elms, group_vec_size * fms_v::vec_size>;
    /*    using dst_chnls_v = hvx::util::VectorParam<fms_v::elms, fms_v::vec_size>; */
    using src_dim  = hvx::util::TensorParam<4, src_chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim  = hvx::util::TensorParam<4, dst_chnls_v, dst_cols_v, dst_rows_v, batch_v>;
    using wgts_dim = hvx::util::TensorParam<4, knl_cols_v, knl_rows_v, chnls_v, fms_v>;
    using bias_dim = hvx::util::TensorParam<1, dst_chnls_v>;

    // dimensions
    static constexpr auto batch                 = batch_v::elms;
    static constexpr auto src_rows              = src_rows_v::elms;
    static constexpr auto src_row_vec_size      = src_rows_v::vec_size;
    static constexpr auto src_row_vec_elms      = src_rows_v::vec_elms;
    static constexpr auto src_cols              = src_cols_v::elms;
    static constexpr auto src_col_vec_size      = src_cols_v::vec_size;
    static constexpr auto src_col_vec_elms      = src_cols_v::vec_elms;
    static constexpr auto dst_rows              = dst_rows_v::elms;
    static constexpr auto dst_row_vec_size      = dst_rows_v::vec_size;
    static constexpr auto dst_row_vec_elms      = dst_rows_v::vec_elms;
    static constexpr auto dst_cols              = dst_cols_v::elms;
    static constexpr auto dst_col_vec_size      = dst_cols_v::vec_size;
    static constexpr auto dst_col_vec_elms      = dst_cols_v::vec_elms;
    static constexpr auto chnls                 = chnls_v::elms;
    static constexpr auto chnl_vec_size         = chnls_v::vec_size;
    static constexpr auto chnl_vec_elms         = chnls_v::elms / chnls_v::vec_size;
    static constexpr auto fms                   = fms_v::elms;
    static constexpr auto fm_vec_size           = fms_v::vec_size;
    static constexpr auto fm_vec_elms           = fms_v::elms / fms_v::vec_size;
    static constexpr auto wgts_vec_size         = wgts_dim::vec_size;
    static constexpr auto wgts_vec_elms         = wgts_dim::vec_elms;
    static constexpr auto wgt_src_chnl_vec_elms = chnl_vec_elms - ((chnl_vec_elms - 1) * (layer_type_ == hvx::util::layer_e::Depthwise));
    static constexpr auto wgt_dst_chnl_vec_elms =
        fm_vec_elms * (layer_type_ != hvx::util::layer_e::Depthwise) + chnl_vec_elms * (layer_type_ == hvx::util::layer_e::Depthwise);
    static constexpr auto bias_vec_size = bias_dim::vec_size;
    static constexpr auto bias_vec_elms = bias_dim::vec_elms;

    // data types
    using src_type  = src_type_;
    using dst_type  = dst_type_;
    using wgts_type = wgts_type_;
    using bias_type = bias_type_;
    using comp_type = hvx::util::def_int_type_t<src_type, wgts_type_>;  
    using src_vec   = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec   = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using wgts_vec  = hvx::util::vector<wgts_type, wgts_dim::vec_size>;
    using bias_vec  = hvx::util::vector<bias_type, bias_dim::vec_size>;
    using comp_vec  = hvx::util::vector<comp_type, fm_vec_size>;
    using knl_vec   = hvx::util::vector<src_type, knl_rows_v::elms * knl_cols_v::elms>;
    using chnl_vec  = hvx::util::vector<src_type, chnls_v::vec_size>;
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
    static constexpr auto pad_rows_up       = pad_rows::rows;
    static constexpr auto pad_rows_down     = pad_rows::cols;
    static constexpr auto pad_cols_left     = pad_cols::rows;
    static constexpr auto pad_cols_right    = pad_cols::cols;
    static constexpr auto dil_rows          = dil_::rows;
    static constexpr auto dil_cols          = dil_::cols;
    static constexpr auto knl_dil_rows      = hvx::util::WinKnlDilLen<knl_rows_v::elms, dil_::rows>();
    static constexpr auto knl_dil_cols      = hvx::util::WinKnlDilLen<knl_cols_v::elms, dil_::cols>();
    static constexpr auto knl_dil_elms      = knl_dil_rows * knl_dil_cols;
    static constexpr auto str_rows          = str_::rows;
    static constexpr auto str_cols          = str_::cols;
    static constexpr auto knl_vec_rows      = knl_rows + (dst_row_vec_size - 1) * str_rows;
    static constexpr auto knl_vec_cols      = knl_cols + (dst_col_vec_size - 1) * str_cols;
    static constexpr auto knl_win_rows = hvx::util::Win_knl_size<knl_dil_rows, src_row_vec_size, dst_row_vec_size, str_rows, pad_rows_up>();
    static constexpr auto knl_win_cols =
        hvx::util::Win_knl_size<knl_dil_cols, src_col_vec_size, dst_col_vec_size, str_cols, pad_cols_left>();
    static constexpr auto knl_sel_cols = knl_dil_cols + (dst_col_vec_size - 1) * str_cols;
    static constexpr auto knl_sel_rows = knl_dil_rows + (dst_row_vec_size - 1) * str_rows;
    static constexpr auto knl_ovr_cols =
        hvx::util::Over_size<knl_win_cols, knl_sel_cols, src_col_vec_size, dst_col_vec_size, str_cols, pad_cols_left>();
    static constexpr auto knl_ovr_rows =
        hvx::util::Over_size<knl_win_rows, knl_sel_rows, src_row_vec_size, dst_row_vec_size, str_rows, pad_rows_up>();

    // summation parameters
    static constexpr auto sum_global_elms = 1;
    static constexpr auto sum_elms        = knl_elms * chnl_vec_size;

    // numerical stability
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;

    // latency
    static constexpr auto ohd_rows  = (pad_rows_up + pad_rows_down) / src_row_vec_size;
    static constexpr auto ohd_cols  = (pad_cols_left + pad_cols_right) / src_col_vec_size;
    static constexpr auto lat_rows  = src_row_vec_elms + ohd_rows;
    static constexpr auto lat_cols  = src_col_vec_elms + ohd_cols;
    static constexpr auto lat_chnls = chnl_vec_elms;
    // static constexpr auto lat_fms   = fm_vec_elms - ((fm_vec_elms - 1) * (layer_type_ != hvx::util::layer_e::Conv));
    static constexpr auto lat_fms = fm_vec_elms;
    static constexpr auto lat     = batch * lat_rows * lat_cols * lat_chnls * lat_fms;

    // buffer parameters
    static constexpr auto row_buf_elms = (src_col_vec_elms + ohd_cols) * chnl_vec_elms;
    static constexpr auto row_buf_num  = hvx::util::Max((knl_win_rows / src_row_vec_size) - 1, static_cast<int64_t>(1));
    static constexpr auto win_buf_elms = chnl_vec_elms;
    static constexpr auto win_buf_num =
        hvx::util::Max((knl_win_cols / src_col_vec_size) - 1, static_cast<int64_t>(1)) * (knl_win_rows / src_row_vec_size);
    static constexpr auto src_buf_elms = chnl_vec_elms;
    static constexpr auto src_buf_num  = 1;
    static constexpr auto win_elms     = knl_sel_rows * knl_sel_cols;
    static constexpr auto win_dil_elms = knl_win_rows * knl_win_cols;
    static constexpr auto buffer_wgts  = buf_wgts_;
    static constexpr auto buffer_bias  = buf_bias_;

    // constructor (verifies the dimensions and types)
    constexpr SuperParam() {
        // TODO: implement the possibility that the kernel does not have to be vectorized
        static_assert(knl_rows == knl_rows_vec_size, "Knl rows are not fully vectorized!");
        static_assert(knl_cols == knl_cols_vec_size, "Knl cols are not fully vectorized!");
        // TODO: also implement a vectorization for the row and col dimension
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, false, false, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, false, false, true, true, true>();
        //
        hvx::util::TensorVerifySameDims<src_dim, dst_dim, 4, false, false, false, true, true, true>();
        hvx::util::BiasVerifyDim<bias_dim, dst_rows, dst_cols, fms, fm_vec_size>();
        hvx::util::WinVerifyDim<src_rows, src_cols, knl_rows, knl_cols, pad_rows_up, pad_cols_left, dil_rows, dil_cols>();
        hvx::nn::impl::ConvVerifyType<src_type, wgts_type, bias_type, dst_type>();
    }
};

/*!
 * @brief applies conv function on an src vector
 */
template<typename param_, hvx::util::pooling_e pool_type_, hvx::util::layer_e layer_type_>
HVX_FORCE_INLINE constexpr auto
SuperComp(int64_t chnl_v,
          hvx::util::array1d<typename param_::comp_vec, param_::sum_global_elms>& sum_global_vec,
          hvx::util::array1d<typename param_::chnl_vec, param_::win_elms>& win,
          typename param_::wgts_vec& wgts_data,
          typename param_::bias_vec& bias_data,
          typename param_::dst_vec& dst_data) noexcept -> void {
    HVX_INLINE_TOP();

    switch (layer_type_) {
        case hvx::util::layer_e::Conv: {
            //for (int64_t fm_p = 0; fm_p < param_::fm_vec_size; ++fm_p) {
            //    HVX_UNROLL();
            //    // buffers needed win and wgts to comp one dst element
            //    hvx::util::vector<typename param_::wgts_type, param_::sum_elms> wgts_tmp{};
            //    hvx::util::vector<typename param_::src_type, param_::sum_elms> win_tmp{};
            //    // get needed win and wgts
            //    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
            //        for (int64_t knl_pix = 0; knl_pix < param_::knl_elms; ++knl_pix) {
            //            const int64_t ptr_fm_p   = fm_p * param_::sum_elms;
            //            const int64_t ptr_chnl_p = chnl_p * param_::knl_elms;
            //            wgts_tmp.Set(wgts_data.Get(ptr_fm_p + ptr_chnl_p + knl_pix), ptr_chnl_p + knl_pix);
            //            win_tmp.Set(win.Get(knl_pix).Get(chnl_p), ptr_chnl_p + knl_pix);
            //        }
            //    }
            //    // applies conv function on a single element
            //    hvx::nn::impl::ConvComp<param_>(chnl_v, sum_global_vec.Get(0).Get(fm_p), win_tmp, wgts_tmp, bias_data.Get(fm_p),
            //                                    dst_data.Get(fm_p));
            //}

             hvx::util::array1d<typename param_::chnl_vec, param_::dst_col_vec_size * param_::dst_row_vec_size> data{};
             for (int64_t fm_p = 0; fm_p < param_::fm_vec_size; ++fm_p) {
                 HVX_UNROLL();
                 hvx::util::vector<typename param_::wgts_type, param_::sum_elms> wgts_tmp{};
                 hvx::util::vector<typename param_::src_type, param_::sum_elms> win_tmp{};                
                 // buffers needed win and weights to comp one dst element
                 // get needed win and weights
                 for (int64_t row_p = 0, row_m = param_::dst_row_vec_size - 1; row_p < param_::dst_row_vec_size; ++row_p, --row_m) {
                     for (int64_t col_p = 0, col_m = param_::dst_col_vec_size - 1; col_p < param_::dst_col_vec_size; ++col_p, --col_m) {
                         for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
                             for (int64_t knl_row = 0, knl_row_sel = 0, knl_row_m = param_::knl_dil_rows - 1; knl_row < param_::knl_rows;
                                     ++knl_row, knl_row_sel += (param_::dil_rows + 1), knl_row_m -= (param_::dil_rows + 1)) {
                                 for (int64_t knl_col = 0, knl_col_sel = 0, knl_col_m = param_::knl_dil_cols - 1; knl_col < param_::knl_cols;
                                         ++knl_col, knl_col_sel += (param_::dil_cols + 1), knl_col_m -= (param_::dil_cols + 1)) {
                                     int64_t knl_mid = (row_m * param_::str_rows + knl_row_sel) * param_::knl_sel_cols + knl_col_sel +
                                                         (col_m * param_::str_cols);
                                     int64_t ptr = knl_row * param_::knl_cols + knl_col;
                                     const int64_t ptr_fm_p   = fm_p * param_::sum_elms;
                                     const int64_t ptr_chnl_p = chnl_p * param_::knl_elms;
                                     wgts_tmp.Set(wgts_data.Get(ptr_fm_p + ptr_chnl_p + ptr), ptr_chnl_p + ptr);
                                     win_tmp.Set(win.Get(knl_mid).Get(chnl_p), ptr_chnl_p + ptr);
                                 }
                             }
                         }
                         // applies depthwise function on a single element (TODO: delete template parameters except param_)
                         hvx::nn::impl::ConvComp<param_>(chnl_v, sum_global_vec.Get(0).Get(fm_p), win_tmp, wgts_tmp, bias_data.Get(fm_p),
                                                         data.Get(row_p * param_::dst_col_vec_size + col_p).Get(fm_p));                        
                     }
                 }
             }
             for (int64_t x = 0; x < param_::dst_row_vec_size; ++x) {
                 for (int64_t i = 0; i < param_::dst_col_vec_size; ++i) {
                     for (int64_t j = 0; j < param_::fm_vec_size; ++j) {
                         int64_t ptr = x * param_::dst_col_vec_size + i;
                         int64_t sto = x * param_::dst_col_vec_size * param_::fm_vec_size + i * param_::fm_vec_size + j;
                         dst_data.Set(data.Get(ptr).Get(j), sto);
                     }
                 }
             }
            break;
        }

        case hvx::util::layer_e::Depthwise: {
            hvx::util::array1d<typename param_::chnl_vec, param_::dst_col_vec_size * param_::dst_row_vec_size> data{};
            for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
                HVX_UNROLL();
                hvx::util::vector<typename param_::wgts_type, param_::knl_elms> wgts_tmp{};
                hvx::util::vector<typename param_::src_type, param_::knl_elms> win_tmp{};
                // buffers needed win and weights to comp one dst element
                // get needed win and weights
                for (int64_t row_p = 0, row_m = param_::dst_row_vec_size - 1; row_p < param_::dst_row_vec_size; ++row_p, --row_m) {
                    for (int64_t col_p = 0, col_m = param_::dst_col_vec_size - 1; col_p < param_::dst_col_vec_size; ++col_p, --col_m) {
                        for (int64_t knl_row = 0, knl_row_sel = 0, knl_row_m = param_::knl_dil_rows - 1; knl_row < param_::knl_rows;
                             ++knl_row, knl_row_sel += (param_::dil_rows + 1), knl_row_m -= (param_::dil_rows + 1)) {
                            for (int64_t knl_col = 0, knl_col_sel = 0, knl_col_m = param_::knl_dil_cols - 1; knl_col < param_::knl_cols;
                                 ++knl_col, knl_col_sel += (param_::dil_cols + 1), knl_col_m -= (param_::dil_cols + 1)) {
                                int64_t knl_mid = (row_m * param_::str_rows + knl_row_sel) * param_::knl_sel_cols + knl_col_sel +
                                                  (col_m * param_::str_cols);
                                int64_t ptr = knl_row * param_::knl_cols + knl_col;
                                win_tmp.Set(win.Get(knl_mid).Get(chnl_p), ptr);
                                wgts_tmp.Set(wgts_data.Get(chnl_p * param_::knl_elms + ptr), ptr);
                            }
                        }
                        // applies depthwise function on a single element (TODO: delete template parameters except param_)
                        hvx::nn::impl::DepthwiseComp<typename param_::wgts_dim, param_>(
                            win_tmp, wgts_tmp, bias_data.Get(chnl_p), data.Get(row_p * param_::dst_col_vec_size + col_p).Get(chnl_p));
                    }
                }
            }
            for (int64_t x = 0; x < param_::dst_row_vec_size; ++x) {
                for (int64_t i = 0; i < param_::dst_col_vec_size; ++i) {
                    for (int64_t j = 0; j < param_::chnl_vec_size; ++j) {
                        int64_t ptr = x * param_::dst_col_vec_size + i;
                        int64_t sto = x * param_::dst_col_vec_size * param_::chnl_vec_size + i * param_::chnl_vec_size + j;
                        dst_data.Set(data.Get(ptr).Get(j), sto);
                    }
                }
            }
            break;
        }
        case hvx::util::layer_e::Pool: {
            hvx::util::array1d<typename param_::chnl_vec, param_::dst_col_vec_size * param_::dst_row_vec_size> data{};

            for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
                HVX_UNROLL();
                // buffers needed win to comp one dst element
                typename param_::knl_vec win_tmp{};
                // get needed win
                for (int64_t row_p = 0, row_m = param_::dst_row_vec_size - 1; row_p < param_::dst_row_vec_size; ++row_p, --row_m) {
                    for (int64_t col_p = 0, col_m = param_::dst_col_vec_size - 1; col_p < param_::dst_col_vec_size; ++col_p, --col_m) {
                        for (int64_t knl_row = 0, knl_row_sel = 0, knl_row_m = param_::knl_dil_rows - 1; knl_row < param_::knl_rows;
                             ++knl_row, knl_row_sel += (param_::dil_rows + 1), knl_row_m -= (param_::dil_rows + 1)) {
                            for (int64_t knl_col = 0, knl_col_sel = 0, knl_col_m = param_::knl_dil_cols - 1; knl_col < param_::knl_cols;
                                 ++knl_col, knl_col_sel += (param_::dil_cols + 1), knl_col_m -= (param_::dil_cols + 1)) {
                                int64_t knl_mid = (row_m * param_::str_rows + knl_row_sel) * param_::knl_sel_cols + knl_col_sel +
                                                  (col_m * param_::str_cols);
                                int64_t ptr = knl_row * param_::knl_cols + knl_col;
                                win_tmp.Set(win.Get(knl_mid).Get(chnl_p), ptr);
                            }
                        }
                        // applies selected pool function on a single element
                        switch (pool_type_) {
                            case hvx::util::pooling_e::kMax:
                                hvx::nn::impl::PoolMax<param_>(
                                    win_tmp, data.Get(row_p * param_::dst_col_vec_size + col_p).Get(chnl_p)); // TODO (different ptr)
                                break;
                            case hvx::util::pooling_e::kAvg:
                                hvx::nn::impl::PoolAvg<param_>(
                                    win_tmp, data.Get(row_p * param_::dst_col_vec_size + col_p).Get(chnl_p)); // TODO (different ptr)
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
            for (int64_t x = 0; x < param_::dst_row_vec_size; ++x) {
                for (int64_t i = 0; i < param_::dst_col_vec_size; ++i) {
                    for (int64_t j = 0; j < param_::chnl_vec_size; ++j) {
                        int64_t ptr = x * param_::dst_col_vec_size + i;
                        int64_t sto = x * param_::dst_col_vec_size * param_::chnl_vec_size + i * param_::chnl_vec_size + j;
                        dst_data.Set(data.Get(ptr).Get(j), sto);
                    }
                }
            }

            break;
        }

        default:
            break;
    }
}

/*!
 * @brief top function of the conv layer
 */
template<typename param_, bool with_bias_ = false, hvx::util::pooling_e pool_type_, hvx::util::layer_e layer_type_>
HVX_FORCE_INLINE auto
SuperTop(typename param_::src_port* src,
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
    static hvx::util::array1d<typename param_::chnl_vec, param_::win_elms> win;
    static hvx::util::array1d<typename param_::chnl_vec, param_::win_dil_elms> win_dil;

    // buffers the global sum for one dst vector [dont initialize]
    static hvx::util::array1d<typename param_::comp_vec, param_::sum_global_elms> sum_global;

    // directives for buffers and windows
    HVX_DATAPACK(bias_buf.data, row_buf.data, win_buf.data, src_buf.data, win.data, win_dil.data); // wgts_buf.data,
    HVX_ARRAY_PARTITION_COMPLETE(row_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(src_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_dil.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(sum_global.data, 0);

    // iterates through the tensor vector by vector
    int64_t ptr_src = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::lat; ++i) {
        HVX_PIPELINE_ON(1, frp);
        // HVX_PRAGMA(HLS dependence variable = sum_global type = inter distance = param_::sum_global_elms true)

        // HVX_PRAGMA(HLS dependence variable = sum_global type = inter direction = WAW|WAR distance = 3 true);

        // buffer the src, dst, wgts and bias vectors
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};
        typename param_::wgts_vec wgts_data{};
        typename param_::bias_vec bias_data{};

        // flattening loop to improve latency (TODO: loop inefficient for stride >= 2)
        const int64_t src_row        = (i / (param_::lat_chnls * param_::lat_fms * param_::lat_cols)) % (param_::lat_rows);
        const int64_t src_col        = (i / (param_::lat_chnls * param_::lat_fms)) % (param_::lat_cols);
        const int64_t fm_v           = (i / (param_::lat_chnls)) % (param_::lat_fms) * (layer_type_ == hvx::util::layer_e::Conv);
        const int64_t chnl_v         = (i % (param_::lat_chnls));
        const int64_t wgt_src_chnl_v = chnl_v * (layer_type_ == hvx::util::layer_e::Conv);
        const int64_t wgt_dst_chnl_v =
            fm_v * (layer_type_ == hvx::util::layer_e::Conv) + chnl_v * (layer_type_ == hvx::util::layer_e::Depthwise);
        // comp conditions for src and dst (TODO: delete template parameters except param_)4
        const auto cond =
            hvx::util::WinCompCond<param_::src_rows, param_::src_cols, param_::dst_rows, param_::dst_cols, param_::src_row_vec_size,
                                   param_::src_col_vec_size, param_::dst_row_vec_size, param_::dst_col_vec_size, param_::knl_win_rows,
                                   param_::knl_win_cols, param_::knl_rows, param_::knl_cols, param_::pad_rows_up, param_::pad_rows_down,
                                   param_::pad_cols_left, param_::pad_cols_right, param_::str_cols, param_::str_rows, param_::dil_rows,
                                   param_::dil_cols>(src_col, src_row);
        const bool cond_chnl = (fm_v == 0);
        const bool cond_fm   = (chnl_v == (param_::chnl_vec_elms - 1)) || (layer_type_ != hvx::util::layer_e::Conv);
        const bool cond_wgts = (cond.dst_row && cond.dst_col && (layer_type_ != hvx::util::layer_e::Pool));
        const bool cond_bias = (with_bias_ && cond.dst_row && cond.dst_col && cond_fm && (layer_type_ != hvx::util::layer_e::Pool));
        // read next src vector
        hvx::util::StreamReadData<>(src, src_data, ptr_src, (cond.src_row && cond.src_col && cond_chnl));

        // updates the window and its buffers (TODO: delete template parameters except param_)                                                        win_dil, win);
        hvx::util::WinUpdate<typename param_::src_type, typename param_::src_dim, param_::ohd_cols, param_::knl_rows, param_::knl_cols,
                             param_::dil_rows, param_::dil_cols, param_::str_rows, param_::str_cols, param_::knl_sel_rows,
                             param_::knl_sel_cols, param_::knl_win_rows, param_::knl_win_cols, param_::knl_vec_rows, param_::knl_vec_cols,
                             param_::knl_ovr_rows, param_::knl_ovr_cols, param_::dst_row_vec_size, param_::dst_col_vec_size>(
            src_row, src_col, chnl_v, fm_v, src_data, row_buf, src_buf, win_buf, win_dil, win);
        // read weights src vector (TODO: delete template parameters except param_)
        hvx::util::WeightsUpdate<typename param_::wgts_type, param_::wgts_vec_size, param_::wgt_src_chnl_vec_elms,
                                 param_::wgt_dst_chnl_vec_elms, param_::buffer_wgts>(wgt_src_chnl_v, wgt_dst_chnl_v, ptr_dst,
                                                                                     wgts_buffered_, cond_wgts, wgts, wgts_buf, wgts_data);

        // std::cout << "    wgt_size:" << param_::knl_sel_cols << "\n";
        // read bias src vector (TODO: delete template parameters except param_)
        hvx::util::BiasUpdate<typename param_::bias_type, param_::dst_chnls_v::vec_size, param_::bias_vec_elms, param_::buffer_bias>(
            ptr_dst, bias_buffered_, cond_bias, bias, bias_buf, bias_data);

        // applies conv function on an src vector
        hvx::nn::SuperComp<param_, pool_type_, layer_type_>(chnl_v, sum_global, win, wgts_data, bias_data, dst_data);

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, (cond.dst_row && cond.dst_col && cond_fm));
    }
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(ptr_src, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_SUPER_H_
