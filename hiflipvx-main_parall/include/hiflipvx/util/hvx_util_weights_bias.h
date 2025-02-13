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
 * @file    hvx_util_weights_bias.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_WEIGHTS_BIAS_H_
#define HVX_UTIL_WEIGHTS_BIAS_H_

#include "hvx_util_tensor.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief Verfies the dimensions of a Bias input
 */
template<typename bias_dim_, int64_t dst_rows_, int64_t dst_cols_, int64_t dst_chnls_, int64_t dst_vec_size_>
HVX_FORCE_INLINE constexpr auto
BiasVerifyDim() noexcept -> void {
    HVX_INLINE_TOP();
    static_assert(((bias_dim_::dim_num == 3) && (bias_dim_::elms == (dst_rows_ * dst_cols_ * dst_chnls_))) ||
                      ((bias_dim_::dim_num == 1) && (bias_dim_::elms == dst_chnls_)),
                  "Bias has 1 dim (channels) or 3 dims (channels, cols, rows) with dame size as output");
    static_assert((bias_dim_::vec_size == dst_vec_size_), "Bias has same vector size as output!");
}

/*!
 * @brief Read bias from input port (dependent on bias size), if bias should be buffered buffer internally, or read bias from internal
 * buffer
 */
template<typename bias_type_, int64_t dst_chnl_vec_size_, int64_t bias_vec_elms_, bool buffer_bias>
HVX_FORCE_INLINE constexpr auto
BiasUpdate(int64_t ptr_out,
           bool& bias_buffered,
           bool cond_chnl_col_row,
           hvx::util::vector<bias_type_, dst_chnl_vec_size_>* bias,
           hvx::util::array1d<hvx::util::vector<bias_type_, dst_chnl_vec_size_>, bias_vec_elms_>& bias_buf,
           hvx::util::vector<bias_type_, dst_chnl_vec_size_>& bias_vec) noexcept -> void {
    HVX_INLINE_TOP();

    // points to the vector
    const int64_t ptr_bias = ptr_out % bias_vec_elms_;
    if (cond_chnl_col_row == true) {
        if (buffer_bias == true) {
            if ((ptr_out < bias_vec_elms_) && (bias_buffered == false)) {
                bias_vec = bias[ptr_bias]; // NOLINT
                bias_buf.Set(bias_vec, ptr_bias);
            } else {
                bias_buffered = true;
                bias_vec      = bias_buf.Get(ptr_bias);
            }
        } else {
            bias_vec = bias[ptr_bias]; // NOLINT
        }
    }
}

/*!
 * @brief read weights from input port, if weights should be buffered buffer internally, or read weights from internal buffer
 */
template<typename wgts_type_, int64_t wgts_vec_size_, int64_t src_chnl_vec_elms_, int64_t dst_chnl_vec_elms_, bool buffer_wgts>
HVX_FORCE_INLINE constexpr auto
WeightsUpdate(int64_t src_chnl_v,
              int64_t dst_chnl_v,
              int64_t ptr_out,
              bool& wgts_buffered,
              bool cond_col_row,
              hvx::util::vector<wgts_type_, wgts_vec_size_>* wgts,
              hvx::util::array1d<hvx::util::vector<wgts_type_, wgts_vec_size_>, src_chnl_vec_elms_ * dst_chnl_vec_elms_>& wgts_buf,
              hvx::util::vector<wgts_type_, wgts_vec_size_>& wgts_vec) noexcept -> void {
    HVX_INLINE_TOP();
    int64_t ptr_wgts = dst_chnl_v * src_chnl_vec_elms_ + src_chnl_v;
    if (cond_col_row == true) {
        if (buffer_wgts == true) {
            if ((ptr_out < dst_chnl_vec_elms_) && (wgts_buffered == false)) {
                wgts_vec = wgts[ptr_wgts]; // NOLINT
                wgts_buf.Set(wgts_vec, ptr_wgts);
            } else {
                wgts_buffered = true;
                wgts_vec      = wgts_buf.Get(ptr_wgts);
            }
        } else {
            wgts_vec = wgts[ptr_wgts]; // NOLINT
        }
    }
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_WEIGHTS_BIAS_H_
