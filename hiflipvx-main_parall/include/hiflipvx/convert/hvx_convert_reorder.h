/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
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
 * @file    hvx_convert_reshape.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_REORDER_H
#define HVX_CONVERT_REORDER_H

#include "../hvx_defs.h"
#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the reordering function
 */
template<typename src_type_                 = hvx::util::dfixed<int16_t, 15>, // data type for the inputs
         typename dst_type_                 = hvx::util::dfixed<int16_t, 15>, // data type for the outputs
         typename batch_v                   = hvx::util::VectorParam<1, 1>,   // batch size
         typename src_rows_v                = hvx::util::VectorParam<1, 1>,   // number of rows in the input tensor
         typename src_cols_v                = hvx::util::VectorParam<1, 1>,   // number of columns in the input tensor
         typename chnls_v                   = hvx::util::VectorParam<1, 1>,   // number of channels (equal to input channels)
         typename fms_v                     = hvx::util::VectorParam<1, 1>,   // number of feature maps (equal to output channels)
         hvx::util::reorder_e reorder_type_ = hvx::util::reorder_e::None>
struct ReorderParam {
    using dst_rows_v =
        decltype(hvx::util::ReorderDstVecParams<src_rows_v>());
    using dst_cols_v =
        decltype(hvx::util::ReorderDstVecParams<src_cols_v>());
    // tensor parameters
    using src_dim  = hvx::util::TensorParam<4, chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim  = hvx::util::TensorParam<4, fms_v, dst_cols_v, dst_rows_v, batch_v>;
    // using wgts_dim = hvx::util::TensorParam<4, knl_cols_v, knl_rows_v, chnls_v, fms_v>;
    // using bias_dim = hvx::util::TensorParam<1, fms_v>;
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
    static constexpr auto fms              = fms_v::elms;
    static constexpr auto fm_vec_size      = fms_v::vec_size;
    static constexpr auto fm_vec_elms      = fms_v::elms / fms_v::vec_size;

    // types
    using src_type = src_type_;
    using dst_type = dst_type_;
    using src_vec  = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec  = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using src_port = src_vec;
    using dst_port = dst_vec;

    // window (kernel) parameters
    // static constexpr auto knl_rows          = knl_rows_v::elms;
    // static constexpr auto knl_rows_vec_size = knl_rows_v::vec_size;
    // static constexpr auto knl_cols          = knl_cols_v::elms;
    // static constexpr auto knl_cols_vec_size = knl_cols_v::vec_size;
    // static constexpr auto knl_elms          = knl_rows * knl_cols;
    // static constexpr auto pad_rows_up       = pad_rows::rows;
    // static constexpr auto pad_rows_down     = pad_rows::cols;
    // static constexpr auto pad_cols_left     = pad_cols::rows;
    // static constexpr auto pad_cols_right    = pad_cols::cols;
    // static constexpr auto dil_rows          = dil_::rows;
    // static constexpr auto dil_cols          = dil_::cols;
    // static constexpr auto knl_dil_rows      = hvx::util::WinKnlDilLen<knl_rows_v::elms, dil_::rows>();
    // static constexpr auto knl_dil_cols      = hvx::util::WinKnlDilLen<knl_cols_v::elms, dil_::cols>();
    // static constexpr auto knl_dil_elms      = knl_dil_rows * knl_dil_cols;
    // static constexpr auto str_rows          = str_::rows;
    // static constexpr auto str_cols          = str_::cols;

    // buffer size
    static constexpr auto buf_elms            = src_cols * src_rows * chnls;
    static constexpr int64_t src_buf_vec_elms = buf_elms / src_dim::vec_size;
    static constexpr int64_t dst_buf_vec_elms = buf_elms / dst_dim::vec_size;

    // assertions
    constexpr ReorderParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, true, true, true, true, true>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Top function of the Reordering function, if (input vector size > output vector size)
 */
// template<typename param_, std::enable_if_t<(param_::src_dim::vec_size > param_::dst_dim::vec_size), bool> = true>
template<typename param_, hvx::util::reorder_e reorder_type_>
HVX_FORCE_INLINE auto
HwReorderTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    static hvx::util::array1d<typename param_::src_type, param_::buf_elms> buffer;
    typename param_::src_vec src_data{};
    typename param_::dst_vec dst_data{};
    // buffer to overcome the vector size missmatch

    switch (reorder_type_) {
        case hvx::util::reorder_e::Positive: {
            for (int64_t i = 0, ptr_src = 0, ptr_dst = 0; i < param_::src_dim::vec_elms + param_::dst_dim::vec_elms; ++i) {
                HVX_PIPELINE_ON(1, frp);
                // pointers
                const int64_t ptr_src_buf_off = i % (param_::src_buf_vec_elms + param_::dst_buf_vec_elms);

                // Read input
                if (ptr_src_buf_off < param_::src_buf_vec_elms) {
                    // read next src vector
                    hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

                    // write data to buffer
                    for (int64_t p = 0; p < param_::src_dim::vec_size; ++p) {
                        const int64_t ptr_src_buf = (ptr_src_buf_off * param_::src_dim::vec_size) + p;
                        buffer.Set(src_data.Get(p), ptr_src_buf);
                    }

                } else {
                    int64_t start_point = 0;
                    for (int64_t m = 0; m < param_::dst_row_vec_size; ++m) {
                        for (int64_t k = 0; k < param_::dst_col_vec_size; ++k) {
                            for (int64_t j = 0; j < param_::fm_vec_size; ++j) {
                                if (i - param_::src_dim::vec_elms == 0) {
                                    start_point = m * param_::fms * param_::src_cols + k * param_::fms;
                                } else {
                                    const int64_t mid = i - param_::src_dim::vec_elms;
                                    start_point       = (mid / (param_::dst_col_vec_elms * param_::chnl_vec_elms)) * param_::src_cols *
                                                      param_::fms * param_::dst_row_vec_size +
                                                  m * param_::fms * param_::src_cols +
                                                  ((mid / param_::chnl_vec_elms) % param_::dst_col_vec_elms) * param_::fms *
                                                      param_::dst_col_vec_size +
                                                  k * param_::fms + (mid % param_::chnl_vec_elms) * param_::fm_vec_size;
                                }
                                const int64_t ptr = m * param_::fm_vec_size * param_::dst_col_vec_size + k * param_::fm_vec_size + j;
                                dst_data.Set(buffer.Get(start_point + j), ptr);
                            }
                        }
                    }
                    // write next dst vector
                    hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
                }
            }
            break;
        }

        case hvx::util::reorder_e::Negative: {
            for (int64_t i = 0, ptr_src = 0, ptr_dst = 0; i < param_::src_dim::vec_elms + param_::dst_dim::vec_elms; ++i) {
                HVX_PIPELINE_ON(1, frp);
                // pointers
                const int64_t ptr_src_buf_off = i % (param_::src_buf_vec_elms + param_::dst_buf_vec_elms);

                // Read input
                if (ptr_src_buf_off < param_::src_buf_vec_elms) {
                    // read next src vector
                    hvx::util::StreamReadData<>(src, src_data, ptr_src, true);

                    int64_t start_point = 0;
                    for (int64_t m = 0; m < param_::dst_row_vec_size; ++m) {
                        for (int64_t k = 0; k < param_::dst_col_vec_size; ++k) {
                            for (int64_t j = 0; j < param_::fm_vec_size; ++j) {
                                if (i == 0) {
                                    start_point = m * param_::fms * param_::src_cols + k * param_::fms;
                                } else {
                                    start_point       = (i / (param_::dst_col_vec_elms * param_::chnl_vec_elms)) * param_::src_cols *
                                                      param_::fms * param_::dst_row_vec_size +
                                                  m * param_::fms * param_::src_cols +
                                                  ((i / param_::chnl_vec_elms) % param_::dst_col_vec_elms) * param_::fms *
                                                      param_::dst_col_vec_size +
                                                  k * param_::fms + (i % param_::chnl_vec_elms) * param_::fm_vec_size;
                                }
                                const int64_t ptr = m * param_::fm_vec_size * param_::dst_col_vec_size + k * param_::fm_vec_size + j;
                                buffer.Set(src_data.Get(ptr), start_point + j);
                            }
                        }
                    }
                } else {
                    for (int64_t p = 0; p < param_::dst_dim::vec_size; ++p) {
                        int64_t ptr_buf = (i - param_::src_dim::vec_elms)*param_::dst_dim::vec_size  + p;
                        // read data from buffer
                        dst_data.Set(buffer.Get(ptr_buf), p);
                    }
                    // write next dst vector
                    hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
                }
            }
            break;
        }

        default:
            break;
    }
}



/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_REORDER_H
