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
 * @file    hvx_nn_pool.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_POOL_H_
#define HVX_NN_POOL_H_

#include "impl/hvx_nn_pool_dfixed.h"
#include "impl/hvx_nn_pool_dfloat.h"

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the pooling functions
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         typename knl_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename knl_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename pad_                          = hvx::util::Array2dParam<0, 0>,
         typename dil_                          = hvx::util::Array2dParam<0, 0>,
         typename str_                          = hvx::util::Array2dParam<1, 1>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         hvx::util::pooling_e pool_type_        = hvx::util::pooling_e::kMax>
struct PoolParam {
    // destination rows/cols
    using dst_rows_v = decltype(hvx::util::WinDstVecParams<src_rows_v, knl_rows_v, pad_::rows, dil_::rows, str_::rows>());
    using dst_cols_v = decltype(hvx::util::WinDstVecParams<src_cols_v, knl_cols_v, pad_::cols, dil_::cols, str_::cols>());

    // tensor parameters
    using src_dim = hvx::util::TensorParam<4, chnls_v, src_cols_v, src_rows_v, batch_v>;
    using dst_dim = hvx::util::TensorParam<4, chnls_v, dst_cols_v, dst_rows_v, batch_v>;

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
    static constexpr auto chnl_vec_elms    = chnls_v::vec_elms;

    // data types
    using src_type = src_type_;
    using dst_type = dst_type_;
    using src_vec  = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec  = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using knl_vec  = hvx::util::vector<src_type, knl_rows_v::elms * knl_cols_v::elms>;
    using src_port = src_vec;
    using dst_port = dst_vec;

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
    constexpr PoolParam() {
        static_assert(knl_rows == knl_rows_vec_size, "Knl rows are not fully vectorized!");
        static_assert(knl_cols == knl_cols_vec_size, "Knl cols are not fully vectorized!");
        hvx::util::TensorVerifyIfVecSizeIs1<src_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifyIfVecSizeIs1<dst_dim, false, true, true, true, true, true>();
        hvx::util::TensorVerifySameDims<src_dim, dst_dim, 4, true, false, false, true, true, true>();
        hvx::util::WinVerifyDim<src_rows, src_cols, knl_rows, knl_cols, pad_rows, pad_cols, dil_rows, dil_cols>();
        hvx::nn::impl::PoolVerifyType<src_type, dst_type>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief applies pool function on an src vector
 */
template<typename param_, hvx::util::pooling_e pool_type_>
HVX_FORCE_INLINE constexpr auto
PoolComp(hvx::util::array1d<typename param_::src_vec, param_::knl_elms>& win, typename param_::dst_vec& dst_data) noexcept -> void {
    HVX_INLINE_TOP();

    for (int64_t chnl_p = 0; chnl_p < param_::chnl_vec_size; ++chnl_p) {
        HVX_UNROLL();

        // buffers needed win to comp one dst element
        typename param_::knl_vec win_tmp{};

        // get needed win
        for (int64_t knl_pix = 0; knl_pix < param_::knl_elms; ++knl_pix)
            win_tmp.Set(win.Get(knl_pix).Get(chnl_p), knl_pix);

        // applies selected pool function on a single element
        switch (pool_type_) {
            case hvx::util::pooling_e::kMax:
                hvx::nn::impl::PoolMax<param_>(win_tmp, dst_data.Get(chnl_p));
                break;
            case hvx::util::pooling_e::kAvg:
                hvx::nn::impl::PoolAvg<param_>(win_tmp, dst_data.Get(chnl_p));
                break;
            default:
                break;
        }
    }
}

/*!
 * @brief top function of the pool layer
 */
template<typename param_, hvx::util::pooling_e pool_type_>
HVX_FORCE_INLINE auto
PoolTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // buffers needed src elements for win to not read same element twice from global memory [dont initialize]
    static hvx::util::array2d<typename param_::src_vec, param_::row_buf_elms, param_::row_buf_num> row_buf;
    static hvx::util::array2d<typename param_::src_vec, param_::win_buf_elms, param_::win_buf_num> win_buf;
    static hvx::util::array2d<typename param_::src_vec, param_::src_buf_elms, param_::src_buf_num> src_buf;
    static hvx::util::array1d<typename param_::src_vec, param_::win_elms> win;
    static hvx::util::array1d<typename param_::src_vec, param_::win_dil_elms> win_dil;

    // directives for buffers and win
    HVX_DATAPACK(row_buf.data, win_buf.data, src_buf.data, win.data, win_dil.data);
    HVX_ARRAY_PARTITION_COMPLETE(row_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(src_buf.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win.data, 1);
    HVX_ARRAY_PARTITION_COMPLETE(win_dil.data, 1);

    // iterates through the tensor vector by vector
    int64_t ptr_src = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::lat; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer the src and dst vectors
        typename param_::src_vec src_data{};
        typename param_::dst_vec dst_data{};

        // flattening loop to improve lat (TODO: loop inefficient for stride >= 2)
        const int64_t src_row = (i / (param_::lat_chnls * param_::lat_cols)) % (param_::lat_rows);
        const int64_t src_col = (i / (param_::lat_chnls)) % (param_::lat_cols);
        const int64_t chnl_v  = (i % param_::lat_chnls);

        // comp conditions for src and dst (TODO: delete template parameters except param_)
        const auto cond = hvx::util::WinCompCond<param_::src_rows, param_::src_cols, param_::dst_rows, param_::dst_cols, param_::knl_rows,
                                                 param_::knl_cols, param_::pad_rows, param_::pad_cols, param_::dil_rows, param_::dil_cols>(
            src_col, src_row);

        // read next src vector
        hvx::util::StreamReadData<>(src, src_data, ptr_src, (cond.src_col && cond.src_row));

        // updates the win and its buffers (TODO: delete template parameters except param_)
        hvx::util::WinUpdate<typename param_::src_type, typename param_::src_dim, param_::knl_rows, param_::knl_cols, param_::dil_rows,
                             param_::dil_cols>(src_row, src_col, chnl_v, 0, src_data, row_buf, src_buf, win_buf, win_dil, win);

        // applies pool function on an src vector
        hvx::nn::PoolComp<param_, pool_type_>(win, dst_data);

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, (cond.dst_col && cond.dst_row));
    }
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(ptr_src, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif // HVX_NN_POOL_H_
