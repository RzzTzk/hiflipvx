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
 * @file    hvx_util_window.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_WINDOW_RE_H_
#define HVX_UTIL_WINDOW_RE_H_

#include "hvx_util_tensor.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief Stores the input and output conditions for a kernel/window based function
 */
struct WindowCond_re {
    int64_t src_col;
    int64_t src_row;
    int64_t dst_col;
    int64_t dst_row;
};

template<typename src_type_,
         typename src_dim_,
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>(),
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t in_vec_size_       = src_chnl_vec_size_ * src_cols_vec_size_ * src_rows_vec_size_>
HVX_FORCE_INLINE constexpr auto
SplitVector_re(hvx::util::vector<src_type_, in_vec_size_>& src_vec,
            hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, src_cols_vec_size_ * src_rows_vec_size_>& src_con) noexcept
    -> void {
    HVX_INLINE_TOP();
    hvx::util::vector<src_type_, src_chnl_vec_size_> src_mid{};
    for (int64_t x = 0; x < src_rows_vec_size_; ++x) {
        for (int64_t i = 0; i < src_cols_vec_size_; ++i) {
            for (int64_t j = 0; j < src_chnl_vec_size_; ++j) {
                int64_t ptr = x * (src_cols_vec_size_ * src_chnl_vec_size_) + (i * src_chnl_vec_size_ + j);
                src_mid.Set(src_vec.Get(ptr), j);
            }
            src_con.Set(src_mid, x * src_cols_vec_size_ + i);
        }
    }
}

template<typename src_type_,
         typename src_dim_,
         int64_t knl_dil_rows_,
         int64_t knl_dil_cols_,
         int64_t knl_win_cols_,
         int64_t knl_win_rows_,
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>(),
         int64_t in_vec_size_       = src_chnl_vec_size_ * src_cols_vec_size_ * src_rows_vec_size_>
HVX_FORCE_INLINE constexpr auto
CombineVector_re(int64_t knl_dil_row,
              int64_t knl_dil_col,
              hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_win_cols_ * knl_win_rows_>& win_dil,
              hvx::util::vector<src_type_, in_vec_size_>& data_mid) noexcept -> void {
    HVX_INLINE_TOP();
    for (int64_t x = 0, x_mid = src_rows_vec_size_ - 1; x < src_rows_vec_size_; ++x, --x_mid) {
        for (int64_t i = 0, i_mid = src_cols_vec_size_ - 1; i < src_cols_vec_size_; ++i, --i_mid) {
            for (int64_t j = 0; j < src_chnl_vec_size_; ++j) {
                int64_t ptr = 0;
                int64_t sto = x * src_cols_vec_size_ * src_chnl_vec_size_ + i * src_chnl_vec_size_ + j;
                if (knl_dil_col == 0) {
                    if (knl_dil_row == 0) {
                        ptr = (x_mid * knl_win_cols_) + src_cols_vec_size_ - (i + 1);
                        data_mid.Set(win_dil.Get(ptr).Get(j), sto);
                        
                    } else {
                        ptr = (((knl_dil_row - 1) * src_rows_vec_size_ + x_mid) * knl_win_cols_) + i_mid;
                        data_mid.Set(win_dil.Get(ptr).Get(j), sto);
                    }
                } else {
                    ptr = ((knl_dil_row * src_rows_vec_size_ + x_mid) * knl_win_cols_) + knl_dil_col * src_cols_vec_size_ - (i + 1);
                    data_mid.Set(win_dil.Get(ptr).Get(j), sto);
                }
            }
        }
    }
}

/*!
 * @brief Read data and write it into window
 */
template<typename src_type_,
         typename src_dim_,
         int64_t ohd_cols,
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t knl_dil_rows_,
         int64_t knl_dil_cols_,
         int64_t knl_win_cols_,
         int64_t knl_win_rows_,
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t src_rows_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 2>() / src_rows_vec_size_,
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_cols_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 1>() / src_cols_vec_size_,                  
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>(),
         int64_t src_chnl_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 0>() / src_chnl_vec_size_,
         int64_t in_vec_size_       = src_rows_vec_size_ * src_cols_vec_size_ * src_chnl_vec_size_,         
         int64_t row_buf_cols_      = (src_cols_vec_elms_ + ohd_cols) * src_chnl_vec_size_,
         int64_t row_buf_rows_      = hvx::util::Max((knl_win_rows_ / src_rows_vec_size_) - 1, static_cast<int64_t>(1)),
         int64_t win_buf_cols_      = hvx::util::Max((knl_win_cols_ / src_cols_vec_size_) - 1, static_cast<int64_t>(1))>
HVX_FORCE_INLINE constexpr auto
WinUpdateElms_re(
    const int64_t src_row,
    const int64_t src_col,
    const int64_t src_chnl_v,
    const int64_t dst_chnl_v,
    hvx::util::vector<src_type_, in_vec_size_>& src_vec,
    hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, row_buf_cols_, row_buf_rows_>& row_buf,
    hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_elms_, 1>& src_buf,
    hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_size_, win_buf_cols_*(knl_win_rows_ / src_rows_vec_size_)>& win_buf,
    hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_win_rows_ * knl_win_cols_>& win_dil) noexcept -> void {
    HVX_INLINE_TOP();

    // constants
    constexpr int64_t src_rows  = hvx::util::TensorGetDimElms<src_dim_, 2>();
    constexpr int64_t src_cols  = hvx::util::TensorGetDimElms<src_dim_, 1>();

    // Read data and write it into window
    for (int64_t knl_dil_row = 0; knl_dil_row < (knl_win_rows_ / src_rows_vec_size_); ++knl_dil_row) {
        for (int64_t knl_dil_col = 0; knl_dil_col < (knl_win_cols_ / src_cols_vec_size_); ++knl_dil_col) {
            // Used for zero padding
            hvx::util::vector<src_type_, in_vec_size_> data_mid{};
            hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, src_rows_vec_size_ * src_cols_vec_size_> data{};
            // Get input from (linebuffer and (src_buf or input))
            if (knl_dil_col == 0) {
                // Only read if input is valid, else output zero
                if (src_col < src_cols) {
                    // Read from input into row 0 of window
                    if (knl_dil_row == 0) {
                        // Only read if input is valid, else output zero
                        if (src_row < src_rows) {
                            // Read from input on 1. ofm
                            if (dst_chnl_v == 0)
                                data_mid = src_vec;

                            // Otherwise read from src_buf (which buffers the dame value)
                            else
                                data_mid = src_buf.Get(src_chnl_v, 0);
                        }

                        // Read from linebuffer into rows other than row 0
                    } else {
                        // Only read from valid linebuffer
                        if (src_row >= knl_dil_row) {
                            data_mid = row_buf.Get((src_col * src_chnl_vec_size_) + (src_chnl_v % src_chnl_vec_size_), knl_dil_row - 1);
                        }
                    }
                }

                // Get input from win_buf
            } else {
                if (knl_dil_col < (src_col + 1))
                    data_mid = win_buf.Get((src_chnl_v % src_chnl_vec_size_), (knl_dil_row * win_buf_cols_) + (knl_dil_col - 1));
            }
            SplitVector_re<src_type_, src_dim_>(data_mid, data);

            // Store data into window
            for (int64_t row_vec = 0, row_dil = src_rows_vec_size_ - 1; row_vec < src_rows_vec_size_; ++row_vec, --row_dil) {
                for (int64_t col_vec = 0, col_dil = src_cols_vec_size_ - 1; col_vec < src_cols_vec_size_; ++col_vec, --col_dil) {
                    int64_t ptr = (knl_dil_row * src_rows_vec_size_ + row_dil) * knl_win_cols_ + (knl_dil_col + 1) * src_cols_vec_size_ -
                                   (col_vec + 1);
                    int64_t red = row_vec * src_cols_vec_size_ + col_vec;
                    win_dil.Set(data.Get(red), ptr);
                }
            }
        }
    }
}

/*!
 * @brief Store data from window into win_buf and (linebuffer or inputbuffer).
 */
template<typename src_type_,
         typename src_dim_,
         int64_t ohd_cols,
         int64_t knl_dil_rows_,
         int64_t knl_dil_cols_,
         int64_t knl_win_cols_,
         int64_t knl_win_rows_,         
         int64_t dst_chnl_vec_elms_,
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t src_rows_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 2>() / src_rows_vec_size_,
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_cols_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 1>() / src_cols_vec_size_,         
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>(),
         int64_t src_chnl_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 0>() / src_chnl_vec_size_,
         int64_t in_vec_size_       = src_rows_vec_size_ * src_cols_vec_size_ * src_chnl_vec_size_,
         int64_t row_buf_cols_      = (src_cols_vec_elms_ + ohd_cols) * src_chnl_vec_size_,
         int64_t row_buf_rows_      = hvx::util::Max((knl_win_rows_ / src_rows_vec_size_) - 1, static_cast<int64_t>(1)),
         int64_t win_buf_cols_      = hvx::util::Max((knl_win_cols_ / src_cols_vec_size_) - 1, static_cast<int64_t>(1))>
HVX_FORCE_INLINE constexpr auto
WinUpdateBufs_re(const int64_t src_col,
              const int64_t src_chnl_v,
              const int64_t dst_chnl_v,
              hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_win_cols_ * knl_win_rows_>& win_dil,
              hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, row_buf_cols_, row_buf_rows_>& row_buf,
              hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_elms_, 1>& src_buf,
              hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_size_, win_buf_cols_ * (knl_win_rows_ / src_rows_vec_size_)>&
                  win_buf) noexcept -> void {
    HVX_INLINE_TOP();

    // constants
    constexpr int64_t src_cols  = hvx::util::TensorGetDimElms<src_dim_, 1>();

    // Store data from window into (win_buf and (linebuffer or src_buf)
    for (int64_t knl_dil_row = 0; knl_dil_row < (knl_win_rows_ / src_rows_vec_size_); ++knl_dil_row) {
        for (int64_t knl_dil_col = 0; knl_dil_col < (knl_win_cols_ / src_cols_vec_size_); ++knl_dil_col) {
            hvx::util::vector<src_type_, in_vec_size_> data_mid{};
            // Store data into (linebuffer or src_buf)
            if (knl_dil_col == 0) {
                if (dst_chnl_v == (dst_chnl_vec_elms_ - 1)) {
                    if ((knl_dil_row > 0) && (src_col < src_cols)) {
                        hvx::util::CombineVector_re<src_type_, src_dim_, knl_dil_rows_, knl_dil_cols_, knl_win_cols_, knl_win_rows_>(
                            knl_dil_row, knl_dil_col, win_dil, data_mid);                     
                        row_buf.Set(data_mid, (src_col * src_chnl_vec_size_) + (src_chnl_v % src_chnl_vec_size_), knl_dil_row - 1);
                    }

                } else {
                    if (knl_dil_row == 0){
                        hvx::util::CombineVector_re<src_type_, src_dim_, knl_dil_rows_, knl_dil_cols_, knl_win_cols_, knl_win_rows_>(
                            knl_dil_row, knl_dil_col, win_dil, data_mid);
                        src_buf.Set(data_mid, src_chnl_v, 0);
                    }

                }

                // Store data into (win_buf)
            } else {
                if (dst_chnl_v == (dst_chnl_vec_elms_ - 1)){
                    hvx::util::CombineVector_re<src_type_, src_dim_, knl_dil_rows_, knl_dil_cols_, knl_win_cols_, knl_win_rows_>(
                        knl_dil_row, knl_dil_col, win_dil, data_mid);
                    win_buf.Set(data_mid, (src_chnl_v % src_chnl_vec_size_), (knl_dil_row * win_buf_cols_) + (knl_dil_col - 1));
                }

            }
        }
    }
}


/*!
 * @brief convert window from dilated version to normal
 */
template<typename src_type_,
         typename src_dim_,
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_,
         int64_t knl_ovr_rows_,
         int64_t knl_ovr_cols_,
         int64_t knl_sel_rows_,
         int64_t knl_sel_cols_,
         int64_t knl_win_rows_,
         int64_t knl_win_cols_,
         int64_t knl_vec_rows_,
         int64_t knl_vec_cols_,
         int64_t dst_row_vec_size_,
         int64_t dst_col_vec_size_,
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>()>
HVX_FORCE_INLINE constexpr auto
WinDilWinConv_re(hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_sel_cols_ * knl_sel_rows_>& win,
              hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_win_cols_ * knl_win_rows_>& win_dil) noexcept
    -> void {
    HVX_INLINE_TOP();

    int64_t knl_row_init = knl_ovr_rows_;
    int64_t knl_col_init = knl_ovr_cols_;
    for (int64_t knl_row = knl_row_init, i = 0; i < knl_sel_rows_; ++knl_row, ++i) {
        for (int64_t knl_col = knl_col_init, j = 0; j < knl_sel_cols_; ++knl_col, ++j) {
            int64_t ptr = (knl_row * knl_win_cols_) + knl_col;
            win.Set(win_dil.Get(ptr), (i * knl_sel_cols_) + j);
        }
    }
}


/*!
 * @brief Updates the window and its buffers
 */
template<typename src_type_,
         typename src_dim_,
         int64_t ohd_cols,         
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_,
         int64_t knl_sel_rows_,
         int64_t knl_sel_cols_,
         int64_t knl_win_rows_,
         int64_t knl_win_cols_,
         int64_t knl_dil_rows_,
         int64_t knl_dil_cols_,         
         int64_t knl_vec_rows_,
         int64_t knl_vec_cols_,
         int64_t knl_ovr_rows_,
         int64_t knl_ovr_cols_,
         int64_t dst_row_vec_size_,
         int64_t dst_col_vec_size_,         
         int64_t dst_chnl_vec_elms_ = 1,
         int64_t src_chnl_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 0>(),
         int64_t src_chnl_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 0>() / src_chnl_vec_size_,
         int64_t src_rows_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 2>(),
         int64_t src_rows_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 2>() / src_rows_vec_size_,
         int64_t src_cols_vec_size_ = hvx::util::TensorGetDimVecSize<src_dim_, 1>(),
         int64_t src_cols_vec_elms_ = hvx::util::TensorGetDimElms<src_dim_, 1>() / src_cols_vec_size_,   
         int64_t in_vec_size_       = src_rows_vec_size_ * src_cols_vec_size_ * src_chnl_vec_size_,      
         int64_t row_buf_cols_      = (src_cols_vec_elms_ + ohd_cols) * src_chnl_vec_size_,
         int64_t row_buf_rows_      = hvx::util::Max((knl_win_cols_ / src_cols_vec_size_) - 1, static_cast<int64_t>(1)),
         int64_t win_buf_cols_      = hvx::util::Max((knl_win_cols_ / src_cols_vec_size_) - 1, static_cast<int64_t>(1))>
HVX_FORCE_INLINE auto
WinUpdate_re(const int64_t src_row,
          const int64_t src_col,
          const int64_t src_chnl_v,
          const int64_t dst_chnl_v,
          hvx::util::vector<src_type_, in_vec_size_>& src,
          hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, row_buf_cols_, row_buf_rows_>& row_buf,
          hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_elms_, 1>& src_buf,
          hvx::util::array2d<hvx::util::vector<src_type_, in_vec_size_>, src_chnl_vec_size_, win_buf_cols_*(knl_win_rows_ / src_rows_vec_size_)>& win_buf,
          hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_win_cols_ * knl_win_rows_>& win_dil,
          hvx::util::array1d<hvx::util::vector<src_type_, src_chnl_vec_size_>, knl_sel_cols_ * knl_sel_rows_>& win) noexcept -> void {
    HVX_INLINE_TOP();

    // Read data and write it into window
    hvx::util::WinUpdateElms_re<src_type_, src_dim_, ohd_cols, knl_rows_, knl_cols_, knl_dil_rows_, knl_dil_cols_, knl_win_cols_, knl_win_rows_>(
        src_row, src_col, src_chnl_v, dst_chnl_v, src, row_buf, src_buf, win_buf, win_dil);

    // Store data from window into (win_buf and (row_buf or src_buf)
    hvx::util::WinUpdateBufs_re<src_type_, src_dim_, ohd_cols,  knl_dil_rows_, knl_dil_cols_, knl_win_cols_, knl_win_rows_,
                             dst_chnl_vec_elms_>(src_col, src_chnl_v, dst_chnl_v, win_dil, row_buf, src_buf, win_buf);

    // Convert window into default format
    hvx::util::WinDilWinConv_re<src_type_, src_dim_, knl_rows_, knl_cols_, dil_rows_, dil_cols_, str_rows_, str_cols_, knl_ovr_rows_,
                             knl_ovr_cols_, knl_sel_rows_, knl_sel_cols_, knl_win_rows_, knl_win_cols_, knl_vec_rows_, knl_vec_cols_,
                             dst_row_vec_size_, dst_col_vec_size_>(win, win_dil);
}


/*!
 * @brief Computes the input and output conditions for a kernel/window based function
 */
template<int64_t src_rows_,
         int64_t src_cols_,
         int64_t dst_rows_,
         int64_t dst_cols_,
         int64_t src_rows_vec,
         int64_t src_cols_vec,
         int64_t dst_row_vec,
         int64_t dst_col_vec,    
         int64_t knl_win_rows,
         int64_t knl_win_cols,              
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t pad_rows_up,
         int64_t pad_rows_down,
         int64_t pad_cols_left,
         int64_t pad_cols_right,
         int64_t str_cols,
         int64_t str_rows,
         int64_t dil_rows_,
         int64_t dil_cols_> 
HVX_FORCE_INLINE constexpr auto
WinCompCond_re(int64_t src_col, int64_t src_row) noexcept -> const WindowCond_re {
    constexpr int64_t dst_col_beg = ((knl_win_cols - pad_cols_left)/ src_cols_vec)  - 1;
    constexpr int64_t dst_row_beg = ((knl_win_rows - pad_rows_up)/ src_rows_vec)  - 1; 
    //constexpr int64_t dst_col_beg = RoundUp< (knl_win_cols - pad_cols_left), src_cols_vec >() - 1;
    //constexpr int64_t dst_row_beg = RoundUp<(knl_win_rows - pad_rows_up), src_rows_vec>()  - 1; 
    constexpr int64_t dst_col_end = dst_col_beg + (dst_cols_/ dst_col_vec) * str_cols;
    constexpr int64_t dst_row_end = dst_row_beg + (dst_rows_/ dst_row_vec) * str_rows; 


    // calculate and return conditions
    const bool cond_src_col = (src_col < src_cols_) && ((src_col - (src_cols_ / src_cols_vec)) < 0);
    const bool cond_src_row = (src_row < src_rows_) && ((src_row - (src_rows_ / src_rows_vec)) < 0);
    const bool cond_dst_col =
        ((((src_col - dst_col_beg) * src_cols_vec) % str_cols) == 0) && (src_col >= dst_col_beg) && (src_col < dst_col_end);
    const bool cond_dst_row =
        ((((src_row - dst_row_beg) * src_rows_vec) % str_rows) == 0) && (src_row >= dst_row_beg) && (src_row < dst_row_end);      
    return {cond_src_col, cond_src_row, cond_dst_col, cond_dst_row};
}


/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_WINDOW_H_
