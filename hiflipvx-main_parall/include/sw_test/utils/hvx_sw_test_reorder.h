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
 * @file    hvx_sw_test_new_reshape.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_REORDER_H
#define HVX_SW_TEST_REORDER_H

#include "hvx_sw_test_helper.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief SW reshape function
 */
template<typename param_, hvx::util::reorder_e reorder_type_>
constexpr auto
SwReorder(typename param_::src_type* src, typename param_::dst_type* dst) noexcept -> void {

    switch (reorder_type_)
    {
        case hvx::util::reorder_e::Positive: {
            for(int64_t i = 0; i < param_::dst_buf_vec_elms; ++i){
                int64_t start_point = 0;
                for (int64_t m = 0; m < param_::dst_row_vec_size; ++m) {
                    for (int64_t k = 0; k < param_::dst_col_vec_size; ++k) {
                        for (int64_t j = 0; j < param_::fm_vec_size; ++j) {
                            if (i == 0) {
                                start_point = m * param_::fms * param_::src_cols + k * param_::fms;
                            } else {
                                start_point =
                                
                                    (i / (param_::dst_col_vec_elms * param_::fm_vec_elms)) * param_::src_cols * param_::fms *
                                        param_::dst_row_vec_size +
                                    m * param_::fms * param_::src_cols +
                                    ((i / param_::fm_vec_elms) % param_::dst_col_vec_elms) * param_::fms * param_::dst_col_vec_size +
                                    k * param_::fms + (i % param_::fm_vec_elms) * param_::fm_vec_size;
                            }
                            int64_t ptr = i * param_::dst_dim::vec_size + m * param_::dst_col_vec_size * param_::dst_col_vec_size + k * param_::fm_vec_size + j;
                            dst[ptr] = src[start_point + j];
                        }
                    }
                }    
            }
            break;            
        }

        case hvx::util::reorder_e::Negative: {
            for(int64_t i = 0; i < param_::dst_buf_vec_elms; ++i){
                int64_t start_point = 0;
                for (int64_t m = 0; m < param_::dst_row_vec_size; ++m) {
                    for (int64_t k = 0; k < param_::dst_col_vec_size; ++k) {
                        for (int64_t j = 0; j < param_::fm_vec_size; ++j) {
                            if (i == 0) {
                                start_point = m * param_::fms * param_::src_cols + k * param_::fms;
                            } else {
                                start_point =
                                
                                    (i / (param_::dst_col_vec_elms * param_::fm_vec_elms)) * param_::src_cols * param_::fms *
                                        param_::dst_row_vec_size +
                                    m * param_::fms * param_::src_cols +
                                    ((i / param_::fm_vec_elms) % param_::dst_col_vec_elms) * param_::fms * param_::dst_col_vec_size +
                                    k * param_::fms + (i % param_::fm_vec_elms) * param_::fm_vec_size;
                            }
                            int64_t ptr = i * param_::dst_dim::vec_size + m * param_::dst_col_vec_size * param_::dst_col_vec_size + k * param_::fm_vec_size + j;
                            dst[start_point + j] = src[ptr];
                        }
                    }
                }    
            }
            break;            
        }
        default:
            break;
    }


}

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_NEW_RESHAPE_H
