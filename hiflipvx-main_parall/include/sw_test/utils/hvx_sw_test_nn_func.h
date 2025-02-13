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
 * @file    hvx_sw_test_nn_func.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_NN_FUNC_H
#define HVX_SW_TEST_NN_FUNC_H

#include "hvx_sw_test_helper.h"

namespace hvx {
namespace sw {

/******************************************************************************************************************************************/
/*!
 * @brief SW function of the pooling layer
 */
template<typename param_, hvx::util::pooling_e pool_type_>
HVX_FORCE_INLINE constexpr auto
SwPool(const float* src, float* dst) noexcept -> void {
    // iterates over the output tensors
    for (int64_t batch = 0; batch < param_::batch; ++batch) {
        for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
            for (int64_t dst_row = 0; dst_row < param_::dst_rows; ++dst_row) {
                for (int64_t dst_col = 0; dst_col < param_::dst_cols; ++dst_col) {
                
                    // initialize pooling
                    float result = (pool_type_ == hvx::util::pooling_e::kMax) ? (std::numeric_limits<float>::lowest()) : (0.0f);

                    // compute pooling
                    for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                        for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                            const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows_up + knl_row * (param_::dil_rows + 1);
                            const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols_left + knl_col * (param_::dil_cols + 1);

                            // read input
                            float data = 0;
                            if ((ptr_row >= 0) && (ptr_row < param_::src_rows) && (ptr_col >= 0) && (ptr_col < param_::src_cols))
                                data = src[hvx::util::TensorGetPtr<typename param_::src_dim>(batch, ptr_row, ptr_col, chnl)]; // NOLINT

                            // update max or average pooling
                            result = (pool_type_ == hvx::util::pooling_e::kMax) ? (hvx::util::Max(result, data)) : (result + data);
                        }
                    }

                    // compute average pooling
                    result = (pool_type_ == hvx::util::pooling_e::kMax) ? (result) : (result / static_cast<float>(param_::knl_elms));

                    // write output
                    if ((dst_row < param_::dst_rows) && (dst_col < param_::dst_cols))
                        dst[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, dst_row, dst_col, chnl)] = result; // NOLINT
                }
            }
        }
    }
}

//template<typename param_, hvx::util::pooling_e pool_type_>
//HVX_FORCE_INLINE constexpr auto
//SwPool(const float* src, float* dst) noexcept -> void {
//    // iterates over the output tensors
//    for (int64_t batch = 0; batch < param_::batch; ++batch) {
//        for (int64_t dst_row = 0; dst_row < param_::dst_rows; ++dst_row) {
//            for (int64_t dst_col = 0; dst_col < param_::dst_cols; ++dst_col) {
//                for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
//                    // initialize pooling
//                    float result = (pool_type_ == hvx::util::pooling_e::kMax) ? (std::numeric_limits<float>::lowest()) : (0.0f);
//
//                    // compute pooling
//                    for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
//                        for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
//                            const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows_up + knl_row * (param_::dil_rows + 1);
//                            const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols_left + knl_col * (param_::dil_cols + 1);
//
//                            // read input
//                            float data = 0;
//                            if ((ptr_row >= 0) && (ptr_row < param_::src_rows) && (ptr_col >= 0) && (ptr_col < param_::src_cols))
//                                data = src[hvx::util::TensorGetPtr<typename param_::src_dim>(batch, ptr_row, ptr_col, chnl)]; // NOLINT
//
//                            // update max or average pooling
//                            result = (pool_type_ == hvx::util::pooling_e::kMax) ? (hvx::util::Max(result, data)) : (result + data);
//                        }
//                    }
//
//                    // compute average pooling
//                    result = (pool_type_ == hvx::util::pooling_e::kMax) ? (result) : (result / static_cast<float>(param_::knl_elms));
//
//                    // write output
//                    if ((dst_row < param_::dst_rows) && (dst_col < param_::dst_cols))
//                        dst[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, dst_row, dst_col, chnl)] = result; // NOLINT
//                }
//            }
//        }
//    }
//}


/******************************************************************************************************************************************/

/*!
 * @brief SW function of the dense layer
 */
template<typename param_, bool with_bias_ = false>
HVX_FORCE_INLINE constexpr auto
SwDense(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
    // iterates over the output tensors
    for (int64_t batch = 0; batch < param_::batch; ++batch) {
        for (int64_t fm = 0; fm < param_::fms; ++fm) {
            float result = 0.0f;

            // compute dense
            for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
                const float weight = wgts[hvx::util::TensorGetPtr<typename param_::wgts_dim>(fm, chnl)];  // NOLINT
                const float data   = src[hvx::util::TensorGetPtr<typename param_::src_dim>(batch, chnl)]; // NOLINT
                result += data * weight;
            }

            // add bias
            if (with_bias_ == true)
                result += bias[hvx::util::TensorGetPtr<typename param_::src_dim>(fm)]; // NOLINT

            // write output
            dst[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, fm)] = result; // NOLINT
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief SW function of the depthwise convolution layer
 */
template<typename param_, bool with_bias_ = false>
HVX_FORCE_INLINE constexpr auto
SwDepthwise(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
    // iterates over the output tensors
    for (int64_t batch = 0; batch < param_::batch; ++batch) {
        for (int64_t dst_row = 0; dst_row < param_::dst_rows; ++dst_row) {
            for (int64_t dst_col = 0; dst_col < param_::dst_cols; ++dst_col) {
                for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
                    float result = 0.0f;

                    // compute depthwise
                    for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                        for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                            const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows_up + knl_row * (param_::dil_rows + 1);
                            const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols_left + knl_col * (param_::dil_cols + 1);                            
                            // const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows + knl_row * (param_::dil_rows + 1);
                            // const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols + knl_col * (param_::dil_cols + 1);

                            // check if input is in image boundary
                            if ((ptr_row >= 0) && (ptr_row < param_::src_rows) && (ptr_col >= 0) && (ptr_col < param_::src_cols)) {
                                result += src[hvx::util::TensorGetPtr<typename param_::src_dim>(batch, ptr_row, ptr_col, chnl)] * // NOLINT
                                          wgts[hvx::util::TensorGetPtr<typename param_::wgts_dim>(chnl, knl_row, knl_col)];       // NOLINT
                            }
                        }
                    }

                    // add bias
                    if (with_bias_ == true) {
                        if (param_::bias_dim::elms == param_::chnls)
                            result += bias[hvx::util::TensorGetPtr<typename param_::bias_dim>(chnl)]; // NOLINT
                        else if (param_::bias_dim::elms == (param_::dst_rows * param_::dst_cols * param_::chnls))
                            result += bias[hvx::util::TensorGetPtr<typename param_::bias_dim>(dst_row, dst_col, chnl)]; // NOLINT
                    }

                    // write output
                    if ((dst_row < param_::dst_rows) && (dst_col < param_::dst_cols))
                        dst[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, dst_row, dst_col, chnl)] = result; // NOLINT
                }
            }
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief SW function of the convolution layer
 */
template<typename param_, bool with_bias_ = false>
HVX_FORCE_INLINE constexpr auto
SwConv(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
    // iterates over the output tensors
    for (int64_t batch = 0; batch < param_::batch; ++batch) {
        for (int64_t dst_row = 0; dst_row < param_::dst_rows; ++dst_row) {
            for (int64_t dst_col = 0; dst_col < param_::dst_cols; ++dst_col) {
                for (int64_t fm = 0; fm < param_::fms; ++fm) {
                    float result = 0.0f;

                    // iterate and sum over channels
                    for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
                        float sum_chnls = 0.0f;

                        // iterate and sum over kernels
                        for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                            for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                                const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows_up + knl_row * (param_::dil_rows + 1);
                                const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols_left + knl_col * (param_::dil_cols + 1);
                                // const int64_t ptr_row = (dst_row * param_::str_rows) - param_::pad_rows + knl_row * (param_::dil_rows + 1);
                                // const int64_t ptr_col = (dst_col * param_::str_cols) - param_::pad_cols + knl_col * (param_::dil_cols + 1);
                                // check if input is in image boundary
                                if ((ptr_row >= 0) && (ptr_row < param_::src_rows) && (ptr_col >= 0) && (ptr_col < param_::src_cols)) {
                                    sum_chnls +=
                                        src[hvx::util::TensorGetPtr<typename param_::src_dim>(batch, ptr_row, ptr_col, chnl)] * // NOLINT
                                        wgts[hvx::util::TensorGetPtr<typename param_::wgts_dim>(fm, chnl, knl_row, knl_col)];   // NOLINT
                                }
                            }
                        }
                        result += sum_chnls;
                    }

                    // add bias
                    if (with_bias_ == true) {
                        if (param_::bias_dim::elms == param_::fms)
                            result += bias[hvx::util::TensorGetPtr<typename param_::bias_dim>(fm)]; // NOLINT
                        else if (param_::bias_dim::elms == (param_::dst_rows * param_::dst_cols * param_::fms))
                            result += bias[hvx::util::TensorGetPtr<typename param_::bias_dim>(dst_row, dst_col, fm)]; // NOLINT
                    }

                    // write output
                    if ((dst_row < param_::dst_rows) && (dst_col < param_::dst_cols))
                        dst[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, dst_row, dst_col, fm)] = result; // NOLINT
                }
            }
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief SW function of the softmax layer
 */
template<typename param_>
HVX_FORCE_INLINE auto
SwSoftmax(const float* src, float* dst) noexcept -> void {
    // Buffer the exponential of inputs and the sum of all exponentials
    std::vector<float> buf(param_::chnls);

    // iterate over tensor
    for (int64_t b = 0; b < (param_::src_dim::elms / param_::chnls); ++b) {
        float sum = 0.0f;

        // Computes: n(i) = exp(src(i)) | N: sum of all n
        for (int64_t c = 0; c < param_::chnls; ++c) {
            const float data        = src[b * param_::chnls + c]; // NOLINT
            const float exponential = std::exp(data);
            sum += exponential;
            buf.at(c) = exponential;
        }

        // Computes: m(i) = n(i) / N
        for (int64_t c = 0; c < param_::chnls; ++c) {
            const float result         = buf.at(c) / sum;
            dst[b * param_::chnls + c] = result; // NOLINT
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief SW function of the layer normalization function
 */
template<typename param_>
void
SwLayernorm(float* src, float* wgts, float* bias, float* dst) {
    for (int64_t i = 0; i < param_::batch * param_::src_rows * param_::src_cols; ++i) {
        for (int64_t chnl = 0; chnl < param_::chnls; ++chnl) {
            dst[i * param_::chnls + chnl] = src[i * param_::chnls + chnl] * wgts[chnl] + bias[chnl]; // NOLINT
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief SW function of the ConvTranspose layer
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
SwConvTranspose(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
    // kernel parameters
    constexpr int64_t str_rows = (param_::dst_rows - param_::knl_dil_rows + 2 * param_::pad_rows) / (param_::src_rows - 1);
    constexpr int64_t str_cols = (param_::dst_cols - param_::knl_dil_cols + 2 * param_::pad_cols) / (param_::src_cols - 1);

    for (int64_t batch = 0; batch < param_::batch; ++batch) {
        // initializes the output with the corresponding bias value
        for (int64_t dst_row = 0; dst_row < param_::dst_rows; ++dst_row) {
            for (int64_t dst_col = 0; dst_col < param_::dst_cols; ++dst_col) {
                for (int64_t dst_chnl = 0; dst_chnl < param_::fms; ++dst_chnl) {
                    const int64_t ptr_dst  = hvx::util::TensorGetPtr<typename param_::dst_dim>(dst_row, dst_col, dst_chnl);
                    const int64_t ptr_bias = hvx::util::TensorGetPtr<typename param_::src_dim>(dst_chnl);
                    dst[ptr_dst]           =                                                                             // NOLINT
                        (param_::bias_dim::elms == param_::fms)                                                //
                                      ? (bias[ptr_bias])                                                                 // NOLINT
                                      : ((param_::bias_dim::elms == (param_::dst_rows * param_::dst_cols * param_::fms)) //
                                             ? (bias[ptr_dst])                                                           // NOLINT
                                             : (0));
                }
            }
        }

        // multiplies wgts with input and adds to output
        for (int64_t src_row = 0; src_row < param_::src_rows; ++src_row) {
            for (int64_t src_col = 0; src_col < param_::src_cols; ++src_col) {
                for (int64_t dst_chnl = 0; dst_chnl < param_::fms; ++dst_chnl) {
                    for (int64_t src_chnl = 0; src_chnl < param_::chnls; ++src_chnl) {
                        // reads input data
                        auto src_data = src[hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, src_row, src_col, src_chnl)];

                        // Iterates over the kernel
                        for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                            const int64_t dst_row = src_row * str_rows + knl_row * (param_::dil_rows + 1) - param_::pad_rows;
                            for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                                const int64_t dst_col = src_col * str_cols + knl_col * (param_::dil_cols + 1) - param_::pad_cols;

                                // writes output data
                                if ((dst_row >= 0) && (dst_row < param_::dst_rows - 2 * param_::pad_rows) && (dst_col >= 0) &&
                                    (dst_col < param_::dst_cols - 2 * param_::pad_cols)) {
                                    const int64_t ptr_dst =
                                        hvx::util::TensorGetPtr<typename param_::dst_dim>(batch, dst_row, dst_col, dst_chnl);
                                    const int64_t ptr_wgt =
                                        hvx::util::TensorGetPtr<typename param_::wgts_dim>(dst_chnl, src_chnl, knl_row, knl_col);
                                    dst[ptr_dst] += (src_data * wgts[ptr_wgt]); // NOLINT
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_NN_FUNC_H
