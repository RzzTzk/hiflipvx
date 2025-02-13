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
 * @file    hvx_sw_test_reduce.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_REDUCE_H
#define HVX_SW_TEST_REDUCE_H

#include "hvx_sw_test_helper.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief Evaluation class for all elementwise functions
 */
template<typename param_, typename eval_>
class ReduceEvaluate {
    // input and output arrays
    std::vector<typename param_::src_port> src_hw_;
    typename eval_::dst_port* dst_hw_;
    std::vector<float> src_sw_;
    std::vector<float> dst_sw_;
    std::vector<float> dst_hw_flt_;

    /*!
     * @brief
     */
    auto ReduceMax(float src, float& dst, int64_t ptr, int64_t dim, std::vector<float>& buf, int64_t& buf_ptr) noexcept -> void {
        dst = (ptr == 0) ? src : std::max(dst, src);
        if (ptr == param_::src_dim::dims[dim] - 1) { // NOLINT
            buf.at(buf_ptr) = dst;
            ++buf_ptr;
        }
    }

    /*!
     * @brief
     */
    auto ReduceMean(float src, float& dst, int64_t ptr, int64_t dim, std::vector<float>& buf, int64_t& buf_ptr) noexcept -> void {
        dst = (ptr == 0) ? (src) : (dst + src);
        if (ptr == param_::src_dim::dims[dim] - 1) {            // NOLINT
            buf.at(buf_ptr) = dst / param_::src_dim::dims[dim]; // NOLINT
            ++buf_ptr;
        }
    }

    /*!
     * @brief
     */
    auto ReduceMin(float src, float& dst, int64_t ptr, int64_t dim, std::vector<float>& buf, int64_t& buf_ptr) noexcept -> void {
        dst = (ptr == 0) ? src : std::min(dst, src);
        if (ptr == param_::src_dim::dims[dim] - 1) { // NOLINT
            buf.at(buf_ptr) = dst;
            ++buf_ptr;
        }
    }

    /*!
     * @brief
     */
    auto ReduceSum(float src, float& dst, int64_t ptr, int64_t dim, std::vector<float>& buf, int64_t& buf_ptr) noexcept -> void {
        dst = (ptr == 0) ? src : (dst + src);
        if (ptr == param_::src_dim::dims[dim] - 1) { // NOLINT
            buf.at(buf_ptr) = dst;
            ++buf_ptr;
        }
    }

    /*!
     * @brief
     */
    auto ReduceComp(int64_t dim_id, std::array<int64_t, hvx::util::limits_e::kTensorDimMax>& dims) {
        // update the total number of elements as they might have changed to to previous reduce functions
        const int64_t elms = std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<int64_t>()); // NOLINT

        // variable used to buffer intermediate results
        float data = 0;

        // iterate over the src tensor
        for (int64_t i = 0, buf_ptr = 0; i < (elms / dims.at(dim_id)); ++i) {
            for (int64_t j = 0; j < dims.at(dim_id); ++j) {
                // calculate the ptr
                const int64_t stride0 = std::accumulate(dims.begin(), dims.begin() + dim_id, 1LL, std::multiplies<int64_t>());     // NOLINT
                const int64_t stride1 = std::accumulate(dims.begin(), dims.begin() + dim_id + 1, 1LL, std::multiplies<int64_t>()); // NOLINT
                const int64_t src_ptr = i % stride0 + (i / stride0) * stride1 + j * stride0;

                // operate reduce functions
                switch (param_::op_type) {
                    case hvx::util::reduce_e::Max: {
                        ReduceMax(src_sw_.at(src_ptr), data, j, dim_id, src_sw_, buf_ptr);
                        break;
                    }
                    case hvx::util::reduce_e::Mean: {
                        ReduceMean(src_sw_.at(src_ptr), data, j, dim_id, src_sw_, buf_ptr);
                        break;
                    }
                    case hvx::util::reduce_e::Min: {
                        ReduceMin(src_sw_.at(src_ptr), data, j, dim_id, src_sw_, buf_ptr);
                        break;
                    }
                    case hvx::util::reduce_e::Sum: {
                        ReduceSum(src_sw_.at(src_ptr), data, j, dim_id, src_sw_, buf_ptr);
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // update dimension to size 1
        dims.at(dim_id) = 1;
    }

    /*!
     * @brief
     */
    auto SwReduce() noexcept -> void {
        // buffers the dimension sizes, as they need to be updated after a tensor dimension has been reduced
        std::array<int64_t, hvx::util::limits_e::kTensorDimMax> dims = {param_::src_dim::dims[0], param_::src_dim::dims[1],
                                                                        param_::src_dim::dims[2], param_::src_dim::dims[3],
                                                                        param_::src_dim::dims[4], param_::src_dim::dims[5]};

        // computes the reduce function on each dimension that should be reduced reparately
        for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i) {
            if (param_::reduce::dims[i] == true) // NOLINT
                ReduceComp(i, dims);
        }

        // update the total number of elements as they might have changed to to previous reduce functions
        int64_t elms = std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<int64_t>()); // NOLINT

        // copy the final results
        std::copy(src_sw_.begin(), src_sw_.begin() + elms, dst_sw_.begin());
    }

public:

    /*!
     * @brief Constructor (allocates memory and creates random inputs)
     */
    ReduceEvaluate(const float src_max): dst_hw_(new typename eval_::dst_port[param_::dst_dim::vec_elms]) {
        src_hw_.resize(param_::src_dim::vec_elms);
        src_sw_.resize(param_::src_dim::elms);
        dst_sw_.resize(param_::dst_dim::elms);
        dst_hw_flt_.resize(param_::dst_dim::elms);
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(src_hw_.data(), src_sw_.data(), src_max);
    };

    /*!
     * @brief Gets the pointer to the input of HW function
     */
    constexpr auto GetSrcHw() noexcept -> typename param_::src_port* {
        return src_hw_.data();
    }

    /*!
     * @brief Gets the pointer to the output of HW function
     */
    constexpr auto GetDstHw() noexcept -> typename eval_::dst_port* {
        return dst_hw_;
    }

    /*!
     * @brief Calls the hardware implementation of the selected reduce function
     */
    auto HwReduce() noexcept -> void {
        switch (param_::op_type) {
            case hvx::util::reduce_e::Max: {
                hvx::HwReduceMax<param_>(src_hw_.data(), dst_hw_);
                break;
            }
            case hvx::util::reduce_e::Mean: {
                hvx::HwReduceMean<param_>(src_hw_.data(), dst_hw_);
                break;
            }
            case hvx::util::reduce_e::Min: {
                hvx::HwReduceMin<param_>(src_hw_.data(), dst_hw_);
                break;
            }
            case hvx::util::reduce_e::Sum: {
                hvx::HwReduceSum<param_>(src_hw_.data(), dst_hw_);
                break;
            }
            default:
                break;
        }
    }

    /*!
     * @brief Runs the software function, converts HW results to float and comapres both results with each other
     */
    auto Evaluation() noexcept -> std::string {
        SwReduce();
        hvx::sw::ConvertDstHwToFloat<typename param_::dst_type, typename param_::dst_dim, eval_::dst_flags>(*dst_hw_, dst_hw_flt_.data());
        return hvx::sw::EvalPrintDiff<typename param_::dst_dim, eval_>(dst_sw_.data(), dst_hw_flt_.data());
    }
};

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_REDUCE_H
