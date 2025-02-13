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
 * @file    hvx_sw_test_ew.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_EW_H
#define HVX_SW_TEST_EW_H

#include "hvx_sw_test_helper.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief Evaluation class for all elementwise functions
 */
template<typename param_, typename eval_>
class EwEvaluate {
    // input and output arrays
    std::vector<typename param_::src1_port> src1_hw_;
    std::vector<typename param_::src2_port> src2_hw_;
    typename eval_::dst_port* dst_hw_;
    std::vector<float> src1_sw_;
    std::vector<float> src2_sw_;
    std::vector<float> dst_sw_;
    std::vector<float> dst_hw_flt_;

    /*!
     * @brief Calls the software implementation of the selected elementwise function
     */
    auto SwElementwise(const float argument1, const float argument2) noexcept -> void {
        for (int64_t i = 0; i < param_::dst_dim::elms; ++i) {
            switch (param_::op_type) {
                case hvx::util::elmwise_e::Abs:
                    dst_sw_.at(i) = hvx::util::Abs(src1_sw_.at(i));
                    break;
                case hvx::util::elmwise_e::Add:
                    dst_sw_.at(i) = src1_sw_.at(i) + src2_sw_.at(i);
                    break;
                case hvx::util::elmwise_e::AddConst:
                    dst_sw_.at(i) = src1_sw_.at(i) + argument1;
                    break;
                case hvx::util::elmwise_e::Clip:
                    dst_sw_.at(i) = hvx::util::Clamp(src1_sw_.at(i), argument1, argument2);
                    break;
                case hvx::util::elmwise_e::Max:
                    dst_sw_.at(i) = hvx::util::Max(src1_sw_.at(i), src2_sw_.at(i));
                    break;
                case hvx::util::elmwise_e::MaxConst:
                    dst_sw_.at(i) = hvx::util::Max(src1_sw_.at(i), argument1);
                    break;
                case hvx::util::elmwise_e::Min:
                    dst_sw_.at(i) = hvx::util::Min(src1_sw_.at(i), src2_sw_.at(i));
                    break;
                case hvx::util::elmwise_e::MinConst:
                    dst_sw_.at(i) = hvx::util::Min(src1_sw_.at(i), argument1);
                    break;
                case hvx::util::elmwise_e::Mul:
                    dst_sw_.at(i) = src1_sw_.at(i) * src2_sw_.at(i);
                    break;
                case hvx::util::elmwise_e::MulConst:
                    dst_sw_.at(i) = src1_sw_.at(i) * argument1;
                    break;
                case hvx::util::elmwise_e::Sigmoid:
                    dst_sw_.at(i) = 1.0f / (1.0f + std::exp(-src1_sw_.at(i)));
                    break;
                case hvx::util::elmwise_e::Sub:
                    dst_sw_.at(i) = src1_sw_.at(i) - src2_sw_.at(i);
                    break;
                case hvx::util::elmwise_e::Tanh:
                    dst_sw_.at(i) = std::tanh(src1_sw_.at(i));
                    break;
                default:
                    dst_sw_.at(i) = 42;
                    break;
            }
        }
        (void)argument2;
    }

public:

    /*!
     * @brief Constructor (allocates memory and creates random inputs)
     */
    EwEvaluate(const float src_max): dst_hw_(new typename eval_::dst_port[param_::dst_dim::vec_elms]) {
        src1_hw_.resize(param_::src1_dim::vec_elms);
        src2_hw_.resize(param_::src2_dim::vec_elms);
        src1_sw_.resize(param_::src1_dim::elms);
        src2_sw_.resize(param_::src2_dim::elms);
        dst_sw_.resize(param_::dst_dim::elms);
        dst_hw_flt_.resize(param_::dst_dim::elms);
        hvx::sw::EvalCreateRndSrc<typename param_::src1_port, typename param_::src1_dim>(src1_hw_.data(), src1_sw_.data(), src_max);
        hvx::sw::EvalCreateRndSrc<typename param_::src2_port, typename param_::src2_dim>(src2_hw_.data(), src2_sw_.data(), src_max);
    };

    /*!
     * @brief Gets the pointer to the 1st input of HW function
     */
    constexpr auto GetSrc1Hw() noexcept -> typename param_::src1_port* {
        return src1_hw_.data();
    }

    /*!
     * @brief Gets the pointer to the 2nd input of HW function
     */
    constexpr auto GetSrc2Hw() noexcept -> typename param_::src2_port* {
        return src2_hw_.data();
    }

    /*!
     * @brief Gets the pointer to the output of HW function
     */
    constexpr auto GetDstHw() noexcept -> typename eval_::dst_port* {
        return dst_hw_;
    }

    /*!
     * @brief Calls the hardware implementation of the selected elementwise function
     */
    auto HwElementwise(const float sw_arg1, const float sw_arg2) noexcept -> void {
        switch (param_::op_type) {
            case hvx::util::elmwise_e::Abs:
                hvx::HwAbs<param_>(src1_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::Add:
                hvx::HwAdd<param_>(src1_hw_.data(), src2_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::AddConst:
                hvx::HwAddConst<param_>(src1_hw_.data(), sw_arg1, dst_hw_);
                break;
            case hvx::util::elmwise_e::Clip:
                hvx::HwClip<param_>(src1_hw_.data(), sw_arg1, sw_arg2, dst_hw_);
                break;
            case hvx::util::elmwise_e::Max:
                hvx::HwMax<param_>(src1_hw_.data(), src2_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::MaxConst:
                hvx::HwMaxConst<param_>(src1_hw_.data(), sw_arg1, dst_hw_);
                break;
            case hvx::util::elmwise_e::Min:
                hvx::HwMin<param_>(src1_hw_.data(), src2_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::MinConst:
                hvx::HwMinConst<param_>(src1_hw_.data(), sw_arg1, dst_hw_);
                break;
            case hvx::util::elmwise_e::Mul:
                hvx::HwMul<param_>(src1_hw_.data(), src2_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::MulConst:
                hvx::HwMulConst<param_>(src1_hw_.data(), sw_arg1, dst_hw_);
                break;
            case hvx::util::elmwise_e::Sigmoid:
                hvx::HwSigmoid<param_>(src1_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::Sub:
                hvx::HwSub<param_>(src1_hw_.data(), src2_hw_.data(), dst_hw_);
                break;
            case hvx::util::elmwise_e::Tanh:
                hvx::HwTanh<param_>(src1_hw_.data(), dst_hw_);
                break;
            default:
                break;
        }
    }

    /*!
     * @brief Runs the software function, converts HW results to float and comapres both results with each other
     */
    auto Evaluation(const float sw_arg1, const float sw_arg2) noexcept -> std::string {
        SwElementwise(sw_arg1, sw_arg2);
        hvx::sw::ConvertDstHwToFloat<typename param_::dst_type, typename param_::dst_dim, eval_::dst_flags>(*dst_hw_, dst_hw_flt_.data());
        return hvx::sw::EvalPrintDiff<typename param_::dst_dim, eval_>(dst_sw_.data(), dst_hw_flt_.data());
    }

    auto Evaluation(const float sw_arg1) noexcept -> std::string {
        SwElementwise(sw_arg1, sw_arg1);
        hvx::sw::ConvertDstHwToFloat<typename param_::dst_type, typename param_::dst_dim, eval_::dst_flags>(*dst_hw_, dst_hw_flt_.data());
        return hvx::sw::EvalPrintDiff<typename param_::dst_dim, eval_>(dst_sw_.data(), dst_hw_flt_.data());
    }

    /*!
     * @brief Runs the software function, converts HW results to float and comapres both results with each other
     */
    auto Evaluation() noexcept -> std::string {
        SwElementwise(0.0f, 0.0f);
        hvx::sw::ConvertDstHwToFloat<typename param_::dst_type, typename param_::dst_dim, eval_::dst_flags>(*dst_hw_, dst_hw_flt_.data());
        return hvx::sw::EvalPrintDiff<typename param_::dst_dim, eval_>(dst_sw_.data(), dst_hw_flt_.data());
    }
};

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_EW_H
