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
 * @file    hvx_sw_test_core.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_CORE_H_
#define HVX_SW_TEST_CORE_H_

#include "utils/hvx_sw_test_convert.h"
#include "utils/hvx_sw_test_ew.h"
#include "utils/hvx_sw_test_nn.h"
#include "utils/hvx_sw_test_reduce.h"

namespace hvx {
/******************************************************************************************************************************************/

/*!
 * @brief stores all additional parameters needed for evaluation
 */
template<bool debug_, int64_t worst_res_, int64_t first_and_last_res_, int64_t repetitions_, typename dst_port_, int64_t dst_flags_>
using eval_param = hvx::sw::EvaluateParam<debug_, worst_res_, first_and_last_res_, repetitions_, dst_port_, dst_flags_>;

/*!
 * @brief Evaluation class for all elementwise functions
 */
template<typename param_, typename eval_>
using elementwise_eval = hvx::sw::EwEvaluate<param_, eval_>;

/*!
 * @brief Evaluation class for all elementwise functions
 */
template<typename param_, typename eval_>
using reduce_eval = hvx::sw::ReduceEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the layer normalization function
 */
template<typename param_, typename eval_>
using layernorm_eval = hvx::sw::LayernormEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the softmax function
 */
template<typename param_, typename eval_>
using softmax_eval = hvx::sw::SoftmaxEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the convolution function
 */
template<typename param_, typename eval_>
using conv_eval = hvx::sw::ConvEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the depthwise function
 */
template<typename param_, typename eval_>
using depthwise_eval = hvx::sw::DepthwiseEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the dense function
 */
template<typename param_, typename eval_>
using dense_eval = hvx::sw::DenseEvaluate<param_, eval_>;

/*!
 * @brief Wrapper classe to evaluate the avg pool function
 */
template<typename param_, typename eval_>
using pool_avg_eval = hvx::sw::PoolEvaluate<param_, eval_, hvx::util::pooling_e::kAvg>;

/*!
 * @brief Wrapper classe to evaluate the max pool function
 */
template<typename param_, typename eval_>
using pool_max_eval = hvx::sw::PoolEvaluate<param_, eval_, hvx::util::pooling_e::kMax>;

/******************************************************************************************************************************************/

/*!
 * @brief Measures the execution time of a function, if it is not used in the HLS environment
 */
template<typename Function, typename... Parameters>
auto
SwMeasureFuncTime(const char* name, int64_t iterations, Function&& function, Parameters&&... parameters) {
    return hvx::sw::MeasureFuncTime(true, name, iterations, std::forward<Function>(function), std::forward<Parameters>(parameters)...);
}

/*!
 * @brief Creates an array with variables (starting with value 0 and then counting upwards)
 */
template<typename dim_, typename type_>
constexpr auto
SwCreateArrayOfVector() noexcept -> decltype(auto) {
    return hvx::sw::CreateArrayOfVector<dim_, type_>();
}

/*!
 * @brief Creates an array with variables (starting with value 0 and then counting upwards)
 */
template<typename dim_, typename type_>
constexpr auto
SwCreateArray() noexcept -> decltype(auto) {
    return hvx::sw::CreateArray<dim_, type_>();
}

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
SwCompareArrayOfVector(hvx::util::array1d<hvx::util::vector<src_type_, src_dim_::vec_size>, src_dim_::vec_elms>& src,
                       hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                       const char* name) noexcept -> void {
    std::cout << hvx::sw::CompareArrayOfVector<src_dim_, src_type_, dst_dim_, dst_type_>(src, dst, name);
}

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
SwCompareArrayOfVector(hvx::util::vector<src_type_, src_dim_::vec_size>* src,
                       hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                       const char* name) noexcept -> void {
    std::cout << hvx::sw::CompareArrayOfVector<src_dim_, src_type_, dst_dim_, dst_type_>(src, dst, name);
}

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
SwCompareArrayOfVector(hvx::util::array1d<src_type_, src_dim_::elms>& src,
                       hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                       const char* name) noexcept -> void {
    std::cout << hvx::sw::CompareArrayOfVector<src_dim_, src_type_, dst_dim_, dst_type_>(src, dst, name);
}

/******************************************************************************************************************************************/
} // namespace hvx

#endif // HVX_SW_TEST_CORE_H_
