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
 * @file    hvx_sw_test_nn_func.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_NN_H
#define HVX_SW_TEST_NN_H

#include "hvx_sw_test_nn_func.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief the core class to evaluate the neural network functions
 */
template<typename eval_,
         typename src_type_,
         typename src_dim_,
         typename src_port_,
         typename dst_type_,
         typename dst_dim_,
         typename dst_port_,
         typename wgts_type_ = hvx::util::dfixed<float, 24>,
         typename wgts_dim_  = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename wgts_port_ = hvx::util::vector<hvx::util::dfixed<float, 24>, 1>,
         typename bias_type_ = hvx::util::dfixed<float, 24>,
         typename bias_dim_  = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename bias_port_ = hvx::util::vector<hvx::util::dfixed<float, 24>, bias_dim_::vec_size>>
class EvaluateCore {
protected:

    // hw containers
    std::vector<src_port_> src_hw_;    // NOLINT
    typename eval_::dst_port* dst_hw_; // NOLINT std::vector does not work with hls stream types
    std::vector<wgts_port_> wgts_hw_;  // NOLINT
    std::vector<bias_port_> bias_hw_;  // NOLINT
    // sw containers
    std::vector<float> src_sw_;  // NOLINT
    std::vector<float> dst_sw_;  // NOLINT
    std::vector<float> wgts_sw_; // NOLINT
    std::vector<float> bias_sw_; // NOLINT
    // testbench containers
    std::vector<float> dst_hw_flt_; // NOLINT

public:

    /*!
     * @brief constructor
     */
    constexpr EvaluateCore(): dst_hw_(new typename eval_::dst_port[dst_dim_::vec_elms]) {
        src_hw_.resize(src_dim_::vec_elms);
        wgts_hw_.resize(wgts_dim_::vec_elms);
        bias_hw_.resize(bias_dim_::vec_elms);
        src_sw_.resize(src_dim_::elms);
        dst_sw_.resize(dst_dim_::elms);
        wgts_sw_.resize(wgts_dim_::elms);
        bias_sw_.resize(bias_dim_::elms);
        dst_hw_flt_.resize(dst_dim_::elms);
    };

    constexpr auto GetSrcHw() noexcept -> src_port_* {
        return src_hw_.data();
    }

    constexpr auto GetDstHw() noexcept -> typename eval_::dst_port* {
        return dst_hw_;
    }

    constexpr auto GetWgtsHw() noexcept -> wgts_port_* {
        return wgts_hw_.data();
    }

    constexpr auto GetBiasHw() noexcept -> bias_port_* {
        return bias_hw_.data();
    }

    /*!
     * @brief create random biases between (upper,-1) for signed or (upper,0) for unsigned
     */
    auto RandomBiases(const float upper) noexcept -> void {
        // lower boundary for values
        const float lower = (bias_type_::is_signed == true) ? (-upper) : (0.0f);

        //
        for (int64_t v = 0; v < bias_dim_::vec_elms; ++v) {
            for (int64_t p = 0; p < bias_dim_::vec_size; ++p) {
                // creating and storing random input for sw
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> distribution(lower, upper);
                const auto sw_bias                       = distribution(rng);
                bias_sw_.at(v * bias_dim_::vec_size + p) = sw_bias;
                // convert and storing to hw type
                auto hw_bias = static_cast<bias_type_>(sw_bias);
                bias_hw_.at(v).Set(hw_bias, p);
            }
        }
    }

    /*!
     * @brief Evaluate Time & Error Rate
     */
    auto Compute() noexcept -> std::string {
        hvx::sw::ConvertDstHwToFloat<dst_type_, dst_dim_, eval_::dst_flags>(*dst_hw_, dst_hw_flt_.data());
        return hvx::sw::EvalPrintDiff<dst_dim_, eval_>(dst_sw_.data(), dst_hw_flt_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Wrapper classe to evaluate the depthwise function
 */
template<typename param_, typename eval_>
class DepthwiseEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::dst_dim,
                        typename param_::dst_port,
                        typename param_::wgts_type,
                        typename param_::wgts_dim,
                        typename param_::wgts_port,
                        typename param_::bias_type,
                        typename param_::bias_dim,
                        typename param_::bias_port> {
private:

    /*!
     * @brief create random weights between (upper,-1) for signed or (upper,0) for unsigned
     */
    auto RandomWeights(const float wgts_max) noexcept -> void {
        // upper/lower boundary for values
        const float upper = wgts_max / static_cast<float>(param_::knl_elms);
        const float lower = (param_::wgts_type::is_signed == true) ? (-upper) : (0.0f);

        // channels
        for (int64_t chnlv = 0; chnlv < param_::chnl_vec_elms; ++chnlv) {
            for (int64_t chnlp = 0; chnlp < param_::chnl_vec_size; ++chnlp) {
                const int64_t chnl = (chnlv * param_::chnl_vec_size) + chnlp;

                // kernel
                for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                    for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                        const int64_t knl = (knl_row * param_::knl_cols) + knl_col;

                        // software
                        std::mt19937 rng(std::random_device{}());
                        std::uniform_real_distribution<float> distribution(lower, upper);
                        const auto sw_wgt                                  = distribution(rng);
                        this->wgts_sw_.at((chnl * param_::knl_elms) + knl) = sw_wgt;

                        // hardware
                        auto hw_wgt = static_cast<typename param_::wgts_type>(sw_wgt);
                        this->wgts_hw_.at(chnlv).Set(hw_wgt, (chnlp * param_::knl_elms) + knl);
                    }
                }
            }
        }
    }

    /*!
     * @brief SW function (with Bias)
     */
    static constexpr auto SwDepthwiseWithBias(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
        hvx::sw::SwDepthwise<param_, true>(src, wgts, bias, dst);
    }

    /*!
     * @brief SW function (without Bias)
     */
    static constexpr auto SwDepthwiseWithoutBias(float* src, float* wgts, float* dst) noexcept -> void {
        hvx::sw::SwDepthwise<param_, false>(src, wgts, nullptr, dst);
    }

public:

    /*!
     * @brief constructor (without bias)
     */
    constexpr DepthwiseEvaluate(float conv_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDepthwiseWithoutBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias)
     */
    constexpr DepthwiseEvaluate(float conv_max, float bias_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        this->RandomBiases(bias_max);
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDepthwiseWithBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }

    /*!
     * @brief constructor (without bias, fixed src, wgts)
     */
    constexpr DepthwiseEvaluate(typename param_::src_vec* src, typename param_::wgts_vec* wgts) {
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::src_type, typename param_::src_dim>(src, this->src_sw_.data());     // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::wgts_type, typename param_::wgts_dim>(wgts, this->wgts_sw_.data()); // NOLINT
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDepthwiseWithoutBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias, fixed src, wgts, bias)
     */
    constexpr DepthwiseEvaluate(typename param_::src_vec* src, typename param_::wgts_vec* wgts, typename param_::bias_vec* bias) {
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::src_type, typename param_::src_dim>(src, this->src_sw_.data());     // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::wgts_type, typename param_::wgts_dim>(wgts, this->wgts_sw_.data()); // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::bias_type, typename param_::bias_dim>(bias, this->bias_sw_.data()); // NOLINT
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDepthwiseWithBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Wrapper classe to evaluate the convolution function
 */
template<typename param_, typename eval_>
class ConvEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::dst_dim,
                        typename param_::dst_port,
                        typename param_::wgts_type,
                        typename param_::wgts_dim,
                        typename param_::wgts_port,
                        typename param_::bias_type,
                        typename param_::bias_dim,
                        typename param_::bias_port> {
private:

    /*!
     * @brief create random weights between (upper,-1) for signed or (upper,0) for unsigned
     */
    auto RandomWeights(const float wgts_max) noexcept -> void {
        // upper/lower boundary for values
        const float upper = wgts_max / static_cast<float>(param_::knl_elms * param_::chnls);
        const float lower = (param_::wgts_type::is_signed == true) ? (-upper) : (0.0f);

        //
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(lower, upper);

        // feature maps
        for (int64_t fmv = 0; fmv < param_::fm_vec_elms; ++fmv) {
            for (int64_t fmp = 0; fmp < param_::fm_vec_size; ++fmp) {
                const int64_t fm = (fmv * param_::fm_vec_size) + fmp;

                // channels
                for (int64_t chnlv = 0; chnlv < param_::chnl_vec_elms; ++chnlv) {
                    for (int64_t chnlp = 0; chnlp < param_::chnl_vec_size; ++chnlp) {
                        const int64_t chnl = (chnlv * param_::chnl_vec_size) + chnlp;

                        // kernel
                        for (int64_t knl_row = 0; knl_row < param_::knl_rows; ++knl_row) {
                            for (int64_t knl_col = 0; knl_col < param_::knl_cols; ++knl_col) {
                                const int64_t knl = (knl_row * param_::knl_cols) + knl_col;

                                // software
                                const auto sw_wgt = distribution(rng);
                                this->wgts_sw_.at((fm * param_::chnls * param_::knl_elms) + (chnl * param_::knl_elms) + knl) = sw_wgt;

                                // hardware
                                auto hw_wgt = static_cast<typename param_::wgts_type>(sw_wgt);
                                this->wgts_hw_.at((fmv * param_::chnl_vec_elms) + chnlv)
                                    .Set(hw_wgt, (fmp * param_::chnl_vec_size * param_::knl_elms) + (chnlp * param_::knl_elms) + knl);
                            }
                        }
                    }
                }
            }
        }
    }

    /*!
     * @brief SW function (with Bias)
     */
    static constexpr auto SwConvWithBias(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
        hvx::sw::SwConv<param_, true>(src, wgts, bias, dst);
    }

    /*!
     * @brief SW function (without Bias)
     */
    static constexpr auto SwConvWithoutBias(float* src, float* wgts, float* dst) noexcept -> void {
        hvx::sw::SwConv<param_, false>(src, wgts, nullptr, dst);
    }

public:

    /*!
     * @brief constructor (without bias)
     */
    constexpr ConvEvaluate(float conv_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwConvWithoutBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias)
     */
    constexpr ConvEvaluate(float conv_max, float bias_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        this->RandomBiases(bias_max);
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwConvWithBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }

    /*!
     * @brief constructor (without bias, fixed src, wgts)
     */
    constexpr ConvEvaluate(typename param_::src_vec* src, typename param_::wgts_vec* wgts) {
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::src_type, typename param_::src_dim>(src, this->src_sw_.data());     // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::wgts_type, typename param_::wgts_dim>(wgts, this->wgts_sw_.data()); // NOLINT
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwConvWithoutBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias, fixed src, wgts, bias)
     */
    constexpr ConvEvaluate(typename param_::src_vec* src, typename param_::wgts_vec* wgts, typename param_::bias_vec* bias) {
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::src_type, typename param_::src_dim>(src, this->src_sw_.data());     // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::wgts_type, typename param_::wgts_dim>(wgts, this->wgts_sw_.data()); // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::bias_type, typename param_::bias_dim>(bias, this->bias_sw_.data()); // NOLINT
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwConvWithBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Wrapper classe to evaluate the dense function
 */
template<typename param_, typename eval_>
class DenseEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::dst_dim,
                        typename param_::dst_port,
                        typename param_::wgts_type,
                        typename param_::wgts_dim,
                        typename param_::wgts_port,
                        typename param_::bias_type,
                        typename param_::bias_dim,
                        typename param_::bias_port> {
private:

    /*!
     * @brief create random weights between (upper,-1) for signed or (upper,0) for unsigned
     */
    auto RandomWeights(const float wgts_max) noexcept -> void {
        // upper/lower boundary for values
        float upper = wgts_max / static_cast<float>(param_::chnls);
        float lower = (param_::wgts_type::is_signed == true) ? (-upper) : (0.0f);

        // feature maps
        for (int64_t fmv = 0; fmv < param_::fm_vec_elms; ++fmv) {
            for (int64_t fmp = 0; fmp < param_::fm_vec_size; ++fmp) {
                const int64_t fm = (fmv * param_::fm_vec_size) + fmp;

                // channels
                for (int64_t chnlv = 0; chnlv < param_::chnl_vec_elms; ++chnlv) {
                    for (int64_t chnlp = 0; chnlp < param_::chnl_vec_size; ++chnlp) {
                        const int64_t chnl = (chnlv * param_::chnl_vec_size) + chnlp;

                        // software
                        std::mt19937 rng(std::random_device{}());
                        std::uniform_real_distribution<float> distribution(lower, upper);
                        const auto sw_wgt                            = distribution(rng);
                        this->wgts_sw_.at(fm * param_::chnls + chnl) = sw_wgt;

                        // hardware
                        auto hw_wgt = static_cast<typename param_::wgts_type>(sw_wgt);
                        this->wgts_hw_.at((fmv * param_::chnl_vec_elms) + chnlv).Set(hw_wgt, (fmp * param_::chnl_vec_size) + chnlp);
                    }
                }
            }
        }
    }

    /*!
     * @brief SW function (with Bias)
     */
    static constexpr auto SwDenseWithBias(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
        hvx::sw::SwDense<param_, true>(src, wgts, bias, dst);
    }

    /*!
     * @brief SW function (without Bias)
     */
    static constexpr auto SwDenseWithoutBias(float* src, float* wgts, float* dst) noexcept -> void {
        hvx::sw::SwDense<param_, false>(src, wgts, nullptr, dst);
    }

public:

    /*!
     * @brief constructor (without bias)
     */
    constexpr DenseEvaluate(float conv_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDenseWithoutBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias)
     */
    constexpr DenseEvaluate(float conv_max, float bias_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        this->RandomBiases(bias_max);
        RandomWeights(conv_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwDenseWithBias, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Wrapper classe to evaluate the pool function
 */
template<typename param_, typename eval_, hvx::util::pooling_e pool_type_>
class PoolEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::dst_dim,
                        typename param_::dst_port> {
private:

    /*!
     * @brief SW function
     */
    static constexpr auto SwPool(float* src, float* dst) noexcept -> void {
        hvx::sw::SwPool<param_, pool_type_>(src, dst);
    }

public:

    /*!
     * @brief constructor
     */
    constexpr PoolEvaluate() {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwPool, this->src_sw_.data(), this->dst_sw_.data());
    }

    /*!
     * @brief constructor (fixed src)
     */
    constexpr PoolEvaluate(typename param_::src_vec* src) {
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::src_type, typename param_::src_dim>(src, this->src_sw_.data());
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwPool, this->src_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Wrapper classe to evaluate the softmax function
 */
template<typename param_, typename eval_>
class SoftmaxEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::src_dim,
                        typename param_::dst_port> {
private:

    /*!
     * @brief SW function
     */
    static constexpr auto SwSoftmax(float* src, float* dst) noexcept -> void {
        hvx::sw::SwSoftmax<param_>(src, dst);
    }

public:

    /*!
     * @brief constructor
     */
    constexpr SoftmaxEvaluate() {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data());
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwSoftmax, this->src_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief wrapper classe to evaluate the layer normalization function
 */
template<typename param_, typename eval_>
class LayernormEvaluate:
    public EvaluateCore<eval_,
                        typename param_::src_type,
                        typename param_::src_dim,
                        typename param_::src_port,
                        typename param_::dst_type,
                        typename param_::dst_dim,
                        typename param_::dst_port,
                        typename param_::wgts_type,
                        typename param_::wgts_dim,
                        typename param_::wgts_port,
                        typename param_::bias_type,
                        typename param_::bias_dim,
                        typename param_::bias_port> {
private:

    /*!
     * @brief create random weights between (upper,-1) for signed or (upper,0) for unsigned
     */
    auto RandomWeights(const float upper) noexcept -> void {
        // lower boundary for values
        const float lower = (param_::wgts_type::is_signed == true) ? (-upper) : (0.0f);

        //
        for (int64_t v = 0; v < param_::wgts_dim::vec_elms; ++v) {
            for (int64_t p = 0; p < param_::wgts_dim::vec_size; ++p) {
                // creating and storing random input for sw
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> distribution(lower, upper);
                const auto sw_wgt                                     = distribution(rng);
                this->wgts_sw_.at(v * param_::wgts_dim::vec_size + p) = sw_wgt;
                // convert and storing to hw type
                auto hw_wgt = static_cast<typename param_::wgts_type>(sw_wgt);
                this->wgts_hw_.at(v).Set(hw_wgt, p);
            }
        }
    }

    /*!
     * @brief SW function
     */
    static constexpr auto SwLayernorm(float* src, float* wgts, float* bias, float* dst) noexcept -> void {
        hvx::sw::SwLayernorm<param_>(src, wgts, bias, dst);
    }

public:

    /*!
     * @brief constructor
     */
    constexpr LayernormEvaluate(float src_max, float wgts_max, float bias_max) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data(), src_max);
        this->RandomBiases(bias_max);
        RandomWeights(wgts_max);
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwLayernorm, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }

    /*!
     * @brief constructor (with bias, fixed src, wgts, bias)
     */
    constexpr LayernormEvaluate(float src_max, typename param_::wgts_vec* wgts, typename param_::bias_vec* bias) {
        hvx::sw::EvalCreateRndSrc<typename param_::src_port, typename param_::src_dim>(this->src_hw_.data(), this->src_sw_.data(), src_max);
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::wgts_type, typename param_::wgts_dim>(wgts, this->wgts_sw_.data()); // NOLINT
        hvx::sw::ConvertDfixedVectorToFloat32<typename param_::bias_type, typename param_::bias_dim>(bias, this->bias_sw_.data()); // NOLINT
        hvx::sw::MeasureFuncTime(eval_::dbg, "SW", eval_::rept, SwLayernorm, this->src_sw_.data(), this->wgts_sw_.data(),
                                 this->bias_sw_.data(), this->dst_sw_.data());
    }
};

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_NN_H
