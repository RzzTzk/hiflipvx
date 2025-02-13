/**
 * Licence: GNU GPLv3 \n
 * You may copy, distribute and modify the software as long as you track
 * changes/dates in source files. Any modifications to or software
 * including (via compiler) GPL-licensed code must also be made available
 * under the GPL along with build & install instructions.
 *
 * @file    hvx_sw_test_nn.cpp
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

constexpr int64_t worst_res   = 4; // printing the worst results
constexpr int64_t outer_res   = 4; // printing the first and last results
constexpr int64_t repetitions = 4;
constexpr auto overflow        = hvx::util::overflow_e::kSaturate;
constexpr auto underflow       = hvx::util::underflow_e::kTrunc;
constexpr auto exec            = hvx::util::execution_e::kExact;
constexpr bool buffer_wgts = false, buffer_bias = false, debug = false;
using batch_v = hvx::util::VectorParam<2, 1>;

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         bool with_bias_,
         int64_t src_rows_,
         int64_t src_cols_,
         int64_t chnls_,
         int64_t fms_,
         int64_t chnl_vec_size_,
         int64_t fm_vec_size_,
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t pad_rows_,
         int64_t pad_cols_,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_>
auto
TestConv(const char* name) noexcept -> std::string {
    // configuration
    using conv = hvx::nn::ConvParam<src_type_, dst_type_, wgts_type_, bias_type_, batch_v, hvx::util::VectorParam<src_rows_, 1>,
                                    hvx::util::VectorParam<src_cols_, 1>, hvx::util::VectorParam<fms_, fm_vec_size_>,
                                    hvx::util::VectorParam<chnls_, chnl_vec_size_>, hvx::util::VectorParam<knl_rows_, knl_rows_>,
                                    hvx::util::VectorParam<knl_cols_, knl_cols_>, hvx::util::Array2dParam<pad_rows_, pad_cols_>,
                                    hvx::util::Array2dParam<dil_rows_, dil_cols_>, hvx::util::Array2dParam<str_rows_, str_cols_>,
                                    buffer_wgts, buffer_bias, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    if (with_bias_ == true) {
        hvx::sw::ConvEvaluate<conv, hvx::sw::EvaluateParam<false, 4, 4, 4, typename conv::dst_port, 0>> eval(0.75f, 0.25f);
        hvx::HwConv<conv>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    } else {
        hvx::sw::ConvEvaluate<conv, hvx::sw::EvaluateParam<false, 4, 4, 4, typename conv::dst_port, 0>> eval(0.75f);
        hvx::HwConv<conv>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    }
}

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename dst_type_>
auto
TestConvMultiple() noexcept -> std::string {
    return "  Convolution: src[(16,1),(32,1),(8,2)] dst[(?,1),(?,1),(16,2)] ker(3,3) pad(1,1) dil(0,0) str(1,1):\n" +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1>("\t(default) ") +
           // test without bias
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1>("\t(no bias) ") +
           // test vector
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 1, 2, 3, 3, 1, 1, 0, 0, 1, 1>("\t(vec=1|2) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 1, 3, 3, 1, 1, 0, 0, 1, 1>("\t(vec=2|1) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 8, 8, 3, 3, 1, 1, 0, 0, 1, 1>("\t(vec=8|8) ") +
           // test kernel
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 1, 3, 0, 1, 0, 0, 1, 1>("\t(ker=1|3) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 1, 1, 0, 0, 0, 1, 1>("\t(ker=3|1) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 5, 5, 2, 2, 0, 0, 1, 1>("\t(ker=5|5) ") +
           // test padding
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1>("\t(pad=0|0) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 2, 0, 0, 1, 1>("\t(pad=1|2) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 2, 1, 0, 0, 1, 1>("\t(pad=2|1) ") +
           // test dilation
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 0, 1, 1>("\t(dil=1|0) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 1, 1, 1>("\t(dil=0|1) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1>("\t(dil=1|1) ") +
           // test stride
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 0, 2, 2>("\t(str=2|2) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 0, 1, 2>("\t(str=1|2) ") +
           TestConv<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 0, 0, 2, 1>("\t(str=2|1) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         bool with_bias_,
         int64_t src_rows_,
         int64_t src_cols_,
         int64_t chnls_,
         int64_t chnls_vec_size_,
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t pad_rows_,
         int64_t pad_cols_,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_>
auto
TestDepth(const char* name) noexcept -> std::string {
    // configuration
    using depth =
        hvx::nn::DepthwiseParam<src_type_, dst_type_, wgts_type_, bias_type_, batch_v, hvx::util::VectorParam<src_rows_, 1>,
                                hvx::util::VectorParam<src_cols_, 1>, hvx::util::VectorParam<chnls_, chnls_vec_size_>,
                                hvx::util::VectorParam<knl_rows_, knl_rows_>, hvx::util::VectorParam<knl_cols_, knl_cols_>,
                                hvx::util::Array2dParam<pad_rows_, pad_cols_>, hvx::util::Array2dParam<dil_rows_, dil_cols_>,
                                hvx::util::Array2dParam<str_rows_, str_cols_>, buffer_wgts, buffer_bias, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    if (with_bias_ == true) {
        hvx::sw::DepthwiseEvaluate<depth, hvx::sw::EvaluateParam<false, 4, 4, 4, typename depth::dst_port, 0>> eval(0.75f, 0.25f);
        hvx::HwDepthwise<depth>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    } else {
        hvx::sw::DepthwiseEvaluate<depth, hvx::sw::EvaluateParam<false, 4, 4, 4, typename depth::dst_port, 0>> eval(0.75f);
        hvx::HwDepthwise<depth>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    }
}

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename dst_type_>
auto
TestDepthMultiple() noexcept -> std::string {
    return "  Depthwise: src[(16,1),(32,1),(8,2)] ker(3,3) pad(1,1) dil(0,0) str(1,1):\n" +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 0, 0, 1, 1>("\t(default) ") +
           // test without bias
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 2, 3, 3, 1, 1, 0, 0, 1, 1>("\t(no bias) ") +
           // test vector
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 1, 3, 3, 1, 1, 0, 0, 1, 1>("\t(vec=1)   ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 8, 3, 3, 1, 1, 0, 0, 1, 1>("\t(vec=8)   ") +
           // test kernel
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 1, 3, 0, 1, 0, 0, 1, 1>("\t(ker=1|3) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 1, 1, 0, 0, 0, 1, 1>("\t(ker=3|1) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 5, 5, 2, 2, 0, 0, 1, 1>("\t(ker=5|5) ") +
           // test padding
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 0, 0, 0, 0, 1, 1>("\t(pad=0|0) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 2, 0, 0, 1, 1>("\t(pad=1|2) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 2, 1, 0, 0, 1, 1>("\t(pad=2|1) ") +
           // test dilation
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 1, 0, 1, 1>("\t(dil=1|0) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 0, 1, 1, 1>("\t(dil=0|1) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 1, 1, 1, 1>("\t(dil=1|1) ") +
           // test stride
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 0, 0, 2, 2>("\t(str=2|2) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 0, 0, 1, 2>("\t(str=1|2) ") +
           TestDepth<src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 2, 3, 3, 1, 1, 0, 0, 2, 1>("\t(str=2|1) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<hvx::util::pooling_e pool_type_,
         typename src_type_,
         typename dst_type_,
         int64_t src_rows_,
         int64_t src_cols_,
         int64_t chnls_,
         int64_t chnls_vec_size_,
         int64_t knl_rows_,
         int64_t knl_cols_,
         int64_t pad_rows_,
         int64_t pad_cols_,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_>
auto
TestPool(const char* name) noexcept -> std::string {
    // configuration
    using pool = hvx::nn::PoolParam<src_type_, dst_type_, batch_v, hvx::util::VectorParam<src_rows_, 1>,
                                    hvx::util::VectorParam<src_cols_, 1>, hvx::util::VectorParam<chnls_, chnls_vec_size_>,
                                    hvx::util::VectorParam<knl_rows_, knl_rows_>, hvx::util::VectorParam<knl_cols_, knl_cols_>,
                                    hvx::util::Array2dParam<pad_rows_, pad_cols_>, hvx::util::Array2dParam<dil_rows_, dil_cols_>,
                                    hvx::util::Array2dParam<str_rows_, str_cols_>, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    hvx::sw::PoolEvaluate<pool, hvx::sw::EvaluateParam<false, 4, 4, 4, typename pool::dst_port, 0>, pool_type_> eval;
    if (pool_type_ == hvx::util::pooling_e::kAvg)
        hvx::HwPoolAvg<pool>(eval.GetSrcHw(), eval.GetDstHw());
    else
        hvx::HwPoolMax<pool>(eval.GetSrcHw(), eval.GetDstHw());
    return name + eval.Compute() + "\n";
}

/*!
 * @brief
 */
template<typename src_type_, typename dst_type_>
auto
TestPoolMultiple() noexcept -> std::string {
    constexpr auto avg_pool = hvx::util::pooling_e::kAvg;
    constexpr auto max_pool = hvx::util::pooling_e::kMax;

    return "  Pooling (AvgPool): src[(16,1),(32,1),(8,2)] ker(2,2) pad(0,0) dil(0,0) str(2,2):\n" +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 0, 2, 2>("\t(default) ") +
           TestPool<max_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 0, 2, 2>("\t(MaxPool) ") +
           // test vector
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 1, 2, 2, 0, 0, 0, 0, 2, 2>("\t(vec=1)   ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 8, 2, 2, 0, 0, 0, 0, 2, 2>("\t(vec=8)   ") +
           // test kernel
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 1, 2, 0, 0, 0, 0, 2, 2>("\t(ker=1|2) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 1, 0, 0, 0, 0, 2, 2>("\t(ker=2|1) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 3, 3, 0, 0, 0, 0, 2, 2>("\t(ker=3|3) ") +
           // test padding
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 1, 0, 0, 2, 2>("\t(pad=0|1) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 1, 0, 0, 0, 2, 2>("\t(pad=1|0) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 1, 1, 0, 0, 2, 2>("\t(pad=1|1) ") +
           // test dilation
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 1, 0, 2, 2>("\t(dil=1|0) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 1, 2, 2>("\t(dil=0|1) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 1, 1, 2, 2>("\t(dil=1|1) ") +
           // test stride
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 0, 1, 2>("\t(str=1|2) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 0, 2, 1>("\t(str=2|1) ") +
           TestPool<avg_pool, src_type_, dst_type_, 16, 32, 8, 2, 2, 2, 0, 0, 0, 0, 3, 3>("\t(str=3|3) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         int64_t chnls_,
         int64_t fms_,
         int64_t chnl_vec_size_,
         int64_t fm_vec_size_,
         bool with_bias_>
auto
TestDense(const char* name) noexcept -> std::string {
    // configuration
    using dense = hvx::nn::DenseParam<src_type_, dst_type_, wgts_type_, bias_type_, batch_v, hvx::util::VectorParam<chnls_, chnl_vec_size_>,
                                      hvx::util::VectorParam<fms_, fm_vec_size_>, buffer_wgts, buffer_bias, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    if (with_bias_ == true) {
        hvx::sw::DenseEvaluate<dense, hvx::sw::EvaluateParam<false, 4, 4, 4, typename dense::dst_port, 0>> eval(0.75f, 0.25f);
        hvx::HwDense<dense>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    } else {
        hvx::sw::DenseEvaluate<dense, hvx::sw::EvaluateParam<false, 4, 4, 4, typename dense::dst_port, 0>> eval(0.75f);
        hvx::HwDense<dense>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetDstHw());
        return name + eval.Compute() + "\n";
    }
}

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename dst_type_>
auto
TestDenseMultiple() noexcept -> std::string {
    return "  Dense: src[(512,2)] dst[(512,2)]\n" + //
           TestDense<src_type_, wgts_type_, bias_type_, dst_type_, 512, 512, 2, 2, true>("\t(default)  ") +
           TestDense<src_type_, wgts_type_, bias_type_, dst_type_, 512, 512, 2, 2, false>("\t(no bias)  ") +
           // test vector
           TestDense<src_type_, wgts_type_, bias_type_, dst_type_, 512, 512, 1, 2, true>("\t(vec=1|2) ") +
           TestDense<src_type_, wgts_type_, bias_type_, dst_type_, 512, 512, 2, 1, true>("\t(vec=2|1) ") +
           TestDense<src_type_, wgts_type_, bias_type_, dst_type_, 512, 512, 8, 8, true>("\t(vec=8|8) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_, typename dst_type_, int64_t rows_, int64_t cols_, int64_t chnls_, int64_t chnls_vec_size_>
auto
TestSoft(const char* name) noexcept -> std::string {
    // configuration
    using softmax = hvx::nn::SoftmaxParam<src_type_, dst_type_, batch_v, hvx::util::VectorParam<rows_, 1>, hvx::util::VectorParam<cols_, 1>,
                                          hvx::util::VectorParam<chnls_, chnls_vec_size_>, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    hvx::sw::SoftmaxEvaluate<softmax, hvx::sw::EvaluateParam<false, 4, 4, 4, typename softmax::dst_port, 0>> eval;
    hvx::HwSoftmax<softmax>(eval.GetSrcHw(), eval.GetDstHw());
    return name + eval.Compute() + "\n";
}

/*!
 * @brief
 */
template<typename src_type_, typename dst_type_>
auto
TestSoftMultiple() noexcept -> std::string {
    return "  SoftMax: src[(1,1),(1,1),(1024,1)]:\n" + //
           TestSoft<src_type_, dst_type_, 1, 1, 1024, 1>("\t(default)      ") +
           TestSoft<src_type_, dst_type_, 1, 1, 512, 1>("\t(chnls=512)    ") +
           TestSoft<src_type_, dst_type_, 1, 1, 1024, 8>("\t(vec=8)        ") +
           TestSoft<src_type_, dst_type_, 8, 8, 1024, 1>("\t(src=8,cols=8) ");
}

/******************************************************************************************************************************************/

///*!
// * @brief
// */
//template<vx_nn_activation_function_e activation_type_,
//         int64_t rows_,
//         int64_t cols_,
//         int64_t chnls,
//         int64_t chnls_vec_size,
//         typename src_type_,
//         typename param_type_,
//         typename dst_type_>
//auto
//TestAct(const char* name) noexcept -> std::string {
//    // parameters and data types of activation layer
//    using rows_v                    = hvx::util::VectorParam<rows_, 1>;
//    using cols_v                    = hvx::util::VectorParam<cols_, 1>;
//    using chnls_v                   = hvx::util::VectorParam<chnls, chnls_vec_size>;
//    using param                     = hvx::util::ActivationParams<activation_type_, overflow, underflow>;
//    using dim                       = hvx::util::TensorParam<4, chnls_v, cols_v, rows_v, batch_v>;
//    constexpr float sw_param1  = 0.5f;
//    constexpr float sw_param2  = 0.5f;
//    constexpr param_type_ hw_param1 = hvx::util::CastFltToDfixed<param_type_, underflow>(sw_param1);
//    constexpr param_type_ hw_param2 = hvx::util::CastFltToDfixed<param_type_, underflow>(sw_param2);
//
//    //
//    hvx::sw::NnEvaluate<debug, worst_res, src_type_, dim, dst_type_, dim> data(enum_name(activation_type_));
//    data.template EvalCreateRndSrc<activation_type_ != vx_nn_activation_function_e::VX_NN_ACTIVATION_SQRT>(false, false);
//    hvx::sw::SwActivation<param, dim>(data.GetSwSrc(), data.GetSwOut(), sw_param1, sw_param2);
//    hvx::user::HwActivation<dim, param, src_type_, dst_type_, param_type_>(&data.GetHwSrc(), &data.GetHwOut(), hw_param1, hw_param2);
//    return name + data.Evaluate() + "\n";
//}

///*!
// * @brief
// */
//template<typename src_type_, typename param_type_, typename dst_type_>
//auto
//TestActMultiple() noexcept -> std::string {
//    constexpr auto f1 = vx_nn_activation_function_e::VX_NN_ACTIVATION_LOGISTIC;
//    constexpr auto f2 = vx_nn_activation_function_e::VX_NN_ACTIVATION_HYPERBOLIC_TAN;
//    constexpr auto f3 = vx_nn_activation_function_e::VX_NN_ACTIVATION_RELU;
//    constexpr auto f4 = vx_nn_activation_function_e::VX_NN_ACTIVATION_BRELU;
//    constexpr auto f5 = vx_nn_activation_function_e::VX_NN_ACTIVATION_SOFTRELU;
//    constexpr auto f6 = vx_nn_activation_function_e::VX_NN_ACTIVATION_ABS;
//    constexpr auto f7 = vx_nn_activation_function_e::VX_NN_ACTIVATION_SQUARE;
//    constexpr auto f8 = vx_nn_activation_function_e::VX_NN_ACTIVATION_SQRT;
//    constexpr auto f9 = vx_nn_activation_function_e::VX_NN_ACTIVATION_LINEAR;
//
//    return "  Activation: src[(16,1),(32,1),(64,8)]:\n" + //
//           TestAct<f1, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tLogistic: ") +
//           TestAct<f2, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tHTan:     ") +
//           TestAct<f3, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tRelu:     ") +
//           TestAct<f4, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tBrelu:    ") +
//           TestAct<f5, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tSoftRelu: ") +
//           TestAct<f6, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tAbs:      ") +
//           TestAct<f7, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tSquare:   ") +
//           TestAct<f8, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tSqrt:     ") +
//           TestAct<f9, 16, 32, 64, 4, src_type_, param_type_, dst_type_>("\tLinear:   ");
//}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_,
         typename wgts_type_,
         typename bias_type_,
         typename dst_type_,
         int64_t rows_,
         int64_t cols_,
         int64_t chnls_,
         int64_t chnls_vec_size_>
auto
TestLayernorm(const char* name) noexcept -> std::string {
    // configuration
    using layernorm = hvx::nn::LayernormParam<src_type_, dst_type_, wgts_type_, bias_type_, batch_v, hvx::util::VectorParam<rows_, 1>,
                                              hvx::util::VectorParam<cols_, 1>, hvx::util::VectorParam<chnls_, chnls_vec_size_>,
                                              buffer_wgts, buffer_bias, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    hvx::sw::LayernormEvaluate<layernorm, hvx::sw::EvaluateParam<false, 4, 4, 4, typename layernorm::dst_port, 0>> eval(1.0, 0.75, 0.25);
    hvx::HwLayernorm<layernorm>(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
    return name + eval.Compute() + "\n";
}

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename dst_type_>
auto
TestLayernormMultiple() {
    return "  Layer Norm: src[(16,1),(16,1),(32,1)\n" + //
           TestLayernorm<src_type_, wgts_type_, bias_type_, dst_type_, 16, 16, 32, 1>("\t(default)        ") +
           TestLayernorm<src_type_, wgts_type_, bias_type_, dst_type_, 16, 16, 32, 8>("\t(vec=8, chnl=32) ") +
           TestLayernorm<src_type_, wgts_type_, bias_type_, dst_type_, 16, 16, 96, 8>("\t(vec=8, chnl=96) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename param_type_, typename dst_type_>
auto
TestLayers(const char* name) noexcept -> void {
    std::string results;
    results.append(TestConvMultiple<src_type_, wgts_type_, bias_type_, dst_type_>());
    results.append(TestDepthMultiple<src_type_, wgts_type_, bias_type_, dst_type_>());
    results.append(TestPoolMultiple<src_type_, dst_type_>());
    results.append(TestDenseMultiple<src_type_, wgts_type_, bias_type_, dst_type_>());
    results.append(TestSoftMultiple<src_type_, dst_type_>());
    //  results.append(TestActMultiple<src_type_, param_type_, dst_type_>());
    results.append(TestLayernormMultiple<src_type_, wgts_type_, bias_type_, dst_type_>());

    std::cout << name << results;
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
auto
main() -> int {
    //
    using type1 = hvx::util::dfixed<int16_t, 15>;
    using type2 = hvx::util::dfixed<uint16_t, 16>;
    using type3 = hvx::util::dfixed<float, 28>;
    using type4 = hvx::util::dfixed<int16_t, 14>;

    //
    std::array<const char*, 5> names{"\nFixed-Point [signed 16-bit, 15-bit fraction]\n",
                                     "\nFixed-Point [unsigned 16-bit, 16-bit fraction]\n", "\nFloating-Point\n",
                                     "\nFixed-Point [signed 16-bit, 15-bit fraction (inputs), 14-bit fraction (outputs)]\n",
                                     "\nFixed-Point [signed 16-bit, 14-bit fraction (inputs), 15-bit fraction (outputs)]\n"};

    //
    std::vector<std::thread> threads;
    threads.reserve(5);

    // test neural network functions (one configuration per thread)
    threads.emplace_back(&TestLayers<type1, type1, type1, type1, type1>, names[0]);
    threads.emplace_back(&TestLayers<type2, type2, type2, type2, type2>, names[1]);
    threads.emplace_back(&TestLayers<type3, type3, type3, type3, type3>, names[2]);
    threads.emplace_back(&TestLayers<type1, type1, type1, type1, type4>, names[3]);
    threads.emplace_back(&TestLayers<type4, type4, type4, type4, type1>, names[4]);

    // wait until all threads have finished
    for (auto& thread: threads)
        thread.join();
}