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

#include "../../include/hiflipvx/nn/hvx_nn_super.h"
#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

constexpr int64_t worst_res   = 4; // printing the worst results
constexpr int64_t outer_res   = 4; // printing the first and last results
constexpr int64_t repetitions = 4;
constexpr auto overflow       = hvx::util::overflow_e::kSaturate;
constexpr auto underflow      = hvx::util::underflow_e::kTrunc;
constexpr auto exec           = hvx::util::execution_e::kExact;
constexpr bool buffer_wgts = false, buffer_bias = false, debug = false;
using batch_v = hvx::util::VectorParam<2, 1>;

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<hvx::util::layer_e laye_type_,
         hvx::util::pooling_e pool_type_,
         typename src_type_,
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
         int64_t pad_rows_up,
         int64_t pad_rows_down,
         int64_t pad_cols_left,
         int64_t pad_cols_right,
         int64_t dil_rows_,
         int64_t dil_cols_,
         int64_t str_rows_,
         int64_t str_cols_>
auto
TestSuper(const char* name) noexcept -> std::string {
    // configuration
    using super = hvx::nn::SuperParam<src_type_, dst_type_, wgts_type_, bias_type_, batch_v, hvx::util::VectorParam<src_rows_, 1>,
                                      hvx::util::VectorParam<src_cols_, 1>, hvx::util::VectorParam<fms_, fm_vec_size_>,
                                      hvx::util::VectorParam<chnls_, chnl_vec_size_>, hvx::util::VectorParam<knl_rows_, knl_rows_>,
                                      hvx::util::VectorParam<knl_cols_, knl_cols_>, hvx::util::Array2dParam<pad_rows_up, pad_rows_down>,
                                      hvx::util::Array2dParam<pad_cols_left, pad_cols_right>, hvx::util::Array2dParam<dil_rows_, dil_cols_>,
                                      hvx::util::Array2dParam<str_rows_, str_cols_>, buffer_wgts, buffer_bias, overflow, underflow, exec>;

    // create random data, compute SW, compute HW and evaluate
    if (laye_type_ == hvx::util::layer_e::Conv) {
        if (with_bias_ == true) {
            hvx::sw::ConvEvaluate<super, hvx::sw::EvaluateParam<false, 4, 4, 4, typename super::dst_port, 0>> eval(0.75f, 0.25f);
            hvx::nn::SuperTop<super, true, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Conv>(eval.GetSrcHw(), eval.GetWgtsHw(),
                                                                                                 eval.GetBiasHw(), eval.GetDstHw());
            return name + eval.Compute() + "\n";
        } else {
            hvx::sw::ConvEvaluate<super, hvx::sw::EvaluateParam<false, 4, 4, 4, typename super::dst_port, 0>> eval(0.75f);
            hvx::nn::SuperTop<super, false, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Conv>(eval.GetSrcHw(), eval.GetWgtsHw(),
                                                                                                  nullptr, eval.GetDstHw());
            return name + eval.Compute() + "\n";
        }
    } else if (laye_type_ == hvx::util::layer_e::Depthwise) {
        if (with_bias_ == true) {
            hvx::sw::DepthwiseEvaluate<super, hvx::sw::EvaluateParam<false, 4, 4, 4, typename super::dst_port, 0>> eval(0.75f, 0.25f);
            hvx::nn::SuperTop<super, true, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Depthwise>(eval.GetSrcHw(), eval.GetWgtsHw(),
                                                                                                      eval.GetBiasHw(), eval.GetDstHw());
            return name + eval.Compute() + "\n";
        } else {
            hvx::sw::DepthwiseEvaluate<super, hvx::sw::EvaluateParam<false, 4, 4, 4, typename super::dst_port, 0>> eval(0.75f);
            hvx::nn::SuperTop<super, false, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Depthwise>(eval.GetSrcHw(), eval.GetWgtsHw(),
                                                                                                       nullptr, eval.GetDstHw());
            return name + eval.Compute() + "\n";
        }
    } else if (laye_type_ == hvx::util::layer_e::Pool) {
        hvx::sw::PoolEvaluate<super, hvx::sw::EvaluateParam<false, 4, 4, 4, typename super::dst_port, 0>, pool_type_> eval;
        if (pool_type_ == hvx::util::pooling_e::kAvg)
            hvx::nn::SuperTop<super, false, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Pool>(eval.GetSrcHw(), nullptr, nullptr,
                                                                                                  eval.GetDstHw());
        else
            hvx::nn::SuperTop<super, false, hvx::util::pooling_e::kMax, hvx::util::layer_e::Pool>(eval.GetSrcHw(), nullptr, nullptr,
                                                                                                  eval.GetDstHw());
        return name + eval.Compute() + "\n";
    }
}

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename dst_type_>
auto
TestSuperMultiple() noexcept -> std::string {
    constexpr auto Conv     = hvx::util::layer_e::Conv;
    constexpr auto Depth    = hvx::util::layer_e::Depthwise;
    constexpr auto Pool     = hvx::util::layer_e::Pool;
    constexpr auto avg_pool = hvx::util::pooling_e::kAvg;
    constexpr auto max_pool = hvx::util::pooling_e::kMax;
    return "  Super:(conv) (pool_type) src[(16,1),(32,1),(8,2)] dst[(?,1),(?,1),(16,2)] ker(3,3) pad_rol(1,1) pad_col(1,1) dil(0,0) "
           "str(1,1):\n" +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1>(
               "\t(default) ") +
           // test without bias
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1,
                     1>("\t(no bias) ") +
           // test vector
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 1, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1>(
               "\t(vec=1|2) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 1, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1>(
               "\t(vec=2|1) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 8, 8, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1>(
               "\t(vec=8|8) ") +
           // test kernel
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 1, 3, 0, 0, 1, 1, 0, 0, 1, 1>(
               "\t(ker=1|3) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1>(
               "\t(ker=3|1) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 5, 5, 2, 2, 2, 2, 0, 0, 1, 1>(
               "\t(ker=5|5) ") +
           // test padding
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1>(
               "\t(pad=0|0) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 2, 2, 0, 0, 1, 1>(
               "\t(pad=1|2) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1>(
               "\t(pad=2|1) ") +
           // test dilation
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1>(
               "\t(dil=1|0) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1>(
               "\t(dil=0|1) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1>(
               "\t(dil=1|1) ") +
           // test stride
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 2, 2>(
               "\t(str=2|2) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1, 2>(
               "\t(str=1|2) ") +
           TestSuper<Conv, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 2, 1>(
               "\t(str=2|1) ") +

           "  Super:(depth) (pool_type) src[(16,1),(32,1),(8,2)] dst[(?,1),(?,1),(16,2)] ker(3,3) pad_rol(1,1) pad_col(1,1) dil(0,0) "
           "str(1,1):\n" +
           TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1,
                     1>("\t(default) ");
    //    // test without bias
    //       TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1,
    //    1>("\t(no bias) ") ;
    //    // test vector
    // //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 8, 1, 1, 3, 3, 1, 1, 1, 1, 0, 0, 1,
    // 1>("\t(vec=1)   ") +
    // //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 0, 0, 1,
    // 1>("\t(vec=8)   ") +
    //    // test kernel
    //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 1, 3, 0, 0, 1, 1, 0, 0, 1,
    //    1>("\t(ker=1|3) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 1, 1,
    //    1, 0, 0, 0, 0, 1, 1>("\t(ker=3|1) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8,
    //    16, 2, 2, 5, 5, 2, 2, 2, 2, 0, 0, 1, 1>("\t(ker=5|5) ") +
    //    // test padding
    //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 1,
    //    1>("\t(pad=0|0) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1,
    //    1, 2, 2, 0, 0, 1, 1>("\t(pad=1|2) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8,
    //    16, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1>("\t(pad=2|1) ") +
    //    // test dilation
    //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 1, 0, 1,
    //    1>("\t(dil=1|0) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1,
    //    1, 1, 1, 0, 1, 1, 1>("\t(dil=0|1) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8,
    //    16, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1>("\t(dil=1|1) ") +
    //    // test stride
    //    TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 2,
    //    2>("\t(str=2|2) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8, 16, 2, 2, 3, 3, 1,
    //    1, 1, 1, 0, 0, 1, 2>("\t(str=1|2) ") + TestSuper<Depth, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, true, 16, 32, 8,
    //    16, 2, 2, 3, 3, 1, 1, 1, 1, 0, 0, 2, 1>("\t(str=2|1) ") ;

    // "  Super:(pool) (pool_type) src[(16,1),(32,1),(8,2)] dst[(?,1),(?,1),(16,2)] ker(3,3) pad_rol(1,1) pad_col(1,1) dil(0,0) str(1,1):\n"
    // +
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 8, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2,
    //    2>("\t(default) ") ; TestSuper<Pool, max_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0,
    //    0, 0, 0, 0, 0, 2, 2>("\t(MaxPool) ") +
    //    // test vector
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2,
    //    2>("\t(vec=1)   ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 8, 8, 2, 2, 0,
    //    0, 0, 0, 0, 0, 2, 2>("\t(vec=8)   ") +
    //    // test kernel
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 2,
    //    2>("\t(ker=1|2) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 1, 0,
    //    0, 0, 0, 0, 0, 2, 2>("\t(ker=2|1) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8,
    //    16, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2>("\t(ker=3|3) ") +
    //    // test padding
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 2,
    //    2>("\t(pad=0|1) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 1,
    //    1, 0, 0, 0, 0, 2, 2>("\t(pad=1|0) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8,
    //    16, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 2, 2>("\t(pad=1|1) ") +
    //    // test dilation
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 2,
    //    2>("\t(dil=1|0) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0,
    //    0, 0, 0, 0, 1, 2, 2>("\t(dil=0|1) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8,
    //    16, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2>("\t(dil=1|1) ") +
    //    // test stride
    //    TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1,
    //    2>("\t(str=1|2) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8, 16, 2, 2, 2, 2, 0,
    //    0, 0, 0, 0, 0, 2, 1>("\t(str=2|1) ") + TestSuper<Pool, avg_pool, src_type_, wgts_type_, bias_type_, dst_type_, false, 16, 32, 8,
    //    16, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3>("\t(str=3|3) ");
}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename src_type_, typename wgts_type_, typename bias_type_, typename param_type_, typename dst_type_>
auto
TestLayers(const char* name) noexcept -> void {
    std::string results;

    results.append(TestSuperMultiple<src_type_, wgts_type_, bias_type_, dst_type_>());


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