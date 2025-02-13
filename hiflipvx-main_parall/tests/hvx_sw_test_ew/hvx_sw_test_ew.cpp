/**
 * Licence: GNU GPLv3 \n
 * You may copy, distribute and modify the software as long as you track
 * changes/dates in source files. Any modifications to or software
 * including (via compiler) GPL-licensed code must also be made available
 * under the GPL along with build & install instructions.
 *
 * @file    hvx_sw_test_ew.cpp
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

constexpr auto overflow  = hvx::util::overflow_e::kSaturate;
constexpr auto underflow = hvx::util::underflow_e::kTrunc;
constexpr auto exec      = hvx::util::execution_e::kExact;

/******************************************************************************************************************************************/

template<typename src_type_, typename arg_type_, typename dst_type_, int64_t vec_size_>
auto
TestConfiguration(const char* name) noexcept -> std::string {
    constexpr float sw_arg1 = 0.25f;
    constexpr float sw_arg2 = 0.75f;
    using dim =
        hvx::tensor_param<4, hvx::vector_param<16, vec_size_>, hvx::vector_param<8, 1>, hvx::vector_param<4, 1>, hvx::vector_param<2, 1>>;
    using eval = hvx::sw::EvaluateParam<false, 4, 4, 4, hvx::util::vector<dst_type_, dim::vec_size>, 0>;

    // configuration
    hvx::sw::EwEvaluate<hvx::abs_param<src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_abs(1.0f);
    hvx::sw::EwEvaluate<hvx::addconst_param<src_type_, arg_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_addconst(0.5f);
    hvx::sw::EwEvaluate<hvx::add_param<src_type_, src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_add(0.5f);
    hvx::sw::EwEvaluate<hvx::clip2_param<src_type_, arg_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_clip(1.0f);
    hvx::sw::EwEvaluate<hvx::maxconst_param<src_type_, arg_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_maxconst(1.0f);
    hvx::sw::EwEvaluate<hvx::max_param<src_type_, src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_max(1.0f);
    hvx::sw::EwEvaluate<hvx::minconst_param<src_type_, arg_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_minconst(1.0f);
    hvx::sw::EwEvaluate<hvx::min_param<src_type_, src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_min(1.0f);
    hvx::sw::EwEvaluate<hvx::mulconst_param<src_type_, arg_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_mulconst(0.5f);
    hvx::sw::EwEvaluate<hvx::mul_param<src_type_, src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_mul(0.5f);
    hvx::sw::EwEvaluate<hvx::sigmoid_param<src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_sigmoid(1.0f);
    hvx::sw::EwEvaluate<hvx::sub_param<src_type_, src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_sub(0.5f);
    hvx::sw::EwEvaluate<hvx::tanh_param<src_type_, dst_type_, dim, overflow, underflow, exec>, eval> eval_tanh(1.0f);

    //
    eval_abs.HwElementwise(sw_arg1, sw_arg2);
    eval_addconst.HwElementwise(sw_arg1, sw_arg2);
    eval_add.HwElementwise(sw_arg1, sw_arg2);
    eval_clip.HwElementwise(sw_arg1, sw_arg2);
    eval_maxconst.HwElementwise(sw_arg1, sw_arg2);
    eval_max.HwElementwise(sw_arg1, sw_arg2);
    eval_minconst.HwElementwise(sw_arg1, sw_arg2);
    eval_min.HwElementwise(sw_arg1, sw_arg2);
    eval_mulconst.HwElementwise(sw_arg1, sw_arg2);
    eval_mul.HwElementwise(sw_arg1, sw_arg2);
    eval_sigmoid.HwElementwise(sw_arg1, sw_arg2);
    eval_sub.HwElementwise(sw_arg1, sw_arg2);
    eval_tanh.HwElementwise(sw_arg1, sw_arg2);

    //
    return std::string(name) + "\n"                                           //
         + "\tAbs     : " + eval_abs.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tAddConst: " + eval_addconst.Evaluation(sw_arg1, sw_arg2) + "\n" //
         + "\tAdd     : " + eval_add.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tClip    : " + eval_clip.Evaluation(sw_arg1, sw_arg2) + "\n"     //
         + "\tMaxConst: " + eval_maxconst.Evaluation(sw_arg1, sw_arg2) + "\n" //
         + "\tMax     : " + eval_max.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tMinConst: " + eval_minconst.Evaluation(sw_arg1, sw_arg2) + "\n" //
         + "\tMin     : " + eval_min.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tMulConst: " + eval_mulconst.Evaluation(sw_arg1, sw_arg2) + "\n" //
         + "\tMul     : " + eval_mul.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tSigmoid : " + eval_sigmoid.Evaluation(sw_arg1, sw_arg2) + "\n"  //
         + "\tSub     : " + eval_sub.Evaluation(sw_arg1, sw_arg2) + "\n"      //
         + "\tTanh    : " + eval_tanh.Evaluation(sw_arg1, sw_arg2) + "\n";    //
}

/*!
 * @brief
 */
template<typename src_type_, typename arg_type_, typename dst_type_>
auto
TestLayers(const char* name) noexcept -> void {
    std::string results;
    results.append(name);
    results.append(TestConfiguration<src_type_, arg_type_, dst_type_, 1>("  Vec1:"));
    results.append(TestConfiguration<src_type_, arg_type_, dst_type_, 2>("  Vec2:"));
    results.append(TestConfiguration<src_type_, arg_type_, dst_type_, 4>("  Vec4:"));
    std::cout << results;
}

/******************************************************************************************************************************************/

auto
main() -> int {
    //
    constexpr int64_t num = 7;
    using type1           = hvx::util::dfixed<int16_t, 15>;
    using type2           = hvx::util::dfixed<uint16_t, 16>;
    using type3           = hvx::util::dfixed<float, 28>;
    using type4           = hvx::util::dfixed<int16_t, 14>;
    // using type5           = dynfloat::std_f32;
    // using type6           = dynfloat::std_f16;

    //
    std::array<const char*, num> names{
        "\nFixed-Point [signed 16-bit, 15-bit fraction]\n",
        "\nFixed-Point [unsigned 16-bit, 16-bit fraction]\n",
        "\nFixed-Point [signed 16-bit, 15-bit fraction (inputs), 14-bit fraction (outputs)]\n",
        "\nFixed-Point [signed 16-bit, 14-bit fraction (inputs), 15-bit fraction (outputs)]\n",
        "\nFloating-Point\n",
        "\nDfloat std32\n",
        "\nDfloat std16\n",
    };

    //
    std::vector<std::thread> threads;
    threads.reserve(num);

    // test neural network functions (one configuration per thread)
    threads.emplace_back(&TestLayers<type1, type1, type1>, names[0]);
    threads.emplace_back(&TestLayers<type2, type2, type2>, names[1]);
    threads.emplace_back(&TestLayers<type1, type1, type4>, names[2]);
    threads.emplace_back(&TestLayers<type4, type4, type1>, names[3]);
    threads.emplace_back(&TestLayers<type3, type3, type3>, names[4]);
    // threads.emplace_back(&TestLayers<type5, type5, type5>, names[5]);
    // threads.emplace_back(&TestLayers<type6, type6, type6>, names[6]);

    // wait until all threads have finished
    for (auto& thread: threads)
        thread.join();
}
