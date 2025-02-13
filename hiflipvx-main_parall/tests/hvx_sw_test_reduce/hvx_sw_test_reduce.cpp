/**
 * Licence: GNU GPLv3 \n
 * You may copy, distribute and modify the software as long as you track
 * changes/dates in source files. Any modifications to or software
 * including (via compiler) GPL-licensed code must also be made available
 * under the GPL along with build & install instructions.
 *
 * @file    hvx_sw_test_reduce.cpp
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

constexpr auto overflow  = hvx::util::overflow_e::kSaturate;
constexpr auto underflow = hvx::util::underflow_e::kRound;
constexpr auto exec      = hvx::util::execution_e::kExact;

/******************************************************************************************************************************************/

template<typename src_type_, typename dst_type_, typename src_dim_, typename reduce_, hvx::util::reduce_e op_type_>
auto
TestConfiguration(const char* name, float max_val) noexcept -> std::string {
    std::string names = std::string(name);

    for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i) {
        if (i < src_dim_::dim_num)
            names.append(std::to_string(reduce_::dims[i]) + ","); // NOLINT
        else
            names.append("  ");
    }

    // configuration
    using param = hvx::red::Reduce<src_type_, dst_type_, src_dim_, reduce_, overflow, underflow, exec, op_type_>;

    // evaluation
    hvx::sw::ReduceEvaluate<param, hvx::sw::EvaluateParam<false, 4, 4, 4, hvx::util::vector<dst_type_, 1>, 0>> eval(max_val);

    // computation
    eval.HwReduce();
    return names + eval.Evaluation() + "\n"; //
}

template<typename src_type_, typename dst_type_, typename src_dim_, typename reduce_>
auto
TestConfigurations() noexcept -> std::string {
    // calculate the max value for sum reduction to prevent overflow
    constexpr auto sum_bits = hvx::red::impl::ExtraBitsCalculate<src_dim_, reduce_, hvx::util::reduce_e::Sum, src_dim_::dim_num>();
    const float sum_max_val = 1.0f / powf(2.0f, sum_bits);

    std::string results;
    results.append(TestConfiguration<src_type_, dst_type_, src_dim_, reduce_, hvx::util::reduce_e::Max>("Max:  ", 1.0f));
    results.append(TestConfiguration<src_type_, dst_type_, src_dim_, reduce_, hvx::util::reduce_e::Mean>("Mean: ", 1.0f));
    results.append(TestConfiguration<src_type_, dst_type_, src_dim_, reduce_, hvx::util::reduce_e::Min>("Min:  ", 1.0f));
    results.append(TestConfiguration<src_type_, dst_type_, src_dim_, reduce_, hvx::util::reduce_e::Sum>("Sum:  ", sum_max_val));
    return results;
}

/*!
 * @brief
 */
template<typename src_type_, typename dst_type_>
auto
TestLayers(const char* name) noexcept -> void {
    //
    using dim0 = hvx::vector_param<4, 1>;
    using dim1 = hvx::vector_param<4, 1>;
    using dim2 = hvx::vector_param<4, 1>;
    using dim3 = hvx::vector_param<4, 1>;
    using dim4 = hvx::vector_param<4, 1>;

    //
    using dims1 = hvx::util::TensorParam<1, dim0>;
    using dims2 = hvx::util::TensorParam<2, dim0, dim1>;
    using dims3 = hvx::util::TensorParam<3, dim0, dim1, dim2>;
    using dims4 = hvx::util::TensorParam<4, dim0, dim1, dim2, dim3>;
    using dims5 = hvx::util::TensorParam<5, dim0, dim1, dim2, dim3, dim4>;

    std::string results = name;

    results.append(TestConfigurations<src_type_, dst_type_, dims1, hvx::util::ReduceParam<false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims1, hvx::util::ReduceParam<true>>());

    results.append(TestConfigurations<src_type_, dst_type_, dims2, hvx::util::ReduceParam<false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims2, hvx::util::ReduceParam<true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims2, hvx::util::ReduceParam<false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims2, hvx::util::ReduceParam<true, true>>());

    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<false, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<true, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<false, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<true, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<false, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<true, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims3, hvx::util::ReduceParam<false, true, true>>());

    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, false, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, false, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, true, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, true, false, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, false, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, false, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, true, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, true, true, false>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, false, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, false, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, true, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, true, false, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, false, true, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, false, true, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<false, true, true, true>>());
    results.append(TestConfigurations<src_type_, dst_type_, dims4, hvx::util::ReduceParam<true, true, true, true>>());

    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, false, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, false, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, false, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, false, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, true, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, true, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, true, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, true, false, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, false, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, false, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, false, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, false, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, true, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, true, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, true, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, true, true, false>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, false, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, false, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, false, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, false, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, true, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, true, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, true, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, true, false, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, false, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, false, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, false, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, false, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, false, true, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, false, true, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<false, true, true, true, true>>());
    // results.append(TestConfigurations<src_type_, dst_type_, dims5, hvx::util::ReduceParam<true, true, true, true, true>>());

    std::cout << results;
}

/******************************************************************************************************************************************/

auto
main() -> int {
    constexpr int64_t num = 3;
    using type1           = hvx::util::dfixed<int32_t, 31>;
    // using type2           = hvx::util::dfixed<uint32_t, 32>;
    // using type3           = hvx::util::dfixed<int16_t, 15>;
    using type4 = hvx::util::dfixed<uint16_t, 16>;
    using type5 = hvx::util::dfixed<float, 24>;
    // using type6 = dynfloat::std_f32;

    //
    std::array<const char*, 6> names{
        "\nFixed-Point [signed 32-bit, 31-bit fraction  ]\n", "\nFixed-Point [unsigned 32-bit, 32-bit fraction]\n",
        "\nFixed-Point [signed 16-bit, 15-bit fraction  ]\n", "\nFixed-Point [unsigned 16-bit, 16-bit fraction]\n",
        "\nFixed-Point [float                           ]\n", "\nDynfloat    [float32                         ]\n",
    };

    //
    std::vector<std::thread> threads;
    threads.reserve(num);

    // test neural network functions (one configuration per thread)
    threads.emplace_back(&TestLayers<type1, type1>, names[0]);
    // threads.emplace_back(&TestLayers<type2, type2>, names[1]);
    // threads.emplace_back(&TestLayers<type3, type3>, names[2]);
    threads.emplace_back(&TestLayers<type4, type4>, names[3]);
    threads.emplace_back(&TestLayers<type5, type5>, names[4]);
    // threads.emplace_back(&TestLayers<type6, type6>, names[5]);

    // wait until all threads have finished
    for (auto& thread: threads)
        thread.join();
}
