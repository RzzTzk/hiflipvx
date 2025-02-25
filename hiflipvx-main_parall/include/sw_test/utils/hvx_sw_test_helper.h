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
 * @file    hvx_sw_test_helper.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_SW_TEST_HELPER_H_
#define HVX_SW_TEST_HELPER_H_

#include "../../hiflipvx/hvx_core.h"

namespace hvx {
namespace sw {
/******************************************************************************************************************************************/

/*!
 * @brief stores all additional parameters needed for evaluation
 */
template<bool dbg_, int64_t num_worst_res_, int64_t num_first_and_last_elms_, int64_t rept_, typename dst_port_, int64_t dst_flags_>
struct EvaluateParam {
    static constexpr auto dbg                     = dbg_;
    static constexpr auto num_worst_res           = num_worst_res_;
    static constexpr auto num_first_and_last_elms = num_first_and_last_elms_;
    static constexpr auto rept                    = rept_;
    using dst_port                                = dst_port_;
    static constexpr auto dst_flags               = dst_flags_;
};

/******************************************************************************************************************************************/

/*!
 * @brief prints result of one element
 */
auto
EvalPrintElm(float* dst_sw,
             float* dst_hw_flt,
             std::vector<std::pair<float, int64_t>>& results,
             const char* name,
             int64_t percentile) noexcept -> void {
    const auto id       = results.at(percentile).second;
    const auto relative = results.at(percentile).first;
    const auto sw_dst   = dst_sw[id];     // NOLINT
    const auto hw_dst   = dst_hw_flt[id]; // NOLINT
    const auto absolute = hvx::util::Abs(sw_dst - hw_dst);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << name;
    std::cout << " | SW:" << std::setw(10) << sw_dst;
    std::cout << " | HW:" << std::setw(10) << hw_dst;
    std::cout << " | abs:" << std::setw(10) << absolute;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " | rel(%):" << std::setw(9) << relative;
    std::cout << "\n";
}

/*!
 * @brief prints result of one element
 */
auto
EvalPrintElm(int64_t id, float sw_dst, float hw_dst, float absolute, float relative) noexcept -> void {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    ID:" << std::setw(8) << id;
    std::cout << " | SW:" << std::setw(10) << sw_dst;
    std::cout << " | HW:" << std::setw(10) << hw_dst;
    std::cout << " | abs:" << std::setw(10) << absolute;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " | rel(%):" << std::setw(9) << relative;
    std::cout << "\n";
}

/*!
 * @brief Print the diff of 2 different output 2d images
 */
template<typename dst_dim_, typename eval_>
auto
EvalPrintDiff(float* dst_sw, float* dst_hw_flt) noexcept -> std::string {
    // buffer relative error between SW and HW output
    std::vector<std::pair<float, int64_t>> results;
    results.reserve(dst_dim_::elms);

    // Variables
    float avg   = 0.0;
    int64_t cnt = 0;

    // separator
    if (eval_::dbg == true) {
        std::cout << "    --------------------------------------------------------------------------------\n";
    }

    // Go through different dimensions
    for (int64_t i = 0; i < dst_dim_::elms; ++i) {
        // get the sw and hw output results (represented as float32)
        const float sw_dst = dst_sw[i];     // NOLINT
        const float hw_dst = dst_hw_flt[i]; // NOLINT

        // absolute error
        const float absolute = hvx::util::Abs(sw_dst - hw_dst);

        // relative error
        // - Ignore if (absolute error is not minimal) and if (division by zero)
        // - Set relative error to 0 if (absolute error in minimal)
        float relative = 0.0f;
        if (hvx::util::Abs(hw_dst) > FLT_EPSILON)
            relative = 100.0f * (absolute / hvx::util::Abs(hw_dst));
        else if (absolute > FLT_EPSILON)
            continue;

        // MAPE (Mean absolute percentage error)
        avg += relative;
        ++cnt;

        // store the relative error of all results
        results.emplace_back(relative, i);

        // printing the frist and last elements of the results
        if (eval_::dbg == true) {
            if (i < eval_::num_first_and_last_elms)
                hvx::sw::EvalPrintElm(i, sw_dst, hw_dst, absolute, relative);
            if (i >= (dst_dim_::elms - eval_::num_first_and_last_elms))
                hvx::sw::EvalPrintElm(i, sw_dst, hw_dst, absolute, relative);
        }
    }
    avg /= hvx::util::Max(static_cast<float>(cnt), 1.0f);

    // sort the output pixels by relative error
    std::sort(results.begin(), results.end(), std::greater<>());
    const int64_t elms = hvx::util::Min(results.size(), static_cast<size_t>(eval_::num_worst_res));

    // mean absolute percentage error
    if (eval_::dbg == true) {
        std::cout << "    --------------------------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(7);
        std::cout << "    " << (dst_dim_::elms - cnt) << " out of " << dst_dim_::elms
                  << " elements have been ommited due to division by zero!\n";
        std::cout << "    " << elms << " worst elements in terms of relative error are shown!\n";
        std::cout << "    Mean absolute percentage error (MAPE) of " << avg << "\n";
    }

    // get ptr to last element of result array
    const int64_t result_end = hvx::util::Max(cnt - 1, static_cast<int64_t>(1));
    if (eval_::dbg == true) {
        std::cout << "    --------------------------------------------------------------------------------\n";
        hvx::sw::EvalPrintElm(dst_sw, dst_hw_flt, results, "    minimum     ", static_cast<int64_t>((result_end * 8) / 8));
        hvx::sw::EvalPrintElm(dst_sw, dst_hw_flt, results, "    1st_quartile", static_cast<int64_t>((result_end * 6) / 8));
        hvx::sw::EvalPrintElm(dst_sw, dst_hw_flt, results, "    median      ", static_cast<int64_t>((result_end * 4) / 8));
        hvx::sw::EvalPrintElm(dst_sw, dst_hw_flt, results, "    3rd_quartile", static_cast<int64_t>((result_end * 2) / 8));
        hvx::sw::EvalPrintElm(dst_sw, dst_hw_flt, results, "    maximum     ", static_cast<int64_t>((result_end * 0) / 8));
        std::cout << "    --------------------------------------------------------------------------------\n";
        for (int64_t i = 0; i < elms; ++i) {
            const auto id       = results.at(i).second;
            const auto sw_dst   = dst_sw[id];     // NOLINT
            const auto hw_dst   = dst_hw_flt[id]; // NOLINT
            const auto absolute = hvx::util::Abs(sw_dst - hw_dst);
            const auto relative = results.at(i).first;
            hvx::sw::EvalPrintElm(id, sw_dst, hw_dst, absolute, relative);
        }
        std::cout << "    --------------------------------------------------------------------------------\n";
    }

    // return most important results
    float quartile1 = results.at((result_end * 3) / 4).first;
    float median    = results.at((result_end * 2) / 4).first;
    float quartile3 = results.at((result_end * 1) / 4).first;
    return "Mape: " + std::to_string(avg) + "   Q1: " + std::to_string(quartile1) + "   Median: " + std::to_string(median) +
           "   Q3: " + std::to_string(quartile3);
}

/******************************************************************************************************************************************/

/*!
 * @brief creates sw/hw input tensor with random values between (upper,-1) for signed or (upper,0) for unsigned
 */
template<typename src_port_, typename dim_>
auto
EvalCreateRndSrc(src_port_* src_hw_, float* src_sw_, const float upper = 1.f) noexcept -> void {
    // lower boundary for values
    const float lower = (src_port_::type::is_signed == true) ? (-upper) : (0.0f);

    // creating random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(lower, upper);

    // loop pointers for different dimensions. sperated for vectorization for HW (ptr_elms_v/ptr_elms_p)
    hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_elms{}, ptr_elms_v{}, ptr_elms_p{};

    for (int64_t i = 0; i < dim_::elms; ++i) {
        // updating pointers of N-dimensional flattened loop
        hvx::util::TensorDimElmsIter<dim_>(ptr_elms, i);
        hvx::util::TensorDimVecElmsIter<dim_>(ptr_elms, ptr_elms_v, ptr_elms_p, i);

        // creating random input
        //const auto sw_src = static_cast<float>(i);
        const auto sw_src = distribution(rng);
        const auto hw_src = static_cast<typename src_port_::type>(sw_src);

        // store in sw and hw containers
        src_sw_[hvx::util::TensorPtrElms<dim_>(ptr_elms)]                                                     = sw_src; // NOLINT
        src_hw_[hvx::util::TensorPtrElmsV<dim_>(ptr_elms_v)].Get(hvx::util::TensorPtrElmsP<dim_>(ptr_elms_p)) = hw_src; // NOLINT
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief Convert to floating point from an integer data type
 */
template<typename dst_type_, typename dim_, int64_t dst_flags>
auto
ConvertDstHwToFloat(hvx::util::vector<dst_type_, dim_::vec_size>& src, float* dst_hw_flt) noexcept -> void {
    for (int64_t v = 0; v < dim_::vec_elms; ++v) {
        for (int64_t p = 0; p < dim_::vec_size; ++p) {
            dst_hw_flt[v * dim_::vec_size + p] = static_cast<float>((&src)[v].Get(p)); // NOLINT
        }
    }
}
//xwq
// #if defined(HVX_SYNTHESIS_ACTIVE)
// /*!
//  * @brief Convert to floating point from an integer data type
//  */
// template<typename dst_type_, typename dst_dim_, int64_t dst_flags>
// auto
// ConvertDstHwToFloat(hvx::convert::hls_stream_port<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_flags>& src,
//                     float* dst_hw_flt) noexcept -> void {
//     for (int64_t v = 0; v < dst_dim_::vec_elms; ++v) {
//         auto src_data = src.read();
//         for (int64_t p = 0; p < dst_dim_::vec_size; ++p) {
//             dst_hw_flt[v * dst_dim_::vec_size + p] = static_cast<float>(src_data.data.Get(p)); // NOLINT
//         }
//     }
// }
// #endif

/******************************************************************************************************************************************/

/*!
 * @brief Converts an array of vector with dfixed type to an array with float type
 */
template<typename type_, typename dim_>
auto
ConvertDfixedVectorToFloat32(hvx::util::vector<type_, dim_::vec_size>* src, hvx::util::array1d<float, dim_::elms>& dst) noexcept -> void {
    constexpr float shift = (type_::is_int == false) ? (1.0f) : (1.0f / static_cast<float>(static_cast<int64_t>(1) << type_::frac_bits));

    // fisxed point data needs to be shifted back by the number of fraction bits
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        for (int64_t j = 0; j < dim_::vec_size; ++j) {
            dst.Set(src[i].Get(j).data * shift, i * dim_::vec_size + j);
        }
    }
}

/******************************************************************************************************************************************/

/*!
 * @brief Converts an array of vector with dfixed type to an array with float type
 */
template<typename type_, typename dim_>
auto
ConvertDfixedVectorToFloat32(hvx::util::vector<type_, dim_::vec_size>* src, float* dst) -> void {
    constexpr float shift = (type_::is_int == false) ? (1.0f) : (1.0f / static_cast<float>(static_cast<int64_t>(1) << type_::frac_bits));

    // fisxed point data needs to be shifted back by the number of fraction bits
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        for (int64_t j = 0; j < dim_::vec_size; ++j)
            dst[i * dim_::vec_size + j] = src[i].Get(j).data * shift; // NOLINT
    }
}

/******************************************************************************************************************************************/
/*!
 * @brief Creates an array with variables (starting with value 0 and then counting upwards)
 */
template<typename dim_, typename type_>
constexpr auto
CreateArrayOfVector() noexcept -> decltype(auto) {
    // the array to store data
    hvx::util::array1d<hvx::util::vector<type_, dim_::vec_size>, dim_::vec_elms> src{};

    // stores pointer and vectorized pointer
    hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_elms{}, ptr_elms_v{}, ptr_elms_p{};

    for (int32_t i = 0; i < dim_::elms; ++i) {
        // creates pointer for vectorization
        hvx::util::TensorDimElmsIter<dim_>(ptr_elms, i);
        hvx::util::TensorDimVecElmsIter<dim_>(ptr_elms, ptr_elms_v, ptr_elms_p, i);
        const int64_t v = hvx::util::TensorPtrElmsV<dim_>(ptr_elms_v);
        const int64_t p = hvx::util::TensorPtrElmsP<dim_>(ptr_elms_p);

        // writes data to the array
        auto data = static_cast<type_>(static_cast<float>(i % 128));
        src.Get(v).Set(data, p);
    }
    return src;
}

/*!
 * @brief Creates an array with variables (starting with value 0 and then counting upwards)
 */
template<typename dim_, typename type_>
constexpr auto
CreateArray() noexcept -> decltype(auto) {
    // the array to store data
    hvx::util::array1d<type_, dim_::elms> src{};

    // writes data to the array
    for (int64_t i = 0; i < dim_::elms; ++i) {
        auto data = static_cast<type_>(static_cast<float>(i % 128));
        src.Set(data, i);
    }

    return src;
}

/******************************************************************************************************************************************/

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
CompareArrayOfVector(hvx::util::array1d<hvx::util::vector<src_type_, src_dim_::vec_size>, src_dim_::vec_elms>& src,
                     hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                     const char* name) noexcept -> std::string {
    static_assert(src_dim_::elms == dst_dim_::elms, "Input and output need same amount of elements!");

    int64_t cnt = 0;
    for (int64_t i = 0; i < src_dim_::elms; ++i) {
        const int64_t src_v = i / src_dim_::vec_size;
        const int64_t src_p = i % src_dim_::vec_size;
        const int64_t dst_v = i / dst_dim_::vec_size;
        const int64_t dst_p = i % dst_dim_::vec_size;
        const float src_flt = static_cast<float>(src.Get(src_v).Get(src_p));
        const float dst_flt = static_cast<float>(dst.Get(dst_v).Get(dst_p));

        // check if same
        if (hvx::util::Abs(src_flt - dst_flt) > DBL_EPSILON)
            ++cnt;
    }

    // output the amount of errors
    return name + std::to_string(cnt) + " of " + std::to_string(dst_dim_::elms) + " output elements wrong!\n";
}

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
CompareArrayOfVector(hvx::util::array1d<src_type_, src_dim_::elms>& src,
                     hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                     const char* name) noexcept -> std::string {
    static_assert(src_dim_::elms == dst_dim_::elms, "Input and output need same amount of elements!");

    int64_t cnt = 0;
    for (int64_t i = 0; i < src_dim_::elms; ++i) {
        const int64_t dst_v = i / dst_dim_::vec_size;
        const int64_t dst_p = i % dst_dim_::vec_size;
        const float src_flt = static_cast<float>(src.Get(i));
        const float dst_flt = static_cast<float>(dst.Get(dst_v).Get(dst_p));

        // check if same
        if ((src_flt - dst_flt) > DBL_EPSILON)
            ++cnt;
    }

    // output the amount of errors
    return name + std::to_string(cnt) + " of " + std::to_string(dst_dim_::elms) + " output elements wrong!\n";
}

/*!
 * @brief Compares if the elements of 2 arrays are the same
 */
template<typename src_dim_, typename src_type_, typename dst_dim_, typename dst_type_>
auto
CompareArrayOfVector(hvx::util::vector<src_type_, src_dim_::vec_size>* src,
                     hvx::util::array1d<hvx::util::vector<dst_type_, dst_dim_::vec_size>, dst_dim_::vec_elms>& dst,
                     const char* name) noexcept -> std::string {
    static_assert(src_dim_::elms == dst_dim_::elms, "Input and output need same amount of elements!");

    // iterate over both arrays
    int64_t cnt = 0;
    for (int64_t i = 0; i < src_dim_::elms; ++i) {
        const int64_t src_v = i / src_dim_::vec_size;
        const int64_t src_p = i % src_dim_::vec_size;
        const int64_t dst_v = i / dst_dim_::vec_size;
        const int64_t dst_p = i % dst_dim_::vec_size;

        // check if same
        if (hvx::util::Abs(src[src_v].Get(src_p).data - dst.Get(dst_v).Get(dst_p).data) > DBL_EPSILON)
            ++cnt;
    }

    // output the amount of errors
    return name + std::to_string(cnt) + " of " + std::to_string(dst_dim_::elms) + " output elements wrong!\n";
}

/******************************************************************************************************************************************/

/*!
 * @brief measures the execution time of a function
 */
const auto MeasureTime = [](auto&& function) -> decltype(auto) {
    return [=](const char* name, int64_t iterations, auto&&... parameters) mutable -> decltype(auto) {
        double time_max = DBL_MIN, time_min = DBL_MAX, time_avg = 0.0;

        // execute function multiple times and get max/min/avg timing
        for (int64_t i = 0; i < iterations; ++i) {
            const auto t1 = std::chrono::high_resolution_clock::now();
            function(std::forward<decltype(parameters)>(parameters)...); // NOLINT
            const auto t2        = std::chrono::high_resolution_clock::now();
            const auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            const auto time      = time_span.count() * 1000000.0;
            time_max             = hvx::util::Max(time, time_max);
            time_min             = hvx::util::Min(time, time_min);
            time_avg += time;
        }
        time_avg /= static_cast<double>(iterations);

        // print timing results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << name << " Time (us):" << std::setw(10) << time_max << "(max)" << std::setw(10) << time_avg << "(avg)"
                  << std::setw(10) << time_min << "(min)\n";
    };
};

/*!
 * @brief Measures the execution time of a function, if it is not used in the HLS environment
 */
template<typename Function, typename... Parameters>
auto
MeasureFuncTime(const bool debug, const char* name, int64_t iterations, Function&& function, Parameters&&... parameters) {
#ifndef HVX_SYNTHESIS_ACTIVE
    if (debug == true)
        return MeasureTime(std::forward<Function>(function))(name, iterations, std::forward<Parameters>(parameters)...);
    else
        std::forward<Function>(function)(std::forward<Parameters>(parameters)...); // NOLINT
#else
    std::forward<Function>(function)(std::forward<Parameters>(parameters)...);
    (void)name;
    (void)iterations;
#endif
}

/******************************************************************************************************************************************/
} // namespace sw
} // namespace hvx

#endif // HVX_SW_TEST_HELPER_H_
