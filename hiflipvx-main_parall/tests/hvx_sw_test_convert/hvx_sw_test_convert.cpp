/**
 * Licence: GNU GPLv3 \n
 * You may copy, distribute and modify the software as long as you track
 * changes/dates in source files. Any modifications to or software
 * including (via compiler) GPL-licensed code must also be made available
 * under the GPL along with build & install instructions.
 *
 * @file    hvx_sw_test_convert.cpp
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

/*!
 * @brief test multicast function
 */
template<typename type_, int64_t elms_, int64_t vec_size_>
auto
TestMulticastSingle(std::string& results) noexcept -> void {
    using config =
        hvx::convert::MulticastParam<type_,
                                     hvx::util::TensorParam<3, hvx::util::VectorParam<3 * elms_, vec_size_>,
                                                            hvx::util::VectorParam<2 * elms_, 1>, hvx::util::VectorParam<1 * elms_, 1>>>;

    // allocate memory
    hvx::util::array1d<typename config::src_port, config::dim::vec_elms> src{};
    hvx::util::array1d<hvx::util::array1d<typename config::dst_port, config::dim::vec_elms>, 4> dst{};

    // create input data, execute accelerator and verify outputs
    hvx::sw::CreateArrayOfVector<typename config::dim, typename config::type>();
    hvx::HwMulticast<config>(src.data, dst.Get(0).data, dst.Get(1).data);
    for (int64_t i = 0; i < 2; ++i)
        results.append(
            hvx::sw::CompareArrayOfVector<typename config::dim, typename config::type, typename config::dim, typename config::type>(
                src, dst.Get(i), "  "));
    hvx::HwMulticast<config>(src.data, dst.Get(0).data, dst.Get(1).data, dst.Get(2).data);
    for (int64_t i = 0; i < 3; ++i)
        results.append(
            hvx::sw::CompareArrayOfVector<typename config::dim, typename config::type, typename config::dim, typename config::type>(
                src, dst.Get(i), "  "));
    hvx::HwMulticast<config>(src.data, dst.Get(0).data, dst.Get(1).data, dst.Get(2).data, dst.Get(3).data);
    for (int64_t i = 0; i < 4; ++i)
        results.append(
            hvx::sw::CompareArrayOfVector<typename config::dim, typename config::type, typename config::dim, typename config::type>(
                src, dst.Get(i), "  "));
}

/*!
 * @brief test multicast functions
 */
auto
TestMulticast() noexcept -> void {
    std::string results = "Multicast:\n";
    TestMulticastSingle<hvx::util::dfixed<uint16_t, 15>, 32, 6>(results);
    std::cout << results;
}

/******************************************************************************************************************************************/

///*!
// * @brief
// */
// template<typename type_, int64_t dim0_, int64_t dim1_, int64_t dim2_, int64_t elms_, int64_t src_vec_size_, int64_t dst_vec_size_>
// auto
// TestTransposeSingle() noexcept -> std::string {
//    // input and output ports/tensor
//    using src_dim  = hvx::util::TensorParam<3, hvx::util::VectorParam<1 * elms_, src_vec_size_>, hvx::util::VectorParam<2 * elms_, 1>,
//                                           hvx::util::VectorParam<3 * elms_, 1>>;
//    using dst_dim  = hvx::util::TensorParam<3, hvx::util::VectorParam<(dim0_ + 1) * elms_, dst_vec_size_>,
//                                           hvx::util::VectorParam<(dim1_ + 1) * elms_, 1>, hvx::util::VectorParam<(dim2_ + 1) * elms_,
//                                           1>>;
//    using src_port = hvx::util::stream_port<type_, src_dim::vec_size>;
//    using dst_port = hvx::util::stream_port<type_, dst_dim::vec_size>;
//
//    // allocate memory
//    std::vector<src_port> src_hw(src_dim::elms / src_dim::vec_size);
//    std::vector<dst_port> dst_hw(dst_dim::elms / dst_dim::vec_size);
//    std::vector<float> src_sw(src_dim::elms);
//    std::vector<float> dst_sw(dst_dim::elms);
//
//    // create inputs, compute HW/SW transpose, compare outputs
//    hvx::sw::CreateInputData<src_dim, src_port>(src_hw.data(), src_sw.data());
//    hvx::user::HwTranspose<type_, src_dim, dst_dim, dim0_, dim1_, dim2_>(*src_hw.data(), *dst_hw.data());
//    hvx::sw::SwTranspose<src_dim, dst_dim, dim0_, dim1_, dim2_>(src_sw.data(), dst_sw.data());
//    return hvx::sw::CompareOutputData<dst_dim, dst_port>(dst_hw.data(), dst_sw.data(), ": ");
//}
//
///*!
// * @brief
// */
// auto
// TestTranspose() noexcept -> void {
//    std::string results = "Transpose:\n";
//    results.append("  (2, 1, 0)" + TestTransposeSingle<hvx::util::dfixed<uint16_t, 15>, 2, 1, 0, 32, 2, 2>());
//    results.append("  (1, 0, 2)" + TestTransposeSingle<hvx::util::dfixed<uint16_t, 15>, 1, 0, 2, 32, 2, 2>());
//    results.append("  (0, 2, 1)" + TestTransposeSingle<hvx::util::dfixed<uint16_t, 15>, 0, 2, 1, 32, 2, 2>());
//    std::cout << results;
//}

/******************************************************************************************************************************************/

///*!
// * @brief
// */
// template<typename port_, typename dim_, typename mid_dim_>
// auto
// TestSplitConcat2(std::vector<port_>& src, std::vector<port_>& dst, std::string& results) noexcept -> void {
//    // allocate memory
//    std::array<std::vector<port_>, 2> mid;
//    for (auto& i: mid)
//        i.resize(mid_dim_::elms / dim_::vec_size);
//
//    // create inputs, compute HW split/concat, compare output with input
//    hvx::sw::CreateStreamPortData<dim_, port_>(src.data());
//    hvx::user::HwSplit<dim_, mid_dim_, mid_dim_>(*src.data(), *mid.at(0).data(), *mid.at(1).data());
//    hvx::user::HwConcat<dim_, mid_dim_, mid_dim_>(*dst.data(), *mid.at(0).data(), *mid.at(1).data());
//    results.append(hvx::sw::CompareStreamPortData<dim_, dim_, port_, port_>(src.data(), dst.data(), "   2 intermediate: "));
//}
//
///*!
// * @brief
// */
// template<typename port_, typename dim_, typename mid_dim_>
// auto
// TestSplitConcat3(std::vector<port_>& src, std::vector<port_>& dst, std::string& results) noexcept -> void {
//    // allocate memory
//    std::array<std::vector<port_>, 3> mid;
//    for (auto& i: mid)
//        i.resize(mid_dim_::elms / dim_::vec_size);
//
//    // create inputs, compute HW split/concat, compare output with input
//    hvx::sw::CreateStreamPortData<dim_, port_>(src.data());
//    hvx::user::HwSplit<dim_, mid_dim_, mid_dim_, mid_dim_>(*src.data(), *mid.at(0).data(), *mid.at(1).data(), *mid.at(2).data());
//    hvx::user::HwConcat<dim_, mid_dim_, mid_dim_, mid_dim_>(*dst.data(), *mid.at(0).data(), *mid.at(1).data(), *mid.at(2).data());
//    results.append(hvx::sw::CompareStreamPortData<dim_, dim_, port_, port_>(src.data(), dst.data(), "   3 intermediate: "));
//}
//
///*!
// * @brief
// */
// template<typename port_, typename dim_, typename mid_dim_>
// auto
// TestSplitConcat4(std::vector<port_>& src, std::vector<port_>& dst, std::string& results) noexcept -> void {
//    // allocate memory
//    std::array<std::vector<port_>, 4> mid;
//    for (auto& i: mid)
//        i.resize(mid_dim_::elms / dim_::vec_size);
//
//    // create inputs, compute HW split/concat, compare output with input
//    hvx::sw::CreateStreamPortData<dim_, port_>(src.data());
//    hvx::user::HwSplit<dim_, mid_dim_, mid_dim_, mid_dim_, mid_dim_>(*src.data(), *mid.at(0).data(), *mid.at(1).data(), *mid.at(2).data(),
//                                                                     *mid.at(3).data());
//    hvx::user::HwConcat<dim_, mid_dim_, mid_dim_, mid_dim_, mid_dim_>(*dst.data(), *mid.at(0).data(), *mid.at(1).data(),
//    *mid.at(2).data(),
//                                                                      *mid.at(3).data());
//    results.append(hvx::sw::CompareStreamPortData<dim_, dim_, port_, port_>(src.data(), dst.data(), "   4 intermediate: "));
//}
//
///*!
// * @brief
// */
// auto
// TestSplitConcat() noexcept -> void {
//    using batch    = hvx::util::VectorParam<2, 1>;
//    using rows     = hvx::util::VectorParam<48, 1>;
//    using cols     = hvx::util::VectorParam<48, 1>;
//    using chan     = hvx::util::VectorParam<48, 4>;
//    using port     = hvx::util::stream_port<hvx::util::dfixed<uint16_t, 15>, chan::vec_size>;
//    using dim      = hvx::util::TensorParam<4, chan, cols, rows, batch>;
//    using mid4_dim = hvx::util::TensorParam<4, hvx::util::VectorParam<chan::elms / 4, chan::vec_size>, cols, rows, batch>;
//    using mid3_dim = hvx::util::TensorParam<4, chan, hvx::util::VectorParam<cols::elms / 3, cols::vec_size>, rows, batch>;
//    using mid2_dim = hvx::util::TensorParam<4, chan, cols, hvx::util::VectorParam<rows::elms / 2, rows::vec_size>, batch>;
//
//    // allocate memory
//    std::vector<port> src(dim::elms / dim::vec_size);
//    std::vector<port> dst(dim::elms / dim::vec_size);
//
//    // test different split transpose function implementations
//    std::string results = "Split & Concat:\n";
//    TestSplitConcat2<port, dim, mid2_dim>(src, dst, results);
//    TestSplitConcat3<port, dim, mid3_dim>(src, dst, results);
//    TestSplitConcat4<port, dim, mid4_dim>(src, dst, results);
//    std::cout << results;
//}

/******************************************************************************************************************************************/

/*!
 * @brief
 */
template<typename type, typename src_dim_, typename dst_dim_>
auto
TestReshapeSingle(std::string& results, const char* name) noexcept -> void {
    // configuration of the reshape layer
    using config = hvx::convert::ReshapeParam<type, src_dim_, src_dim_>;

    // allocate memory
    hvx::util::array1d<typename config::src_port, config::src_dim::vec_elms> src{};
    hvx::util::array1d<typename config::dst_port, config::dst_dim::vec_elms> dst{};

    // create input data, test HW/SW functions and compare outputs
    hvx::sw::CreateArrayOfVector<typename config::src_dim, typename config::type>();
    hvx::HwReshape<config>(src.data, dst.data);
    results.append(
        hvx::sw::CompareArrayOfVector<typename config::src_dim, typename config::type, typename config::dst_dim, typename config::type>(
            src, dst, name));
}

/*!
 * @brief
 */
auto
TestReshape() noexcept -> void {
    using batch    = hvx::util::VectorParam<2, 1>;
    using rows     = hvx::util::VectorParam<32, 1>;
    using cols     = hvx::util::VectorParam<64, 1>;
    using vec3_dim = hvx::util::TensorParam<4, hvx::util::VectorParam<96, 3>, cols, rows, batch>;
    using vec4_dim = hvx::util::TensorParam<4, hvx::util::VectorParam<96, 4>, cols, rows, batch>;
    using vec6_dim = hvx::util::TensorParam<4, hvx::util::VectorParam<96, 6>, cols, rows, batch>;
    using kern     = hvx::util::VectorParam<3, 3>;
    using wgt1_dim = hvx::util::TensorParam<4, kern, kern, hvx::util::VectorParam<256, 4>, hvx::util::VectorParam<128, 1>>;
    using wgt2_dim = hvx::util::TensorParam<4, kern, kern, hvx::util::VectorParam<128, 8>, hvx::util::VectorParam<256, 1>>;

    //
    std::string results = "Reshape:\n";
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, vec4_dim, vec4_dim>(results, "  Passthrough:  ");
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, vec6_dim, vec3_dim>(results, "  Src multiple: ");
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, vec3_dim, vec6_dim>(results, "  Dst multiple: ");
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, vec6_dim, vec4_dim>(results, "  Src Bigger:   ");
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, vec4_dim, vec6_dim>(results, "  Dst Bigger:   ");
    TestReshapeSingle<hvx::util::dfixed<uint16_t, 15>, wgt1_dim, wgt2_dim>(results, "  Vec on Dim2:  ");
    std::cout << results;
}

/******************************************************************************************************************************************/

auto
main() -> int {
    // std::vector<std::thread> threads;
    // threads.reserve(4);

    //// test conversion functions (one function per thread)
    // threads.emplace_back(TestMulticast);
    // threads.emplace_back(TestReshape);
    // threads.emplace_back(TestSplitConcat);
    // threads.emplace_back(TestTranspose);

    //// wait until all threads have finished
    // for (auto& thread: threads)
    //     thread.join();

    return 0;
}
