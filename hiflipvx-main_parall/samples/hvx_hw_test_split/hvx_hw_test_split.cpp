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
 * @file    hvx_hw_test_split.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::split_param<
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,
    hvx::tensor_param<4, hvx::vector_param<32, 4>, hvx::vector_param<16, 1>, hvx::vector_param<8, 1>, hvx::vector_param<2, 1>>,
    hvx::util::ConcatSplitParam<0, 8, 4, 16, 4>>;
using stream0 = hvx::stream_param<typename param::type, typename param::split0::dim, hvx::axis_e::kEof>;
using stream1 = hvx::stream_param<typename param::type, typename param::split1::dim, hvx::axis_e::kEof>;
using stream2 = hvx::stream_param<typename param::type, typename param::split2::dim, hvx::axis_e::kEof>;
using stream3 = hvx::stream_param<typename param::type, typename param::split3::dim, hvx::axis_e::kEof>;

// HW accelerator
void
TestHw(param::port* src, stream0::port* dst0, stream1::port* dst1, stream2::port* dst2, stream3::port* dst3) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst0, dst1, dst2, dst3);
    auto dst0_fifo = hvx::HwFifo<param::split0_vec, param::split0::dim::vec_elms, 2>();
    auto dst1_fifo = hvx::HwFifo<param::split1_vec, param::split1::dim::vec_elms, 2>();
    auto dst2_fifo = hvx::HwFifo<param::split2_vec, param::split2::dim::vec_elms, 2>();
    auto dst3_fifo = hvx::HwFifo<param::split3_vec, param::split3::dim::vec_elms, 2>();
    hvx::HwSplit<param>(src, dst0_fifo.data, dst1_fifo.data, dst2_fifo.data, dst3_fifo.data);
    hvx::HwHvxToStream<stream0>(dst0_fifo.data, *dst0);
    hvx::HwHvxToStream<stream1>(dst1_fifo.data, *dst1);
    hvx::HwHvxToStream<stream2>(dst2_fifo.data, *dst2);
    hvx::HwHvxToStream<stream3>(dst3_fifo.data, *dst3);
}

// testbench
auto
main() -> int {
    auto src = hvx::SwCreateArrayOfVector<param::dim, param::type>();
    hvx::array1d<stream0::port, stream0::dim::vec_elms> split0_hls{};
    hvx::array1d<stream1::port, stream1::dim::vec_elms> split1_hls{};
    hvx::array1d<stream2::port, stream2::dim::vec_elms> split2_hls{};
    hvx::array1d<stream3::port, stream3::dim::vec_elms> split3_hls{};
    hvx::array1d<stream0::vec, stream0::dim::vec_elms> split0_hvx{};
    hvx::array1d<stream1::vec, stream1::dim::vec_elms> split1_hvx{};
    hvx::array1d<stream2::vec, stream2::dim::vec_elms> split2_hvx{};
    hvx::array1d<stream3::vec, stream3::dim::vec_elms> split3_hvx{};
    decltype(src) dst;
    TestHw(src.data, split0_hls.data, split1_hls.data, split2_hls.data, split3_hls.data);
    hvx::HwStreamToHvx<stream0>(*split0_hls.data, split0_hvx.data);
    hvx::HwStreamToHvx<stream1>(*split1_hls.data, split1_hvx.data);
    hvx::HwStreamToHvx<stream2>(*split2_hls.data, split2_hvx.data);
    hvx::HwStreamToHvx<stream3>(*split3_hls.data, split3_hvx.data);
    hvx::HwConcat<param>(split0_hvx.data, split1_hvx.data, split2_hvx.data, split3_hvx.data, dst.data);
    hvx::SwCompareArrayOfVector<param::dim, param::type, param::dim, param::type>(src, dst, "  ");
    return 0;
}
