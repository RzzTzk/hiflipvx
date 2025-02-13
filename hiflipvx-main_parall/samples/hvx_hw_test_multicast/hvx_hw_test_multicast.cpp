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
 * @file    hvx_hw_test_multicast.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::multicast_param<
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,
    hvx::tensor_param<4, hvx::vector_param<32, 4>, hvx::vector_param<16, 1>, hvx::vector_param<8, 1>, hvx::vector_param<2, 1>>>;
using stream = hvx::stream_param<typename param::type, typename param::dim, hvx::axis_e::kEof>;

// HW accelerator
auto
TestHw(param::src_port* src, stream::port* dst0, stream::port* dst1, stream::port* dst2, stream::port* dst3) noexcept -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst0, dst1, dst2, dst3);
    auto dst_fifo0 = hvx::HwFifo<param::vec, param::dim::vec_elms, 2>();
    auto dst_fifo1 = hvx::HwFifo<param::vec, param::dim::vec_elms, 2>();
    auto dst_fifo2 = hvx::HwFifo<param::vec, param::dim::vec_elms, 2>();
    auto dst_fifo3 = hvx::HwFifo<param::vec, param::dim::vec_elms, 2>();
    hvx::HwMulticast<param>(src, dst_fifo0.data, dst_fifo1.data, dst_fifo2.data, dst_fifo3.data);
    hvx::HwHvxToStream<stream>(dst_fifo0.data, *dst0);
    hvx::HwHvxToStream<stream>(dst_fifo1.data, *dst1);
    hvx::HwHvxToStream<stream>(dst_fifo2.data, *dst2);
    hvx::HwHvxToStream<stream>(dst_fifo3.data, *dst3);
}

// testbench
auto
main() -> int {
    auto src = hvx::SwCreateArrayOfVector<param::dim, param::type>();
    hvx::array1d<hvx::array1d<stream::port, stream::dim::vec_elms>, 4> dst{};
    TestHw(src.data, dst.Get(0).data, dst.Get(1).data, dst.Get(2).data, dst.Get(3).data);
    for (int64_t i = 0; i < 4; ++i) {
        hvx::array1d<stream::vec, stream::dim::vec_elms> buffer{};
        hvx::HwStreamToHvx<stream>(*dst.Get(i).data, buffer.data);
        hvx::SwCompareArrayOfVector<param::dim, param::type, param::dim, param::type>(src, buffer, "  ");
    }
    return 0;
}
