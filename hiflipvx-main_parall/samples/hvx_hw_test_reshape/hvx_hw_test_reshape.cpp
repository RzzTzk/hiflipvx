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
 * @file    hvx_hw_test_reshape.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::reshape_param<
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,
    hvx::tensor_param<4, hvx::vector_param<24, 6>, hvx::vector_param<16, 1>, hvx::vector_param<8, 1>, hvx::vector_param<4, 1>>,
    hvx::tensor_param<4, hvx::vector_param<24, 4>, hvx::vector_param<16, 1>, hvx::vector_param<8, 1>, hvx::vector_param<4, 1>>>;
using stream = hvx::stream_param<typename param::type, typename param::dst_dim, hvx::axis_e::kEof>;

// HW accelerator
void
TestHw(param::src_port* src, stream::port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst);
    auto dst_fifo = hvx::HwFifo<param::dst_vec, param::dst_dim::vec_elms, 2>();
    hvx::HwReshape<param>(src, dst_fifo.data);
    hvx::HwHvxToStream<stream>(dst_fifo.data, *dst);
}

// testbench
auto
main() -> int {
    auto src = hvx::SwCreateArrayOfVector<param::src_dim, param::type>();
    hvx::array1d<stream::port, stream::dim::vec_elms> dst_hls{};
    hvx::array1d<stream::vec, stream::dim::vec_elms> dst_hvx{};
    TestHw(src.data, dst_hls.data);
    hvx::HwStreamToHvx<stream>(*dst_hls.data, dst_hvx.data);
    hvx::SwCompareArrayOfVector<param::src_dim, param::type, param::dst_dim, param::type>(src, dst_hvx, "  ");
    return 0;
}
