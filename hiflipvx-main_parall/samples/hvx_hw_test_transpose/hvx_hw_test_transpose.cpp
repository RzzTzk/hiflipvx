/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ´┐¢Software´┐¢), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ´┐¢AS IS´┐¢, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_hw_test_transpose.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::transpose_param<
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,
    hvx::tensor_param<4, hvx::vector_param<4, 2>, hvx::vector_param<4, 1>, hvx::vector_param<4, 1>, hvx::vector_param<1, 1>>,
    hvx::transpose_perm<0, 2, 1, 3>,
    1>;
using stream = hvx::stream_param<typename param::type, typename param::dst_dim, hvx::axis_e::kEof>;

// HW accelerator
void
TestHw(param::src_port* src, stream::port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst);
    auto dst_fifo = hvx::HwFifo<param::dst_vec, param::dst_dim::vec_elms, 2>();
    hvx::HwTranspose<param>(src, dst_fifo.data);
    hvx::HwHvxToStream<stream>(dst_fifo.data, *dst);
}

// testbench
auto
main() -> int {
    auto src_hw = hvx::SwCreateArrayOfVector<param::src_dim, param::type>();
    auto src_sw = hvx::SwCreateArray<param::src_dim, param::type>();
    hvx::array1d<param::type, param::dst_dim::elms> dst_sw{};
    hvx::array1d<stream::port, stream::dim::vec_elms> dst_hw_hls{};
    hvx::array1d<param::dst_port, param::dst_dim::vec_elms> dst_hw_hvx{};
    hvx::sw::SwTranspose<param>(src_sw.data, dst_sw.data);
    TestHw(src_hw.data, dst_hw_hls.data);
    hvx::HwStreamToHvx<stream>(*dst_hw_hls.data, dst_hw_hvx.data);
    hvx::SwCompareArrayOfVector<param::src_dim, param::type, param::dst_dim, param::type>(dst_sw, dst_hw_hvx, "  ");
    return 0;
}
