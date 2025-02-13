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
 * @file    hvx_hw_test_reduce_sum.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::reduce_sum_param<
    // dynfloat::std_f16,
    // dynfloat::std_f16,
    hvx::dfixed<int16_t, 15>,
    hvx::dfixed<int16_t, 15>,
    hvx::tensor_param<4, hvx::vector_param<32, 1>, hvx::vector_param<32, 1>, hvx::vector_param<32, 1>, hvx::vector_param<32, 1>>,
    hvx::util::ReduceParam<false, true, false, true, false, false>,
    hvx::overflow_e::kSaturate,
    hvx::underflow_e::kFloor,
    hvx::execution_e::kExact>;
using stream = hvx::stream_param<typename param::dst_type, typename param::dst_dim, hvx::axis_e::kEof>;

// HW accelerator
void
TestHw(param::src_port* src, stream::port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst);
    auto dst_fifo = hvx::HwFifo<param::dst_vec, param::dst_dim::vec_elms, 2>();
    hvx::HwReduceSum<param>(src, dst_fifo.data);
    hvx::HwHvxToStream<stream>(dst_fifo.data, *dst);
}

// testbench
auto
main() -> int {
    hvx::reduce_eval<param, hvx::eval_param<true, 4, 4, 4, stream::port, stream::flags>> eval(1.0f);
    TestHw(eval.GetSrcHw(), eval.GetDstHw());
    eval.Evaluation();
    return 0;
}
