/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * “Software?), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS?, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_hw_test_conv.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::conv_param<
    // dynfloat::dfloat<8, 17>,
    // dynfloat::dfloat<8, 17>,
    // dynfloat::dfloat<8, 17>,
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,  // src_type  / src_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // dst_type  / dst_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // wgts_type / wgts_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // bias_type / bias_type_frac_bits
    hvx::vector_param<1, 1>,   // batch     / batch_vec_size
    hvx::vector_param<8, 1>,  // src_rows  / src_rows_vec_size
    hvx::vector_param<8, 1>,  // src_cols  / src_cols_vec_size
    hvx::vector_param<2, 2>,  // chnls     / chnls_vec_size
    hvx::vector_param<2, 2>,  // fms       / fms_vec_size
    hvx::vector_param<3, 3>,   // knl_rows  / knl_rows_vec_size
    hvx::vector_param<3, 3>,   // knl_cols  / knl_cols_vec_size
    hvx::array2d_param<1, 1>,  // pad_rows  / pad_cols
    hvx::array2d_param<0, 0>,  // dil_rows  / dil_cols
    hvx::array2d_param<1, 1>,  // str_rows  / str_cols
    true,                      // buf_wgts
    true,                      // buf_bias
    hvx::overflow_e::kWrap,    // overflow_type
    hvx::underflow_e::kTrunc,  // underflow_type
    hvx::execution_e::kExact>; // exec_type
using stream = hvx::stream_param<typename param::dst_type, typename param::dst_dim, hvx::axis_e::kEof>;

// HW accelerator
auto
TestHw(param::src_port* src, param::wgts_port* wgts, param::bias_port* bias, stream::port* dst) noexcept -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    auto dst_fifo = hvx::HwFifo<param::dst_vec, param::dst_dim::vec_elms, 2>();
    hvx::HwConv<param>(src, wgts, bias, dst_fifo.data);
    hvx::HwHvxToStream<stream>(dst_fifo.data, *dst);
}

// testbench
auto
main() -> int {
    hvx::conv_eval<param, hvx::eval_param<true, 4, 4, 4, stream::port, stream::flags>> eval(0.75f, 0.25f);
    hvx::SwMeasureFuncTime("HW", 4, TestHw, eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
    eval.Compute();
    return 0;
}
