/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ï¿½AS ISï¿½, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_hw_test_new_reshape.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"
#include "../../include/hiflipvx/convert/hvx_convert_reorder.h"
#include "../../include/sw_test/utils/hvx_sw_test_reorder.h"

/******************************************************************************************************************************************/

// configuration
using param = hvx::convert::ReorderParam <
    
    // dynfloat::dfloat<8, 17>,
    hvx::dfixed<int16_t, 15>,  // src_type  / src_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // dst_type  / dst_type_frac_bits
    hvx::vector_param<1, 1>,   // batch     / batch_vec_size
    hvx::vector_param<16, 2>,  // src_rows  / src_rows_vec_size
    hvx::vector_param<16, 2>,  // src_cols  / src_cols_vec_size
    hvx::vector_param<16, 2>,  // chnls     / chnls_vec_size
    hvx::vector_param<16, 2>,  // fms       / fms_vec_size
    hvx::util::reorder_e::Negative>; 
using stream = hvx::stream_param<typename param::dst_type, typename param::dst_dim, hvx::axis_e::kEof>;

// HW accelerator
void
TestHw(param::src_port* src, stream::port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst);
    auto dst_fifo = hvx::HwFifo<param::dst_vec, param::dst_dim::vec_elms, 2>();
    hvx::convert::HwReorderTop<param, hvx::util::reorder_e::Negative>(src, dst_fifo.data);
    hvx::HwHvxToStream<stream>(dst_fifo.data, *dst);
}

// testbench
auto
main() -> int {
    auto src_hw = hvx::SwCreateArrayOfVector<param::src_dim, param::src_type>();
    auto src_sw = hvx::SwCreateArray<param::src_dim, param::src_type>();
    hvx::array1d<param::dst_type, param::dst_dim::elms> dst_sw{};
    hvx::array1d<stream::port, stream::dim::vec_elms> dst_hw_hls{};
    hvx::array1d<param::dst_port, param::dst_dim::vec_elms> dst_hw_hvx{};
    hvx::sw::SwReorder<param, hvx::util::reorder_e::Negative>(src_sw.data, dst_sw.data);
    TestHw(src_hw.data, dst_hw_hls.data);
    hvx::HwStreamToHvx<stream>(*dst_hw_hls.data, dst_hw_hvx.data);
    hvx::SwCompareArrayOfVector<param::src_dim, param::src_type, param::dst_dim, param::dst_type>(dst_sw, dst_hw_hvx, "  ");
    return 0;
}
