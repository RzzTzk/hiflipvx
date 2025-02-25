/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
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
 * @file    hvx_nn_conv.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_NN_HBM_H_
#define HVX_NN_HBM_H_

#include "impl/hvx_nn_conv_dfixed.h"
#include "impl/hvx_nn_conv_dfloat.h"
// #include <hls_stream.h>
// #include <ap_int.h>

namespace hvx {
namespace nn {
/******************************************************************************************************************************************/

void hbm_example(ap_uint<512>* hbm_in, ap_uint<512>* hbm_out) {
#pragma HLS INTERFACE m_axi port = hbm_in offset = slave bundle = HBM
#pragma HLS INTERFACE m_axi port = hbm_out offset = slave bundle = HBM
#pragma HLS INTERFACE s_axilite port = return bundle = control

    for (int i = 0; i < SIZE; i++) {
        hbm_out[i] = hbm_in[i]; // simple copy operation to/from HBM
    }
}

/******************************************************************************************************************************************/
} // namespace nn
} // namespace hvx

#endif 
