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
 * @file    hvx_hw_test_conv.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"
#include "../../include/hiflipvx/nn/hvx_nn_super.h"
#include "../../include/hiflipvx/convert/hvx_convert_reorder.h"
#include "../../include/sw_test/utils/hvx_sw_test_reorder.h"
#include "hvx_hw_test_super.cpp"
/******************************************************************************************************************************************/


// testbench
auto
main() -> int {

    // Conv test

     //hvx::conv_eval<param_super, hvx::eval_param<true, 4, 4, 4, stream::port, stream::flags>> eval(0.75f, 0.25f);
     //TestHw(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
     //eval.Compute();
     //return 0;


    // Depth test

    //  hvx::depthwise_eval<param_super, hvx::eval_param<true, 4, 4, 4, param_super::dst_port, 0>> eval(0.75f, 0.25f);
    //  TestHw(eval.GetSrcHw(), eval.GetWgtsHw(), eval.GetBiasHw(), eval.GetDstHw());
    // //  eval.Compute();
    //  return 0;

    // //pool test

    hvx::pool_avg_eval<param_super, hvx::eval_param<true, 4, 4, 4, param_super::dst_port, 0>> eval;
    TestHw(eval.GetSrcHw(), eval.GetDstHw());
    eval.Compute();
    return 0;
}
