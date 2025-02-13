/**
 * Licence: GNU GPLv3 \n
 * You may copy, distribute and modify the software as long as you track
 * changes/dates in source files. Any modifications to or software
 * including (via compiler) GPL-licensed code must also be made available
 * under the GPL along with build & install instructions.
 *
 * @file    hvx_hw_test_conv_fixed.cpp
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#include "../../include/sw_test/hvx_sw_test_core.h"
#include "../hvx_vitis.h"

/******************************************************************************************************************************************/

// testbench
auto
main() -> int {
    // main configuration
    VitisParam vitis(false,                                       // do C simulation
                     true,                                       // do C synthesis
                     true,                                       // do RTL synthesis
                     "xczu7ev-ffvc1156-2-e",                     // FPGA part number
                     "5",                                        // target clock period
                     "C:/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls", // vitis path
                     "D:/DatenTUD/PostDoc/Repos/hiflipvx/"       // HiFlipVX path
    );

    //
    std::string csyn_res_file_name = "hvx_csyn.csv";
    std::string syn_res_file_name  = "hvx_syn.csv";
    constexpr int64_t node_num     = 29;
    constexpr int64_t trhead_num   = 15;
    //
    std::array<std::array<const char*, 3>, node_num> names{
        {
         {"SynthAbs", "samples/hvx_hw_test_abs/hvx_hw_test_abs.cpp", "TestHw"},
         {"SynthAdd", "samples/hvx_hw_test_add/hvx_hw_test_add.cpp", "TestHw"},
         {"SynthAddConst", "samples/hvx_hw_test_addconst/hvx_hw_test_addconst.cpp", "TestHw"},
         {"SynthClip", "samples/hvx_hw_test_clip/hvx_hw_test_clip.cpp", "TestHw"},
         {"SynthConcat", "samples/hvx_hw_test_concat/hvx_hw_test_concat.cpp", "TestHw"},
         {"SynthConv", "samples/hvx_hw_test_conv/hvx_hw_test_conv.cpp", "TestHw"},
         {"SynthDense", "samples/hvx_hw_test_dense/hvx_hw_test_dense.cpp", "TestHw"},
         {"SynthDepthwise", "samples/hvx_hw_test_depthwise/hvx_hw_test_depthwise.cpp", "TestHw"},
         {"SynthLayerNorm", "samples/hvx_hw_test_layernorm/hvx_hw_test_layernorm.cpp", "TestHw"},
         {"SynthMax", "samples/hvx_hw_test_max/hvx_hw_test_max.cpp", "TestHw"},
         {"SynthMaxConst", "samples/hvx_hw_test_maxconst/hvx_hw_test_maxconst.cpp", "TestHw"},
         {"SynthMin", "samples/hvx_hw_test_min/hvx_hw_test_min.cpp", "TestHw"},
         {"SynthMinConst", "samples/hvx_hw_test_minconst/hvx_hw_test_minconst.cpp", "TestHw"},
         {"SynthMul", "samples/hvx_hw_test_mul/hvx_hw_test_mul.cpp", "TestHw"},
         {"SynthMulConst", "samples/hvx_hw_test_mulconst/hvx_hw_test_mulconst.cpp", "TestHw"},
         {"SynthMulticast", "samples/hvx_hw_test_multicast/hvx_hw_test_multicast.cpp", "TestHw"},
         {"SynthPoolAvg", "samples/hvx_hw_test_pool_avg/hvx_hw_test_pool_avg.cpp", "TestHw"},
         {"SynthPoolMax", "samples/hvx_hw_test_pool_max/hvx_hw_test_pool_max.cpp", "TestHw"},
         {"SynthReduceMax", "samples/hvx_hw_test_reduce_max/hvx_hw_test_reduce_max.cpp", "TestHw"},
         {"SynthReduceMean", "samples/hvx_hw_test_reduce_mean/hvx_hw_test_reduce_mean.cpp", "TestHw"},
         {"SynthReduceMin", "samples/hvx_hw_test_reduce_min/hvx_hw_test_reduce_min.cpp", "TestHw"},
         {"SynthReduceSum", "samples/hvx_hw_test_reduce_sum/hvx_hw_test_reduce_sum.cpp", "TestHw"},
         {"SynthReshape", "samples/hvx_hw_test_reshape/hvx_hw_test_reshape.cpp", "TestHw"},
         {"SynthSigmoid", "samples/hvx_hw_test_sigmoid/hvx_hw_test_sigmoid.cpp", "TestHw"},
         {"SynthSoftmax", "samples/hvx_hw_test_softmax/hvx_hw_test_softmax.cpp", "TestHw"},
         {"SynthSplit", "samples/hvx_hw_test_split/hvx_hw_test_split.cpp", "TestHw"},
         {"SynthSub", "samples/hvx_hw_test_sub/hvx_hw_test_sub.cpp", "TestHw"},
         {"SynthTanh", "samples/hvx_hw_test_tanh/hvx_hw_test_tanh.cpp", "TestHw"},
         {"SynthTranspose", "samples/hvx_hw_test_transpose/hvx_hw_test_transpose.cpp", "TestHw"},
        }
    };
    vitis.Compute<node_num, trhead_num>(names, csyn_res_file_name, syn_res_file_name);

    return 0;
}

// std::array<std::array<const char*, 3>, node_num> mul_names{
//     {
//      {"HwMulHlsF32", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHlsF32"},
//      {"HwMulHlsF16", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHlsF16"},
//      {"HwMulHvxF32S4", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF32S4"},
//      {"HwMulHvxF32S3", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF32S3"},
//      {"HwMulHvxF32S2", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF32S2"},
//      {"HwMulHvxF32S1", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF32S1"},
//      {"HwMulHvxF32S0", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF32S0"},
//      {"HwMulHvxDspS4", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxDspS4"},
//      {"HwMulHvxDspS3", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxDspS3"},
//      {"HwMulHvxDspS2", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxDspS2"},
//      {"HwMulHvxDspS1", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxDspS1"},
//      {"HwMulHvxDspS0", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxDspS0"},
//      {"HwMulHvxF16S4", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF16S4"},
//      {"HwMulHvxF16S3", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF16S3"},
//      {"HwMulHvxF16S2", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF16S2"},
//      {"HwMulHvxF16S1", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF16S1"},
//      {"HwMulHvxF16S0", "samples/dfloat_mul/dfloat_mul.cpp", "HwMulHvxF16S0"},
//      {"HwAddHlsF32", "samples/dfloat_add/dfloat_add.cpp", "HwAddHlsF32"},
//      {"HwAddHlsF16", "samples/dfloat_add/dfloat_add.cpp", "HwAddHlsF16"},
//      {"HwAddHvxF32S4", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF32S4"},
//      {"HwAddHvxF32S3", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF32S3"},
//      {"HwAddHvxF32S2", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF32S2"},
//      {"HwAddHvxF32S1", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF32S1"},
//      {"HwAddHvxF32S0", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF32S0"},
//      {"HwAddHvxDspS4", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxDspS4"},
//      {"HwAddHvxDspS3", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxDspS3"},
//      {"HwAddHvxDspS2", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxDspS2"},
//      {"HwAddHvxDspS1", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxDspS1"},
//      {"HwAddHvxDspS0", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxDspS0"},
//      {"HwAddHvxF16S4", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF16S4"},
//      {"HwAddHvxF16S3", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF16S3"},
//      {"HwAddHvxF16S2", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF16S2"},
//      {"HwAddHvxF16S1", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF16S1"},
//      {"HwAddHvxF16S0", "samples/dfloat_add/dfloat_add.cpp", "HwAddHvxF16S0"},
//      {"HwDivHlsF32", "samples/dfloat_div/dfloat_div.cpp", "HwDivHlsF32"},
//      {"HwDivHlsF16", "samples/dfloat_div/dfloat_div.cpp", "HwDivHlsF16"},
//      {"HwDivHvxF32S4", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF32S4"},
//      {"HwDivHvxF32S3", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF32S3"},
//      {"HwDivHvxF32S2", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF32S2"},
//      {"HwDivHvxF32S1", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF32S1"},
//      {"HwDivHvxF32S0", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF32S0"},
//      {"HwDivHvxDspS4", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxDspS4"},
//      {"HwDivHvxDspS3", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxDspS3"},
//      {"HwDivHvxDspS2", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxDspS2"},
//      {"HwDivHvxDspS1", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxDspS1"},
//      {"HwDivHvxDspS0", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxDspS0"},
//      {"HwDivHvxF16S4", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF16S4"},
//      {"HwDivHvxF16S3", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF16S3"},
//      {"HwDivHvxF16S2", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF16S2"},
//      {"HwDivHvxF16S1", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF16S1"},
//      {"HwDivHvxF16S0", "samples/dfloat_div/dfloat_div.cpp", "HwDivHvxF16S0"},
//      {"HwSqrtHlsF32", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHlsF32"},
//      {"HwSqrtHlsF16", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHlsF16"},
//      {"HwSqrtHvxF32S4", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF32S4"},
//      {"HwSqrtHvxF32S3", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF32S3"},
//      {"HwSqrtHvxF32S2", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF32S2"},
//      {"HwSqrtHvxF32S1", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF32S1"},
//      {"HwSqrtHvxF32S0", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF32S0"},
//      {"HwSqrtHvxDspS4", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxDspS4"},
//      {"HwSqrtHvxDspS3", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxDspS3"},
//      {"HwSqrtHvxDspS2", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxDspS2"},
//      {"HwSqrtHvxDspS1", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxDspS1"},
//      {"HwSqrtHvxDspS0", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxDspS0"},
//      {"HwSqrtHvxF16S4", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF16S4"},
//      {"HwSqrtHvxF16S3", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF16S3"},
//      {"HwSqrtHvxF16S2", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF16S2"},
//      {"HwSqrtHvxF16S1", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF16S1"},
//      {"HwSqrtHvxF16S0", "samples/dfloat_sqrt/dfloat_sqrt.cpp", "HwSqrtHvxF16S0"},
//      {"HwExpHlsF32", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHlsF32"},
//      {"HwExpHlsF16", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHlsF16"},
//      {"HwExpHvxF32S4", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF32S4"},
//      {"HwExpHvxF32S3", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF32S3"},
//      {"HwExpHvxF32S2", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF32S2"},
//      {"HwExpHvxF32S1", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF32S1"},
//      {"HwExpHvxF32S0", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF32S0"},
//      {"HwExpHvxDspS4", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxDspS4"},
//      {"HwExpHvxDspS3", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxDspS3"},
//      {"HwExpHvxDspS2", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxDspS2"},
//      {"HwExpHvxDspS1", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxDspS1"},
//      {"HwExpHvxDspS0", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxDspS0"},
//      {"HwExpHvxF16S4", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF16S4"},
//      {"HwExpHvxF16S3", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF16S3"},
//      {"HwExpHvxF16S2", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF16S2"},
//      {"HwExpHvxF16S1", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF16S1"},
//      {"HwExpHvxF16S0", "samples/dfloat_exp/dfloat_exp.cpp", "HwExpHvxF16S0"},
//      {"HwTanhHlsF32", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHlsF32"},
//      {"HwTanhHlsF16", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHlsF16"},
//      {"HwTanhHvxF32S4", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF32S4"},
//      {"HwTanhHvxF32S3", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF32S3"},
//      {"HwTanhHvxF32S2", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF32S2"},
//      {"HwTanhHvxF32S1", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF32S1"},
//      {"HwTanhHvxF32S0", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF32S0"},
//      {"HwTanhHvxDspS4", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxDspS4"},
//      {"HwTanhHvxDspS3", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxDspS3"},
//      {"HwTanhHvxDspS2", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxDspS2"},
//      {"HwTanhHvxDspS1", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxDspS1"},
//      {"HwTanhHvxDspS0", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxDspS0"},
//      {"HwTanhHvxF16S4", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF16S4"},
//      {"HwTanhHvxF16S3", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF16S3"},
//      {"HwTanhHvxF16S2", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF16S2"},
//      {"HwTanhHvxF16S1", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF16S1"},
//      {"HwTanhHvxF16S0", "samples/dfloat_tanh/dfloat_tanh.cpp", "HwTanhHvxF16S0"},
//      {"HwLog2HlsF32", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HlsF32"},
//      {"HwLog2HlsF16", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HlsF16"},
//      {"HwLog2HvxF32S4", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF32S4"},
//      {"HwLog2HvxF32S3", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF32S3"},
//      {"HwLog2HvxF32S2", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF32S2"},
//      {"HwLog2HvxF32S1", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF32S1"},
//      {"HwLog2HvxF32S0", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF32S0"},
//      {"HwLog2HvxDspS4", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxDspS4"},
//      {"HwLog2HvxDspS3", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxDspS3"},
//      {"HwLog2HvxDspS2", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxDspS2"},
//      {"HwLog2HvxDspS1", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxDspS1"},
//      {"HwLog2HvxDspS0", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxDspS0"},
//      {"HwLog2HvxF16S4", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF16S4"},
//      {"HwLog2HvxF16S3", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF16S3"},
//      {"HwLog2HvxF16S2", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF16S2"},
//      {"HwLog2HvxF16S1", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF16S1"},
//      {"HwLog2HvxF16S0", "samples/dfloat_log2/dfloat_log2.cpp", "HwLog2HvxF16S0"},
//      {"HwLnHlsF32", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHlsF32"},
//      {"HwLnHlsF16", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHlsF16"},
//      {"HwLnHvxF32S4", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF32S4"},
//      {"HwLnHvxF32S3", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF32S3"},
//      {"HwLnHvxF32S2", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF32S2"},
//      {"HwLnHvxF32S1", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF32S1"},
//      {"HwLnHvxF32S0", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF32S0"},
//      {"HwLnHvxDspS4", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxDspS4"},
//      {"HwLnHvxDspS3", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxDspS3"},
//      {"HwLnHvxDspS2", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxDspS2"},
//      {"HwLnHvxDspS1", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxDspS1"},
//      {"HwLnHvxDspS0", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxDspS0"},
//      {"HwLnHvxF16S4", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF16S4"},
//      {"HwLnHvxF16S3", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF16S3"},
//      {"HwLnHvxF16S2", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF16S2"},
//      {"HwLnHvxF16S1", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF16S1"},
//      {"HwLnHvxF16S0", "samples/dfloat_ln/dfloat_ln.cpp", "HwLnHvxF16S0"},
//     }
// };