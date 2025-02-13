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
    VitisParam vitis(false,                                      // do C simulation
                     true,                                       // do C synthesis
                     true,                                       // do RTL synthesis
                     "xczu7ev-ffvc1156-2-e",                     // FPGA part number
                     "5",                                        // target clock period
                     "G:/xilinx/Vitis_HLS/2024.2/bin/vitis_hls", // vitis path
                     "F:/Master_thesis/Git/my_project/hiflipvx-main_parall/" // HiFlipVX path
    );

    //
    std::string csyn_res_file_name = "hvx_csyn.csv";
    std::string syn_res_file_name  = "hvx_syn.csv";
    constexpr int64_t node_num     = 1;
    constexpr int64_t trhead_num   = 1;
    //
    std::array<std::array<const char*, 3>, node_num> names{
        {
         {"SynthSuper", "samples/hvx_hw_test_super/hvx_hw_test_super.cpp", "TestHw"},
         }
    };
    vitis.Compute<node_num, trhead_num>(names, csyn_res_file_name, syn_res_file_name);

    return 0;
}