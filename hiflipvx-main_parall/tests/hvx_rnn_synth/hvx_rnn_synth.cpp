#include "../../include/sw_test/hvx_sw_test_core.h"
#include "../hvx_vitis.h"

// testbench
auto
main() -> int {
    // main configuration
    VitisParam vitis(false,                                      // do C simulation
                     true,                                       // do C synthesis
                     true,                                      // do RTL synthesis
                     "xczu7ev-ffvc1156-2-e",                     // FPGA part number
                     "5",                                        // target clock period
                     "C:/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls", // vitis path
                     "D:/DatenTUD/PostDoc/Repos/hiflipvx/"       // HiFlipVX path
    );

    //
    std::string csyn_res_file_name = "hvx_csyn.csv";
    std::string syn_res_file_name  = "hvx_syn.csv";
    constexpr int64_t node_num     = 6;
    constexpr int64_t trhead_num   = 6;

    //
    std::array<std::array<const char*, 3>, node_num> names{
        {
         {"SynthRnnReader", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnReader"},
         {"SynthRnnMatMulWx", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnMatMulWx"},
         {"SynthRnnMatMulWh", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnMatMulWh"},
         {"SynthRnnAddHw", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnAddHw"},
         {"SynthRnnTanhHw", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnTanhHw"},
         {"SynthRnnWriter", "tests/hvx_rnn_test/hvx_rnn_test.cpp", "RnnWriter"},
         }
    };
    vitis.Compute<node_num, trhead_num>(names, csyn_res_file_name, syn_res_file_name);

    return 0;
}