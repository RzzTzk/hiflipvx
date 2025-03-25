#include "../../include/sw_test/hvx_sw_test_core.h"
#include "../../include/hiflipvx/nn/hvx_nn_super.h"
#include "../../include/hiflipvx/convert/hvx_convert_reorder.h"
#include "../../include/sw_test/utils/hvx_sw_test_reorder.h"
/******************************************************************************************************************************************/

// configuration
using param_super = hvx::nn::SuperParam<

    //hvx::dfixed<float, 24>,     // src_type  / src_type_frac_bits
    //hvx::dfixed<float, 24>,     // dst_type  / dst_type_frac_bits
    //hvx::dfixed<float, 24>,     // wgts_type / wgts_type_frac_bits
    //hvx::dfixed<float, 24>,     // bias_type / bias_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // src_type  / src_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // dst_type  / dst_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // wgts_type / wgts_type_frac_bits
    hvx::dfixed<int16_t, 15>,  // bias_type / bias_type_frac_bits
    hvx::vector_param<1, 1>,   // batch     / batch_vec_size
    hvx::vector_param<8, 1>,  // src_rows  / src_rows_vec_size
    hvx::vector_param<8, 1>,  // src_cols  / src_cols_vec_size
    hvx::vector_param<8, 1>,  // chnls     / chnls_vec_size
    hvx::vector_param<1, 1>,  // fms       / fms_vec_size
    hvx::vector_param<2, 2>,   // knl_rows  / knl_rows_vec_size
    hvx::vector_param<2, 2>,   // knl_cols  / knl_cols_vec_size
    hvx::array2d_param<0, 0>,  // pad_rows
    hvx::array2d_param<0, 0>,  // pad_cols
    hvx::array2d_param<0, 0>,  // dil_rows  / dil_cols
    hvx::array2d_param<2, 2>,  // str_rows  / str_cols
    true,                      // buf_wgt
    true,                      // buf_bias
    hvx::overflow_e::kWrap,    // overflow_type
    hvx::underflow_e::kTrunc,  // underflow_type
    hvx::execution_e::kExact,  // exec_type
    hvx::util::layer_e::Pool,
    hvx::util::pooling_e::kAvg >;

auto
krnl_TestHw(param_super::src_port* src1, param_super::src2_port* src2, param_super::dst_port* dst1, param_super::dst2_port* dst2) noexcept -> void {

   // pool test
    // #pragma HLS INTERFACE m_axi port=src offset=slave bundle=gmem0 depth=32768
    // #pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem1 depth=8192
    #pragma HLS INTERFACE m_axi port=src1 offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=src2 offset=slave bundle=gmem1   
    #pragma HLS INTERFACE m_axi port=dst1 offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=dst2 offset=slave bundle=gmem3    
    #pragma HLS INTERFACE s_axilite port=src1
    #pragma HLS INTERFACE s_axilite port=src2
    #pragma HLS INTERFACE s_axilite port=dst1
    #pragma HLS INTERFACE s_axilite port=dst2
    #pragma HLS INTERFACE s_axilite port=return 
    hvx::nn::SuperTop<param_super, true, hvx::util::pooling_e::kAvg, hvx::util::layer_e::Pool>(src1, src2, nullptr, nullptr, dst1, dst2);
}