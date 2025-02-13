#include "../../include/sw_test/hvx_sw_test_core.h"

using dense_param0 = hvx::dense_param<hvx::dfixed<int16_t, 15>,
                                      hvx::dfixed<int16_t, 15>,
                                      hvx::dfixed<int16_t, 15>,
                                      hvx::dfixed<int16_t, 15>,
                                      hvx::vector_param<2, 1>,
                                      hvx::vector_param<256, 2>,
                                      hvx::vector_param<256, 2>,
                                      true,
                                      true,
                                      hvx::overflow_e::kWrap,
                                      hvx::underflow_e::kTrunc,
                                      hvx::execution_e::kExact>;

using dense_param1 = hvx::dense_param<hvx::dfixed<uint16_t, 16>,
                                      hvx::dfixed<uint16_t, 16>,
                                      hvx::dfixed<uint16_t, 16>,
                                      hvx::dfixed<uint16_t, 16>,
                                      hvx::vector_param<2, 1>,
                                      hvx::vector_param<256, 2>,
                                      hvx::vector_param<256, 2>,
                                      true,
                                      true,
                                      hvx::overflow_e::kWrap,
                                      hvx::underflow_e::kTrunc,
                                      hvx::execution_e::kExact>;

using dense_param2 = hvx::dense_param<hvx::dfixed<int32_t, 31>,
                                      hvx::dfixed<int32_t, 31>,
                                      hvx::dfixed<int32_t, 31>,
                                      hvx::dfixed<int32_t, 31>,
                                      hvx::vector_param<2, 1>,
                                      hvx::vector_param<256, 2>,
                                      hvx::vector_param<256, 2>,
                                      true,
                                      true,
                                      hvx::overflow_e::kWrap,
                                      hvx::underflow_e::kTrunc,
                                      hvx::execution_e::kExact>;

/******************************************************************************************************************************************/

auto
SynthDense0(dense_param0::src_port* src, dense_param0::wgts_port* wgts, dense_param0::bias_port* bias, dense_param0::dst_port* dst) noexcept
    -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    hvx::HwDense<dense_param0>(src, wgts, bias, dst);
}

auto
SynthDense1(dense_param1::src_port* src, dense_param1::wgts_port* wgts, dense_param1::bias_port* bias, dense_param1::dst_port* dst) noexcept
    -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    hvx::HwDense<dense_param1>(src, wgts, bias, dst);
}

auto
SynthDense2(dense_param2::src_port* src, dense_param2::wgts_port* wgts, dense_param2::bias_port* bias, dense_param2::dst_port* dst) noexcept
    -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    hvx::HwDense<dense_param2>(src, wgts, bias, dst);
}

/******************************************************************************************************************************************/

// testbench
auto
main() -> int {
    return 0;
}