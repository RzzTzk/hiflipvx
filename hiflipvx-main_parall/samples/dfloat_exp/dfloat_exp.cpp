#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using exp_hls_f16_t = half;
#else
using exp_hls_f16_t = float;
#endif
using exp_hls_f32_t = float;
using exp_hvx_f32_t = dynfloat::std_f32;
using exp_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwExpHlsF32(exp_hls_f32_t& src1, exp_hls_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::exp(src1);
#else
    dst = exp(src1);
#endif
}

void
SwExpHlsF32() {
    exp_hls_f32_t A(1.3f), C(0.0f);
    HwExpHlsF32(A, C);
    std::cout << "HwExpHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwExpHlsF16(exp_hls_f16_t& src1, exp_hls_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::half_exp(src1);
#else
    dst = exp(src1);
#endif
}

void
SwExpHlsF16() {
    exp_hls_f16_t A(1.3f), C(0.0f);
    HwExpHlsF16(A, C);
    std::cout << "HwExpHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwExpHvxF32S4(exp_hvx_f32_t& src1, exp_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwExpHvxF32S3(exp_hvx_f32_t& src1, exp_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwExpHvxF32S2(exp_hvx_f32_t& src1, exp_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwExpHvxF32S1(exp_hvx_f32_t& src1, exp_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwExpHvxF32S0(exp_hvx_f32_t& src1, exp_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwExpHvxF32() {
    exp_hvx_f32_t A(1.3f), C(0.0f);
    HwExpHvxF32S4(A, C);
    std::cout << "HwExpHvxF32S4: " << static_cast<float>(C) << "\n";
    HwExpHvxF32S3(A, C);
    std::cout << "HwExpHvxF32S3: " << static_cast<float>(C) << "\n";
    HwExpHvxF32S2(A, C);
    std::cout << "HwExpHvxF32S2: " << static_cast<float>(C) << "\n";
    HwExpHvxF32S1(A, C);
    std::cout << "HwExpHvxF32S1: " << static_cast<float>(C) << "\n";
    HwExpHvxF32S0(A, C);
    std::cout << "HwExpHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwExpHvxDspS4(exp_hvx_dsp_t& src1, exp_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwExpHvxDspS3(exp_hvx_dsp_t& src1, exp_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwExpHvxDspS2(exp_hvx_dsp_t& src1, exp_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwExpHvxDspS1(exp_hvx_dsp_t& src1, exp_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwExpHvxDspS0(exp_hvx_dsp_t& src1, exp_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwExpHvxDsp() {
    exp_hvx_dsp_t A(1.3f), C(0.0f);
    HwExpHvxDspS4(A, C);
    std::cout << "HwExpHvxDspS4: " << static_cast<float>(C) << "\n";
    HwExpHvxDspS3(A, C);
    std::cout << "HwExpHvxDspS3: " << static_cast<float>(C) << "\n";
    HwExpHvxDspS2(A, C);
    std::cout << "HwExpHvxDspS2: " << static_cast<float>(C) << "\n";
    HwExpHvxDspS1(A, C);
    std::cout << "HwExpHvxDspS1: " << static_cast<float>(C) << "\n";
    HwExpHvxDspS0(A, C);
    std::cout << "HwExpHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using exp_hvx_f16_t = dynfloat::std_f16;

void
HwExpHvxF16S4(exp_hvx_f16_t& src1, exp_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwExpHvxF16S3(exp_hvx_f16_t& src1, exp_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwExpHvxF16S2(exp_hvx_f16_t& src1, exp_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwExpHvxF16S1(exp_hvx_f16_t& src1, exp_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwExpHvxF16S0(exp_hvx_f16_t& src1, exp_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::exp<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwExpHvxF16() {
    exp_hvx_f16_t A(1.3f), C(0.0f);
    HwExpHvxF16S4(A, C);
    std::cout << "HwExpHvxF16S4: " << static_cast<float>(C) << "\n";
    HwExpHvxF16S3(A, C);
    std::cout << "HwExpHvxF16S3: " << static_cast<float>(C) << "\n";
    HwExpHvxF16S2(A, C);
    std::cout << "HwExpHvxF16S2: " << static_cast<float>(C) << "\n";
    HwExpHvxF16S1(A, C);
    std::cout << "HwExpHvxF16S1: " << static_cast<float>(C) << "\n";
    HwExpHvxF16S0(A, C);
    std::cout << "HwExpHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwExpHlsF32();
    SwExpHlsF16();
    SwExpHvxF32();
    SwExpHvxDsp();
    SwExpHvxF16();

    return 0;
}
