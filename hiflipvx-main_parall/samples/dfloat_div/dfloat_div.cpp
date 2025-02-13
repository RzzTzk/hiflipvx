#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;


#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using div_hls_f32_t = float;
using div_hls_f16_t = hls_half;
using div_hvx_f32_t = dynfloat::std_f32;
using div_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwDivHlsF32(div_hls_f32_t& src1, div_hls_f32_t& src2, div_hls_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 / src2;
}

void
SwDivHlsF32() {
    div_hls_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwDivHlsF32(A, B, C);
    std::cout << "HwDivHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwDivHlsF16(div_hls_f16_t& src1, div_hls_f16_t& src2, div_hls_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 / src2;
}

void
SwDivHlsF16() {
    div_hls_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwDivHlsF16(A, B, C);
    std::cout << "HwDivHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwDivHvxF32S4(div_hvx_f32_t& src1, div_hvx_f32_t& src2, div_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF32S3(div_hvx_f32_t& src1, div_hvx_f32_t& src2, div_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF32S2(div_hvx_f32_t& src1, div_hvx_f32_t& src2, div_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF32S1(div_hvx_f32_t& src1, div_hvx_f32_t& src2, div_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF32S0(div_hvx_f32_t& src1, div_hvx_f32_t& src2, div_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwDivHvxF32() {
    div_hvx_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwDivHvxF32S4(A, B, C);
    std::cout << "HwDivHvxF32S4: " << static_cast<float>(C) << "\n";
    HwDivHvxF32S3(A, B, C);
    std::cout << "HwDivHvxF32S3: " << static_cast<float>(C) << "\n";
    HwDivHvxF32S2(A, B, C);
    std::cout << "HwDivHvxF32S2: " << static_cast<float>(C) << "\n";
    HwDivHvxF32S1(A, B, C);
    std::cout << "HwDivHvxF32S1: " << static_cast<float>(C) << "\n";
    HwDivHvxF32S0(A, B, C);
    std::cout << "HwDivHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwDivHvxDspS4(div_hvx_dsp_t& src1, div_hvx_dsp_t& src2, div_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxDspS3(div_hvx_dsp_t& src1, div_hvx_dsp_t& src2, div_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxDspS2(div_hvx_dsp_t& src1, div_hvx_dsp_t& src2, div_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwDivHvxDspS1(div_hvx_dsp_t& src1, div_hvx_dsp_t& src2, div_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwDivHvxDspS0(div_hvx_dsp_t& src1, div_hvx_dsp_t& src2, div_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwDivHvxDsp() {
    div_hvx_dsp_t A(1.3f), B(2.3f), C(0.0f);
    HwDivHvxDspS4(A, B, C);
    std::cout << "HwDivHvxDspS4: " << static_cast<float>(C) << "\n";
    HwDivHvxDspS3(A, B, C);
    std::cout << "HwDivHvxDspS3: " << static_cast<float>(C) << "\n";
    HwDivHvxDspS2(A, B, C);
    std::cout << "HwDivHvxDspS2: " << static_cast<float>(C) << "\n";
    HwDivHvxDspS1(A, B, C);
    std::cout << "HwDivHvxDspS1: " << static_cast<float>(C) << "\n";
    HwDivHvxDspS0(A, B, C);
    std::cout << "HwDivHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using div_hvx_f16_t = dynfloat::std_f16;

void
HwDivHvxF16S4(div_hvx_f16_t& src1, div_hvx_f16_t& src2, div_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF16S3(div_hvx_f16_t& src1, div_hvx_f16_t& src2, div_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF16S2(div_hvx_f16_t& src1, div_hvx_f16_t& src2, div_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF16S1(div_hvx_f16_t& src1, div_hvx_f16_t& src2, div_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwDivHvxF16S0(div_hvx_f16_t& src1, div_hvx_f16_t& src2, div_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::div<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwDivHvxF16() {
    div_hvx_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwDivHvxF16S4(A, B, C);
    std::cout << "HwDivHvxF16S4: " << static_cast<float>(C) << "\n";
    HwDivHvxF16S3(A, B, C);
    std::cout << "HwDivHvxF16S3: " << static_cast<float>(C) << "\n";
    HwDivHvxF16S2(A, B, C);
    std::cout << "HwDivHvxF16S2: " << static_cast<float>(C) << "\n";
    HwDivHvxF16S1(A, B, C);
    std::cout << "HwDivHvxF16S1: " << static_cast<float>(C) << "\n";
    HwDivHvxF16S0(A, B, C);
    std::cout << "HwDivHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwDivHlsF32();
    SwDivHlsF16();
    SwDivHvxF32();
    SwDivHvxDsp();
    SwDivHvxF16();

    return 0;
}
