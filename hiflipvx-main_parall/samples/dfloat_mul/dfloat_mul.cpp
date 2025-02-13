#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;


#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using mul_hls_f32_t = float;
using mul_hls_f16_t = hls_half;
using mul_hvx_f32_t = dynfloat::std_f32;
using mul_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwMulHlsF32(mul_hls_f32_t& src1, mul_hls_f32_t& src2, mul_hls_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 * src2;
}

void
SwMulHlsF32() {
    mul_hls_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwMulHlsF32(A, B, C);
    std::cout << "HwMulHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwMulHlsF16(mul_hls_f16_t& src1, mul_hls_f16_t& src2, mul_hls_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 * src2;
}

void
SwMulHlsF16() {
    mul_hls_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwMulHlsF16(A, B, C);
    std::cout << "HwMulHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwMulHvxF32S4(mul_hvx_f32_t& src1, mul_hvx_f32_t& src2, mul_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF32S3(mul_hvx_f32_t& src1, mul_hvx_f32_t& src2, mul_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF32S2(mul_hvx_f32_t& src1, mul_hvx_f32_t& src2, mul_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF32S1(mul_hvx_f32_t& src1, mul_hvx_f32_t& src2, mul_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF32S0(mul_hvx_f32_t& src1, mul_hvx_f32_t& src2, mul_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwMulHvxF32() {
    mul_hvx_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwMulHvxF32S4(A, B, C);
    std::cout << "HwMulHvxF32S4: " << static_cast<float>(C) << "\n";
    HwMulHvxF32S3(A, B, C);
    std::cout << "HwMulHvxF32S3: " << static_cast<float>(C) << "\n";
    HwMulHvxF32S2(A, B, C);
    std::cout << "HwMulHvxF32S2: " << static_cast<float>(C) << "\n";
    HwMulHvxF32S1(A, B, C);
    std::cout << "HwMulHvxF32S1: " << static_cast<float>(C) << "\n";
    HwMulHvxF32S0(A, B, C);
    std::cout << "HwMulHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwMulHvxDspS4(mul_hvx_dsp_t& src1, mul_hvx_dsp_t& src2, mul_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxDspS3(mul_hvx_dsp_t& src1, mul_hvx_dsp_t& src2, mul_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxDspS2(mul_hvx_dsp_t& src1, mul_hvx_dsp_t& src2, mul_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwMulHvxDspS1(mul_hvx_dsp_t& src1, mul_hvx_dsp_t& src2, mul_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwMulHvxDspS0(mul_hvx_dsp_t& src1, mul_hvx_dsp_t& src2, mul_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwMulHvxDsp() {
    mul_hvx_dsp_t A(1.3f), B(2.3f), C(0.0f);
    HwMulHvxDspS4(A, B, C);
    std::cout << "HwMulHvxDspS4: " << static_cast<float>(C) << "\n";
    HwMulHvxDspS3(A, B, C);
    std::cout << "HwMulHvxDspS3: " << static_cast<float>(C) << "\n";
    HwMulHvxDspS2(A, B, C);
    std::cout << "HwMulHvxDspS2: " << static_cast<float>(C) << "\n";
    HwMulHvxDspS1(A, B, C);
    std::cout << "HwMulHvxDspS1: " << static_cast<float>(C) << "\n";
    HwMulHvxDspS0(A, B, C);
    std::cout << "HwMulHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using mul_hvx_f16_t = dynfloat::std_f16;

void
HwMulHvxF16S4(mul_hvx_f16_t& src1, mul_hvx_f16_t& src2, mul_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF16S3(mul_hvx_f16_t& src1, mul_hvx_f16_t& src2, mul_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF16S2(mul_hvx_f16_t& src1, mul_hvx_f16_t& src2, mul_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF16S1(mul_hvx_f16_t& src1, mul_hvx_f16_t& src2, mul_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwMulHvxF16S0(mul_hvx_f16_t& src1, mul_hvx_f16_t& src2, mul_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::mul<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwMulHvxF16() {
    mul_hvx_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwMulHvxF16S4(A, B, C);
    std::cout << "HwMulHvxF16S4: " << static_cast<float>(C) << "\n";
    HwMulHvxF16S3(A, B, C);
    std::cout << "HwMulHvxF16S3: " << static_cast<float>(C) << "\n";
    HwMulHvxF16S2(A, B, C);
    std::cout << "HwMulHvxF16S2: " << static_cast<float>(C) << "\n";
    HwMulHvxF16S1(A, B, C);
    std::cout << "HwMulHvxF16S1: " << static_cast<float>(C) << "\n";
    HwMulHvxF16S0(A, B, C);
    std::cout << "HwMulHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwMulHlsF32();
    SwMulHlsF16();
    SwMulHvxF32();
    SwMulHvxDsp();
    SwMulHvxF16();

    return 0;
}
