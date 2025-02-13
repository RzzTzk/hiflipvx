#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using sqrt_hls_f32_t = float;
using sqrt_hls_f16_t = hls_half;
using sqrt_hvx_f32_t = dynfloat::std_f32;
using sqrt_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwSqrtHlsF32(sqrt_hls_f32_t& src1, sqrt_hls_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::sqrt(src1);
#else
    dst = sqrt(src1);
#endif
}

void
SwSqrtHlsF32() {
    sqrt_hls_f32_t A(1.3f), C(0.0f);
    HwSqrtHlsF32(A, C);
    std::cout << "HwSqrtHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwSqrtHlsF16(sqrt_hls_f16_t& src1, sqrt_hls_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::half_sqrt(src1);
#else
    dst = sqrt(src1);
#endif
}

void
SwSqrtHlsF16() {
    sqrt_hls_f16_t A(1.3f), C(0.0f);
    HwSqrtHlsF16(A, C);
    std::cout << "HwSqrtHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwSqrtHvxF32S4(sqrt_hvx_f32_t& src1, sqrt_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxF32S3(sqrt_hvx_f32_t& src1, sqrt_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxF32S2(sqrt_hvx_f32_t& src1, sqrt_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwSqrtHvxF32S1(sqrt_hvx_f32_t& src1, sqrt_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwSqrtHvxF32S0(sqrt_hvx_f32_t& src1, sqrt_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwSqrtHvxF32() {
    sqrt_hvx_f32_t A(1.3f), C(0.0f);
    HwSqrtHvxF32S4(A, C);
    std::cout << "HwSqrtHvxF32S4: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF32S3(A, C);
    std::cout << "HwSqrtHvxF32S3: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF32S2(A, C);
    std::cout << "HwSqrtHvxF32S2: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF32S1(A, C);
    std::cout << "HwSqrtHvxF32S1: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF32S0(A, C);
    std::cout << "HwSqrtHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwSqrtHvxDspS4(sqrt_hvx_dsp_t& src1, sqrt_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxDspS3(sqrt_hvx_dsp_t& src1, sqrt_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxDspS2(sqrt_hvx_dsp_t& src1, sqrt_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwSqrtHvxDspS1(sqrt_hvx_dsp_t& src1, sqrt_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwSqrtHvxDspS0(sqrt_hvx_dsp_t& src1, sqrt_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwSqrtHvxDsp() {
    sqrt_hvx_dsp_t A(1.3f), C(0.0f);
    HwSqrtHvxDspS4(A, C);
    std::cout << "HwSqrtHvxDspS4: " << static_cast<float>(C) << "\n";
    HwSqrtHvxDspS3(A, C);
    std::cout << "HwSqrtHvxDspS3: " << static_cast<float>(C) << "\n";
    HwSqrtHvxDspS2(A, C);
    std::cout << "HwSqrtHvxDspS2: " << static_cast<float>(C) << "\n";
    HwSqrtHvxDspS1(A, C);
    std::cout << "HwSqrtHvxDspS1: " << static_cast<float>(C) << "\n";
    HwSqrtHvxDspS0(A, C);
    std::cout << "HwSqrtHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using sqrt_hvx_f16_t = dynfloat::std_f16;

void
HwSqrtHvxF16S4(sqrt_hvx_f16_t& src1, sqrt_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxF16S3(sqrt_hvx_f16_t& src1, sqrt_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwSqrtHvxF16S2(sqrt_hvx_f16_t& src1, sqrt_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwSqrtHvxF16S1(sqrt_hvx_f16_t& src1, sqrt_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwSqrtHvxF16S0(sqrt_hvx_f16_t& src1, sqrt_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::sqrt<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwSqrtHvxF16() {
    sqrt_hvx_f16_t A(1.3f), C(0.0f);
    HwSqrtHvxF16S4(A, C);
    std::cout << "HwSqrtHvxF16S4: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF16S3(A, C);
    std::cout << "HwSqrtHvxF16S3: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF16S2(A, C);
    std::cout << "HwSqrtHvxF16S2: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF16S1(A, C);
    std::cout << "HwSqrtHvxF16S1: " << static_cast<float>(C) << "\n";
    HwSqrtHvxF16S0(A, C);
    std::cout << "HwSqrtHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwSqrtHlsF32();
    SwSqrtHlsF16();
    SwSqrtHvxF32();
    SwSqrtHvxDsp();
    SwSqrtHvxF16();

    return 0;
}
