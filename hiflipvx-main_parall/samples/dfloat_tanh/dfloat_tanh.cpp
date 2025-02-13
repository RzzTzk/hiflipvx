#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using tanh_hls_f32_t = float;
using tanh_hls_f16_t = hls_half;
using tanh_hvx_f32_t = dynfloat::std_f32;
using tanh_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwTanhHlsF32(tanh_hls_f32_t& src1, tanh_hls_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::tanh(src1);
#else
    dst = tanh(src1);
#endif
}

void
SwTanhHlsF32() {
    tanh_hls_f32_t A(1.3f), C(0.0f);
    HwTanhHlsF32(A, C);
    std::cout << "HwTanhHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwTanhHlsF16(tanh_hls_f16_t& src1, tanh_hls_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::half_tanh(src1);
#else
    dst = tanh(src1);
#endif
}

void
SwTanhHlsF16() {
    tanh_hls_f16_t A(1.3f), C(0.0f);
    HwTanhHlsF16(A, C);
    std::cout << "HwTanhHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwTanhHvxF32S4(tanh_hvx_f32_t& src1, tanh_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwTanhHvxF32S3(tanh_hvx_f32_t& src1, tanh_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwTanhHvxF32S2(tanh_hvx_f32_t& src1, tanh_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwTanhHvxF32S1(tanh_hvx_f32_t& src1, tanh_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwTanhHvxF32S0(tanh_hvx_f32_t& src1, tanh_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwTanhHvxF32() {
    tanh_hvx_f32_t A(1.3f), C(0.0f);
    HwTanhHvxF32S4(A, C);
    std::cout << "HwTanhHvxF32S4: " << static_cast<float>(C) << "\n";
    HwTanhHvxF32S3(A, C);
    std::cout << "HwTanhHvxF32S3: " << static_cast<float>(C) << "\n";
    HwTanhHvxF32S2(A, C);
    std::cout << "HwTanhHvxF32S2: " << static_cast<float>(C) << "\n";
    HwTanhHvxF32S1(A, C);
    std::cout << "HwTanhHvxF32S1: " << static_cast<float>(C) << "\n";
    HwTanhHvxF32S0(A, C);
    std::cout << "HwTanhHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwTanhHvxDspS4(tanh_hvx_dsp_t& src1, tanh_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwTanhHvxDspS3(tanh_hvx_dsp_t& src1, tanh_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwTanhHvxDspS2(tanh_hvx_dsp_t& src1, tanh_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwTanhHvxDspS1(tanh_hvx_dsp_t& src1, tanh_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwTanhHvxDspS0(tanh_hvx_dsp_t& src1, tanh_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwTanhHvxDsp() {
    tanh_hvx_dsp_t A(1.3f), C(0.0f);
    HwTanhHvxDspS4(A, C);
    std::cout << "HwTanhHvxDspS4: " << static_cast<float>(C) << "\n";
    HwTanhHvxDspS3(A, C);
    std::cout << "HwTanhHvxDspS3: " << static_cast<float>(C) << "\n";
    HwTanhHvxDspS2(A, C);
    std::cout << "HwTanhHvxDspS2: " << static_cast<float>(C) << "\n";
    HwTanhHvxDspS1(A, C);
    std::cout << "HwTanhHvxDspS1: " << static_cast<float>(C) << "\n";
    HwTanhHvxDspS0(A, C);
    std::cout << "HwTanhHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using tanh_hvx_f16_t = dynfloat::std_f16;

void
HwTanhHvxF16S4(tanh_hvx_f16_t& src1, tanh_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwTanhHvxF16S3(tanh_hvx_f16_t& src1, tanh_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwTanhHvxF16S2(tanh_hvx_f16_t& src1, tanh_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwTanhHvxF16S1(tanh_hvx_f16_t& src1, tanh_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwTanhHvxF16S0(tanh_hvx_f16_t& src1, tanh_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::tanh<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwTanhHvxF16() {
    tanh_hvx_f16_t A(1.3f), C(0.0f);
    HwTanhHvxF16S4(A, C);
    std::cout << "HwTanhHvxF16S4: " << static_cast<float>(C) << "\n";
    HwTanhHvxF16S3(A, C);
    std::cout << "HwTanhHvxF16S3: " << static_cast<float>(C) << "\n";
    HwTanhHvxF16S2(A, C);
    std::cout << "HwTanhHvxF16S2: " << static_cast<float>(C) << "\n";
    HwTanhHvxF16S1(A, C);
    std::cout << "HwTanhHvxF16S1: " << static_cast<float>(C) << "\n";
    HwTanhHvxF16S0(A, C);
    std::cout << "HwTanhHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwTanhHlsF32();
    SwTanhHlsF16();
    SwTanhHvxF32();
    SwTanhHvxDsp();
    SwTanhHvxF16();

    return 0;
}
