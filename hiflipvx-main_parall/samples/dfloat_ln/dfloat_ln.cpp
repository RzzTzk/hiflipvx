#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using ln_hls_f32_t = float;
using ln_hls_f16_t = hls_half;
using ln_hvx_f32_t = dynfloat::std_f32;
using ln_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwLnHlsF32(ln_hls_f32_t& src1, ln_hls_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::log(src1);
#else
    dst = std::log(src1);
#endif
}

void
SwLnHlsF32() {
    ln_hls_f32_t A(1.3f), C(0.0f);
    HwLnHlsF32(A, C);
    std::cout << "HwLnHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLnHlsF16(ln_hls_f16_t& src1, ln_hls_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::half_log(src1);
#else
    dst = std::log(src1);
#endif
}

void
SwLnHlsF16() {
    ln_hls_f16_t A(1.3f), C(0.0f);
    HwLnHlsF16(A, C);
    std::cout << "HwLnHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLnHvxF32S4(ln_hvx_f32_t& src1, ln_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLnHvxF32S3(ln_hvx_f32_t& src1, ln_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLnHvxF32S2(ln_hvx_f32_t& src1, ln_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLnHvxF32S1(ln_hvx_f32_t& src1, ln_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLnHvxF32S0(ln_hvx_f32_t& src1, ln_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLnHvxF32() {
    ln_hvx_f32_t A(1.3f), C(0.0f);
    HwLnHvxF32S4(A, C);
    std::cout << "HwLnHvxF32S4: " << static_cast<float>(C) << "\n";
    HwLnHvxF32S3(A, C);
    std::cout << "HwLnHvxF32S3: " << static_cast<float>(C) << "\n";
    HwLnHvxF32S2(A, C);
    std::cout << "HwLnHvxF32S2: " << static_cast<float>(C) << "\n";
    HwLnHvxF32S1(A, C);
    std::cout << "HwLnHvxF32S1: " << static_cast<float>(C) << "\n";
    HwLnHvxF32S0(A, C);
    std::cout << "HwLnHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLnHvxDspS4(ln_hvx_dsp_t& src1, ln_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLnHvxDspS3(ln_hvx_dsp_t& src1, ln_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLnHvxDspS2(ln_hvx_dsp_t& src1, ln_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLnHvxDspS1(ln_hvx_dsp_t& src1, ln_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLnHvxDspS0(ln_hvx_dsp_t& src1, ln_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLnHvxDsp() {
    ln_hvx_dsp_t A(1.3f), C(0.0f);
    HwLnHvxDspS4(A, C);
    std::cout << "HwLnHvxDspS4: " << static_cast<float>(C) << "\n";
    HwLnHvxDspS3(A, C);
    std::cout << "HwLnHvxDspS3: " << static_cast<float>(C) << "\n";
    HwLnHvxDspS2(A, C);
    std::cout << "HwLnHvxDspS2: " << static_cast<float>(C) << "\n";
    HwLnHvxDspS1(A, C);
    std::cout << "HwLnHvxDspS1: " << static_cast<float>(C) << "\n";
    HwLnHvxDspS0(A, C);
    std::cout << "HwLnHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using ln_hvx_f16_t = dynfloat::std_f16;

void
HwLnHvxF16S4(ln_hvx_f16_t& src1, ln_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLnHvxF16S3(ln_hvx_f16_t& src1, ln_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLnHvxF16S2(ln_hvx_f16_t& src1, ln_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLnHvxF16S1(ln_hvx_f16_t& src1, ln_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLnHvxF16S0(ln_hvx_f16_t& src1, ln_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::ln<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLnHvxF16() {
    ln_hvx_f16_t A(1.3f), C(0.0f);
    HwLnHvxF16S4(A, C);
    std::cout << "HwLnHvxF16S4: " << static_cast<float>(C) << "\n";
    HwLnHvxF16S3(A, C);
    std::cout << "HwLnHvxF16S3: " << static_cast<float>(C) << "\n";
    HwLnHvxF16S2(A, C);
    std::cout << "HwLnHvxF16S2: " << static_cast<float>(C) << "\n";
    HwLnHvxF16S1(A, C);
    std::cout << "HwLnHvxF16S1: " << static_cast<float>(C) << "\n";
    HwLnHvxF16S0(A, C);
    std::cout << "HwLnHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwLnHlsF32();
    SwLnHlsF16();
    SwLnHvxF32();
    SwLnHvxDsp();
    SwLnHvxF16();

    return 0;
}
