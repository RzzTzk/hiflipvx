#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using log2_hls_f32_t = float;
using log2_hls_f16_t = hls_half;
using log2_hvx_f32_t = dynfloat::std_f32;
using log2_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

void
HwLog2HlsF32(log2_hls_f32_t& src1, log2_hls_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::log2(src1);
#else
    dst = log2(src1);
#endif
}

void
SwLog2HlsF32() {
    log2_hls_f32_t A(1.3f), C(0.0f);
    HwLog2HlsF32(A, C);
    std::cout << "HwLog2HlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLog2HlsF16(log2_hls_f16_t& src1, log2_hls_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
#if defined(HVX_SYNTHESIS_ACTIVE)
    dst = hls::half_log2(src1);
#else
    dst = log2(src1);
#endif
}

void
SwLog2HlsF16() {
    log2_hls_f16_t A(1.3f), C(0.0f);
    HwLog2HlsF16(A, C);
    std::cout << "HwLog2HlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLog2HvxF32S4(log2_hvx_f32_t& src1, log2_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLog2HvxF32S3(log2_hvx_f32_t& src1, log2_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLog2HvxF32S2(log2_hvx_f32_t& src1, log2_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLog2HvxF32S1(log2_hvx_f32_t& src1, log2_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLog2HvxF32S0(log2_hvx_f32_t& src1, log2_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLog2HvxF32() {
    log2_hvx_f32_t A(1.3f), C(0.0f);
    HwLog2HvxF32S4(A, C);
    std::cout << "HwLog2HvxF32S4: " << static_cast<float>(C) << "\n";
    HwLog2HvxF32S3(A, C);
    std::cout << "HwLog2HvxF32S3: " << static_cast<float>(C) << "\n";
    HwLog2HvxF32S2(A, C);
    std::cout << "HwLog2HvxF32S2: " << static_cast<float>(C) << "\n";
    HwLog2HvxF32S1(A, C);
    std::cout << "HwLog2HvxF32S1: " << static_cast<float>(C) << "\n";
    HwLog2HvxF32S0(A, C);
    std::cout << "HwLog2HvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwLog2HvxDspS4(log2_hvx_dsp_t& src1, log2_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLog2HvxDspS3(log2_hvx_dsp_t& src1, log2_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLog2HvxDspS2(log2_hvx_dsp_t& src1, log2_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLog2HvxDspS1(log2_hvx_dsp_t& src1, log2_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLog2HvxDspS0(log2_hvx_dsp_t& src1, log2_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLog2HvxDsp() {
    log2_hvx_dsp_t A(1.3f), C(0.0f);
    HwLog2HvxDspS4(A, C);
    std::cout << "HwLog2HvxDspS4: " << static_cast<float>(C) << "\n";
    HwLog2HvxDspS3(A, C);
    std::cout << "HwLog2HvxDspS3: " << static_cast<float>(C) << "\n";
    HwLog2HvxDspS2(A, C);
    std::cout << "HwLog2HvxDspS2: " << static_cast<float>(C) << "\n";
    HwLog2HvxDspS1(A, C);
    std::cout << "HwLog2HvxDspS1: " << static_cast<float>(C) << "\n";
    HwLog2HvxDspS0(A, C);
    std::cout << "HwLog2HvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using log2_hvx_f16_t = dynfloat::std_f16;

void
HwLog2HvxF16S4(log2_hvx_f16_t& src1, log2_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value> >(src1);
}

void
HwLog2HvxF16S3(log2_hvx_f16_t& src1, log2_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value> >(src1);
}

void
HwLog2HvxF16S2(log2_hvx_f16_t& src1, log2_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value> >(src1);
}

void
HwLog2HvxF16S1(log2_hvx_f16_t& src1, log2_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value> >(src1);
}

void
HwLog2HvxF16S0(log2_hvx_f16_t& src1, log2_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_1SRC();
    dst = dynfloat::lg2<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value> >(src1);
}

void
SwLog2HvxF16() {
    log2_hvx_f16_t A(1.3f), C(0.0f);
    HwLog2HvxF16S4(A, C);
    std::cout << "HwLog2HvxF16S4: " << static_cast<float>(C) << "\n";
    HwLog2HvxF16S3(A, C);
    std::cout << "HwLog2HvxF16S3: " << static_cast<float>(C) << "\n";
    HwLog2HvxF16S2(A, C);
    std::cout << "HwLog2HvxF16S2: " << static_cast<float>(C) << "\n";
    HwLog2HvxF16S1(A, C);
    std::cout << "HwLog2HvxF16S1: " << static_cast<float>(C) << "\n";
    HwLog2HvxF16S0(A, C);
    std::cout << "HwLog2HvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwLog2HlsF32();
    SwLog2HlsF16();
    SwLog2HvxF32();
    SwLog2HvxDsp();
    SwLog2HvxF16();

    return 0;
}
