#include "../../include/dynfloat/dynfloat.h"
#include "../../include/hiflipvx/hvx_core.h"

constexpr dynfloat::rounding_behavior rounding   = dynfloat::rounding_behavior::to_zero;
constexpr dynfloat::special_values special_value = dynfloat::special_values::zero;

#if defined(HVX_SYNTHESIS_ACTIVE)
using hls_half = half;
#else
using hls_half = float;
#endif

using add_hls_f32_t = float;
using add_hls_f16_t = hls_half;
using add_hvx_f32_t = dynfloat::std_f32;
using add_hvx_dsp_t = dynfloat::pltfrm_dsp_opt;

/**********************************************************************************************************************/

////#include "../../include/thls/tops/fp_flopoco_add_single_v5.hpp"
//#include "../../include/thls/tops/fp_flopoco.hpp"
//#include "../../include/thls/tops/fw_uint_on_ac_uint.hpp"
//#define THLS_SYNTHESIS
////#define THLS_FW_UINT_ON_MASKED_UINT
//void
//HwAddThls(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
//    HVX_EXEC_UNIT_2SRC();
//    using float_type = dynfloat::std_f32;
//    const thls::fw_uint<2> flags(0);
//    const thls::fw_uint<1> sign{};//  src1.sign_;
//    const thls::fw_uint<1> exp{}; //  src1.sign_;
//    const thls::fw_uint<1> frac{}; //  src1.sign_;    
//
//    //thls::concat(_flags, _sign, _exp, _frac);
//    //thls::fp_flopoco<add_hvx_f32_t::exp_bits, add_hvx_f32_t::man_bits> x(4);
//    thls::fp_flopoco<add_hvx_f32_t::exp_bits, add_hvx_f32_t::man_bits> x(1,2,3,4);

//}

/**********************************************************************************************************************/

void
HwAddHlsF32(add_hls_f32_t& src1, add_hls_f32_t& src2, add_hls_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 + src2;
}

void
SwAddHlsF32() {
    add_hls_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwAddHlsF32(A, B, C);
    std::cout << "HwAddHlsF32: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwAddHlsF16(add_hls_f16_t& src1, add_hls_f16_t& src2, add_hls_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = src1 + src2;
}

void
SwAddHlsF16() {
    add_hls_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwAddHlsF16(A, B, C);
    std::cout << "HwAddHlsF16: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwAddHvxF32S4(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF32S3(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF32S2(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF32S1(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF32S0(add_hvx_f32_t& src1, add_hvx_f32_t& src2, add_hvx_f32_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwAddHvxF32() {
    add_hvx_f32_t A(1.3f), B(2.3f), C(0.0f);
    HwAddHvxF32S4(A, B, C);
    std::cout << "HwAddHvxF32S4: " << static_cast<float>(C) << "\n";
    HwAddHvxF32S3(A, B, C);
    std::cout << "HwAddHvxF32S3: " << static_cast<float>(C) << "\n";
    HwAddHvxF32S2(A, B, C);
    std::cout << "HwAddHvxF32S2: " << static_cast<float>(C) << "\n";
    HwAddHvxF32S1(A, B, C);
    std::cout << "HwAddHvxF32S1: " << static_cast<float>(C) << "\n";
    HwAddHvxF32S0(A, B, C);
    std::cout << "HwAddHvxF32S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

void
HwAddHvxDspS4(add_hvx_dsp_t& src1, add_hvx_dsp_t& src2, add_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxDspS3(add_hvx_dsp_t& src1, add_hvx_dsp_t& src2, add_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxDspS2(add_hvx_dsp_t& src1, add_hvx_dsp_t& src2, add_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwAddHvxDspS1(add_hvx_dsp_t& src1, add_hvx_dsp_t& src2, add_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwAddHvxDspS0(add_hvx_dsp_t& src1, add_hvx_dsp_t& src2, add_hvx_dsp_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwAddHvxDsp() {
    add_hvx_dsp_t A(1.3f), B(2.3f), C(0.0f);
    HwAddHvxDspS4(A, B, C);
    std::cout << "HwAddHvxDspS4: " << static_cast<float>(C) << "\n";
    HwAddHvxDspS3(A, B, C);
    std::cout << "HwAddHvxDspS3: " << static_cast<float>(C) << "\n";
    HwAddHvxDspS2(A, B, C);
    std::cout << "HwAddHvxDspS2: " << static_cast<float>(C) << "\n";
    HwAddHvxDspS1(A, B, C);
    std::cout << "HwAddHvxDspS1: " << static_cast<float>(C) << "\n";
    HwAddHvxDspS0(A, B, C);
    std::cout << "HwAddHvxDspS0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/
using add_hvx_f16_t = dynfloat::std_f16;

void
HwAddHvxF16S4(add_hvx_f16_t& src1, add_hvx_f16_t& src2, add_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF16S3(add_hvx_f16_t& src1, add_hvx_f16_t& src2, add_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_exact, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF16S2(add_hvx_f16_t& src1, add_hvx_f16_t& src2, add_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF16S1(add_hvx_f16_t& src1, add_hvx_f16_t& src2, add_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::refined_fast, rounding, special_value>>(src1, src2);
}

void
HwAddHvxF16S0(add_hvx_f16_t& src1, add_hvx_f16_t& src2, add_hvx_f16_t& dst) {
    HVX_EXEC_UNIT_2SRC();
    dst = dynfloat::add<dynfloat::execution<dynfloat::strategy::fast, rounding, special_value>>(src1, src2);
}

void
SwAddHvxF16() {
    add_hvx_f16_t A(1.3f), B(2.3f), C(0.0f);
    HwAddHvxF16S4(A, B, C);
    std::cout << "HwAddHvxF16S4: " << static_cast<float>(C) << "\n";
    HwAddHvxF16S3(A, B, C);
    std::cout << "HwAddHvxF16S3: " << static_cast<float>(C) << "\n";
    HwAddHvxF16S2(A, B, C);
    std::cout << "HwAddHvxF16S2: " << static_cast<float>(C) << "\n";
    HwAddHvxF16S1(A, B, C);
    std::cout << "HwAddHvxF16S1: " << static_cast<float>(C) << "\n";
    HwAddHvxF16S0(A, B, C);
    std::cout << "HwAddHvxF16S0: " << static_cast<float>(C) << "\n\n";
}

/**********************************************************************************************************************/

// testbench
auto
main() -> int {
    SwAddHlsF32();
    SwAddHlsF16();
    SwAddHvxF32();
    SwAddHvxDsp();
    SwAddHvxF16();

    return 0;
}
