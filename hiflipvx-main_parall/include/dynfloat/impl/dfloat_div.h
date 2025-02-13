#pragma once
#include "dfloat_reciprocal.h"

namespace dynfloat {
namespace details_div {
template<typename execution_, typename asm1, typename asm2, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
div(const dfloat<exp_bits_, man_bits_> lhs, const dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    return mul<execution<execution_::str, execution_::rnd, execution_::spv>, asm1, modify_asm::reciprocal<asm2>>(
        lhs, dynfloat::reciprocal<execution_, asm2>(rhs));
}
} // namespace details_div

template<typename execution_, typename asm1, typename asm2, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
div(const dfloat<exp_bits_, man_bits_> lhs, const dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t = dfloat<exp_bits_, man_bits_>;
    using imm_t = expand_to_dsp_opt<res_t>;
    return static_cast<res_t>(details_div::div<execution_, asm1, asm2>(static_cast<imm_t>(lhs), static_cast<imm_t>(rhs)));
}

} // namespace dynfloat
