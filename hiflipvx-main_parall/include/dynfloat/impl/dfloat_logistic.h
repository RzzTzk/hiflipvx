#pragma once
#include "dfloat.h"

namespace dynfloat {
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
logistic(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t            = dfloat<exp_bits_, man_bits_>;
    using imm_t            = expand_to_dsp_opt<res_t>;
    constexpr auto half    = static_cast<imm_t>(0.5);
    const auto tanh_half_x = tanh<execution_>(static_cast<imm_t>(x) >> 1);
    return static_cast<res_t>(add<make_exact_execution<execution_>, positive_nz_asm, default_assumptions>(half, tanh_half_x >> 1));
}
} // namespace dynfloat
