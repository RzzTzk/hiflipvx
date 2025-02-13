#pragma once
#include "dfloat.h"

namespace dynfloat {
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
ln(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                  = dfloat<exp_bits_, man_bits_>;
    using exact_execution        = make_exact_execution<execution_>;
    using imm_t                  = expand_to_dsp_opt<res_t>;
    constexpr auto one_over_lg2e = static_cast<imm_t>(1. / constants::log2_e);
    const auto lg2_res           = lg2<execution_>(static_cast<imm_t>(x));
    const auto mul_res           = mul<exact_execution>(lg2_res, one_over_lg2e);
    return static_cast<res_t>(mul_res);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace dynfloat
