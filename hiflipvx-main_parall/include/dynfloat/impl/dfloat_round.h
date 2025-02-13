#pragma once
#include "dfloat.h"
#include "dfloat_add.h"

namespace dynfloat {
namespace details_round {

template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add_one_ulp(const dfloat<exponent_bits_, mantissa_bits_> lhs, const bool ulp) noexcept -> dfloat<exponent_bits_, mantissa_bits_> {
    using res_t              = dfloat<exponent_bits_, mantissa_bits_>;
    const auto lhs_dman      = lhs.dman();
    const auto res_dman      = lhs_dman + ulp;
    const auto man_overflow  = ((res_dman >> res_t::dman_bits) & 1u) == 1u;
    const auto corrected_man = man_overflow ? res_dman >> 1 : res_dman;
    const auto res_exp       = man_overflow ? lhs.unsigned_exp() + 1 : lhs.unsigned_exp();
    return res_t::from_components_saturate(lhs.unsigned_sign(), res_dman != 0 ? res_exp : 0, corrected_man);
}
} // namespace details_round

template<typename smaller_dfloat, rounding_behavior rnd, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
round(dfloat<exp_bits_, man_bits_> x) noexcept -> smaller_dfloat {
    using x_t                = dfloat<exp_bits_, man_bits_>;
    using res_t              = smaller_dfloat;
    const auto unrounded_res = dfloat_cast<res_t, asm1>(x);
    switch (rnd) {
        case rounding_behavior::to_zero:
            return unrounded_res;
        case rounding_behavior::nearest_even: {
            if (x_t::man_bits - res_t::man_bits <= 1)
                return unrounded_res;

            constexpr auto s_bits   = std::max(x_t::man_bits - res_t::man_bits - 1, 1LL);
            constexpr auto s_mask   = utils::make_mask<s_bits>(true);
            const auto s            = (x.unsigned_man() & s_mask) != 0;
            const auto g            = (x.unsigned_man() >> s_bits) & 1;
            const auto r            = (x.unsigned_man() >> (s_bits + 1)) & 1;
            const auto should_round = g && (s || r);
            const auto ulp          = res_t::from_components(unrounded_res.unsigned_sign(), unrounded_res.unsigned_exp(), 1);
            return details_round::add_one_ulp(unrounded_res, should_round);
        }
    }
    return unrounded_res;
}
} // namespace dynfloat
