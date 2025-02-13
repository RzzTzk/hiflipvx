#pragma once
#include "dfloat.h"

namespace dynfloat {
namespace details_dfloat_sqrt {

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
direct_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                    = dfloat<exp_bits_, man_bits_>;
    using int_type                 = typename res_t::int_type;
    constexpr auto new_frac_base   = res_t::dman_bits & ~1ULL; // make it even
    constexpr auto base_correction = new_frac_base / 2;
    constexpr auto dman_bits_odd   = res_t::dman_bits & 1ULL;

    const auto comp_exp     = static_cast<int_type>(x.unsigned_exp()) - x.zero_exp;
    const auto comp_exp_odd = comp_exp & 1;

    const auto man_sqrt      = utils::sqrt<new_frac_base + 1 + dman_bits_odd, dman_bits_odd != 0>(x.dman() >> comp_exp_odd);
    const auto res_dman      = (man_sqrt << (base_correction + dman_bits_odd));
    const auto overflow_bit  = (res_dman >> res_t::dman_bits) & 1;
    const auto corrected_man = res_dman >> static_cast<std::uint8_t>(overflow_bit);

    const auto res_exp       = (comp_exp >> 1) + x.zero_exp;
    const auto corrected_exp = res_exp + overflow_bit - !comp_exp_odd;

    return res_t::from_components(0, corrected_exp, corrected_man);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
fast_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t             = dfloat<exp_bits_, man_bits_>;
    using int_repr_type     = typename res_t::int_repr_type;
    constexpr auto ofs      = static_cast<int_repr_type>(utils::int_repr_sqrt_approx_factor(res_t::exp_bits, res_t::man_bits) / 2);
    const auto integer_repr = x.int_repr() >> 1;
    const auto imm_res      = ofs + integer_repr;
    const auto res          = res_t::from_int_repr(0, static_cast<int_repr_type>(imm_res));
    return res;
}

template<std::int64_t expand_man_bits_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
expand_direct_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t = dfloat<exp_bits_, man_bits_>;
    using imm_t = dfloat<res_t::exp_bits, expand_man_bits_>;
    return static_cast<res_t>(details_dfloat_sqrt::direct_sqrt(static_cast<imm_t>(x)));
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
cordic_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                = dfloat<exp_bits_, man_bits_>;
    using int_type             = typename res_t::int_type;
    const auto exp             = x.signed_exp() - res_t::zero_exp;
    const auto dman            = x.dman() << (exp & 1);
    const auto sqrt_fixed_dman = dynfloat::fixed_sqrt_limited_range<man_bits_, man_bits_ / 2 + 2>(static_cast<int_type>(dman));
    const auto half_exp        = (exp >> 1) + res_t::zero_exp;
    return res_t::from_components(false, half_exp, sqrt_fixed_dman);
}
} // namespace details_dfloat_sqrt

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                = dfloat<exp_bits_, man_bits_>;
    constexpr auto man_even    = !(man_bits_ & 1);
    constexpr auto exact_bits_ = (res_t::man_bits + man_even - 2 * (man_bits_ > 15)) << 1;
    constexpr auto exact_bits  = std::min(exact_bits_, std_f64::man_bits);
    using res_16man_t          = dfloat<exp_bits_, std::min(exact_bits, 16LL)>;
    using res_24man_t          = dfloat<exp_bits_, std::min(exact_bits, 24LL)>;
    using res_30man_t          = dfloat<exp_bits_, std::min(exact_bits, 30LL)>;
    using res_max_t            = dfloat<exp_bits_, exact_bits>;

    if (x.is_zero())
        return {};
    constexpr strategy str = execution_::str;
    switch (str) {
        case strategy::fast:
            return details_dfloat_sqrt::fast_sqrt(x);
        case strategy::refined_fast: {
            return static_cast<res_t>(details_dfloat_sqrt::direct_sqrt(static_cast<res_16man_t>(x)));
        }
        case strategy::refined: {
            return static_cast<res_t>(details_dfloat_sqrt::direct_sqrt(static_cast<res_24man_t>(x)));
        }
        case strategy::refined_exact: {
            return static_cast<res_t>(details_dfloat_sqrt::direct_sqrt(static_cast<res_30man_t>(x)));
        }
        case strategy::exact: {
            if (execution_::nfi) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
                if (man_bits_ <= std_f32::man_bits && 30 < exact_bits) {
                    return static_cast<res_t>(std::sqrt(static_cast<float>(static_cast<std_f32>(x))));
                }
#endif
            }
            return static_cast<res_t>(details_dfloat_sqrt::direct_sqrt(static_cast<res_max_t>(x)));
            // return details_dfloat_sqrt::cordic_sqrt(x);
        }
    }
    return {};
}
} // namespace dynfloat
