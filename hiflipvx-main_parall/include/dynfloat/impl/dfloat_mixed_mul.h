#pragma once
#include "dfloat_mul.h"

namespace dynfloat {
namespace details_mixed_mul {

template<bool>
struct mixed_mul_fn_ {
    template<typename execution_,
             typename asm1 = default_assumptions,
             typename asm2 = default_assumptions,
             std::int64_t exp_bits1_,
             std::int64_t man_bits1_,
             std::int64_t exp_bits2_,
             std::int64_t man_bits2_>
    DYNFLOAT_FORCE_INLINE static constexpr auto mixed_mul(dfloat<exp_bits1_, man_bits1_> lhs, dfloat<exp_bits2_, man_bits2_> rhs) noexcept
        -> max_dfloat<dfloat<exp_bits1_, man_bits1_>, dfloat<exp_bits2_, man_bits2_>> {
        using lhs_t                    = dfloat<exp_bits1_, man_bits1_>;
        using rhs_t                    = dfloat<exp_bits2_, man_bits2_>;
        using res_t                    = max_dfloat<lhs_t, rhs_t>;
        using double_dman_t            = typename res_t::double_dman_type;
        constexpr auto saturate        = execution_{} == special_values::zero_and_saturation;
        constexpr zero_behavior z_bhv1 = asm1::zero_behavior;
        constexpr zero_behavior z_bhv2 = asm2::zero_behavior;
        const bool either_zero =
            z_bhv1 == zero_behavior::non_zero && z_bhv2 == zero_behavior::non_zero ? false : lhs.is_zero() || rhs.is_zero();
        const auto res_man_mask = utils::make_mask<res_t::dman_bits + 1>(!either_zero);
        const auto res_exp_mask = utils::make_mask<res_t::exp_bits>(!either_zero);
        static_assert(utils::non_overflowing_sub(lhs_t::dman_bits + rhs_t::dman_bits, std::numeric_limits<double_dman_t>::digits) == 0,
                      "mul will overflow");

        constexpr auto res_mul_shift = lhs_t::dman_bits + rhs_t::dman_bits - 1 - res_t::dman_bits;

        const auto rhs_exp_mag   = rhs.signed_exp() - rhs_t::zero_exp;
        const auto lhs_exp_mag   = lhs.signed_exp() - lhs_t::zero_exp;
        const auto res_exp_mag   = rhs_exp_mag + lhs_exp_mag;
        const auto res_exp       = res_exp_mag + res_t::zero_exp;
        const auto no_underflow  = res_exp > 0;
        const auto res_exp_mask2 = utils::make_mask<res_t::exp_bits>(no_underflow) & res_exp_mask;
        const auto res_man_mask2 = utils::make_mask<res_t::dman_bits + 1>(no_underflow) & res_man_mask;

        // Calculate result's mantissa and exponent correction.
        const auto lhs_dman = static_cast<double_dman_t>(lhs.dman());
        const auto rhs_dman = static_cast<double_dman_t>(rhs.dman());

        const auto res_dman_double = (lhs_dman * rhs_dman) >> res_mul_shift;
        const auto overflowing     = utils::bit_select<res_t::dman_bits>(res_dman_double);

        // Correct result
        const auto corrected_exp = res_exp + overflowing;
        const auto corrected_man = res_dman_double >> static_cast<std::uint8_t>(overflowing);
        if (saturate) {
            return res_t::from_components_saturate(lhs.unsigned_sign() ^ rhs.unsigned_sign(), res_exp_mask2 ? corrected_exp : 0,
                                                   res_man_mask2 ? corrected_man : 0);
        }

        return res_t::from_components(lhs.unsigned_sign() ^ rhs.unsigned_sign(), corrected_exp & res_exp_mask2,
                                      corrected_man & res_man_mask2);
    }
};

template<>
struct mixed_mul_fn_<false> {
    template<typename execution_,
             typename asm1 = default_assumptions,
             typename asm2 = default_assumptions,
             std::int64_t exp_bits1_,
             std::int64_t man_bits1_,
             std::int64_t exp_bits2_,
             std::int64_t man_bits2_>
    DYNFLOAT_FORCE_INLINE static constexpr auto mixed_mul(dfloat<exp_bits1_, man_bits1_> lhs, dfloat<exp_bits2_, man_bits2_> rhs) noexcept
        -> max_dfloat<dfloat<exp_bits1_, man_bits1_>, dfloat<exp_bits2_, man_bits2_>> {
        using lhs_t = dfloat<exp_bits1_, man_bits1_>;
        using rhs_t = dfloat<exp_bits2_, man_bits2_>;
        using res_t = max_dfloat<lhs_t, rhs_t>;
        return mul<execution_>(dfloat_cast<res_t, asm1>(lhs), dfloat_cast<res_t, asm2>(rhs));
    }
};

template<typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exp_bits1_,
         std::int64_t man_bits1_,
         std::int64_t exp_bits2_,
         std::int64_t man_bits2_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mixed_mul(dfloat<exp_bits1_, man_bits1_> lhs, dfloat<exp_bits2_, man_bits2_> rhs) noexcept
    -> max_dfloat<dfloat<exp_bits1_, man_bits1_>, dfloat<exp_bits2_, man_bits2_>> {
    using lhs_t         = dfloat<exp_bits1_, man_bits1_>;
    using rhs_t         = dfloat<exp_bits2_, man_bits2_>;
    using res_t         = max_dfloat<lhs_t, rhs_t>;
    using double_dman_t = typename res_t::double_dman_type;
    constexpr auto selection =
        utils::non_overflowing_sub(lhs_t::dman_bits + rhs_t::dman_bits, std::numeric_limits<double_dman_t>::digits) == 0;

    return mixed_mul_fn_<selection>{}.template mixed_mul<execution_, asm1, asm2>(lhs, rhs);
}
} // namespace details_mixed_mul

template<typename result_type_,
         typename execution_,
         typename asm1,
         typename asm2,
         std::int64_t exp_bits1_,
         std::int64_t man_bits1_,
         std::int64_t exp_bits2_,
         std::int64_t man_bits2_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mixed_mul(dfloat<exp_bits1_, man_bits1_> lhs, dfloat<exp_bits2_, man_bits2_> rhs) noexcept -> result_type_ {
    using max_t = max_dfloat<dfloat<exp_bits1_, man_bits1_>, dfloat<exp_bits2_, man_bits2_>>;

    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            return static_cast<result_type_>(mul<execution_, asm1, asm2>(static_cast<max_t>(lhs), static_cast<max_t>(rhs)));
        }
        case strategy::refined_fast:
        case strategy::refined: {
            using imm_t = dfloat<max_t::exp_bits, 1 + max_t::man_bits / 2>;
            return static_cast<result_type_>(details_mul::mul<make_exact_execution<execution_>, asm1, asm2>(
                dynfloat::round<imm_t, execution_::rnd, asm1>(lhs), dynfloat::round<imm_t, execution_::rnd, asm2>(rhs)));
        }
        case strategy::refined_exact:
        case strategy::exact: {
            return static_cast<result_type_>(details_mixed_mul::mixed_mul<execution_, asm1, asm2>(lhs, rhs));
        }
    }
    return {};
}

} // namespace dynfloat
