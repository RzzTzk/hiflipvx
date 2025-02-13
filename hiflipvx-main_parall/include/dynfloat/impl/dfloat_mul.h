#pragma once
#include "dfloat.h"

namespace dynfloat {
namespace details_mul {
template<typename execution_, typename asm1, typename asm2, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                    = dfloat<exp_bits_, man_bits_>;
    using double_dman_t            = typename res_t::double_dman_type;
    using int_type                 = typename res_t::int_type;
    constexpr auto saturate        = execution_{} == special_values::zero_and_saturation;
    constexpr zero_behavior z_bhv1 = asm1::zero_behavior;
    constexpr zero_behavior z_bhv2 = asm2::zero_behavior;
    const bool either_zero =
        z_bhv1 == zero_behavior::non_zero && z_bhv2 == zero_behavior::non_zero ? false : lhs.is_zero() || rhs.is_zero();
    const auto res_man_mask      = utils::make_mask<res_t::dman_bits + 1>(!either_zero);
    const auto res_exp_mask      = utils::make_mask<res_t::exp_bits>(!either_zero);
    const auto res_int_repr_mask = utils::make_mask<res_t::exp_bits + res_t::dman_bits>(!either_zero);

    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            constexpr auto approx_factor_dbl = utils::int_repr_mul_approx_factor(exp_bits_, man_bits_);
            constexpr auto approx_factor     = static_cast<double_dman_t>(approx_factor_dbl);
            const auto res = static_cast<double_dman_t>(lhs.int_repr()) + static_cast<double_dman_t>(rhs.int_repr()) - approx_factor;
            return res_t::from_int_repr(lhs.unsigned_sign() ^ rhs.unsigned_sign(),
                                        static_cast<typename res_t::int_repr_type>(res) & res_int_repr_mask);
        }
        case strategy::refined_fast:
        case strategy::refined:
        case strategy::refined_exact:
        case strategy::exact: {
            constexpr auto scale_up     = utils::non_overflowing_sub(res_t::dman_bits * 2, std::numeric_limits<double_dman_t>::digits);
            constexpr auto lhs_scale_up = scale_up / 2;
            constexpr auto rhs_scale_up = scale_up - lhs_scale_up;
            constexpr auto res_dman_double_min_width = res_t::dman_bits * 2 - 1 - scale_up;
            constexpr auto res_mul_shift             = res_dman_double_min_width - res_t::dman_bits;
            constexpr auto res_exp_initial_offset    = scale_up - res_t::man_bits + res_mul_shift - res_t::zero_exp;

            // Calculate initial result's exponent
            const auto res_exp =
                static_cast<int_type>(lhs.unsigned_exp()) + static_cast<int_type>(rhs.unsigned_exp()) + res_exp_initial_offset;
            const auto no_underflow  = res_exp > 0;
            const auto res_exp_mask2 = utils::make_mask<res_t::exp_bits>(no_underflow) & res_exp_mask;
            const auto res_man_mask2 = utils::make_mask<res_t::dman_bits + 1>(no_underflow) & res_man_mask;

            // Calculate result's mantissa and exponent correction.
            const auto lhs_dman = static_cast<double_dman_t>(lhs.dman()) >> lhs_scale_up;
            const auto rhs_dman = static_cast<double_dman_t>(rhs.dman()) >> rhs_scale_up;

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
    }
    return {};
}

template<typename execution_, typename asm1, typename asm2, std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
round_mul(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    constexpr rounding_behavior rnd = execution_::rnd;
    if (rnd == rounding_behavior::to_zero)
        return details_mul::mul<execution_, asm1, asm2>(lhs, rhs);

    using res_t     = dfloat<exponent_bits_, mantissa_bits_>;
    using tmp_float = expand_for_rounding<res_t, execution_>;

    return dynfloat::round<res_t, rnd>(
        details_mul::mul<execution_, asm1, asm2>(dynfloat::dfloat_cast<tmp_float, asm1>(lhs), dynfloat::dfloat_cast<tmp_float, asm2>(rhs)));
}
} // namespace details_mul

template<typename execution_, typename asm1, typename asm2, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            return details_mul::mul<execution_, asm1, asm2>(lhs, rhs);
        }
        case strategy::refined_fast:
        case strategy::refined: {
            if (man_bits_ > pltfrm_dsp_opt::man_bits) {
                using res_t = dfloat<exp_bits_, man_bits_>;
                using imm_t = dfloat<exp_bits_, pltfrm_dsp_opt::man_bits>;
                return static_cast<res_t>(details_mul::round_mul<execution_, asm1, asm2>(static_cast<imm_t>(lhs), static_cast<imm_t>(rhs)));
            }
            return details_mul::round_mul<execution_, asm1, asm2>(lhs, rhs);
        }
        case strategy::refined_exact:
        case strategy::exact: {
            return details_mul::round_mul<execution_, asm1, asm2>(lhs, rhs);
        }
    }
    return {};
}

namespace tree {

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(utils::array<dfloat<exp_bits_, man_bits_>, 1> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return eles[0];
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(utils::array<dfloat<exp_bits_, man_bits_>, 2> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return mul<execution_, asm1, asm1>(eles[0], eles[1]);
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(utils::array<dfloat<exp_bits_, man_bits_>, 3> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return mul<execution_, asm1, asm1>(mul<execution_, asm1, asm1>(eles[0], eles[1]), eles[2]);
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(utils::array<dfloat<exp_bits_, man_bits_>, size_> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    constexpr std::size_t middle_estimate = size_ / 2;
    constexpr std::size_t middle          = middle_estimate & ~1;
    utils::array<dfloat<exp_bits_, man_bits_>, middle> left{};
    utils::array<dfloat<exp_bits_, man_bits_>, size_ - middle> right{};
    for (size_t i = 0; i < middle; ++i)
        left[i] = eles[i]; // NOLINT
    for (size_t i = middle; i < size_; ++i)
        right[i - middle] = eles[i]; // NOLINT
    return mul<execution_, asm1>(mul<execution_, asm1>(left), mul<execution_, asm1>(right));
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_, typename... Args>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(dfloat<exp_bits_, man_bits_> ele1, dfloat<exp_bits_, man_bits_> ele2, dfloat<exp_bits_, man_bits_> ele3, Args... eles) noexcept
    -> dfloat<exp_bits_, man_bits_> {
    return mul<execution_, asm1>(utils::array<dfloat<exp_bits_, man_bits_>, 3 + sizeof...(eles)>{ele1, ele2, ele3, eles...});
}
} // namespace tree
} // namespace dynfloat
