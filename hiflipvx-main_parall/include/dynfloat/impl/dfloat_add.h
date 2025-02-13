#pragma once
#include "dfloat.h"

namespace dynfloat {
namespace details_operator_plus {
template<typename execution_, typename asm1, typename asm2, std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
preconditioned_add(const dfloat<exponent_bits_, mantissa_bits_> lhs, const dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    using res_t                    = dfloat<exponent_bits_, mantissa_bits_>;
    constexpr auto saturate        = execution_{} == special_values::zero_and_saturation;
    constexpr sign_behavior s_bhv1 = asm1::sign_behavior;
    constexpr sign_behavior s_bhv2 = asm2::sign_behavior;

    // Result sign is always the same as the sign of lhs as lhs is always greater equal rhs
    const auto res_sign = lhs.assumed_sign(s_bhv1);

    // Check if same signs. This determines whether to add or subtract the mantissas
    const auto same_sign = lhs.assumed_sign(s_bhv1) == rhs.assumed_sign(s_bhv2);

    // Calculate difference between exponents (exp_dif), to use later for shifting the mantissa of rhs
    const auto exp_dif = lhs.unsigned_exp() - rhs.unsigned_exp();

    // Calculate denormalized mantissas. Mantissa of rhs is shifted right by exp_dif
    const auto lhs_dman = lhs.dman();
    const auto rhs_dman = rhs.dman() >> utils::min(exp_dif, res_t::dman_bits);

    // Calculate denormalized mantissa of result (res_dman).
    constexpr auto res_man_mask = utils::make_mask<res_t::dman_bits + 1>(true);
    const auto dman_sum         = lhs_dman + rhs_dman;
    const auto dman_sub         = lhs_dman - rhs_dman;
    const auto res_dman         = (same_sign ? dman_sum : dman_sub) & res_man_mask;
    const auto res_dman_width   = utils::bit_width(res_dman);

    // Determine whether res_dman overflowed and needed exponent correction
    const auto man_overflow    = ((res_dman >> res_t::dman_bits) & 1u) == 1u;
    const auto overflow_amount = res_t::dman_bits - res_dman_width;

    // Calculate exp_mask. exp_mask is zero if res_dman is zero
    const auto res_is_not_zero = res_dman != 0;
    const auto exp_mask        = utils::make_mask<res_t::exp_bits>(res_is_not_zero);

    // Correct result
    const auto corrected_man = (man_overflow ? res_dman >> 1 : res_dman << overflow_amount);
    const auto lhs_exp_pls1  = lhs.unsigned_exp() + 1;
    const auto res_exp       = man_overflow ? lhs_exp_pls1 : lhs.unsigned_exp() - overflow_amount;

    if (saturate)
        return res_t::from_components_saturate(lhs.unsigned_sign(), res_is_not_zero ? res_exp : 0, corrected_man);
    return res_t::from_components(res_sign, res_exp & exp_mask, corrected_man);
}

template<typename execution_, typename asm1, typename asm2, std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
round_add(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    constexpr rounding_behavior rnd = execution_::rnd;
    if (rnd == rounding_behavior::to_zero)
        return details_operator_plus::preconditioned_add<execution_, asm1, asm2>(lhs, rhs);

    using res_t     = dfloat<exponent_bits_, mantissa_bits_>;
    using tmp_float = expand_for_rounding<res_t, execution_>;

    return dynfloat::round<res_t, rnd>(details_operator_plus::preconditioned_add<execution_, asm1, asm2>(
        dynfloat::dfloat_cast<tmp_float, asm1>(lhs), dynfloat::dfloat_cast<tmp_float, asm2>(rhs)));
}
} // namespace details_operator_plus

template<typename execution_, typename asm1, typename asm2, std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    if (execution_{} == special_values::zero || execution_{} == special_values::zero_and_saturation) {
        constexpr zero_behavior z_bhv1 = asm1::zero_behavior;
        constexpr zero_behavior z_bhv2 = asm2::zero_behavior;
        const auto lhs_zero            = z_bhv1 == zero_behavior::non_zero ? false : lhs.is_zero();
        const auto rhs_zero            = z_bhv2 == zero_behavior::non_zero ? false : rhs.is_zero();
        if (lhs_zero)
            return rhs;
        if (rhs_zero)
            return lhs;
    }

    const auto swap_params = lhs.int_repr() < rhs.int_repr();
    return details_operator_plus::round_add<execution_, asm2, asm1>(swap_params ? rhs : lhs, swap_params ? lhs : rhs);
}

template<typename execution_, typename asm1, typename asm2, std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
sub(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    return add<execution_, asm1, modify_asm::negate<asm2>>(lhs, -rhs);
}

namespace tree {
template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(utils::array<dfloat<exp_bits_, man_bits_>, 1> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return eles[0];
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(utils::array<dfloat<exp_bits_, man_bits_>, 2> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return add<execution_, asm1, asm1>(eles[0], eles[1]);
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(utils::array<dfloat<exp_bits_, man_bits_>, 3> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    return add<execution_, asm1>(eles[0], add<execution_, asm1>(eles[1], eles[2]));
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(utils::array<dfloat<exp_bits_, man_bits_>, size_> eles) noexcept -> dfloat<exp_bits_, man_bits_> {
    constexpr std::size_t middle_estimate = size_ / 2;
    constexpr std::size_t middle          = middle_estimate & ~1ULL;
    utils::array<dfloat<exp_bits_, man_bits_>, middle> left{};
    utils::array<dfloat<exp_bits_, man_bits_>, size_ - middle> right{};
    for (size_t i = 0; i < middle; ++i)
        left[i] = eles[i]; // NOLINT
    for (size_t i = middle; i < size_; ++i)
        right[i - middle] = eles[i]; // NOLINT
    return add<execution_, asm1>(add<execution_, asm1>(left), add<execution_, asm1>(right));
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(const dfloat<exp_bits_, man_bits_> (&eles)[size_]) noexcept -> dfloat<exp_bits_, man_bits_> // NOLINT
{
    utils::array<dfloat<exp_bits_, man_bits_>, size_> eles_arr{};
    for (size_t i = 0; i < size_; ++i)
        eles_arr[i] = eles[i]; // NOLINT
    return add<execution_, asm1>(eles_arr);
}

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_, typename... Args>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(dfloat<exp_bits_, man_bits_> ele, dfloat<exp_bits_, man_bits_> ele2, dfloat<exp_bits_, man_bits_> ele3, Args... args) noexcept
    -> dfloat<exp_bits_, man_bits_> {
    return add<execution_, asm1>(utils::array<dfloat<exp_bits_, man_bits_>, 3 + sizeof...(args)>{ele, ele2, ele3, args...});
}
} // namespace tree

} // namespace dynfloat
