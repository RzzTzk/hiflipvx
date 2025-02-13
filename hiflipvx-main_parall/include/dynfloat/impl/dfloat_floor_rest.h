#pragma once
#include "dfloat_abs.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor_int(const dfloat<exp_bits_, man_bits_> x) noexcept -> std::int64_t {
    using res_t    = dfloat<exp_bits_, man_bits_>;
    using int_type = typename res_t::int_type;

    const auto shift_amount = static_cast<int_type>(x.unsigned_exp()) - static_cast<int_type>(res_t::zero_exp);

    // Special case: exponent is negative -> integer will always be zero
    {
        if (shift_amount < 0)
            return x.unsigned_sign() ? -1 : 0;
    }

    const auto right_shift        = res_t::man_bits >= shift_amount;
    const auto right_shift_amount = res_t::man_bits - shift_amount;
    const auto left_shift_amount  = shift_amount - res_t::man_bits;
    const auto extended_man       = static_cast<std::uint64_t>(x.unsigned_man());
    const auto shifted_res        = right_shift ? extended_man >> right_shift_amount : extended_man << left_shift_amount;
    const auto abs_res            = static_cast<std::int64_t>(std::uint64_t{1} << shift_amount | shifted_res);

    const auto lost_shift_amount = 64 - right_shift_amount;
    const auto lost_dman         = right_shift ? extended_man << lost_shift_amount >> lost_shift_amount : 0;

    return x.unsigned_sign() ? -abs_res - !!lost_dman : abs_res;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor_to_zero(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t    = dfloat<exp_bits_, man_bits_>;
    using int_type = typename res_t::int_type;

    const auto shift_amount = static_cast<int_type>(x.unsigned_exp()) - static_cast<int_type>(res_t::zero_exp);
    if (shift_amount < 0)
        return {};

    const auto right_shift = res_t::man_bits >= shift_amount;
    if (!right_shift)
        return x;

    const auto right_shift_amount = res_t::man_bits - shift_amount;
    return res_t::from_components(x.unsigned_sign(), x.unsigned_exp(), x.unsigned_man() >> right_shift_amount << right_shift_amount);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t         = dfloat<exp_bits_, man_bits_>;
    using int_type      = typename res_t::int_type;
    constexpr auto zero = make_dfloat<res_t>(0);
    constexpr auto one  = make_dfloat<res_t>(1);

    const auto shift_amount = static_cast<int_type>(x.unsigned_exp()) - static_cast<int_type>(res_t::zero_exp);
    if (shift_amount < 0)
        return x.unsigned_sign() ? -one : zero;

    const auto right_shift = res_t::man_bits >= shift_amount;
    if (!right_shift)
        return x;

    const auto right_shift_amount = res_t::man_bits - shift_amount;
    const auto shifted_res        = x.unsigned_man() >> right_shift_amount << right_shift_amount;
    const auto non_rounded_res    = res_t::from_components(x.unsigned_sign(), x.unsigned_exp(), shifted_res);
    return x.unsigned_sign() && x.unsigned_man() != shifted_res ? sub<execution_>(non_rounded_res, one) : non_rounded_res;
}
} // namespace dynfloat
