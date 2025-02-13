#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename integer_, std::enable_if_t<utils::is_signed_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::operator integer_() const noexcept {
    const auto shift_amount = real_exp();

    // Special case: exponent is negative -> integer will always be zero
    {
        if (shift_amount < 0)
            return integer_{};
    }

    const auto right_shift        = man_bits >= shift_amount;
    const auto right_shift_amount = man_bits - shift_amount;
    const auto left_shift_amount  = shift_amount - man_bits;
    const auto extended_man       = static_cast<std::uint64_t>(unsigned_man());
    const auto shifted_res        = right_shift ? extended_man >> right_shift_amount : extended_man << left_shift_amount;
    const auto abs_res            = static_cast<integer_>(std::uint64_t{1} << shift_amount | shifted_res);
    const auto res                = unsigned_sign() ? -abs_res : abs_res;
    return res;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename integer_, std::enable_if_t<utils::is_unsigned_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::operator integer_() const noexcept {
    const auto shift_amount = real_exp();

    // Special case: exponent is negative -> integer will always be zero
    {
        if (shift_amount < 0)
            return integer_{};
    }

    const auto extended_man = static_cast<std::uint64_t>(unsigned_man());
    const auto shifted_res  = utils::safe_rshift(extended_man, man_bits - shift_amount);
    const auto abs_res      = static_cast<integer_>(std::uint64_t{1} << shift_amount | shifted_res);
    return abs_res;
}

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
to_ufixed(const dfloat<exp_bits_, man_bits_> lhs) noexcept -> std::uint64_t {
    using float_type        = dfloat<exp_bits_, man_bits_>;
    using int_type          = typename float_type::int_type;
    const auto shift_amount = static_cast<int_type>(lhs.unsigned_exp()) - float_type::zero_exp + scale;

    // Special case: exponent is negative -> integer will always be zero
    {
        if (shift_amount < 0)
            return 0;
    }

    const auto right_shift        = float_type::man_bits >= shift_amount;
    const auto right_shift_amount = float_type::man_bits - shift_amount;
    const auto left_shift_amount  = shift_amount - float_type::man_bits;
    const auto extended_man       = static_cast<std::uint64_t>(lhs.unsigned_man());
    const auto shifted_res        = right_shift ? extended_man >> right_shift_amount : extended_man << left_shift_amount;
    const auto abs_res            = static_cast<std::uint64_t>(std::uint64_t{1} << shift_amount | shifted_res);
    return abs_res;
}

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
to_fixed(const dfloat<exp_bits_, man_bits_> lhs) noexcept -> std::int64_t {
    return lhs.unsigned_sign() ? -static_cast<std::int64_t>(to_ufixed<scale>(lhs)) : static_cast<std::int64_t>(to_ufixed<scale>(lhs));
}

} // namespace dynfloat
