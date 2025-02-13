#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::int_repr() const noexcept -> typename dfloat<exponent_bits_, mantissa_bits_>::int_repr_type {
    const auto extended_exp = static_cast<int_repr_type>(unsigned_exp());
    const auto extended_man = static_cast<int_repr_type>(unsigned_man());
    return static_cast<int_repr_type>(extended_exp << man_bits | extended_man);
}

template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::from_int_repr(const bool sign, const int_repr_type in) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    return dfloat<exponent_bits_, mantissa_bits_>::from_components(sign, in >> man_bits, in);
}
} // namespace dynfloat
