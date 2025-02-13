#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::is_zero() const noexcept -> bool {
    return int_repr() == 0;
}

template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::is_saturated() const noexcept -> bool {
    return unsigned_exp() == max_exp && unsigned_man() == max_man;
}
} // namespace dynfloat
